/*
 * utility_agent.h  -  Classical Utility-Based Agent (NON-ML AI layer)
 *
 * Shared by RF_AI.ino and SVM.ino. This is the SAME agent regardless of which
 * frozen base model (Random Forest / SVR / Decision Tree) produced the ML
 * brightness - the agent only consumes that value as a comfort anchor.
 *
 * DESIGN
 *   - Config lives in RAM as a single global (g_util_cfg). It does NOT touch
 *     the EEPROM layout used by the base-model + KNN pipeline.
 *   - utilityAgentEvaluate() is READ-ONLY w.r.t. the prediction pipeline: it
 *     takes the ML brightness, the previous applied brightness, and the raw
 *     sensor/clock inputs, and writes only to the caller-provided result.
 *   - ADVISORY by default: the .ino keeps a `util_apply` flag and only lets
 *     the recommendation drive the LED when the user types "utilon".
 *
 * OBJECTIVE  (maximize)
 *     U(b) = w_comfort*Comfort(b) - w_energy*Energy(b) - w_smooth*Smooth(b)
 *       Comfort(b) = 1 - ((b - ml_prediction)/100)^2   // anchored on ML output
 *       Energy(b)  = b/100
 *       Smooth(b)  = ((b - prev_brightness)/100)^2
 *
 * HARD CONSTRAINT  (guaranteed, applied before the argmax search)
 *     if (night AND motion)  =>  b >= min_safe_night     // default 40%
 *     night := hour in [20:00, 04:00)
 *
 * DECISION
 *     argmax over b in {0, 5, 10, ..., 100}; ties -> lower b (energy saving).
 */

#ifndef UTILITY_AGENT_H
#define UTILITY_AGENT_H

#if defined(ARDUINO)
  #include <Arduino.h>
#endif

// --- Search grid for the brightness candidate b ----------------------------
#define UTIL_B_MIN     0
#define UTIL_B_MAX     100
#define UTIL_B_STEP    5

// --- Night window for the hard safety constraint (20:00 .. 03:59) -----------
#ifndef UTIL_NIGHT_START_HOUR
#define UTIL_NIGHT_START_HOUR  20
#endif
#ifndef UTIL_NIGHT_END_HOUR
#define UTIL_NIGHT_END_HOUR    4
#endif

// ---------------------------------------------------------------------------
// Configuration (RAM only). Defaults: comfort-led, mild energy/smoothness
// penalties, 40% night-motion safety floor.
//   w_comfort / w_energy / w_smooth : objective weights (all >= 0)
//   min_safe_night                  : guaranteed floor (%) when night+motion
// ---------------------------------------------------------------------------
struct UtilityConfig {
  float w_comfort;
  float w_energy;
  float w_smooth;
  int   min_safe_night;
};

// Single global config instance, mutated by the `setutil` serial command.
static UtilityConfig g_util_cfg = { 1.0f, 0.3f, 0.5f, 40 };

// ---------------------------------------------------------------------------
// Full breakdown of one evaluation. First member is a bool so the sketch can
// brace-initialize with `UtilityResult g_util_result = { false };`.
// The *_term fields are the WEIGHTED contributions at `recommended`, so
//   utility == comfort_term - energy_term - smooth_term  (exactly).
// ---------------------------------------------------------------------------
struct UtilityResult {
  bool  valid;             // has an evaluation been run yet?
  float ml_anchor;         // ML brightness used as the comfort anchor
  float prev_brightness;   // previous applied brightness (smoothness anchor)
  float ambient;           // ambient lux at evaluation (for the printout)
  int   motion;            // motion flag (0/1) at evaluation
  int   hour;              // hour-of-day at evaluation
  bool  is_night;          // derived: hour in night window
  float recommended;       // chosen brightness b* (%)
  float comfort_raw;       // Comfort(b*)  (unweighted)
  float energy_raw;        // Energy(b*)   (unweighted)
  float smooth_raw;        // Smooth(b*)   (unweighted)
  float comfort_term;      // +w_comfort*Comfort(b*)
  float energy_term;       //  w_energy *Energy(b*)
  float smooth_term;       //  w_smooth *Smooth(b*)
  float utility;           // U(b*)
  bool  floor_active;      // night && motion (constraint in force)
  bool  floor_binding;     // floor actually raised the choice
  int   floor_value;       // the floor (%) applied
};

// --- Night test -------------------------------------------------------------
static inline bool utilityIsNight(int hour) {
  return (hour >= UTIL_NIGHT_START_HOUR) || (hour < UTIL_NIGHT_END_HOUR);
}

// --- Per-term helpers (unweighted) -----------------------------------------
static inline float utilityComfort(float b, float ml) {
  float d = (b - ml) / 100.0f;
  return 1.0f - d * d;
}
static inline float utilityEnergy(float b) {
  return b / 100.0f;
}
static inline float utilitySmooth(float b, float prev) {
  float d = (b - prev) / 100.0f;
  return d * d;
}

// ---------------------------------------------------------------------------
// utilityAgentEvaluate()
//   Constrained argmax of U(b) over the discrete grid.
//
//   INPUTS (all read-only):
//     ml_prediction   - existing base-model + KNN brightness (comfort anchor)
//     prev_brightness - the brightness applied on the previous cycle
//     ambient         - ambient lux (stored for the printout; not in U)
//     motion          - PIR motion flag (0/1)
//     hour            - hour-of-day (0..23); night is derived from this
//   OUTPUT:
//     out             - optional; filled with the full breakdown if non-null
//   RETURNS:
//     the recommended brightness b* (%). The caller decides whether to use it.
//
//   Allocates nothing; has no side effects beyond *out.
// ---------------------------------------------------------------------------
static inline float utilityAgentEvaluate(float ml_prediction,
                                         float prev_brightness,
                                         float ambient,
                                         int   motion,
                                         int   hour,
                                         UtilityResult *out) {
  bool  is_night       = utilityIsNight(hour);
  bool  motion_present = (motion != 0);
  bool  floor_active   = is_night && motion_present;
  float floor_value    = (float)g_util_cfg.min_safe_night;

  // Unconstrained optimum (for floor_binding diagnostics).
  float best_b_unc = (float)UTIL_B_MIN;
  float best_u_unc = -1e30f;

  // Constrained optimum (the actual decision).
  float best_b = -1.0f;
  float best_u = -1e30f;

  for (int bi = UTIL_B_MIN; bi <= UTIL_B_MAX; bi += UTIL_B_STEP) {
    float b = (float)bi;

    float comfort = utilityComfort(b, ml_prediction);
    float energy  = utilityEnergy(b);
    float smooth  = utilitySmooth(b, prev_brightness);

    float u = g_util_cfg.w_comfort * comfort
            - g_util_cfg.w_energy  * energy
            - g_util_cfg.w_smooth  * smooth;

    if (u > best_u_unc) { best_u_unc = u; best_b_unc = b; }

    // Feasibility: enforce the night+motion floor BEFORE selecting.
    if (floor_active && b < floor_value) continue;

    if (u > best_u) { best_u = u; best_b = b; }
  }

  // Safety net for floors that fall between grid points (e.g. 42 with step 5).
  if (floor_active && best_b < floor_value) {
    best_b = floor_value;
    if (best_b > (float)UTIL_B_MAX) best_b = (float)UTIL_B_MAX;
    best_u = g_util_cfg.w_comfort * utilityComfort(best_b, ml_prediction)
           - g_util_cfg.w_energy  * utilityEnergy(best_b)
           - g_util_cfg.w_smooth  * utilitySmooth(best_b, prev_brightness);
  }

  if (out) {
    out->valid           = true;
    out->ml_anchor       = ml_prediction;
    out->prev_brightness = prev_brightness;
    out->ambient         = ambient;
    out->motion          = motion;
    out->hour            = hour;
    out->is_night        = is_night;
    out->recommended     = best_b;
    out->comfort_raw     = utilityComfort(best_b, ml_prediction);
    out->energy_raw      = utilityEnergy(best_b);
    out->smooth_raw      = utilitySmooth(best_b, prev_brightness);
    out->comfort_term    = g_util_cfg.w_comfort * out->comfort_raw;
    out->energy_term     = g_util_cfg.w_energy  * out->energy_raw;
    out->smooth_term     = g_util_cfg.w_smooth  * out->smooth_raw;
    out->utility         = best_u;
    out->floor_active    = floor_active;
    out->floor_binding   = floor_active && (best_b_unc < floor_value);
    out->floor_value     = g_util_cfg.min_safe_night;
  }

  return best_b;
}

// ---------------------------------------------------------------------------
// utilityAgentPrint()  -  human-readable breakdown of the last evaluation.
// Lives in the header (Arduino build); a no-op stub keeps host builds happy.
// ---------------------------------------------------------------------------
#if defined(ARDUINO)
static inline void utilityAgentPrint(const UtilityResult &r) {
  Serial.println();
  Serial.println(F("=== UTILITY-BASED AGENT (non-ML AI layer) ==="));

  Serial.println(F("--- Weights (RAM only) ---"));
  Serial.print(F("  w_comfort     : ")); Serial.println(g_util_cfg.w_comfort, 3);
  Serial.print(F("  w_energy      : ")); Serial.println(g_util_cfg.w_energy, 3);
  Serial.print(F("  w_smooth      : ")); Serial.println(g_util_cfg.w_smooth, 3);
  Serial.print(F("  min_safe_night: ")); Serial.print(g_util_cfg.min_safe_night); Serial.println(F("%"));

  if (!r.valid) {
    Serial.println(F("--- No evaluation yet (waiting for first loop cycle) ---"));
    Serial.println();
    return;
  }

  Serial.println(F("--- Inputs (last cycle) ---"));
  Serial.print(F("  Hour          : ")); Serial.println(r.hour);
  Serial.print(F("  Ambient (lux) : ")); Serial.println(r.ambient, 1);
  Serial.print(F("  Motion        : ")); Serial.println(r.motion ? F("YES") : F("NO"));
  Serial.print(F("  Night (20-04) : ")); Serial.println(r.is_night ? F("YES") : F("NO"));
  Serial.print(F("  ML anchor     : ")); Serial.print(r.ml_anchor, 1); Serial.println(F("%"));
  Serial.print(F("  Prev bright.  : ")); Serial.print(r.prev_brightness, 1); Serial.println(F("%"));

  Serial.println(F("--- Safety floor ---"));
  if (!r.floor_active) {
    Serial.println(F("  Status        : not applicable (not night+motion)"));
  } else if (r.floor_binding) {
    Serial.print  (F("  Status        : ENFORCED - raised choice to >= "));
    Serial.print(r.floor_value); Serial.println(F("%"));
  } else {
    Serial.print  (F("  Status        : active, not binding (choice already >= "));
    Serial.print(r.floor_value); Serial.println(F("%)"));
  }

  Serial.println(F("--- Per-term contributions at recommended b ---"));
  Serial.print(F("  +comfort      : +")); Serial.print(r.comfort_term, 4);
  Serial.print(F("  (raw "));            Serial.print(r.comfort_raw, 4); Serial.println(F(")"));
  Serial.print(F("  -energy       : -")); Serial.print(r.energy_term, 4);
  Serial.print(F("  (raw "));            Serial.print(r.energy_raw, 4); Serial.println(F(")"));
  Serial.print(F("  -smooth       : -")); Serial.print(r.smooth_term, 4);
  Serial.print(F("  (raw "));            Serial.print(r.smooth_raw, 4); Serial.println(F(")"));
  Serial.print(F("  = utility U   : ")); Serial.println(r.utility, 4);

  Serial.println(F("--- Decision ---"));
  Serial.print(F("  ML anchor     : ")); Serial.print(r.ml_anchor, 1);  Serial.println(F("%"));
  Serial.print(F("  Recommended   : ")); Serial.print(r.recommended, 1); Serial.println(F("%"));
  Serial.print(F("  Delta vs ML   : ")); Serial.print(r.recommended - r.ml_anchor, 1); Serial.println(F("%"));
  Serial.println();
}
#else
static inline void utilityAgentPrint(const UtilityResult &) {}
#endif

#endif  // UTILITY_AGENT_H
