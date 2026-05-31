/*
 * utility_agent.h  -  Classical Utility-Based Agent (NON-ML AI layer)
 *
 * PURPOSE
 *   A small, self-contained decision layer that sits ON TOP of the existing
 *   SVM + KNN brightness pipeline. It performs constrained utility
 *   maximization to recommend an LED brightness, balancing three competing
 *   goals (comfort, energy, smoothness) while GUARANTEEING a safety floor at
 *   night when motion is present.
 *
 *   This header is intentionally:
 *     - PURE C++  : no Arduino.h, no Serial, no EEPROM, no globals.
 *                   It can be unit-tested on a host PC with plain g++.
 *     - READ-ONLY : utility_evaluate() takes inputs by value and writes only
 *                   to the caller-provided UtilityResult. It NEVER touches the
 *                   prediction pipeline, the model header, or persistent state.
 *     - ADVISORY  : it only computes a recommendation. Whether that
 *                   recommendation actually drives the LED is decided by the
 *                   caller (the .ino sketch) via its own `util_apply` flag.
 *
 * OBJECTIVE  (maximize)
 *     U(b) = w_comfort*Comfort(b) - w_energy*Energy(b) - w_smooth*Smooth(b)
 *       Comfort(b) = 1 - ((b - ml_prediction)/100)^2   // anchored on ML output
 *       Energy(b)  = b/100
 *       Smooth(b)  = ((b - prev_brightness)/100)^2
 *
 * HARD CONSTRAINT  (guaranteed, applied before the argmax search)
 *     if (is_night && motion)  =>  b >= min_safe_night   // default 40%
 *
 * DECISION
 *     argmax over b in {0, 5, 10, ..., 100}.
 *     Ties are broken toward the LOWER brightness (favours energy saving).
 */

#ifndef UTILITY_AGENT_H
#define UTILITY_AGENT_H

// ---------------------------------------------------------------------------
// Search grid for the brightness candidate b.
// ---------------------------------------------------------------------------
#define UTIL_B_MIN     0
#define UTIL_B_MAX     100
#define UTIL_B_STEP    5

// ---------------------------------------------------------------------------
// Configuration (lives in RAM on the caller side; defaults below).
//   w_comfort / w_energy / w_smooth : objective weights (all >= 0)
//   min_safe_night                  : guaranteed floor (%) when night+motion
// ---------------------------------------------------------------------------
struct UtilityConfig {
  float w_comfort;
  float w_energy;
  float w_smooth;
  float min_safe_night;
};

// ---------------------------------------------------------------------------
// Full breakdown of a single evaluation, filled in by utility_evaluate().
// All term fields are the WEIGHTED contributions evaluated at `recommended`,
// so utility == comfort_term - energy_term - smooth_term exactly.
// ---------------------------------------------------------------------------
struct UtilityResult {
  float recommended;       // chosen brightness b* (%)
  float ml_anchor;         // ml_prediction that was passed in (comfort anchor)
  float prev_brightness;   // prev_brightness that was passed in
  float comfort_raw;       // Comfort(b*)            (unweighted)
  float energy_raw;        // Energy(b*)             (unweighted)
  float smooth_raw;        // Smooth(b*)             (unweighted)
  float comfort_term;      // +w_comfort*Comfort(b*)
  float energy_term;       //  w_energy *Energy(b*)  (subtracted in utility)
  float smooth_term;       //  w_smooth *Smooth(b*)  (subtracted in utility)
  float utility;           // U(b*)
  bool  is_night;          // night flag that was passed in
  bool  motion;            // motion flag that was passed in
  bool  floor_active;      // true if the night+motion floor was in force
  float floor_value;       // the floor (%) that was applied (cfg.min_safe_night)
  bool  floor_binding;     // true if the floor actually raised the choice
                           // (i.e. the unconstrained argmax was below floor)
};

// ---------------------------------------------------------------------------
// Sensible defaults (confirmed): comfort-led, mild energy/smoothness penalties,
// 40% night-motion safety floor.
// ---------------------------------------------------------------------------
static inline UtilityConfig utility_default_config() {
  UtilityConfig c;
  c.w_comfort      = 1.0f;
  c.w_energy       = 0.3f;
  c.w_smooth       = 0.5f;
  c.min_safe_night = 40.0f;
  return c;
}

// ---------------------------------------------------------------------------
// Per-term helpers (unweighted). Kept tiny and branch-free.
// ---------------------------------------------------------------------------
static inline float utility_comfort(float b, float ml_prediction) {
  float d = (b - ml_prediction) / 100.0f;
  return 1.0f - d * d;
}
static inline float utility_energy(float b) {
  return b / 100.0f;
}
static inline float utility_smooth(float b, float prev_brightness) {
  float d = (b - prev_brightness) / 100.0f;
  return d * d;
}

// ---------------------------------------------------------------------------
// utility_evaluate()
//   Constrained argmax of U(b) over the discrete grid.
//
//   INPUTS (all read-only):
//     cfg             - weights + night floor
//     ml_prediction   - the existing SVM+KNN brightness (comfort anchor)
//     prev_brightness - the brightness from the previous cycle (smoothness)
//     is_night        - caller-computed night flag
//     motion          - caller-computed motion flag (PIR)
//   OUTPUT:
//     out             - optional; if non-null, filled with the full breakdown
//   RETURNS:
//     the recommended brightness b* (%). The caller decides whether to use it.
//
//   This function allocates nothing and has no side effects beyond *out.
// ---------------------------------------------------------------------------
static inline float utility_evaluate(const UtilityConfig &cfg,
                                     float ml_prediction,
                                     float prev_brightness,
                                     bool  is_night,
                                     bool  motion,
                                     UtilityResult *out) {
  // Hard constraint: restrict the feasible domain BEFORE searching.
  bool  floor_active = (is_night && motion);
  float floor_value  = cfg.min_safe_night;

  // --- Unconstrained argmax (for floor_binding diagnostics) ---
  float best_b_unc   = (float)UTIL_B_MIN;
  float best_u_unc   = -1e30f;

  // --- Constrained argmax (the actual decision) ---
  float best_b       = -1.0f;
  float best_u       = -1e30f;

  for (int bi = UTIL_B_MIN; bi <= UTIL_B_MAX; bi += UTIL_B_STEP) {
    float b = (float)bi;

    float comfort = utility_comfort(b, ml_prediction);
    float energy  = utility_energy(b);
    float smooth  = utility_smooth(b, prev_brightness);

    float u = cfg.w_comfort * comfort
            - cfg.w_energy  * energy
            - cfg.w_smooth  * smooth;

    // Track the unconstrained optimum (tie -> lower b, since we use '>').
    if (u > best_u_unc) {
      best_u_unc = u;
      best_b_unc = b;
    }

    // Feasibility check for the constrained optimum.
    if (floor_active && b < floor_value) {
      continue;  // infeasible: below the guaranteed night-motion floor
    }

    if (u > best_u) {
      best_u = u;
      best_b = b;
    }
  }

  // Safety net: if the floor sits between grid points (e.g. floor=42 with a
  // step of 5) no candidate may satisfy b >= floor exactly. In that case we
  // clamp the recommendation up to the floor to keep the guarantee absolute.
  if (floor_active && (best_b < floor_value)) {
    best_b = floor_value;
    if (best_b > (float)UTIL_B_MAX) best_b = (float)UTIL_B_MAX;
    best_u = cfg.w_comfort * utility_comfort(best_b, ml_prediction)
           - cfg.w_energy  * utility_energy(best_b)
           - cfg.w_smooth  * utility_smooth(best_b, prev_brightness);
  }

  if (out) {
    out->recommended     = best_b;
    out->ml_anchor       = ml_prediction;
    out->prev_brightness = prev_brightness;
    out->comfort_raw     = utility_comfort(best_b, ml_prediction);
    out->energy_raw      = utility_energy(best_b);
    out->smooth_raw      = utility_smooth(best_b, prev_brightness);
    out->comfort_term    = cfg.w_comfort * out->comfort_raw;
    out->energy_term     = cfg.w_energy  * out->energy_raw;
    out->smooth_term     = cfg.w_smooth  * out->smooth_raw;
    out->utility         = best_u;
    out->is_night        = is_night;
    out->motion          = motion;
    out->floor_active    = floor_active;
    out->floor_value     = floor_value;
    out->floor_binding   = floor_active && (best_b_unc < floor_value);
  }

  return best_b;
}

#endif  // UTILITY_AGENT_H
