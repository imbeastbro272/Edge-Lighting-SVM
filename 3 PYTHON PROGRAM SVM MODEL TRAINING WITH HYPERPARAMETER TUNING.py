# ============================================================================
# SVM (SUPPORT VECTOR REGRESSION) MODEL — AMBIENT LIGHT AS INFLUENTIAL FEATURE
# Dataset : Brightness Preference Prediction (52-day dataset)
# Target  : Bulb Intensity (Preferred Brightness %)
# Split   : 80:20
# Goal    : Train an SVR locally, then EXPORT the trained model as a C++
#           header (svm_model.h) so the ESP32 firmware can run inference
#           on-device — no cloud, no Wi-Fi, no Python at runtime.
#
# Why this mirrors the Decision Tree pipeline:
#   The proven fix for the original problem (model ignoring ambient_light_lux)
#   was *feature engineering + data augmentation*, not the choice of algorithm.
#   We therefore reuse:
#       - lux_norm                 : ambient_light_lux / MAX_LUX
#       - effective_need           : (1 - lux_norm) * motion_detected
#       - hour_sin / hour_cos      : cyclic 24-hour encoding
#       - time_of_day_enc          : LabelEncoder over time-of-day strings
#       - synthetic Night+High-lux and Evening+Very-High-lux rows
#   so the SVM also learns "bright environment -> low bulb intensity".
#
# What is different for SVM:
#   1. SVMs are scale-sensitive  -> we wrap the model in a sklearn Pipeline
#      [StandardScaler -> SVR] and tune both inside GridSearchCV.
#   2. SVR has no feature_importances_ -> we use permutation_importance.
#   3. We export the fitted SVM (support vectors, dual coefficients,
#      gamma, intercept, scaler mean/scale, feature order, MAX_LUX,
#      time-of-day mapping) to svm_model.h for the ESP32.
# ============================================================================

# ============================================================================
# STEP 1: INSTALL & IMPORT LIBRARIES
# ============================================================================
# Uncomment below if running in Google Colab:
# !pip install scikit-learn pandas numpy matplotlib seaborn

import math
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("SVR (SUPPORT VECTOR REGRESSION) — EDGE-LIGHTING MODEL FOR ESP32")
print("=" * 80)

# ============================================================================
# STEP 2: LOAD DATASET
# ============================================================================
print("\n[STEP 1] Loading Dataset...")
print("-" * 80)

# --- Google Colab upload (matches the Decision Tree workflow) -----------------
# If you are running locally instead of on Colab, comment out the three Colab
# lines below and uncomment the local-path line.
from google.colab import files
print("Upload your CSV file when prompted:")
uploaded = files.upload()
csv_filename = list(uploaded.keys())[0]

# csv_filename = "brightness_dataset.csv"   # <-- local fallback
print(f"File loaded: {csv_filename}")

df = pd.read_csv(csv_filename)
print(f"\nDataset Shape : {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns       : {df.columns.tolist()}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDataset Statistics:\n{df.describe()}")

# ============================================================================
# STEP 3: ROOT CAUSE DIAGNOSIS
# ============================================================================
print("\n[STEP 2] Root Cause Diagnosis...")
print("-" * 80)

print("\n--- Feature-Target Correlations ---")
le_diag = LabelEncoder()
df_diag = df.copy()
df_diag['time_enc_diag'] = le_diag.fit_transform(df_diag['time of day'])
numeric_cols = ['ambient_light_lux', 'motion_detected',
                'hour_sin', 'hour_cos', 'time_enc_diag']
corr = (df_diag[numeric_cols + ['Bulb Intensity']]
        .corr()['Bulb Intensity']
        .drop('Bulb Intensity'))
for feat, val in corr.items():
    print(f"  {feat:<25} -> correlation with Bulb Intensity: {val:+.4f}")

print("\n--- Lux Coverage by Time of Day ---")
print(df.groupby('time of day')['ambient_light_lux']
        .describe()[['count', 'min', 'max', 'mean']].to_string())

night_high = ((df['time of day'] == 'Night') &
              (df['ambient_light_lux'] > 1000)).sum()
print(f"\n  Night + lux > 1000 samples: {night_high}")
print(f"  -> If this is 0, the raw model never sees 'bright + night'.")

print("\n--- Mean Bulb Intensity by Motion ---")
print(df.groupby('motion_detected')['Bulb Intensity'].mean().to_string())

# ============================================================================
# STEP 4: FEATURE ENGINEERING (same fix as the DT pipeline)
# ============================================================================
print("\n[STEP 3] Feature Engineering...")
print("-" * 80)

data = df.copy()

MAX_LUX = float(data['ambient_light_lux'].max())
print(f"\n  MAX_LUX (used for lux_norm)  : {MAX_LUX:.2f}")

data['lux_norm']       = data['ambient_light_lux'] / MAX_LUX
data['effective_need'] = (1 - data['lux_norm']) * data['motion_detected']

print("  New features:")
print("    lux_norm       = ambient_light_lux / MAX_LUX  in [0, 1]")
print("    effective_need = (1 - lux_norm) * motion_detected")
print("        -> bright + motion -> low effective_need -> low intensity")
print("        -> dark   + motion -> high effective_need -> bright bulb")
print("        -> no motion       -> 0                    -> bulb off")

le = LabelEncoder()
data['time_of_day_enc'] = le.fit_transform(data['time of day'])
TIME_OF_DAY_MAP = dict(zip(le.classes_, [int(v) for v in le.transform(le.classes_)]))
print(f"\n  Time of Day Encoding: {TIME_OF_DAY_MAP}")

# ============================================================================
# STEP 5: DATA AUGMENTATION (high-lux edge cases)
# ============================================================================
print("\n[STEP 4] Data Augmentation for High-Lux Edge Cases...")
print("-" * 80)

n_synth = 400

hours_night = np.random.choice(list(range(20, 24)) + list(range(0, 5)),
                               n_synth // 2)
synth_night = pd.DataFrame({
    'ambient_light_lux': np.random.uniform(800, 15000, n_synth // 2),
    'motion_detected'  : np.random.choice([0, 1], n_synth // 2),
    'time of day'      : 'Night',
    'hour_sin'         : [math.sin(2 * math.pi * h / 24) for h in hours_night],
    'hour_cos'         : [math.cos(2 * math.pi * h / 24) for h in hours_night],
    'Bulb Intensity'   : np.random.uniform(2, 15, n_synth // 2),
})

hours_eve = np.random.randint(17, 21, n_synth // 2)
synth_eve = pd.DataFrame({
    'ambient_light_lux': np.random.uniform(5000, 45000, n_synth // 2),
    'motion_detected'  : np.random.choice([0, 1], n_synth // 2),
    'time of day'      : 'Evening',
    'hour_sin'         : [math.sin(2 * math.pi * h / 24) for h in hours_eve],
    'hour_cos'         : [math.cos(2 * math.pi * h / 24) for h in hours_eve],
    'Bulb Intensity'   : np.random.uniform(0, 10, n_synth // 2),
})

df_aug = pd.concat([data, synth_night, synth_eve], ignore_index=True)
df_aug['lux_norm']        = df_aug['ambient_light_lux'] / MAX_LUX
df_aug['effective_need']  = (1 - df_aug['lux_norm']) * df_aug['motion_detected']
df_aug['time_of_day_enc'] = le.transform(df_aug['time of day'])

print(f"  Original rows : {len(data)}")
print(f"  Synthetic rows: {n_synth}")
print(f"  Augmented rows: {len(df_aug)}")

# ============================================================================
# STEP 6: FEATURES & TARGET
# ============================================================================
print("\n[STEP 5] Preparing Features & Target...")
print("-" * 80)

features_final = ['lux_norm', 'motion_detected', 'hour_sin', 'hour_cos',
                  'time_of_day_enc', 'effective_need']
target = 'Bulb Intensity'

print("  Final feature order (THIS ORDER MUST BE MIRRORED ON THE ESP32):")
for i, f in enumerate(features_final, 1):
    print(f"    {i}. {f}")

X = df_aug[features_final].astype(float)
y = df_aug[target].astype(float)
print(f"\n  X shape: {X.shape}, y shape: {y.shape}")
print(f"  Missing values: {X.isnull().sum().sum()}")

# ============================================================================
# STEP 7: TRAIN-TEST SPLIT (80:20)
# ============================================================================
print("\n[STEP 6] Train-Test Split (80:20)...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set : {X_test.shape[0]} samples")

# ============================================================================
# STEP 8: PIPELINE + HYPERPARAMETER TUNING (StandardScaler + SVR)
# ============================================================================
print("\n[STEP 7] Hyperparameter Tuning (GridSearchCV over SVR)...")
print("-" * 80)

# StandardScaler is REQUIRED for SVMs — the kernel distance is meaningless
# if features are on wildly different scales. We embed it in a Pipeline so
# the same scaler is used at fit time, predict time, AND export time.
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr',    SVR()),
])

param_grid = [
    # RBF kernel — usually the best for non-linear regression problems
    {
        'svr__kernel' : ['rbf'],
        'svr__C'      : [1, 10, 50, 100, 200],
        'svr__gamma'  : ['scale', 0.05, 0.1, 0.3, 1.0],
        'svr__epsilon': [0.1, 0.5, 1.0, 2.0],
    },
    # Linear kernel — included as a fallback. If it wins, the exported header
    # collapses to a single weight vector (smallest possible footprint).
    {
        'svr__kernel' : ['linear'],
        'svr__C'      : [0.1, 1, 10, 50],
        'svr__epsilon': [0.1, 0.5, 1.0, 2.0],
    },
]

grid_search = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train, y_train)

best_pipe   = grid_search.best_estimator_
best_params = grid_search.best_params_
best_svr    = best_pipe.named_steps['svr']
best_scaler = best_pipe.named_steps['scaler']

print(f"\n  Best Parameters     : {best_params}")
print(f"  Best CV neg-MSE     : {grid_search.best_score_:.4f}")
print(f"  Selected kernel     : {best_svr.kernel}")
print(f"  Number of support vectors: {best_svr.support_vectors_.shape[0]}")

# ============================================================================
# STEP 9: EVALUATION
# ============================================================================
print("\n[STEP 8] Model Evaluation...")
print("-" * 80)

y_pred = best_pipe.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

verdict = ('Excellent' if r2 > 0.95 else
           'Good'      if r2 > 0.85 else 'Acceptable')
print(f"  R^2  : {r2:.4f}  ({verdict})")
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")

cv_scores = cross_val_score(best_pipe, X, y, cv=5, scoring='r2')
print(f"\n  5-Fold Cross-Validation R^2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# ============================================================================
# STEP 10: PERMUTATION FEATURE IMPORTANCE (SVM has no feature_importances_)
# ============================================================================
print("\n[STEP 9] Permutation Feature Importance...")
print("-" * 80)

perm = permutation_importance(
    best_pipe, X_test, y_test,
    n_repeats=20, random_state=42, n_jobs=-1, scoring='r2',
)

feat_imp_df = (pd.DataFrame({
    'Feature'   : features_final,
    'Importance': perm.importances_mean,
    'Std'       : perm.importances_std,
}).sort_values('Importance', ascending=False).reset_index(drop=True))

# Normalise to make the bar chart comparable to the DT one (sums to 1)
total = feat_imp_df['Importance'].clip(lower=0).sum()
if total > 0:
    feat_imp_df['ImportanceNorm'] = feat_imp_df['Importance'].clip(lower=0) / total
else:
    feat_imp_df['ImportanceNorm'] = 0.0

print("\n  Permutation Importance (drop in R^2 when feature is shuffled):")
for _, row in feat_imp_df.iterrows():
    bar = '#' * int(max(row['ImportanceNorm'], 0) * 40)
    print(f"    {row['Feature']:<22} {row['Importance']:+.4f} +/- {row['Std']:.4f}  {bar}")

plt.figure(figsize=(10, 5))
sns.barplot(data=feat_imp_df, x='ImportanceNorm', y='Feature', palette='viridis')
plt.title('Permutation Feature Importance — SVR Model', fontsize=13)
plt.xlabel('Normalised Importance (drop in R^2)')
plt.tight_layout()
plt.savefig('svm_feature_importance.png', dpi=150)
plt.show()
print("  Saved: svm_feature_importance.png")

# ============================================================================
# STEP 11: VERIFY THE FIX — TEST SCENARIOS
# ============================================================================
print("\n[STEP 10] Verifying the Fix — Test Scenarios...")
print("-" * 80)

def predict_scenario(lux_raw, motion, hour, time_of_day_str):
    """Predict bulb intensity (%) for a single scenario."""
    lux_n = lux_raw / MAX_LUX
    hs    = math.sin(2 * math.pi * hour / 24)
    hc    = math.cos(2 * math.pi * hour / 24)
    enc   = int(le.transform([time_of_day_str])[0])
    eff   = (1 - lux_n) * motion
    row   = pd.DataFrame([[lux_n, motion, hs, hc, enc, eff]],
                         columns=features_final)
    return float(best_pipe.predict(row)[0])

scenarios = [
    (1100,  1, 22, 'Night',   '< 15%',   'bright + night + motion -> bulb low'),
    (5000,  1, 22, 'Night',   '< 10%',   'very bright at night -> bulb off'),
    (10000, 1, 22, 'Night',   '< 10%',   'street-lit at night -> bulb off'),
    (0,     1, 22, 'Night',   '70-90%',  'dark + motion -> bulb bright'),
    (0,     0, 22, 'Night',   '< 10%',   'dark + no motion -> bulb off'),
    (7000,  1, 10, 'Morning', '< 10%',   'bright morning -> no bulb'),
    (200,   1, 10, 'Morning', '50-90%',  'low light + motion -> bulb bright'),
]
print("\n  +---------------------------------------------------------------------+")
print("  | SCENARIO                                EXPECTED   PREDICTED        |")
print("  +---------------------------------------------------------------------+")
for lux, mot, hr, tod, expected, note in scenarios:
    pred = predict_scenario(lux, mot, hr, tod)
    print(f"  | lux={lux:<6} motion={mot} {tod:<14} {expected:<10} {pred:>5.1f}%   {note}")
print("  +---------------------------------------------------------------------+")

# ============================================================================
# STEP 12: ACTUAL vs PREDICTED PLOT
# ============================================================================
print("\n[STEP 11] Actual vs Predicted Plot...")
print("-" * 80)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.4,
            edgecolors='steelblue', facecolors='none', s=40)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Bulb Intensity (%)')
plt.ylabel('Predicted Bulb Intensity (%)')
plt.title(f'Actual vs Predicted — SVR ({best_svr.kernel} kernel, R^2 = {r2:.4f})')
plt.legend()
plt.tight_layout()
plt.savefig('svm_actual_vs_predicted.png', dpi=150)
plt.show()
print("  Saved: svm_actual_vs_predicted.png")

# ============================================================================
# STEP 13: MANUAL PREDICTION HELPER (Python side, for sanity-checking ESP32)
# ============================================================================
print("\n[STEP 12] Manual Prediction Helper...")
print("-" * 80)
print(textwrap.dedent("""
    Use the helper below in Colab to test any scenario from Python.
    The exported svm_model.h MUST produce the SAME prediction on the ESP32
    (within ~1e-3 due to float vs double).

        def predict_bulb_intensity(lux_raw, motion, hour, time_of_day_str):
            lux_n = lux_raw / MAX_LUX
            hs    = math.sin(2 * math.pi * hour / 24)
            hc    = math.cos(2 * math.pi * hour / 24)
            enc   = int(le.transform([time_of_day_str])[0])
            eff   = (1 - lux_n) * motion
            row   = pd.DataFrame([[lux_n, motion, hs, hc, enc, eff]],
                                 columns=features_final)
            return float(best_pipe.predict(row)[0])

        result = predict_bulb_intensity(1100, 1, 22, 'Night')
        print(f"Predicted intensity: {result:.1f}%")   # should be < 15
"""))

# ============================================================================
# STEP 14: EXPORT TRAINED MODEL TO svm_model.h (FOR ESP32)
# ============================================================================
print("\n[STEP 13] Exporting trained SVM to svm_model.h ...")
print("-" * 80)


def _format_float_array_1d(arr, per_line=8, indent=4):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    pad = ' ' * indent
    out = []
    for i in range(0, len(arr), per_line):
        chunk = ', '.join(f'{v:+.8e}f' for v in arr[i:i + per_line])
        out.append(pad + chunk + ',')
    if out:
        out[-1] = out[-1].rstrip(',')
    return '\n'.join(out)


def _format_float_array_2d(arr, indent=4):
    arr = np.asarray(arr, dtype=np.float32)
    pad = ' ' * indent
    rows = []
    for i, row in enumerate(arr):
        vals = ', '.join(f'{v:+.8e}f' for v in row)
        suffix = ',' if i < len(arr) - 1 else ''
        rows.append(f'{pad}{{ {vals} }}{suffix}')
    return '\n'.join(rows)


def export_svm_to_header(pipeline, max_lux, label_encoder,
                         feature_names, output_path):
    """Serialise the fitted Pipeline(StandardScaler, SVR) to a C++ header."""
    scaler = pipeline.named_steps['scaler']
    svr    = pipeline.named_steps['svr']

    kernel    = svr.kernel
    sv        = svr.support_vectors_              # (n_sv, n_feat)  scaled space
    dual      = svr.dual_coef_.ravel()            # (n_sv,)
    intercept = float(svr.intercept_[0])
    # sklearn stores the resolved gamma (e.g. when gamma='scale') in _gamma
    gamma     = float(getattr(svr, '_gamma', svr.gamma))

    mean  = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    n_sv, n_feat = sv.shape

    tod_map = {str(c): int(v) for c, v in zip(label_encoder.classes_,
                                              label_encoder.transform(label_encoder.classes_))}

    # If the kernel is linear we collapse the SVs into a single weight vector w
    # in scaled-feature space. Inference becomes a 6-multiply dot product.
    if kernel == 'linear':
        w = (dual @ sv).astype(np.float32)        # shape (n_feat,)
    else:
        w = None

    header = []
    header.append('/' + '*' * 78)
    header.append(' * svm_model.h  -  AUTO-GENERATED by')
    header.append(' *   "3 PYTHON PROGRAM SVM MODEL TRAINING WITH HYPERPARAMETER TUNING.py"')
    header.append(' *')
    header.append(' * DO NOT EDIT BY HAND. Re-run the training script to regenerate.')
    header.append(' *')
    header.append(f' * Kernel              : {kernel}')
    header.append(f' * Support vectors     : {n_sv}')
    header.append(f' * Features            : {n_feat}  (order shown below)')
    header.append(f' * MAX_LUX             : {max_lux:.6f}')
    header.append(f' * Time-of-day mapping : {json.dumps(tod_map)}')
    header.append(' *')
    header.append(' * Feature order (MUST match on ESP32):')
    for i, f in enumerate(feature_names):
        header.append(f' *   [{i}] {f}')
    header.append(' ' + '*' * 77 + '/')
    header.append('')
    header.append('#ifndef SVM_MODEL_H')
    header.append('#define SVM_MODEL_H')
    header.append('')
    header.append('#include <math.h>')
    header.append('')
    header.append(f'#define SVM_N_FEATURES   {n_feat}')
    header.append(f'#define SVM_N_SV         {n_sv}')
    header.append(f'#define SVM_KERNEL_RBF    {1 if kernel == "rbf" else 0}')
    header.append(f'#define SVM_KERNEL_LINEAR {1 if kernel == "linear" else 0}')
    header.append(f'static const float SVM_GAMMA      = {gamma:+.8e}f;')
    header.append(f'static const float SVM_INTERCEPT  = {intercept:+.8e}f;')
    header.append(f'static const float SVM_MAX_LUX    = {max_lux:+.8e}f;')
    header.append('')
    # Time-of-day mapping as preprocessor constants
    header.append('// Time-of-day encoding (LabelEncoder result, alphabetic order)')
    for name, idx in sorted(tod_map.items(), key=lambda kv: kv[1]):
        macro = 'TOD_' + ''.join(ch.upper() if ch.isalnum() else '_' for ch in name)
        header.append(f'#define {macro:<20} {idx}')
    header.append('')

    # Scaler params
    header.append('// StandardScaler parameters: x_scaled = (x - mean) / scale')
    header.append('static const float SVM_MEAN[SVM_N_FEATURES] = {')
    header.append(_format_float_array_1d(mean))
    header.append('};')
    header.append('static const float SVM_SCALE[SVM_N_FEATURES] = {')
    header.append(_format_float_array_1d(scale))
    header.append('};')
    header.append('')

    if kernel == 'linear':
        header.append('// Linear kernel collapses the SVs into a single weight vector')
        header.append('static const float SVM_W[SVM_N_FEATURES] = {')
        header.append(_format_float_array_1d(w))
        header.append('};')
    else:
        header.append('// Support vectors (already in scaled feature space)')
        header.append('static const float SVM_SV[SVM_N_SV][SVM_N_FEATURES] = {')
        header.append(_format_float_array_2d(sv))
        header.append('};')
        header.append('static const float SVM_DUAL_COEF[SVM_N_SV] = {')
        header.append(_format_float_array_1d(dual))
        header.append('};')
    header.append('')

    # Inline predict() — minimal, no dynamic allocation, ESP32-friendly
    header.append('// ---------------------------------------------------------------------------')
    header.append('// svm_predict_raw()')
    header.append('//   feats_raw must be in the EXACT order shown at the top of this file:')
    header.append('//     [lux_norm, motion_detected, hour_sin, hour_cos,')
    header.append('//      time_of_day_enc, effective_need]')
    header.append('//   Returns predicted bulb intensity (clamp to [0,100] in the caller).')
    header.append('// ---------------------------------------------------------------------------')
    header.append('static inline float svm_predict_raw(const float feats_raw[SVM_N_FEATURES]) {')
    header.append('    float x[SVM_N_FEATURES];')
    header.append('    for (int i = 0; i < SVM_N_FEATURES; ++i) {')
    header.append('        x[i] = (feats_raw[i] - SVM_MEAN[i]) / SVM_SCALE[i];')
    header.append('    }')
    if kernel == 'linear':
        header.append('    float y = SVM_INTERCEPT;')
        header.append('    for (int i = 0; i < SVM_N_FEATURES; ++i) {')
        header.append('        y += SVM_W[i] * x[i];')
        header.append('    }')
        header.append('    return y;')
    else:
        header.append('    float y = SVM_INTERCEPT;')
        header.append('    for (int s = 0; s < SVM_N_SV; ++s) {')
        header.append('        float d2 = 0.0f;')
        header.append('        for (int i = 0; i < SVM_N_FEATURES; ++i) {')
        header.append('            float d = x[i] - SVM_SV[s][i];')
        header.append('            d2 += d * d;')
        header.append('        }')
        header.append('        y += SVM_DUAL_COEF[s] * expf(-SVM_GAMMA * d2);')
        header.append('    }')
        header.append('    return y;')
    header.append('}')
    header.append('')

    # Convenience wrapper that mirrors the DT.ino predict_brightness() signature
    header.append('// ---------------------------------------------------------------------------')
    header.append('// svm_predict_brightness()')
    header.append('//   Drop-in replacement for the DT.ino predict_brightness() helper.')
    header.append('//     ambient_lux : raw LDR reading converted to lux')
    header.append('//     motion      : 0 or 1 (PIR)')
    header.append('//     sin_h, cos_h: cyclic hour encoding (sin/cos of 2*PI*hour/24)')
    header.append('//     period      : LabelEncoder index for the time-of-day string')
    header.append('//   Returns clamped bulb intensity in [0, 100].')
    header.append('// ---------------------------------------------------------------------------')
    header.append('static inline float svm_predict_brightness(float ambient_lux,')
    header.append('                                           int   motion,')
    header.append('                                           float sin_h,')
    header.append('                                           float cos_h,')
    header.append('                                           int   period) {')
    header.append('    float lux_norm       = ambient_lux / SVM_MAX_LUX;')
    header.append('    if (lux_norm < 0.0f) lux_norm = 0.0f;')
    header.append('    if (lux_norm > 1.0f) lux_norm = 1.0f;')
    header.append('    float effective_need = (1.0f - lux_norm) * (float)motion;')
    header.append('    float feats[SVM_N_FEATURES] = {')
    header.append('        lux_norm,')
    header.append('        (float)motion,')
    header.append('        sin_h,')
    header.append('        cos_h,')
    header.append('        (float)period,')
    header.append('        effective_need')
    header.append('    };')
    header.append('    float y = svm_predict_raw(feats);')
    header.append('    if (y < 0.0f)   y = 0.0f;')
    header.append('    if (y > 100.0f) y = 100.0f;')
    header.append('    return y;')
    header.append('}')
    header.append('')
    header.append('#endif  // SVM_MODEL_H')
    header.append('')

    Path(output_path).write_text('\n'.join(header))
    return output_path


out_path = export_svm_to_header(
    pipeline       = best_pipe,
    max_lux        = MAX_LUX,
    label_encoder  = le,
    feature_names  = features_final,
    output_path    = 'svm_model.h',
)
print(f"  Wrote: {out_path}")
print(f"  Header size: {Path(out_path).stat().st_size / 1024:.1f} KB")
print( "  Drop this file next to your .ino and #include \"svm_model.h\"")
print( "  Then call:  float pct = svm_predict_brightness(lux, motion, sin_h, cos_h, period);")

# Sanity check: Python prediction must match what ESP32 will compute.
# We re-implement svm_predict_raw() in pure Python and compare to sklearn.
def _python_reference_predict(row_raw):
    x = (np.asarray(row_raw, dtype=np.float64) - best_scaler.mean_) / best_scaler.scale_
    if best_svr.kernel == 'linear':
        w = (best_svr.dual_coef_.ravel() @ best_svr.support_vectors_)
        return float(w @ x + best_svr.intercept_[0])
    gamma = getattr(best_svr, '_gamma', best_svr.gamma)
    diff = best_svr.support_vectors_ - x
    k = np.exp(-gamma * np.einsum('ij,ij->i', diff, diff))
    return float((best_svr.dual_coef_.ravel() * k).sum() + best_svr.intercept_[0])

sample = X_test.iloc[0].values
sk_pred  = float(best_pipe.predict(X_test.iloc[[0]])[0])
ref_pred = _python_reference_predict(sample)
print(f"\n  Sanity check (must match within ~1e-5):")
print(f"    sklearn pipeline predict : {sk_pred:.6f}")
print(f"    pure-python ESP32 predict: {ref_pred:.6f}")
print(f"    abs diff                 : {abs(sk_pred - ref_pred):.2e}")

# ============================================================================
# DONE
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE — svm_model.h is ready for ESP32 deployment")
print("=" * 80)
print(f"  Kernel              : {best_svr.kernel}")
print(f"  Support vectors     : {best_svr.support_vectors_.shape[0]}")
print(f"  R^2                 : {r2:.4f}")
print(f"  MAE                 : {mae:.4f}")
print(f"  RMSE                : {rmse:.4f}")
print(f"  CV R^2              : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
print(f"  Header file written : svm_model.h")
