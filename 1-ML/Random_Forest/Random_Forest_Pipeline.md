# Random Forest Pipeline - SOH Estimation

## Overview

Two Random Forest pipelines for SOH estimation, differing only in feature selection strategy:

| File | Feature Strategy | Feature Count | Output Folder |
|---|---|---|---|
| `Random_Forest.py` | All features (RF selects internally) | ~200 | `Output_Random_Forest/` |
| `Random_Forest_Top30.py` | Top 30 features by SOH correlation | 30 | `Output_Random_Forest_Top30/` |

Steps 1-6 are identical. Steps 7-9 differ in feature handling.

## Data

- **Source:** `recording_senaryo_1_Battery_A_N100.mat` (Simulink Recording)
- **Sampling rate:** 1 Hz
- **Signals:** SOC, Current, Cell_V_Max/Min/Avg, Cell_T_Max/Min/Avg, SOH
- **Cycles:** ~100 (SOH: 100% -> 80%)

---

## Pipeline Steps

### STEP 1: Raw Signal Visualization (EDA)

The .mat file is read in HDF5 format. 9 signals are extracted into a DataFrame (1 Hz, each row = 1 second). Raw signals are visualized in 5 subplots: SOC, Current, Voltage (Max & Min), Temperature (Max & Min), SOH.

**Output:** `step1_raw_signals.png`

---

### STEP 2: mV -> V Conversion

Simscape voltage signals come in millivolts. Cell_V_Max, Cell_V_Min, Cell_V_Avg are divided by 1000 to convert to Volts. The converted signals are visualized.

**Output:** `step2_volt_conversion.png`

---

### STEP 3: Cycle Boundary Detection

1 cycle = full discharge (SOC 100% -> 10%) + full charge (SOC 10% -> 100%).

**Method:** Peak points in the SOC signal (>99%) are detected. A minimum distance filter (>5000 seconds) prevents false peaks. Peak-to-peak defines 1 cycle.

**Definitions:**
- **Active period:** Moments where |Current| > 0.5A (discharge + charge)
- **Rest period:** Moments where |Current| <= 0.5A

**Outputs:** `df_step3_cycle_boundaries.csv`, `step3_cycle_analysis.png` (Cycle vs SOH + Cycle vs Duration)

---

### STEP 4: Cycle-Level Summarization (Feature Extraction)

2.3M rows of 1 Hz data -> ~100 rows (1 row per cycle). Three categories of features are computed:

#### A) Raw Signal Statistics (56 features)

8 signals x 7 statistics: min, max, mean, std, range, discharge_end, charge_start.

#### B) Domain-Specific Features (9 feature groups)

| # | Feature | Calculation | Includes Rest? |
|---|---------|-------------|----------------|
| 1 | **Voltage Spread** (mean/max/std) | `Cell_V_Max - Cell_V_Min` per second, cycle average | Yes |
| 2 | **Voltage Spread Under Load** | Same, only active moments (|Current| > 0.5A) | No |
| 3 | **Voltage Asymmetry** (mean) | `(V_Max - V_Avg) / (V_Avg - V_Min + 1e-3)` | Yes |
| 4 | **Voltage Asymmetry Under Load** | Same, only active moments | No |
| 5 | **Thermal Spread** (mean/max/std) | `Cell_T_Max - Cell_T_Min` per second | Yes |
| 6 | **Thermal Spread Under Load** | Same, only active moments | No |
| 7 | **Proxy Internal Resistance** (mean/median/std) | `|dV/dI|` when |dI| > 2A, filtered < 1 Ohm | No |
| 8 | **Energy Efficiency** | discharge energy / charge energy (|Current| > 0.5A) | No |
| 9 | **Thermal Response Ratio** | 60-second windows: `d(T_spread) / d(|Current|)` | No |

#### C) Timing Features (9 features)

charge_time_to_50/80/95/full, discharge_duration, soc_range, cycle_duration, active_duration, rest_duration.

**Total:** ~85 features per cycle.

**Outputs:** `df_step4_cycle_features.csv`, `step4_domain_features.png` (3x3 subplot)

---

### STEP 5: Trend Features

Tree-based models (RF, XGBoost) treat each row independently -- they have no memory of previous cycles. Trend features provide "trajectory" information so the model knows whether a value is increasing, decreasing, or stable.

**For 17 selected domain features, the following are computed:**

| Trend Type | Description | Count per Feature |
|---|---|---|
| Rolling Mean (3, 5, 10) | Last N cycle average (noise smoothing) | 3 |
| Rolling Std (3, 5, 10) | Last N cycle volatility | 3 |
| Delta | Difference from previous cycle | 1 |
| Rate of Change | Average change over 2 cycles | 1 |

**Additional:** `proxy_IR_trend_slope_10` -- Linear slope of proxy_IR_mean over the last 10 cycles.

**Why not compute trends for all 85 features?**
~100 rows / 680+ features = overfitting risk. Only domain features most correlated with aging are selected.

**Result:** ~85 features -> ~205 features

**Outputs:** `df_step5_trend_features.csv`, `step5_trend_features.png`

---

### STEP 6: Correlation Analysis

Pearson correlation of all ~205 features with SOH is computed. Constant features (std=0) are removed first. All feature correlations are printed in ranked order and saved to CSV.

**Outputs:**
- `df_step6_correlation.csv` (all features with SOH correlation values)
- `step6_correlation.png` (Top 30 feature correlation bar chart -- `Random_Forest.py`)
- `step6_correlation_distribution.png` (Correlation histogram -- `Random_Forest_Top30.py`)

---

### STEP 7: Feature Selection

#### Random_Forest.py (All Features)

**No feature selection.** All ~200 features are passed to the model. Random Forest handles selection internally -- at each tree split, it considers a random subset (`max_features='sqrt'`) and assigns importance scores.

**Output:** `step7_feature_distribution.png` (Feature category bar chart: raw signal, domain, trend)

#### Random_Forest_Top30.py (Top 30)

The 30 features with the highest absolute correlation to SOH are selected as model inputs. The rest are discarded.

**Why 30?** 100 rows / 30 features is a reasonable ratio. Correlation-based selection eliminates noise.

**Outputs:** `df_step7_selected_features.csv`, `step7_top30_correlation.png`

**Note:** The `cycle` column is never fed to the model (data leakage prevention).

---

### STEP 8: Model Training (Random Forest Baseline)

#### Data Split: Interleaved (75/25)

```
Train: cycle 1, 2, 3, 5, 6, 7, 9, 10, 11, ...  (75%)
Test:  cycle 4, 8, 12, 16, ...                    (25%)
```

Why interleaved? With single-battery data, a chronological split only tests a narrow SOH range.

#### Sample Weight

Higher weight assigned to low-SOH cycles (the critical degradation region).

#### Baseline Parameters

```python
RandomForestRegressor(
    n_estimators=300,    # 300 trees
    max_depth=10,        # max 10 levels deep
    min_samples_split=5, # min 5 samples to split
    min_samples_leaf=2,  # min 2 samples per leaf
)
```

#### Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (%) -- average prediction error |
| **RMSE** | Root Mean Squared Error (%) -- penalizes large errors more |
| **R2** | Fraction of variance explained (1.0 = perfect) |
| **CV MAE** | 5-Fold Cross Validation MAE -- more reliable than single split |

#### Overfitting Check

Train vs Test metrics compared with automatic verdict:
- Diff < 0.5 -> **OK**
- Diff 0.5-1.0 -> **WARNING**
- Diff > 1.0 -> **OVERFITTING?**

(R2 thresholds: <0.01 OK, 0.01-0.05 WARNING, >0.05 OVERFITTING?)

**Outputs:** `df_step8_model_results.csv`, `model_rf_baseline.joblib`, 5 plots (train vs test metrics, SOH prediction, residuals, feature importance, test metrics summary)

---

### STEP 9: Hyperparameter Tuning (Grid Search)

#### Grid Search Parameters

| Parameter | Values Tested |
|---|---|
| n_estimators | 100, 200, 300, 500 |
| max_depth | 5, 8, 10, 15, None |
| min_samples_split | 2, 3, 5 |
| min_samples_leaf | 1, 2, 3 |

Total: 180 combinations x 5-Fold CV = 900 fits.

#### How It Works

1. Takes only the train set (75%)
2. Splits train set internally into 5 folds
3. Tests each parameter combination via 5-Fold CV MAE
4. Best parameters = lowest CV MAE
5. Retrains on full train set with best parameters
6. Evaluates on held-out test set (25%)

**The test set is never used during Grid Search** (data leakage prevention).

#### 5-Fold CV Detailed Analysis

| Fold | Test Cycles | Real-World Equivalent |
|---|---|---|
| 1 | 1-16 | Predicting the initial region (extrapolation) |
| 2-4 | 17-64 | Predicting the middle region (interpolation) |
| 5 | 65-79 | **Real-world scenario** -- predicting the future (extrapolation) |

#### Key Finding: Interpolation vs Extrapolation

| Method | MAE | What It Measures |
|---|---|---|
| Interleaved split | ~0.10% | Interpolation between neighboring cycles |
| CV Folds 2-4 | ~0.79% | Middle region prediction (interpolation) |
| CV Folds 1, 5 | ~2.20% | Edge region prediction (extrapolation) |

The interleaved split MAE is misleadingly low. Real-world performance is closer to CV Fold 5. **Tree-based models cannot extrapolate** -- predictions are averages of training leaf values.

#### Tuned Model May Not Improve

With ~100 cycles, Grid Search may overfit to CV folds. When tuned performs worse than baseline, a warning is printed.

**Outputs:** `df_step9_tuning_results.csv`, `df_step9_best_params.csv`, `model_rf_tuned.joblib`, `Result_*.csv`, 2 plots (baseline vs tuned, tuned SOH prediction)

---

## Top 30 vs All Features Comparison

| Aspect | All Features (Random_Forest.py) | Top 30 (Random_Forest_Top30.py) |
|---|---|---|
| Feature count | ~200 | 30 |
| Selection method | Model handles internally | Correlation-based (manual) |
| Advantage | More information available | Less noise, faster training |
| Disadvantage | Slightly slower | May lose useful features |

Random Forest is robust with many features -- each tree considers a random subset at each split.

---

## Output Files

### Random_Forest.py -> `Output_Random_Forest/`

| File | Content |
|---|---|
| `step1_raw_signals.png` | Raw signal EDA (5 subplots) |
| `step2_volt_conversion.png` | Voltage signals after mV->V |
| `step3_cycle_analysis.png` | Cycle vs SOH + Cycle vs Duration |
| `step4_domain_features.png` | 9 domain features (3x3 subplot) |
| `step5_trend_features.png` | Trend feature plots |
| `step6_correlation.png` | Top 30 SOH correlation bar chart |
| `step7_feature_distribution.png` | Feature category distribution |
| `step8_*.png` | 5 training plots (metrics, prediction, residuals, importance) |
| `step9_*.png` | 2 tuning plots (baseline vs tuned, tuned prediction) |
| `df_step3_cycle_boundaries.csv` | Cycle boundaries, durations, SOH |
| `df_step4_cycle_features.csv` | ~85 cycle-level features |
| `df_step5_trend_features.csv` | ~205 features with trends |
| `df_step6_correlation.csv` | All features' SOH correlation |
| `df_step8_model_results.csv` | Baseline model metrics |
| `df_step9_tuning_results.csv` | Baseline vs Tuned comparison |
| `df_step9_best_params.csv` | Grid Search best parameters |
| `feature_columns.txt` | Feature list (for inference) |
| `model_rf_baseline.joblib` | Baseline Random Forest model |
| `model_rf_tuned.joblib` | Tuned Random Forest model |
| `Result_Random_Forest.csv` | Summary: comparison + overfitting checks |

### Random_Forest_Top30.py -> `Output_Random_Forest_Top30/`

Same structure, plus:
- `df_step7_selected_features.csv` (Top 30 feature list)
- `step7_top30_correlation.png` (Top 30 bar chart)
- `step6_correlation_distribution.png` (instead of `step6_correlation.png`)
- `Result_Random_Forest_Top30.csv` (instead of `Result_Random_Forest.csv`)
- No `feature_columns.txt` or `step7_feature_distribution.png`
