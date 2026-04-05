# LightGBM Pipeline - SOH Estimation

## Overview

Two LightGBM pipelines for SOH estimation, differing only in feature selection strategy:

| File | Feature Strategy | Feature Count | Output Folder |
|---|---|---|---|
| `LightGBM.py` | All features (LightGBM selects internally) | ~200 | `Output_LightGBM/` |
| `LightGBM_Top30.py` | Top 30 features by SOH correlation | 30 | `Output_LightGBM_Top30/` |

Steps 1-6 are identical to the Random Forest pipeline. Steps 7-9 differ in feature handling and model specifics.

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

Simscape voltage signals come in millivolts. Cell_V_Max, Cell_V_Min, Cell_V_Avg are divided by 1000 to convert to Volts.

**Output:** `step2_volt_conversion.png`

---

### STEP 3: Cycle Boundary Detection

1 cycle = full discharge (SOC 100% -> 10%) + full charge (SOC 10% -> 100%).

**Method:** Peak points in the SOC signal (>99%) are detected. A minimum distance filter (>5000 seconds) prevents false peaks. Peak-to-peak defines 1 cycle.

**Definitions:**
- **Active period:** Moments where |Current| > 0.5A (discharge + charge)
- **Rest period:** Moments where |Current| <= 0.5A

**Outputs:** `df_step3_cycle_boundaries.csv`, `step3_cycle_analysis.png`

---

### STEP 4: Cycle-Level Summarization (Feature Extraction)

2.3M rows of 1 Hz data -> ~100 rows (1 row per cycle). Three categories of features:

#### A) Raw Signal Statistics (56 features)
8 signals x 7 statistics: min, max, mean, std, range, discharge_end, charge_start.

#### B) Domain-Specific Features (9 feature groups)

| # | Feature | Calculation | Includes Rest? |
|---|---------|-------------|----------------|
| 1 | **Voltage Spread** (mean/max/std) | `Cell_V_Max - Cell_V_Min` per second | Yes |
| 2 | **Voltage Spread Under Load** | Same, only active moments (|Current| > 0.5A) | No |
| 3 | **Voltage Asymmetry** (mean) | `(V_Max - V_Avg) / (V_Avg - V_Min + 1e-3)` | Yes |
| 4 | **Voltage Asymmetry Under Load** | Same, only active moments | No |
| 5 | **Thermal Spread** (mean/max/std) | `Cell_T_Max - Cell_T_Min` per second | Yes |
| 6 | **Thermal Spread Under Load** | Same, only active moments | No |
| 7 | **Proxy Internal Resistance** (mean/median/std) | `|dV/dI|` when |dI| > 2A, filtered < 1 Ohm | No |
| 8 | **Energy Efficiency** | discharge energy / charge energy | No |
| 9 | **Thermal Response Ratio** | 60-second windows: `d(T_spread) / d(|Current|)` | No |

#### C) Timing Features (9 features)
charge_time_to_50/80/95/full, discharge_duration, soc_range, cycle_duration, active_duration, rest_duration.

**Total:** ~85 features per cycle.

**Outputs:** `df_step4_cycle_features.csv`, `step4_domain_features.png`

---

### STEP 5: Trend Features

Tree-based models treat each row independently. Trend features provide trajectory information (increasing/decreasing/stable).

**For 17 selected domain features:**

| Trend Type | Description | Count per Feature |
|---|---|---|
| Rolling Mean (3, 5, 10) | Last N cycle average | 3 |
| Rolling Std (3, 5, 10) | Last N cycle volatility | 3 |
| Delta | Difference from previous cycle | 1 |
| Rate of Change | Average change over 2 cycles | 1 |

**Additional:** `proxy_IR_trend_slope_10` -- Linear slope over last 10 cycles.

**Result:** ~85 features -> ~205 features

**Outputs:** `df_step5_trend_features.csv`, `step5_trend_features.png`

---

### STEP 6: Correlation Analysis

Pearson correlation of all ~205 features with SOH. Constant features (std=0) removed.

**Outputs:**
- `df_step6_correlation.csv`
- `step6_correlation.png` (Top 30 bar chart -- `LightGBM.py`)
- `step6_correlation_distribution.png` (Histogram -- `LightGBM_Top30.py`)

---

### STEP 7: Feature Selection

#### LightGBM.py (All Features)
**No feature selection.** All ~200 features passed to the model. LightGBM handles feature importance internally.

**Output:** `step7_feature_distribution.png`

#### LightGBM_Top30.py (Top 30)
The 30 features with highest absolute SOH correlation are selected.

**Outputs:** `df_step7_selected_features.csv`, `step7_top30_correlation.png`

**Note:** `cycle` column is never fed to the model (data leakage prevention).

---

### STEP 8: Model Training (LightGBM Baseline)

#### Data Split: Interleaved (75/25)

```
Train: cycle 1, 2, 3, 5, 6, 7, 9, 10, 11, ...  (75%)
Test:  cycle 4, 8, 12, 16, ...                    (25%)
```

#### Sample Weight
Higher weight assigned to low-SOH cycles.

#### Baseline Parameters

```python
LGBMRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    num_leaves=31,
    min_child_samples=10,
)
```

#### LightGBM vs Random Forest

| Aspect | Random Forest | LightGBM |
|---|---|---|
| Training method | Parallel (independent trees) | Sequential (each tree corrects previous) |
| Small data | More robust | Prone to overfitting |
| Feature count | Handles many features well | Benefits from more features |
| Speed | Slower | Faster |

#### Metrics
MAE, RMSE, R2, CV MAE (5-Fold Cross Validation)

#### Overfitting Check
Train vs Test metrics compared (OK / WARNING / OVERFITTING?)

**Outputs:** `df_step8_model_results.csv`, `model_lgbm_baseline.joblib`, 5 plots

---

### STEP 9: Hyperparameter Tuning (Grid Search)

#### Grid Search Parameters

| Parameter | Values Tested |
|---|---|
| n_estimators | 100, 200, 300, 500 |
| max_depth | 3, 4, 6, 8 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| num_leaves | 15, 20, 31 |
| min_child_samples | 5, 10, 20 |

#### Key Difference from RF

LightGBM is a boosting model -- each tree depends on previous trees. This makes it more sensitive to hyperparameters than RF. With ~100 cycles, Grid Search may find parameters that overfit to CV folds but perform worse on test set. When tuned performs worse, a warning is printed.

#### 5-Fold CV Detailed Analysis

| Fold | Test Cycles | Real-World Equivalent |
|---|---|---|
| 1 | 1-16 | Extrapolation (initial region) |
| 2-4 | 17-64 | Interpolation (middle region) |
| 5 | 65-79 | **Real-world scenario** (future prediction) |

**Outputs:** `df_step9_tuning_results.csv`, `df_step9_best_params.csv`, `model_lgbm_tuned.joblib`, `Result_*.csv`, 2 plots

---

## All Features vs Top 30 Comparison

| Aspect | All Features (LightGBM.py) | Top 30 (LightGBM_Top30.py) |
|---|---|---|
| Feature count | ~200 | 30 |
| Selection method | Model handles internally | Correlation-based (manual) |
| Expected performance | Better (more info for boosting) | Worse (limited features) |

LightGBM generally benefits from more features compared to RF, since boosting can leverage weak signals that RF might ignore.

---

## Output Files

### LightGBM.py -> `Output_LightGBM/`

| File | Content |
|---|---|
| `step1_raw_signals.png` | Raw signal EDA |
| `step2_volt_conversion.png` | Voltage after mV->V |
| `step3_cycle_analysis.png` | Cycle vs SOH + Duration |
| `step4_domain_features.png` | 9 domain features |
| `step5_trend_features.png` | Trend features |
| `step6_correlation.png` | Top 30 SOH correlation |
| `step7_feature_distribution.png` | Feature category distribution |
| `step8_*.png` | 5 training plots |
| `step9_*.png` | 2 tuning plots |
| `df_step3_cycle_boundaries.csv` | Cycle boundaries |
| `df_step4_cycle_features.csv` | ~85 cycle features |
| `df_step5_trend_features.csv` | ~205 features with trends |
| `df_step6_correlation.csv` | SOH correlation values |
| `df_step8_model_results.csv` | Baseline metrics |
| `df_step9_tuning_results.csv` | Baseline vs Tuned |
| `df_step9_best_params.csv` | Best parameters |
| `feature_columns.txt` | Feature list (for inference) |
| `model_lgbm_baseline.joblib` | Baseline LightGBM model |
| `model_lgbm_tuned.joblib` | Tuned LightGBM model |
| `Result_LightGBM.csv` | Summary with overfitting checks |

### LightGBM_Top30.py -> `Output_LightGBM_Top30/`

Same structure, plus:
- `df_step7_selected_features.csv` (Top 30 list)
- `step7_top30_correlation.png`
- `step6_correlation_distribution.png`
- `Result_LightGBM_Top30.csv`
- No `feature_columns.txt` or `step7_feature_distribution.png`
