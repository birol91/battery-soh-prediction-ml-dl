# XGBoost Pipeline - SOH Estimation

## Overview

Two XGBoost pipelines for SOH estimation, differing only in feature selection strategy:

| File | Feature Strategy | Feature Count | Output Folder |
|---|---|---|---|
| `XGBoost.py` | All features (XGBoost selects internally) | ~200 | `Output_XGBoost/` |
| `XGBoost_Top30.py` | Top 30 features by SOH correlation | 30 | `Output_XGBoost_Top30/` |

Steps 1-6 are identical to Random Forest and LightGBM pipelines. Steps 7-9 differ in feature handling and model specifics.

## Data

- **Source:** `recording_senaryo_1_Battery_A_N100.mat` (Simulink Recording)
- **Sampling rate:** 1 Hz
- **Signals:** SOC, Current, Cell_V_Max/Min/Avg, Cell_T_Max/Min/Avg, SOH
- **Cycles:** ~100 (SOH: 100% -> 80%)

---

## Pipeline Steps

### STEP 1: Raw Signal Visualization (EDA)

The .mat file is read in HDF5 format. 9 signals are extracted into a DataFrame (1 Hz). Raw signals visualized in 5 subplots.

**Output:** `step1_raw_signals.png`

---

### STEP 2: mV -> V Conversion

Cell_V_Max, Cell_V_Min, Cell_V_Avg divided by 1000 (mV -> V).

**Output:** `step2_volt_conversion.png`

---

### STEP 3: Cycle Boundary Detection

SOC peak points (>99%) detected with minimum distance filter (>5000s). Peak-to-peak = 1 cycle.

**Outputs:** `df_step3_cycle_boundaries.csv`, `step3_cycle_analysis.png`

---

### STEP 4: Cycle-Level Summarization (Feature Extraction)

~85 features per cycle: raw signal statistics (56), domain features (9 groups), timing features (9).

**Outputs:** `df_step4_cycle_features.csv`, `step4_domain_features.png`

---

### STEP 5: Trend Features

17 domain features x 8 trend types = ~136 trend features + proxy_IR_slope.

**Result:** ~85 -> ~205 features

**Outputs:** `df_step5_trend_features.csv`, `step5_trend_features.png`

---

### STEP 6: Correlation Analysis

Pearson correlation of all features with SOH.

**Outputs:** `df_step6_correlation.csv`, `step6_correlation.png` / `step6_correlation_distribution.png`

---

### STEP 7: Feature Selection

#### XGBoost.py (All Features)
No feature selection. All ~200 features passed to the model.

**Output:** `step7_feature_distribution.png`

#### XGBoost_Top30.py (Top 30)
30 features with highest absolute SOH correlation selected.

**Outputs:** `df_step7_selected_features.csv`, `step7_top30_correlation.png`

---

### STEP 8: Model Training (XGBoost Baseline)

#### Data Split: Interleaved (75/25)

```
Train: cycle 1, 2, 3, 5, 6, 7, ...  (75%)
Test:  cycle 4, 8, 12, 16, ...       (25%)
```

#### Baseline Parameters

```python
XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
```

#### XGBoost vs Random Forest vs LightGBM

| Aspect | Random Forest | XGBoost | LightGBM |
|---|---|---|---|
| Method | Bagging (parallel) | Boosting (sequential) | Boosting (sequential) |
| Tree growth | Level-wise | Level-wise | Leaf-wise |
| Regularization | Limited | L1 + L2 (reg_alpha, reg_lambda) | L1 + L2 |
| Small data | Most robust | Good with regularization | Prone to overfitting |

XGBoost has built-in L1/L2 regularization (`reg_alpha`, `reg_lambda`) which helps prevent overfitting on small datasets compared to LightGBM.

**Outputs:** `df_step8_model_results.csv`, `model_xgb_baseline.joblib`, 5 plots

---

### STEP 9: Hyperparameter Tuning (Grid Search)

#### Grid Search Parameters

| Parameter | Values Tested |
|---|---|
| n_estimators | 100, 200, 300, 500 |
| max_depth | 3, 4, 6, 8 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| subsample | 0.7, 0.8, 0.9 |

**Outputs:** `df_step9_tuning_results.csv`, `df_step9_best_params.csv`, `model_xgb_tuned.joblib`, `Result_*.csv`, 2 plots

---

## Output Files

### XGBoost.py -> `Output_XGBoost/`

| File | Content |
|---|---|
| `step[1-9]_*.png` | Visualization plots for each step |
| `df_step[3-6,8,9]_*.csv` | Data outputs per step |
| `feature_columns.txt` | Feature list (for inference) |
| `model_xgb_baseline.joblib` | Baseline XGBoost model |
| `model_xgb_tuned.joblib` | Tuned XGBoost model |
| `Result_XGBoost.csv` | Summary with overfitting checks |

### XGBoost_Top30.py -> `Output_XGBoost_Top30/`

Same structure, plus `df_step7_selected_features.csv` and `step7_top30_correlation.png`. Result file: `Result_XGBoost_Top30.csv`.
