# Gradient Boosting Pipeline - SOH Estimation

## Overview

Two Gradient Boosting pipelines for SOH estimation, differing only in feature selection strategy:

| File | Feature Strategy | Feature Count | Output Folder |
|---|---|---|---|
| `GradientBoosting.py` | All features (model selects internally) | ~200 | `Output_GradientBoosting/` |
| `GradientBoosting_Top30.py` | Top 30 features by SOH correlation | 30 | `Output_GradientBoosting_Top30/` |

Steps 1-6 are identical to other ML pipelines. Steps 7-9 differ in feature handling and model specifics.

## Data

- **Source:** `recording_senaryo_1_Battery_A_N100.mat` (Simulink Recording)
- **Sampling rate:** 1 Hz
- **Signals:** SOC, Current, Cell_V_Max/Min/Avg, Cell_T_Max/Min/Avg, SOH
- **Cycles:** ~100 (SOH: 100% -> 80%)

---

## Pipeline Steps

Steps 1-7 are identical to the Random Forest pipeline. See `Random_Forest_Pipeline.md` for details.

### STEP 8: Model Training (Gradient Boosting Baseline)

#### Baseline Parameters

```python
GradientBoostingRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
)
```

#### Gradient Boosting vs Other Models

| Aspect | Random Forest | XGBoost | Gradient Boosting |
|---|---|---|---|
| Library | sklearn | xgboost | sklearn |
| Method | Bagging | Boosting | Boosting |
| Speed | Medium | Fast | Slow |
| Regularization | Limited | L1+L2 | Limited |
| Implementation | Parallel trees | Optimized C++ | Pure Python |

sklearn's GradientBoostingRegressor is the "vanilla" gradient boosting implementation. Slower than XGBoost/LightGBM but part of sklearn with no extra dependencies.

**Outputs:** `df_step8_model_results.csv`, `model_gb_baseline.joblib`, 5 plots

---

### STEP 9: Hyperparameter Tuning (Grid Search)

| Parameter | Values Tested |
|---|---|
| n_estimators | 100, 150, 200, 300 |
| max_depth | 3, 4, 5, 7 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |
| subsample | 0.7, 0.8, 0.9 |

**Outputs:** `df_step9_tuning_results.csv`, `df_step9_best_params.csv`, `model_gb_tuned.joblib`, `Result_*.csv`, 2 plots

---

## Output Files

### GradientBoosting.py -> `Output_GradientBoosting/`

Same structure as Random Forest output. Model files: `model_gb_baseline.joblib`, `model_gb_tuned.joblib`. Result: `Result_GradientBoosting.csv`.

### GradientBoosting_Top30.py -> `Output_GradientBoosting_Top30/`

Same + `df_step7_selected_features.csv`. Result: `Result_GradientBoosting_Top30.csv`.
