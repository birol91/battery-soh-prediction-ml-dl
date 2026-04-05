# CatBoost Pipeline - SOH Estimation

## Overview

Two CatBoost pipelines for SOH estimation:

| File | Feature Strategy | Feature Count | Output Folder |
|---|---|---|---|
| `CatBoost.py` | All features (model selects internally) | ~200 | `Output_CatBoost/` |
| `CatBoost_Top30.py` | Top 30 features by SOH correlation | 30 | `Output_CatBoost_Top30/` |

Steps 1-6 identical to other ML pipelines. Steps 7-9 differ in model specifics.

## Data

- **Source:** `recording_senaryo_1_Battery_A_N100.mat`
- **Sampling rate:** 1 Hz
- **Cycles:** ~100 (SOH: 100% -> 80%)

---

## Pipeline Steps

Steps 1-7 identical to Random Forest pipeline.

### STEP 8: Model Training (CatBoost Baseline)

```python
CatBoostRegressor(
    iterations=300,       # number of boosting rounds
    depth=6,              # tree depth
    learning_rate=0.05,
    verbose=0,            # suppress training output
)
```

#### CatBoost vs Other Boosting Models

| Aspect | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise | Leaf-wise | Symmetric (balanced) |
| Categorical features | Manual encoding | Built-in | Built-in (best) |
| Overfitting protection | L1/L2 reg | L1/L2 reg | Ordered boosting |
| Speed | Fast | Fastest | Medium |

CatBoost uses "ordered boosting" which reduces prediction shift (a form of overfitting). No extra dependencies needed beyond `pip install catboost`.

**Outputs:** `model_cb_baseline.joblib`, 5 plots

### STEP 9: Hyperparameter Tuning

| Parameter | Values Tested |
|---|---|
| iterations | 200, 300, 500 |
| depth | 4, 6, 8 |
| learning_rate | 0.01, 0.03, 0.05, 0.1 |

**Outputs:** `model_cb_tuned.joblib`, `Result_CatBoost.csv` / `Result_CatBoost_Top30.csv`
