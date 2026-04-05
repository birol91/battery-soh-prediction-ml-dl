# Extra Trees Pipeline - SOH Estimation

## Overview

Two Extra Trees pipelines for SOH estimation:

| File | Feature Strategy | Feature Count | Output Folder |
|---|---|---|---|
| `ExtraTrees.py` | All features (model selects internally) | ~200 | `Output_ExtraTrees/` |
| `ExtraTrees_Top30.py` | Top 30 features by SOH correlation | 30 | `Output_ExtraTrees_Top30/` |

Steps 1-6 identical to other ML pipelines. Steps 7-9 differ in model specifics.

## Data

- **Source:** `recording_senaryo_1_Battery_A_N100.mat`
- **Sampling rate:** 1 Hz
- **Cycles:** ~100 (SOH: 100% -> 80%)

---

## Pipeline Steps

Steps 1-7 identical to Random Forest pipeline.

### STEP 8: Model Training (Extra Trees Baseline)

```python
ExtraTreesRegressor(
    n_estimators=300,
    max_depth=None,        # unlimited depth
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
)
```

#### Extra Trees vs Random Forest

| Aspect | Random Forest | Extra Trees |
|---|---|---|
| Split selection | Best split | Random split |
| Bias-variance | Lower variance | Even lower variance |
| Speed | Slower (evaluates all splits) | Faster (random splits) |
| Overfitting | Some risk | Less risk (more randomness) |

Extra Trees adds more randomness than RF by using random split thresholds instead of optimal ones. This often reduces overfitting on small datasets.

**Outputs:** `model_et_baseline.joblib`, 5 plots

### STEP 9: Hyperparameter Tuning

| Parameter | Values Tested |
|---|---|
| n_estimators | 200, 300, 500 |
| max_depth | 10, 15, 20, None |
| min_samples_split | 2, 3, 5 |
| min_samples_leaf | 1, 2, 3 |

**Outputs:** `model_et_tuned.joblib`, `Result_ExtraTrees.csv` / `Result_ExtraTrees_Top30.csv`
