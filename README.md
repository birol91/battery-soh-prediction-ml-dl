# Battery SOH Prediction using ML and DL

End-to-end machine learning and deep learning pipeline for **State of Health (SOH)** estimation of Li-ion battery packs. Models are trained on simulated data (Battery A) and progressively adapted to real-world conditions (Battery B) through a 7-step transfer learning pipeline.

## Dataset

**Download the dataset from Google Drive and place in `0_Dataset/` folder:**

**[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1eajA52NDAM5b8E3uwGmTe8Y0owhxnlI4?usp=sharing)**

| File | Size | Description |
|---|---|---|
| `recording_senaryo_1_Battery_A_N100.mat` | ~157 MB | Training data (100 cycles, UDDS) |
| `recording_senaryo_1_Battery_B_SOH88.mat` | ~86 MB | Phase 1 test (GT=88%, UDDS) |
| `recording_senaryo_1_Batarya_B_faz2.mat` | ~183 MB | Phase 2 test (mixed drive cycle) |

## Battery Configuration
- **Pack:** 5s1p (5 series, 1 parallel) LG 18650HG2 NMC cells
- **Simulation:** MATLAB/Simulink with Simscape Battery
- **Drive Cycle:** UDDS (training), mixed UDDS/WLTP/US06 (testing)
- **SOH Range:** 100% to 80% (End of Life)

---

## Pipeline Architecture (7 Steps)

```
Step 1: ML Model Training (Battery A)
  |
Step 2: DL Model Training (Battery A)
  |
Step 3: ML vs DL Comparison
  |
Step 4: Battery B Phase 1 Inference (same drive cycle)
  |
Step 5: Battery B Phase 2 Zero-Shot Transfer (different drive cycle)
  |
Step 6: Anchor Point Fine-Tuning (improve Phase 2 results)
  |
Step 7: Domain Adaptation (self-training + meta-learner)
```

---

## Step 1: Classical ML Training (`1-ML/`)

Train 6 ML models on Battery A data (100 cycles, UDDS drive cycle).

### Models
| Model | Type | Best MAE (%) | R2 |
|---|---|---:|---:|
| **Extra Trees** | Bagging | **0.064** | 0.9998 |
| XGBoost | Boosting | 0.083 | 0.9996 |
| Random Forest | Bagging | 0.106 | 0.9994 |
| Gradient Boosting | Boosting | 0.156 | 0.9987 |
| CatBoost | Boosting | 0.170 | 0.9983 |
| LightGBM | Boosting | 0.696 | 0.9727 |

### Pipeline (9 sub-steps per model)
1. Raw signal visualization (EDA)
2. mV to V conversion
3. Cycle boundary detection (SOC peaks)
4. Cycle-level feature extraction (~85 domain features)
5. Trend features (rolling mean/std, delta, rate of change -> ~205 features)
6. Correlation analysis
7. Feature selection (All ~200 features OR Top 30 by correlation)
8. Model training (Baseline) + overfitting check + 5-Fold CV
9. Hyperparameter tuning (Grid Search)

### Key Finding
- **Bagging > Boosting** on small data (100 cycles)
- **Top 30 features often outperform all features** (less noise)
- Tree-based models **cannot extrapolate** beyond training range

---

## Step 2: Deep Learning Training (`2-DL/`)

Train 5 DL models on Battery A data using sequences of cycles.

### Models
| Model | Type | Best MAE (%) | R2 |
|---|---|---:|---:|
| **GRU** | RNN (2 gates) | **0.249** | 0.9970 |
| LSTM | RNN (3 gates) | 0.410 | 0.9889 |
| Transformer | Attention | 0.441 | 0.9881 |
| MLP | ANN (flat) | 0.787 | 0.9759 |
| 1D-CNN | CNN | 0.923 | 0.9563 |

### Key Differences from ML Pipeline
- **No trend features** -- sequence models (LSTM, GRU) learn temporal patterns from last 5 cycles
- **Feature scaling required** (StandardScaler) -- neural networks need normalized inputs
- **Sequence input** (samples, 5, features) instead of flat (samples, features)
- **Training via epochs** with early stopping, dropout, gradient clipping

### Key Finding
- **GRU > LSTM** on small data (fewer parameters = less overfitting)
- **Feature engineering still matters** -- Top 30 improved GRU by 16%
- **Hyperparameter tuning is critical** -- MLP Baseline R2 = -5.58 (catastrophic)

---

## Step 3: ML vs DL Comparison (`3-MLvsDL/`)

Direct comparison of best 5 ML and best 5 DL models.

| | Best ML | Best DL |
|---|---|---|
| Model | ExtraTrees Top30 | GRU Top30 Tuned |
| MAE | **0.064%** | 0.249% |
| Winner | **ML** (on this dataset) | -- |

**ML wins on small data (100 cycles).** DL's advantages emerge with larger datasets, transfer learning, and extrapolation.

---

## Step 4: Battery B Phase 1 Inference (`4-Inference_Battery_B_Phase_1/`)

Apply trained models to Battery B (unseen battery, **same UDDS drive cycle**).
No retraining -- models predict using saved weights.

**Ground Truth: SOH = 88%**

| Rank | Model | SOH Pred | Error |
|---:|---|---:|---:|
| 1 | **DL_GRU_Tuned** | **87.93%** | **+0.07%** |
| 2 | ML_ExtraTrees_Top30_TN | 88.29% | -0.29% |
| 3 | ML_RF_Top30_TN | 88.30% | -0.30% |

### Key Finding
- **DL won on unseen battery!** GRU generalized better than ML
- ML models were consistent (~88.3-88.6%)
- DL Top30 models collapsed (GRU_Top30_BL = 9.1%) -- scaler mismatch

---

## Step 5: Battery B Phase 2 Zero-Shot Transfer (`5-Inference_Battery_B_Phase_2/`)

Apply same models to Battery B **with different drive cycle** (mixed UDDS/WLTP/US06 + random charge patterns). No adaptation.

**Expected SOH: ~80% (no exact ground truth)**

| Rank | Model | SOH Pred | Error |
|---:|---|---:|---:|
| 1 | **DL_GRU_Top30_BL** | **80.4%** | **-0.4%** |
| 2 | ML_XGBoost_Top30_BL | 80.9% | -0.9% |
| 3 | ML_XGBoost_Top30_TN | 81.2% | -1.2% |
| 4-6 | ML ExtraTrees, RF | 86-87% | -6 to -7% |
| 7-9 | DL GRU_Top30_TN, LSTM | 12-64% | Catastrophic |

### Key Finding
- **Some models surprisingly robust** -- GRU_Top30_BL and XGBoost zero-shot worked well
- **Domain shift affects most models** -- ExtraTrees dropped from 0.064% to 6% error
- **Baseline often better than Tuned** in cross-domain scenarios (less overfit to training domain)

---

## Step 6: Anchor Point Fine-Tuning (`6-Fine_Tuning_Battery_B_Phase_2/`)

Improve Phase 2 results using "anchor points" -- full charge/discharge cycles that provide reliable reference data.

### Method
1. **Find anchors:** Cycles where SOC_min <= 10% AND SOC_max >= 99.5%
2. **Generate pseudo SOH labels:** Fit quadratic polynomial on Battery A's proxy_IR vs SOH, apply to anchors
3. **Incremental fine-tuning:** As anchor points are encountered chronologically:
   - ML: Retrain with Battery A + accumulated anchors (5x weight)
   - DL: Fine-tune with low LR (5e-5), 50 epochs
4. **Skip good models:** GRU_Top30_BL, XGBoost_Top30_BL, XGBoost_Top30_TN already performing well -- fine-tuning could degrade them

### Key Finding
- Fine-tuning **improved most models** but not all
- **Skipping already-good models was critical** -- fine-tuning GRU_Top30_BL degraded it from 80.4% to 91%
- Pseudo labels may be inaccurate, causing some models to shift in wrong direction

---

## Step 7: Domain Adaptation (`7-Domain_Adaptation_Phase_2/`)

Most advanced adaptation using self-training loop + meta-learner calibration.

### Method
1. **Ensemble pseudo labels:** All model predictions weighted (ML 2x, DL 1x) with confidence scores
2. **Self-training (3 rounds):**
   - ML retrain with Battery A + all pseudo labels (confidence-weighted)
   - DL fine-tune with confidence-weighted MSE loss
   - Update pseudo labels, check convergence
3. **Meta-learner:** Ridge Regression + Isotonic Regression for monotonic SOH calibration
4. **Skip models in SKIP_FINETUNE set**

---

## Project Structure

```
calb_predict/
|-- 0_Dataset/                              # Raw .mat files
|   |-- recording_senaryo_1_Battery_A_N100.mat    # Training data
|   |-- recording_senaryo_1_Battery_B_SOH88.mat   # Phase 1 test (GT=88%)
|   |-- recording_senaryo_1_Batarya_B_faz2.mat    # Phase 2 test (GT~80%)
|
|-- 1-ML/                                  # Classical ML models
|   |-- Random_Forest/                     # RF (All + Top30)
|   |-- ExtraTrees/                        # ET (All + Top30)
|   |-- XGBoost/                           # XGB (All + Top30)
|   |-- GradientBoosting/                  # GB (All + Top30)
|   |-- LightGBM/                          # LGBM (All + Top30)
|   |-- Catboost/                          # CB (All + Top30)
|   |-- ML_Comparison.py                   # Compare all ML models
|   |-- ML_Comparison.csv / .png
|
|-- 2-DL/                                  # Deep Learning models
|   |-- LSTM/                              # BiLSTM (All + Top30)
|   |-- GRU/                               # BiGRU (All + Top30)
|   |-- CNN/                               # 1D-CNN (All + Top30)
|   |-- MLP/                               # MLP (All + Top30)
|   |-- Transformer/                       # Encoder-only (All + Top30)
|   |-- DL_Comparison.py                   # Compare all DL models
|   |-- DL_Comparison.csv / .png
|
|-- 3-MLvsDL/                              # ML vs DL head-to-head
|   |-- ML_vs_DL_Comparison.py
|
|-- 4-Inference_Battery_B_Phase_1/         # Same drive cycle inference
|   |-- Battery_B_Inference.py
|   |-- Output_Inference/
|
|-- 5-Inference_Battery_B_Phase_2/         # Zero-shot transfer
|   |-- Battery_B_Phase2_Inference.py
|   |-- Output_Inference_Phase2/
|
|-- 6-Fine_Tuning_Battery_B_Phase_2/       # Anchor point fine-tuning
|   |-- Battery_B_Phase2_FineTuning.py
|   |-- Output_FineTuning/
|
|-- 7-Domain_Adaptation_Phase_2/           # Self-training + meta-learner
|   |-- Battery_B_Phase2_DomainAdaptation.py
|   |-- Output_DomainAdaptation/
|
|-- kitap/                                 # Reference materials & guides
|   |-- ML_DL_Pratik_Rehber.pdf            # 28-page ML/DL learning guide
|
|-- README.md                              # This file
|-- requirements.txt
```

---

## Key Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error (%) -- average prediction error |
| **RMSE** | Root Mean Squared Error (%) -- penalizes large errors |
| **R2** | R-Squared -- fraction of variance explained (1.0 = perfect) |
| **CV MAE** | 5-Fold Cross Validation MAE -- more reliable than single split |

---

## Results Summary

### Training Performance (Battery A)

| Rank | Model | Type | MAE (%) | R2 |
|---:|---|---|---:|---:|
| 1 | ExtraTrees Top30 BL | ML | 0.064 | 0.9998 |
| 2 | XGBoost Top30 BL | ML | 0.083 | 0.9996 |
| 3 | GRU Top30 Tuned | DL | 0.249 | 0.9970 |
| 4 | GRU Tuned | DL | 0.295 | 0.9954 |
| 5 | LSTM Top30 Tuned | DL | 0.410 | 0.9889 |

### Battery B Phase 1 Inference (same drive cycle, GT=88%)

| Rank | Model | SOH Pred | Error |
|---:|---|---:|---:|
| 1 | DL_GRU_Tuned | 87.93% | +0.07% |
| 2 | ML_ExtraTrees_Top30_TN | 88.29% | -0.29% |

### Battery B Phase 2 Zero-Shot (different drive cycle, expected ~80%)

| Rank | Model | SOH Pred | Error |
|---:|---|---:|---:|
| 1 | DL_GRU_Top30_BL | 80.4% | -0.4% |
| 2 | ML_XGBoost_Top30_BL | 80.9% | -0.9% |

---

## Key Learnings

1. **ML beats DL on small data (100 cycles)** -- tree models need less data
2. **DL generalizes better to unseen batteries** -- GRU outperformed all ML on Battery B Phase 1
3. **Feature engineering matters even for DL** -- Top 30 selection improved results with limited data
4. **Baseline can beat Tuned in cross-domain** -- simpler models are more robust to domain shift
5. **Fine-tuning doesn't always help** -- already-good models can be degraded by incorrect pseudo labels
6. **Tree models cannot extrapolate** -- predictions bounded by training range
7. **Bagging > Boosting on small data** -- Extra Trees' randomness reduces overfitting

---

## Requirements

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost h5py matplotlib joblib torch fpdf2
```

---

## How to Run

### Train all models
```bash
# ML (12 pipelines)
cd 1-ML && for d in */; do cd "$d" && python *.py; cd ..; done

# DL (10 pipelines)
cd 2-DL && for d in */; do cd "$d" && python *.py; cd ..; done

# Compare
python 1-ML/ML_Comparison.py
python 2-DL/DL_Comparison.py
python 3-MLvsDL/ML_vs_DL_Comparison.py
```

### Run inference pipeline
```bash
python 4-Inference_Battery_B_Phase_1/Battery_B_Inference.py
python 5-Inference_Battery_B_Phase_2/Battery_B_Phase2_Inference.py
python 6-Fine_Tuning_Battery_B_Phase_2/Battery_B_Phase2_FineTuning.py
python 7-Domain_Adaptation_Phase_2/Battery_B_Phase2_DomainAdaptation.py
```
