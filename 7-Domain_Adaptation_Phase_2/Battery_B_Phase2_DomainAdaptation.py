"""
BMS Predictive Maintenance - Battery B Phase 2 Domain Adaptation
================================================================

Layer 3: Self-Training + Meta-Learner on Battery B Phase 2 data.
Builds upon Layer 2 (Anchor Point Fine-Tuning) to further reduce domain shift.

Strategy:
  1. Load Layer 2 fine-tuned predictions as starting point
  2. Generate ensemble pseudo labels (weighted by model reliability)
  3. Self-Training loop (3 rounds):
     - ML: Retrain with Battery A + ALL pseudo labels (anchor=10x, pseudo=3x*conf)
     - DL: Fine-tune ALL cycles with confidence-weighted MSE, LR=1e-4, 100 epochs
     - Update pseudo labels from new predictions
     - Check convergence (stop if anchor MAE improvement < 0.1%)
  4. Meta-Learner calibration (Ridge + Isotonic Regression)
  5. Compare Layer 1 vs Layer 2 vs Layer 3

Key Differences from Layer 2:
  - Uses ALL cycles (not just anchors) for training via pseudo labels
  - Confidence-weighted loss function
  - Self-training iteratively refines pseudo labels
  - Meta-learner combines all model predictions with calibration
  - ML models get 2x weight in ensemble (closer to target domain)

DL Models: SINGLE HEAD (return single value, NOT tuple)
  model(x) returns single tensor, NOT model(x)[0]

Pipeline:
  Step 1-4: Data loading, cycle detection, features, trends (cached from Layer 2)
  Step 5:   Load Layer 2 results (finetuned_predictions.csv)
  Step 6:   Ensemble pseudo label generation
  Step 7:   Self-training loop (3 rounds)
  Step 8:   Meta-learner calibration (Ridge + Isotonic)
  Step 9:   Results comparison & visualization (Layer 1 vs 2 vs 3)
"""

import copy
import math
import os
import time
import warnings

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.patches import Patch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MAT_PATH = os.path.join(PROJECT_ROOT, "0_Dataset", "recording_senaryo_1_Batarya_B_faz2.mat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output_DomainAdaptation")

# Battery A reference data (for pseudo SOH polynomial fitting)
BATT_A_FEAT_PATH = os.path.join(
    PROJECT_ROOT, "1-ML", "Random_Forest", "Output_Random_Forest", "df_step4_cycle_features.csv"
)

# Layer 1 (zero-shot) results
ZEROSHOT_CSV = os.path.join(
    PROJECT_ROOT, "5-Inference_Battery_B_Phase_2", "Output_Inference_Phase2",
    "phase2_cycle_predictions.csv"
)

# Layer 2 (fine-tuning) results
FINETUNED_CSV = os.path.join(
    PROJECT_ROOT, "6-Fine_Tuning_Battery_B_Phase_2", "Output_FineTuning",
    "finetuned_cycle_predictions.csv"
)

# Layer 2 output directory (for cached features if available)
LAYER2_OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "6-Fine_Tuning_Battery_B_Phase_2", "Output_FineTuning"
)

EXPECTED_SOH = 80.0
SEQ_LEN = 5

# Anchor point thresholds (same as Layer 2)
ANCHOR_SOC_MIN_THRESHOLD = 10.0
ANCHOR_SOC_MAX_THRESHOLD = 99.5

# Self-training hyperparameters
SELF_TRAIN_ROUNDS = 3
ANCHOR_WEIGHT_ML = 10.0       # Anchor samples weight in ML retraining
PSEUDO_WEIGHT_ML = 3.0        # Pseudo label base weight (multiplied by confidence)
DL_FINETUNE_LR = 1e-4         # Learning rate for DL fine-tuning
DL_FINETUNE_EPOCHS = 100      # Epochs per self-training round
DL_EARLY_STOP_PATIENCE = 15   # Early stopping patience
CONVERGENCE_THRESHOLD = 0.1   # Stop if anchor MAE improvement < 0.1%

# Battery B stream map (no SOH!)
STREAM_MAP_B = {
    0: "SOC", 1: "Current",
    2: "Cell_V_Max", 3: "Cell_V_Min", 4: "Cell_V_Avg",
    5: "Cell_T_Max", 6: "Cell_T_Min", 7: "Cell_T_Avg",
}
MV_SIGNALS = {"Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"}
RAW_SIGNALS = ["Current", "Cell_V_Max", "Cell_V_Min", "Cell_V_Avg",
               "Cell_T_Max", "Cell_T_Min", "Cell_T_Avg", "SOC"]

# --- ML models ---
ML_ROOT = os.path.join(PROJECT_ROOT, "1-ML")
ML_MODELS = [
    ("ExtraTrees_Top30_BL",
     os.path.join(ML_ROOT, "ExtraTrees", "Output_ExtraTrees_Top30", "model_et_baseline.joblib"),
     os.path.join(ML_ROOT, "ExtraTrees", "Output_ExtraTrees_Top30", "df_step7_selected_features.csv"),
     "top30"),
    ("ExtraTrees_Top30_TN",
     os.path.join(ML_ROOT, "ExtraTrees", "Output_ExtraTrees_Top30", "model_et_tuned.joblib"),
     os.path.join(ML_ROOT, "ExtraTrees", "Output_ExtraTrees_Top30", "df_step7_selected_features.csv"),
     "top30"),
    ("XGBoost_Top30_BL",
     os.path.join(ML_ROOT, "XGBoost", "Output_XGBoost_Top30", "model_xgb_baseline.joblib"),
     os.path.join(ML_ROOT, "XGBoost", "Output_XGBoost_Top30", "df_step7_selected_features.csv"),
     "top30"),
    ("XGBoost_Top30_TN",
     os.path.join(ML_ROOT, "XGBoost", "Output_XGBoost_Top30", "model_xgb_tuned.joblib"),
     os.path.join(ML_ROOT, "XGBoost", "Output_XGBoost_Top30", "df_step7_selected_features.csv"),
     "top30"),
    ("RF_Top30_TN",
     os.path.join(ML_ROOT, "Random_Forest", "Output_Random_Forest_Top30", "model_rf_tuned.joblib"),
     os.path.join(ML_ROOT, "Random_Forest", "Output_Random_Forest_Top30", "df_step7_selected_features.csv"),
     "top30"),
]

# --- DL models ---
DL_ROOT = os.path.join(PROJECT_ROOT, "2-DL")
DL_MODELS = [
    ("GRU_Top30_TN",
     os.path.join(DL_ROOT, "GRU", "Output_GRU_Top30", "model_gru_tuned.pt"),
     os.path.join(DL_ROOT, "GRU", "Output_GRU_Top30", "feature_columns.txt"),
     os.path.join(DL_ROOT, "GRU", "Output_GRU_Top30", "scaler.joblib"),
     "GRU", {"hidden_dim": 128, "n_layers": 2}, "seq"),
    ("GRU_Tuned",
     os.path.join(DL_ROOT, "GRU", "Output_GRU", "model_gru_tuned.pt"),
     os.path.join(DL_ROOT, "GRU", "Output_GRU", "feature_columns.txt"),
     os.path.join(DL_ROOT, "GRU", "Output_GRU", "scaler.joblib"),
     "GRU", {"hidden_dim": 128, "n_layers": 3}, "seq"),
    ("GRU_Top30_BL",
     os.path.join(DL_ROOT, "GRU", "Output_GRU_Top30", "model_gru_baseline.pt"),
     os.path.join(DL_ROOT, "GRU", "Output_GRU_Top30", "feature_columns.txt"),
     os.path.join(DL_ROOT, "GRU", "Output_GRU_Top30", "scaler.joblib"),
     "GRU", {"hidden_dim": 64, "n_layers": 2}, "seq"),
    ("LSTM_Top30_TN",
     os.path.join(DL_ROOT, "LSTM", "Output_LSTM_Top30", "model_lstm_tuned.pt"),
     os.path.join(DL_ROOT, "LSTM", "Output_LSTM_Top30", "feature_columns.txt"),
     os.path.join(DL_ROOT, "LSTM", "Output_LSTM_Top30", "scaler.joblib"),
     "LSTM", {"hidden_dim": 64, "n_layers": 1}, "seq"),
    ("Transformer_TN",
     os.path.join(DL_ROOT, "Transformer", "Output_Transformer", "model_transformer_tuned.pt"),
     os.path.join(DL_ROOT, "Transformer", "Output_Transformer", "feature_columns.txt"),
     os.path.join(DL_ROOT, "Transformer", "Output_Transformer", "scaler.joblib"),
     "Transformer", {"d_model": 32, "n_layers": 3}, "seq"),
]

# Models to SKIP fine-tuning (already performing well in zero-shot)
# These will still be used for prediction but their weights won't be updated
SKIP_FINETUNE = {
    "GRU_Top30_BL",      # Zero-shot: 80.4% (error -0.4%) - already excellent
    "XGBoost_Top30_BL",  # Zero-shot: 80.9% (error -0.9%) - already very good
    "XGBoost_Top30_TN",  # Zero-shot: 81.2% (error -1.2%) - already very good
}

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

# Trend features (for ML models)
TREND_FEATURES = [
    "voltage_spread_mean", "voltage_spread_under_load", "voltage_asymmetry_mean",
    "voltage_asymmetry_under_load", "thermal_spread_mean", "thermal_spread_under_load",
    "proxy_IR_mean", "energy_efficiency", "thermal_response_ratio",
    "voltage_spread_max", "thermal_spread_max", "charge_time_to_full",
    "discharge_duration", "Cell_V_Avg_mean", "Cell_V_Avg_range", "Current_std", "SOC_range",
]
ROLLING_WINDOWS = [3, 5, 10]


# ============================================================
# DL MODEL DEFINITIONS (SINGLE HEAD - SOH only)
# ============================================================

class SOH_GRU(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        h = self.dropout(self.ln(out[:, -1, :]))
        return self.head(h).squeeze(-1)


class SOH_LSTM(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, n_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.dropout(self.ln(out[:, -1, :]))
        return self.head(h).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = (torch.cos(pos * div[:d_model // 2])
                       if d_model % 2 == 0
                       else torch.cos(pos * div[:-1]))
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SOH_Transformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pe = PositionalEncoding(d_model)
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(el, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, x):
        h = self.pe(self.proj(x))
        h = self.dropout(self.ln(self.enc(h)[:, -1, :]))
        return self.head(h).squeeze(-1)


def create_dl_model(model_type, n_features, kwargs):
    """Instantiate a DL model by type string."""
    if model_type == "GRU":
        return SOH_GRU(n_features, **kwargs)
    elif model_type == "LSTM":
        return SOH_LSTM(n_features, **kwargs)
    elif model_type == "Transformer":
        return SOH_Transformer(n_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def make_sequences(X, seq_len):
    """Create sliding window sequences with left-padding for early cycles."""
    n, nf = X.shape
    seqs = np.zeros((n, seq_len, nf), dtype=np.float32)
    for i in range(n):
        if i < seq_len:
            pad = seq_len - (i + 1)
            seqs[i, pad:, :] = X[:i + 1, :]
        else:
            seqs[i] = X[i - seq_len + 1:i + 1, :]
    return seqs


def load_feature_list_csv(csv_path):
    """Load feature names from df_step7_selected_features.csv (ML models)."""
    df = pd.read_csv(csv_path, nrows=0)
    return [c for c in df.columns if c not in ["cycle", "soh_label"]]


def load_feature_list_txt(txt_path):
    """Load feature names from feature_columns.txt (DL models)."""
    with open(txt_path) as f:
        return [line.strip() for line in f if line.strip()]


def align(features_df, feat_list):
    """Align DataFrame columns to the expected feature list, filling missing with 0."""
    df = features_df.copy()
    for c in feat_list:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feat_list].values.astype(np.float32)
    return np.nan_to_num(X, nan=0.0)


# ============================================================
# STEP 1: DATA LOADING
# ============================================================

def load_battery_b(mat_path):
    """Read Battery B Phase 2 .mat file (8 streams, no SOH, mV->V conversion)."""
    print(f"\n[STEP 1] Reading Battery B Phase 2 .mat...")
    with h5py.File(mat_path, "r") as f:
        n = int(f["#sigstream#/0/#length#"][0])
        data = {}
        for sid, col in STREAM_MAP_B.items():
            raw = f[f"#sigstream#/{sid}/#data#"][:]
            arr = np.frombuffer(raw.tobytes(), dtype=np.float64)[:n]
            if col in MV_SIGNALS:
                arr = arr / 1000.0
            data[col] = arr
    df = pd.DataFrame(data)
    print(f"  {len(df):,} rows, SOC: {df['SOC'].min():.1f}-{df['SOC'].max():.1f}%")
    return df


# ============================================================
# STEP 2: CYCLE DETECTION (Phase 2 method)
# ============================================================

def detect_cycles(df, min_dur=2000):
    """
    Phase 2 cycle detection for mixed drive cycles with random charge patterns.

    Strategy: Find discharge->charge transitions using SOC rate of change.
    1. Smooth SOC signal (60s window)
    2. Compute SOC change rate (300s window)
    3. Find negative-to-positive transitions (end of discharge)
    4. Apply minimum distance filter
    """
    print(f"\n[STEP 2] Detecting cycle boundaries (Phase 2 method)...")
    soc = df["SOC"].values

    # Smooth SOC (60s window to reduce noise)
    window = 60
    soc_smooth = np.convolve(soc, np.ones(window) / window, mode="same")

    # SOC change rate (300s window)
    step = 300
    dsoc = np.zeros(len(soc_smooth))
    dsoc[step:] = soc_smooth[step:] - soc_smooth[:-step]

    # Discharge->charge transitions: dsoc goes from negative to positive
    sign = np.sign(dsoc)
    sign_change = np.diff(sign)
    transitions = np.where(sign_change == 2)[0]

    # Minimum distance filter
    filtered = []
    for t in transitions:
        if not filtered or (t - filtered[-1]) >= min_dur:
            filtered.append(t)

    # Create cycle boundaries
    bounds = []
    if filtered:
        bounds.append((0, filtered[0]))
    for i in range(len(filtered) - 1):
        bounds.append((filtered[i], filtered[i + 1]))
    if filtered and (len(df) - filtered[-1]) > min_dur:
        bounds.append((filtered[-1], len(df) - 1))

    print(f"  {len(transitions)} discharge->charge transitions found")
    print(f"  {len(filtered)} filtered (min_dist={min_dur}s)")
    print(f"  {len(bounds)} cycles created")
    if bounds:
        durations = [(e - s) for s, e in bounds]
        print(f"  Duration: min={min(durations)}s ({min(durations)/3600:.1f}h), "
              f"max={max(durations)}s ({max(durations)/3600:.1f}h), "
              f"avg={np.mean(durations):.0f}s ({np.mean(durations)/3600:.1f}h)")
    return bounds


# ============================================================
# STEP 3: FEATURE EXTRACTION
# ============================================================

def compute_cycle_features(df, start, end, cycle_num):
    """Extract ~85 features from a single cycle segment. Matches training pipeline exactly."""
    seg = df.iloc[start:end + 1]
    row = {"cycle": cycle_num}
    soc_min_idx = np.argmin(seg["SOC"].values)

    for sig in RAW_SIGNALS:
        v = seg[sig].values
        row[f"{sig}_min"] = np.min(v)
        row[f"{sig}_max"] = np.max(v)
        row[f"{sig}_mean"] = np.mean(v)
        row[f"{sig}_std"] = np.std(v)
        row[f"{sig}_range"] = np.max(v) - np.min(v)
        row[f"{sig}_discharge_end"] = v[soc_min_idx]
        row[f"{sig}_charge_start"] = v[soc_min_idx + 1] if soc_min_idx < len(v) - 1 else v[soc_min_idx]

    v_max = seg["Cell_V_Max"].values
    v_min = seg["Cell_V_Min"].values
    v_avg = seg["Cell_V_Avg"].values
    t_max = seg["Cell_T_Max"].values
    t_min = seg["Cell_T_Min"].values
    cur = seg["Current"].values
    soc = seg["SOC"].values
    abs_cur = np.abs(cur)

    # Voltage spread
    v_sp = v_max - v_min
    row["voltage_spread_mean"] = np.mean(v_sp)
    row["voltage_spread_max"] = np.max(v_sp)
    row["voltage_spread_std"] = np.std(v_sp)

    # Voltage asymmetry
    vts = v_max - v_min
    mm = vts > 0.001
    if np.any(mm):
        va = (v_max[mm] - v_avg[mm]) / (v_avg[mm] - v_min[mm] + 1e-2)
        row["voltage_asymmetry_mean"] = np.mean(va)
    else:
        row["voltage_asymmetry_mean"] = 1.0

    # Under load
    am = abs_cur > 0.5
    row["voltage_spread_under_load"] = np.mean(v_sp[am]) if np.any(am) else np.mean(v_sp)
    amm = am & (vts > 0.001)
    if np.any(amm):
        row["voltage_asymmetry_under_load"] = np.mean(
            (v_max[amm] - v_avg[amm]) / (v_avg[amm] - v_min[amm] + 1e-2)
        )
    else:
        row["voltage_asymmetry_under_load"] = 1.0

    # Thermal spread
    t_sp = t_max - t_min
    row["thermal_spread_mean"] = np.mean(t_sp)
    row["thermal_spread_max"] = np.max(t_sp)
    row["thermal_spread_std"] = np.std(t_sp)
    row["thermal_spread_under_load"] = np.mean(t_sp[am]) if np.any(am) else np.mean(t_sp)

    # Proxy internal resistance
    di, dv = np.diff(cur), np.diff(v_avg)
    ldm = np.abs(di) > 2.0
    if np.sum(ldm) > 5:
        pir = np.abs(dv[ldm] / di[ldm])
        pir = pir[pir < 1.0]
        if len(pir) > 0:
            row["proxy_IR_mean"] = np.mean(pir)
            row["proxy_IR_median"] = np.median(pir)
            row["proxy_IR_std"] = np.std(pir)
        else:
            row["proxy_IR_mean"] = row["proxy_IR_median"] = row["proxy_IR_std"] = np.nan
    else:
        row["proxy_IR_mean"] = row["proxy_IR_median"] = row["proxy_IR_std"] = np.nan

    # Energy efficiency
    pw = v_avg * cur
    ce = np.sum(pw[cur > 0.5]) if np.any(cur > 0.5) else 0
    de = np.abs(np.sum(pw[cur < -0.5])) if np.any(cur < -0.5) else 0
    row["energy_efficiency"] = de / ce if ce > 0 else np.nan

    dsoc = np.diff(soc)
    si = dsoc > 0.01
    if np.any(si):
        if np.mean(cur[1:][si]) < 0:
            cef = np.abs(np.sum(pw[cur < -0.5])) if np.any(cur < -0.5) else 0
            def_ = np.sum(pw[cur > 0.5]) if np.any(cur > 0.5) else 0
            if cef > 0:
                row["energy_efficiency"] = def_ / cef

    # Charge time features
    sam = soc[soc_min_idx:]
    for thr, name in [(50, "50"), (80, "80"), (95, "95"), (99.5, "full")]:
        idx = np.where(sam >= thr)[0]
        row[f"charge_time_to_{name}"] = idx[0] if len(idx) > 0 else np.nan
    row["discharge_duration"] = soc_min_idx
    row["soc_range"] = np.max(soc) - np.min(soc)

    # Thermal response ratio
    W = 60
    if len(t_sp) > W:
        ts_s = pd.Series(t_sp).rolling(W, min_periods=W).mean().values
        ac_s = pd.Series(abs_cur).rolling(W, min_periods=W).mean().values
        dts, dai = np.diff(ts_s), np.diff(ac_s)
        vm = (~np.isnan(dts)) & (~np.isnan(dai)) & (np.abs(dai) > 0.5)
        if np.sum(vm) > 3:
            r = dts[vm] / dai[vm]
            r = r[np.abs(r) < 10]
            row["thermal_response_ratio"] = np.mean(r) if len(r) > 0 else np.nan
        else:
            row["thermal_response_ratio"] = np.nan
    else:
        row["thermal_response_ratio"] = np.nan

    # Duration features
    row["cycle_duration"] = end - start + 1
    row["active_duration"] = int((abs_cur > 0.5).sum())
    row["rest_duration"] = (end - start + 1) - row["active_duration"]
    return row


def extract_features(df, bounds):
    """Extract features for all detected cycles."""
    print(f"\n[STEP 3] Extracting cycle features...")
    rows = [compute_cycle_features(df, s, e, i + 1) for i, (s, e) in enumerate(bounds)]
    cdf = pd.DataFrame(rows)
    cdf = cdf.ffill().bfill()
    for c in cdf.columns:
        if cdf[c].isna().any():
            cdf[c] = cdf[c].fillna(cdf[c].median())
    print(f"  {len(cdf)} cycles, {len(cdf.columns)} features")
    return cdf


# ============================================================
# STEP 4: TREND FEATURES (for ML models)
# ============================================================

def add_trends(cdf, silent=False):
    """Add rolling mean/std, delta, rate-of-change, and IR slope trend features."""
    if not silent:
        print(f"\n[STEP 4] Adding trend features (for ML models)...")
    df = cdf.copy()
    for feat in TREND_FEATURES:
        if feat not in df.columns:
            continue
        vals = df[feat]
        for w in ROLLING_WINDOWS:
            df[f"{feat}_rolling_mean_{w}"] = vals.rolling(w, min_periods=1).mean()
            df[f"{feat}_rolling_std_{w}"] = vals.rolling(w, min_periods=1).std().fillna(0)
        df[f"{feat}_delta"] = vals.diff().fillna(0)
        df[f"{feat}_roc"] = vals.diff(periods=2).fillna(0) / 2.0
    if "proxy_IR_mean" in df.columns:
        pir = df["proxy_IR_mean"].values
        sl = np.zeros(len(pir))
        for i in range(len(pir)):
            ws = max(0, i - 9)
            wd = pir[ws:i + 1]
            if len(wd) >= 3:
                sl[i] = np.polyfit(np.arange(len(wd)), wd, 1)[0]
        df["proxy_IR_trend_slope_10"] = sl
    if not silent:
        print(f"  {len(df.columns)} features (with trends)")
    return df


# ============================================================
# ANCHOR POINT DETECTION
# ============================================================

def detect_anchor_points(cycle_df, bounds, raw_df):
    """
    Identify anchor cycles where SOC_min <= 10% AND SOC_max >= 99.5%.
    These are full discharge + full charge cycles that closely resemble
    Phase 1 (UDDS) conditions, making them reliable calibration points.
    """
    print(f"\n  Detecting anchor points...")
    print(f"  Criteria: SOC_min <= {ANCHOR_SOC_MIN_THRESHOLD}% AND "
          f"SOC_max >= {ANCHOR_SOC_MAX_THRESHOLD}%")

    anchor_mask = np.zeros(len(cycle_df), dtype=bool)

    for i, (start, end) in enumerate(bounds):
        if i >= len(cycle_df):
            break
        seg_soc = raw_df["SOC"].values[start:end + 1]
        soc_min = np.min(seg_soc)
        soc_max = np.max(seg_soc)
        if soc_min <= ANCHOR_SOC_MIN_THRESHOLD and soc_max >= ANCHOR_SOC_MAX_THRESHOLD:
            anchor_mask[i] = True

    anchor_indices = np.where(anchor_mask)[0]
    anchor_cycles = cycle_df["cycle"].values[anchor_indices]

    print(f"  Found {len(anchor_indices)} anchor cycles out of {len(cycle_df)} total")
    if len(anchor_indices) > 0:
        print(f"  Anchor cycles: {anchor_cycles.tolist()}")
    else:
        print("  WARNING: No anchor points found!")

    return anchor_mask, anchor_indices


# ============================================================
# PSEUDO SOH GENERATION (Battery A polynomial)
# ============================================================

def generate_pseudo_soh_polynomial(cycle_df, anchor_mask, anchor_indices):
    """
    Generate pseudo SOH labels for anchor cycles using Battery A's
    proxy_IR -> SOH quadratic polynomial relationship.
    """
    print(f"\n  Generating pseudo SOH via Battery A polynomial...")

    if not os.path.exists(BATT_A_FEAT_PATH):
        print(f"  ERROR: Battery A reference data not found: {BATT_A_FEAT_PATH}")
        return None

    df_a = pd.read_csv(BATT_A_FEAT_PATH)
    print(f"  Battery A: {len(df_a)} cycles, SOH range: "
          f"{df_a['soh_label'].min():.1f}-{df_a['soh_label'].max():.1f}%")

    # Fit quadratic polynomial: SOH = f(proxy_IR_mean)
    mask_valid = ~df_a["proxy_IR_mean"].isna() & ~df_a["soh_label"].isna()
    ir_a = df_a.loc[mask_valid, "proxy_IR_mean"].values
    soh_a = df_a.loc[mask_valid, "soh_label"].values

    coeffs = np.polyfit(ir_a, soh_a, 2)
    poly = np.poly1d(coeffs)
    print(f"  Polynomial: SOH = {coeffs[0]:.2e}*IR^2 + {coeffs[1]:.2e}*IR + {coeffs[2]:.4f}")

    soh_pred_a = poly(ir_a)
    mae_a = np.mean(np.abs(soh_a - soh_pred_a))
    print(f"  Battery A fit quality: MAE = {mae_a:.3f}%")

    # Apply to anchor cycles
    pseudo_soh = np.full(len(cycle_df), np.nan)

    if len(anchor_indices) == 0:
        print("  No anchor points - skipping pseudo SOH generation")
        return pseudo_soh

    for idx in anchor_indices:
        ir_val = cycle_df.iloc[idx]["proxy_IR_mean"]
        if np.isnan(ir_val):
            print(f"  WARNING: Cycle {cycle_df.iloc[idx]['cycle']} has NaN proxy_IR, skipping")
            continue
        soh_val = poly(ir_val)
        soh_val = np.clip(soh_val, 50.0, 105.0)
        pseudo_soh[idx] = soh_val

    # Enforce monotonic constraint (SOH can only decrease over time)
    valid_anchors = [(idx, pseudo_soh[idx]) for idx in anchor_indices
                     if not np.isnan(pseudo_soh[idx])]
    if len(valid_anchors) > 1:
        valid_anchors.sort(key=lambda x: x[0])
        min_seen = valid_anchors[0][1]
        for i, (idx, val) in enumerate(valid_anchors):
            if val > min_seen:
                pseudo_soh[idx] = min_seen
            else:
                min_seen = val

    # Print pseudo SOH values
    print(f"\n  Pseudo SOH labels for anchor cycles:")
    print(f"  {'Cycle':<8} {'proxy_IR_mean':<16} {'Pseudo SOH':<12}")
    print(f"  {'-' * 36}")
    for idx in anchor_indices:
        if not np.isnan(pseudo_soh[idx]):
            print(f"  {cycle_df.iloc[idx]['cycle']:<8.0f} "
                  f"{cycle_df.iloc[idx]['proxy_IR_mean']:<16.6f} "
                  f"{pseudo_soh[idx]:<12.2f}")

    return pseudo_soh


# ============================================================
# STEP 5: LOAD LAYER 2 RESULTS
# ============================================================

def load_layer2_results():
    """Load Layer 2 (fine-tuned) predictions for use as starting point."""
    print(f"\n[STEP 5] Loading Layer 2 fine-tuned results...")

    if not os.path.exists(FINETUNED_CSV):
        print(f"  ERROR: Layer 2 results not found: {FINETUNED_CSV}")
        return None

    df = pd.read_csv(FINETUNED_CSV)
    print(f"  Layer 2 results loaded: {len(df)} cycles")

    # List available model columns
    model_cols = [c for c in df.columns if c not in ["cycle", "is_anchor", "pseudo_soh"]]
    print(f"  Available models: {len(model_cols)}")
    for col in model_cols:
        last_val = df[col].values[-1]
        err = abs(EXPECTED_SOH - last_val)
        print(f"    {col}: last={last_val:.2f}%, |err|={err:.2f}%")

    return df


def load_zeroshot_results():
    """Load Layer 1 (zero-shot) predictions for comparison."""
    if not os.path.exists(ZEROSHOT_CSV):
        print(f"  WARNING: Zero-shot results not found: {ZEROSHOT_CSV}")
        return None
    df = pd.read_csv(ZEROSHOT_CSV)
    print(f"  Zero-shot results loaded: {len(df)} cycles, {len(df.columns) - 1} models")
    return df


# ============================================================
# STEP 6: ENSEMBLE PSEUDO LABEL GENERATION
# ============================================================

def generate_ensemble_pseudo_labels(layer2_df, anchor_mask, anchor_pseudo_soh):
    """
    Generate pseudo SOH labels for ALL cycles using weighted ensemble of
    Layer 2 model predictions.

    Strategy:
    - ML models get 2x weight (they are closer to target domain)
    - DL predictions outside [70%, 105%] are filtered out
    - Confidence = 1.0 - std/10 (clipped 0.1-0.95)
    - Anchor points get confidence=1.0
    - Monotonic constraint: SOH can only decrease
    """
    print(f"\n[STEP 6] Generating ensemble pseudo labels for ALL cycles...")

    n_cycles = len(layer2_df)
    model_cols = [c for c in layer2_df.columns if c not in ["cycle", "is_anchor", "pseudo_soh"]]

    # Separate ML and DL model columns
    ml_cols = [c for c in model_cols if c.startswith("ML_")]
    dl_cols = [c for c in model_cols if c.startswith("DL_")]

    print(f"  ML models ({len(ml_cols)}): {ml_cols}")
    print(f"  DL models ({len(dl_cols)}): {dl_cols}")

    pseudo_labels = np.zeros(n_cycles)
    confidence = np.zeros(n_cycles)

    for i in range(n_cycles):
        weighted_preds = []
        weights = []

        # ML predictions (2x weight)
        for col in ml_cols:
            val = layer2_df[col].values[i]
            if not np.isnan(val) and 70.0 <= val <= 105.0:
                weighted_preds.append(val)
                weights.append(2.0)

        # DL predictions (1x weight, filtered)
        for col in dl_cols:
            val = layer2_df[col].values[i]
            if not np.isnan(val) and 70.0 <= val <= 105.0:
                weighted_preds.append(val)
                weights.append(1.0)

        if len(weighted_preds) == 0:
            # Fallback: use all ML predictions without filtering
            for col in ml_cols:
                val = layer2_df[col].values[i]
                if not np.isnan(val):
                    weighted_preds.append(val)
                    weights.append(2.0)

        if len(weighted_preds) > 0:
            wp = np.array(weighted_preds)
            w = np.array(weights)
            pseudo_labels[i] = np.average(wp, weights=w)

            # Confidence = 1.0 - std/10 (clipped 0.1-0.95)
            if len(wp) > 1:
                std_val = np.std(wp)
                conf = 1.0 - std_val / 10.0
                confidence[i] = np.clip(conf, 0.1, 0.95)
            else:
                confidence[i] = 0.5
        else:
            pseudo_labels[i] = EXPECTED_SOH
            confidence[i] = 0.1

    # Anchor points: use polynomial pseudo SOH with confidence=1.0
    for idx in np.where(anchor_mask)[0]:
        if not np.isnan(anchor_pseudo_soh[idx]):
            pseudo_labels[idx] = anchor_pseudo_soh[idx]
            confidence[idx] = 1.0

    # Enforce monotonic constraint: SOH can only decrease over time
    for i in range(1, n_cycles):
        if pseudo_labels[i] > pseudo_labels[i - 1]:
            pseudo_labels[i] = pseudo_labels[i - 1]

    # Print summary
    print(f"\n  Pseudo label statistics:")
    print(f"    Range: {pseudo_labels.min():.2f}% - {pseudo_labels.max():.2f}%")
    print(f"    Mean:  {pseudo_labels.mean():.2f}%")
    print(f"    Last:  {pseudo_labels[-1]:.2f}%")
    print(f"    Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")
    print(f"    Anchor points (conf=1.0): {int(anchor_mask.sum())}")
    print(f"    High confidence (>0.8): {int((confidence > 0.8).sum())}")
    print(f"    Low confidence (<0.3): {int((confidence < 0.3).sum())}")

    return pseudo_labels, confidence


# ============================================================
# STEP 7: SELF-TRAINING LOOP
# ============================================================

def load_battery_a_training_data(feat_list):
    """Load Battery A training features + SOH labels aligned to a specific feature list."""
    df_a = pd.read_csv(BATT_A_FEAT_PATH)
    soh_a = df_a["soh_label"].values
    df_a_with_trends = add_trends(df_a, silent=True)
    X_a = align(df_a_with_trends, feat_list)
    return X_a, soh_a


def self_training_ml(cycle_df_with_trends, anchor_mask, anchor_indices,
                     pseudo_labels, confidence, round_num):
    """
    Self-training round for ML models.

    Retrain with: Battery A data + ALL Phase 2 pseudo labels
    - Anchor samples: weight = ANCHOR_WEIGHT_ML (10x)
    - Pseudo samples: weight = PSEUDO_WEIGHT_ML * confidence (3x * conf)
    - Skip models in SKIP_FINETUNE (predict only, no weight updates)
    """
    print(f"\n    ML Models (Round {round_num}):")

    n_cycles = len(cycle_df_with_trends)
    results = {}

    for label, model_path, feat_path, feat_type in ML_MODELS:
        if not os.path.exists(model_path):
            print(f"      [SKIP] {label}: model not found")
            continue

        skip = label in SKIP_FINETUNE
        t0 = time.time()

        # Load original model and feature list
        feat_list = load_feature_list_csv(feat_path)
        original_model = joblib.load(model_path)

        # Load Battery A training data
        X_a, soh_a = load_battery_a_training_data(feat_list)

        # Prepare Phase 2 features
        X_b = align(cycle_df_with_trends, feat_list)

        if skip:
            # Predict only with original model
            predictions = original_model.predict(X_b)
            elapsed = time.time() - t0
            print(f"      {label} (SKIP): last={predictions[-1]:.2f}%, time={elapsed:.1f}s")
            results[f"ML_{label}"] = predictions
            continue

        # Build training set: Battery A + ALL pseudo-labeled Phase 2 cycles
        # Create sample weights
        n_a = len(X_a)
        n_b = n_cycles

        # Battery A base weight = 1.0
        w_a = np.ones(n_a)

        # Phase 2 pseudo label weights
        w_b = np.zeros(n_b)
        for i in range(n_b):
            if anchor_mask[i]:
                w_b[i] = ANCHOR_WEIGHT_ML  # Anchor: 10x weight
            else:
                w_b[i] = PSEUDO_WEIGHT_ML * confidence[i]  # Pseudo: 3x * confidence

        # Combine datasets
        X_combined = np.vstack([X_a, X_b])
        y_combined = np.concatenate([soh_a, pseudo_labels])
        w_combined = np.concatenate([w_a, w_b])

        # Repeat samples according to integer weights for tree-based models
        # (sample_weight is not always supported or well-behaved)
        repeat_counts = np.maximum(np.round(w_combined).astype(int), 1)
        X_train = np.repeat(X_combined, repeat_counts, axis=0)
        y_train = np.repeat(y_combined, repeat_counts)

        try:
            model = copy.deepcopy(original_model)
            model.fit(X_train, y_train)
            predictions = model.predict(X_b)
        except Exception as e:
            print(f"      [WARN] {label}: Retrain failed: {e}")
            predictions = original_model.predict(X_b)

        elapsed = time.time() - t0
        print(f"      {label}: last={predictions[-1]:.2f}%, "
              f"train_size={len(X_train)}, time={elapsed:.1f}s")
        results[f"ML_{label}"] = predictions

    return results


def self_training_dl(cycle_df_no_trends, anchor_mask, pseudo_labels, confidence, round_num):
    """
    Self-training round for DL models.

    Fine-tune ALL cycles with confidence-weighted MSE loss.
    - LR=1e-4, 100 epochs, early stopping (patience=15)
    - Skip models in SKIP_FINETUNE (predict only, no weight updates)
    """
    print(f"\n    DL Models (Round {round_num}):")

    n_cycles = len(cycle_df_no_trends)
    results = {}

    for label, model_path, feat_path, scaler_path, model_type, kwargs, input_type in DL_MODELS:
        if not os.path.exists(model_path):
            print(f"      [SKIP] {label}: model not found")
            continue
        if not os.path.exists(scaler_path):
            print(f"      [SKIP] {label}: scaler not found")
            continue

        skip = label in SKIP_FINETUNE
        t0 = time.time()

        try:
            # Load feature list and scaler
            feat_list = load_feature_list_txt(feat_path)
            n_features = len(feat_list)
            scaler = joblib.load(scaler_path)

            # Prepare Phase 2 features (scaled)
            X_b = align(cycle_df_no_trends, feat_list)
            X_b_scaled = scaler.transform(X_b).astype(np.float32)

            # Create sequences for all cycles
            X_b_seq = make_sequences(X_b_scaled, SEQ_LEN)

            # Load original model weights
            original_state = torch.load(model_path, map_location="cpu", weights_only=True)

            # Initialize model
            model = create_dl_model(model_type, n_features, kwargs)
            model.load_state_dict(original_state)
            model = model.to(DEVICE)

            if skip:
                # Predict only
                model.eval()
                with torch.no_grad():
                    x_t = torch.tensor(X_b_seq, dtype=torch.float32).to(DEVICE)
                    predictions = model(x_t).cpu().numpy()
                elapsed = time.time() - t0
                print(f"      {label} (SKIP): last={predictions[-1]:.2f}%, time={elapsed:.1f}s")
                results[f"DL_{label}"] = predictions
                continue

            # Fine-tune with confidence-weighted MSE
            _finetune_dl_all_cycles(
                model, X_b_seq, pseudo_labels, confidence, anchor_mask,
                lr=DL_FINETUNE_LR, epochs=DL_FINETUNE_EPOCHS,
                patience=DL_EARLY_STOP_PATIENCE
            )

            # Predict with fine-tuned model
            model.eval()
            with torch.no_grad():
                x_t = torch.tensor(X_b_seq, dtype=torch.float32).to(DEVICE)
                predictions = model(x_t).cpu().numpy()

            elapsed = time.time() - t0
            print(f"      {label}: last={predictions[-1]:.2f}%, time={elapsed:.1f}s")
            results[f"DL_{label}"] = predictions

        except Exception as e:
            print(f"      [ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()

    return results


def _finetune_dl_all_cycles(model, X_seq, pseudo_labels, confidence, anchor_mask,
                            lr, epochs, patience):
    """
    Fine-tune a DL model on ALL Phase 2 cycles with confidence-weighted MSE loss.

    Loss = sum(confidence_i * (pred_i - pseudo_label_i)^2) / sum(confidence_i)

    Early stopping based on weighted loss plateau.
    """
    model.train()

    X = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(pseudo_labels, dtype=torch.float32).to(DEVICE)

    # Build confidence weights: anchors get higher weight
    weights = np.copy(confidence)
    weights[anchor_mask] = np.maximum(weights[anchor_mask], 1.0)
    w = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)

        # Confidence-weighted MSE loss
        sq_err = (pred - y) ** 2
        loss = (w * sq_err).sum() / w.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_val = loss.item()
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break


def self_training_loop(cycle_df_with_trends, cycle_df_no_trends,
                       anchor_mask, anchor_indices, anchor_pseudo_soh,
                       initial_pseudo_labels, initial_confidence):
    """
    Main self-training loop. Runs SELF_TRAIN_ROUNDS iterations.

    Each round:
    1. ML retrain with Battery A + pseudo labels
    2. DL fine-tune with confidence-weighted MSE
    3. Generate new predictions
    4. Update pseudo labels from new predictions
    5. Check convergence
    """
    print(f"\n{'=' * 70}")
    print(f"[STEP 7] SELF-TRAINING LOOP ({SELF_TRAIN_ROUNDS} rounds)")
    print(f"{'=' * 70}")

    pseudo_labels = initial_pseudo_labels.copy()
    confidence = initial_confidence.copy()
    n_cycles = len(pseudo_labels)

    round_history = []  # Track convergence
    all_round_predictions = {}  # model -> list of predictions per round
    prev_anchor_mae = None

    for round_num in range(1, SELF_TRAIN_ROUNDS + 1):
        round_t0 = time.time()
        print(f"\n  {'~' * 60}")
        print(f"  ROUND {round_num}/{SELF_TRAIN_ROUNDS}")
        print(f"  {'~' * 60}")
        print(f"  Pseudo label range: {pseudo_labels.min():.2f}% - {pseudo_labels.max():.2f}%")
        print(f"  Confidence range: {confidence.min():.3f} - {confidence.max():.3f}")

        # --- ML self-training ---
        ml_results = self_training_ml(
            cycle_df_with_trends, anchor_mask, anchor_indices,
            pseudo_labels, confidence, round_num
        )

        # --- DL self-training ---
        dl_results = self_training_dl(
            cycle_df_no_trends, anchor_mask, pseudo_labels, confidence, round_num
        )

        # Combine results for this round
        round_results = {**ml_results, **dl_results}

        # Store predictions per model
        for model_name, preds in round_results.items():
            if model_name not in all_round_predictions:
                all_round_predictions[model_name] = []
            all_round_predictions[model_name].append(preds.copy())

        # --- Update pseudo labels from new predictions ---
        new_pseudo_labels = np.zeros(n_cycles)
        new_confidence = np.zeros(n_cycles)

        for i in range(n_cycles):
            weighted_preds = []
            weights = []

            for model_name, preds in round_results.items():
                val = preds[i]
                if not np.isnan(val) and 70.0 <= val <= 105.0:
                    is_ml = model_name.startswith("ML_")
                    w = 2.0 if is_ml else 1.0
                    weighted_preds.append(val)
                    weights.append(w)

            if len(weighted_preds) == 0:
                # Fallback to ML-only without filtering
                for model_name, preds in round_results.items():
                    if model_name.startswith("ML_"):
                        val = preds[i]
                        if not np.isnan(val):
                            weighted_preds.append(val)
                            weights.append(2.0)

            if len(weighted_preds) > 0:
                wp = np.array(weighted_preds)
                w = np.array(weights)
                new_pseudo_labels[i] = np.average(wp, weights=w)
                if len(wp) > 1:
                    std_val = np.std(wp)
                    new_confidence[i] = np.clip(1.0 - std_val / 10.0, 0.1, 0.95)
                else:
                    new_confidence[i] = 0.5
            else:
                new_pseudo_labels[i] = pseudo_labels[i]
                new_confidence[i] = 0.1

        # Anchor points: restore polynomial pseudo SOH with confidence=1.0
        for idx in np.where(anchor_mask)[0]:
            if not np.isnan(anchor_pseudo_soh[idx]):
                new_pseudo_labels[idx] = anchor_pseudo_soh[idx]
                new_confidence[idx] = 1.0

        # Enforce monotonic constraint
        for i in range(1, n_cycles):
            if new_pseudo_labels[i] > new_pseudo_labels[i - 1]:
                new_pseudo_labels[i] = new_pseudo_labels[i - 1]

        # --- Check convergence ---
        # Compute anchor point MAE
        valid_anchor_idx = [idx for idx in anchor_indices if not np.isnan(anchor_pseudo_soh[idx])]
        if len(valid_anchor_idx) > 0 and len(round_results) > 0:
            # Mean prediction at anchor points vs anchor pseudo SOH
            anchor_preds = []
            for idx in valid_anchor_idx:
                model_preds_at_anchor = []
                for model_name, preds in round_results.items():
                    val = preds[idx]
                    if not np.isnan(val) and 70.0 <= val <= 105.0:
                        model_preds_at_anchor.append(val)
                if model_preds_at_anchor:
                    anchor_preds.append(np.mean(model_preds_at_anchor))
                else:
                    anchor_preds.append(anchor_pseudo_soh[idx])
            anchor_preds = np.array(anchor_preds)
            anchor_true = np.array([anchor_pseudo_soh[idx] for idx in valid_anchor_idx])
            anchor_mae = np.mean(np.abs(anchor_preds - anchor_true))
        else:
            anchor_mae = float("inf")

        # Label drift
        label_drift = np.mean(np.abs(new_pseudo_labels - pseudo_labels))

        round_elapsed = time.time() - round_t0
        round_info = {
            "round": round_num,
            "anchor_mae": anchor_mae,
            "label_drift": label_drift,
            "pseudo_mean": new_pseudo_labels.mean(),
            "pseudo_last": new_pseudo_labels[-1],
            "conf_mean": new_confidence.mean(),
            "time_s": round_elapsed,
        }
        round_history.append(round_info)

        print(f"\n  Round {round_num} Summary:")
        print(f"    Anchor MAE: {anchor_mae:.4f}%")
        print(f"    Label drift: {label_drift:.4f}%")
        print(f"    Pseudo label last: {new_pseudo_labels[-1]:.2f}%")
        print(f"    Time: {round_elapsed:.1f}s")

        # Convergence check
        converged = False
        if prev_anchor_mae is not None:
            improvement = prev_anchor_mae - anchor_mae
            print(f"    Anchor MAE improvement: {improvement:+.4f}%")
            if improvement < CONVERGENCE_THRESHOLD and round_num > 1:
                print(f"    CONVERGED (improvement < {CONVERGENCE_THRESHOLD}%)")
                converged = True

        prev_anchor_mae = anchor_mae

        # Update pseudo labels for next round
        pseudo_labels = new_pseudo_labels.copy()
        confidence = new_confidence.copy()

        if converged:
            break

    # Return final results (last round predictions)
    final_results = {}
    for model_name in all_round_predictions:
        final_results[model_name] = all_round_predictions[model_name][-1]

    return final_results, pseudo_labels, confidence, round_history, all_round_predictions


# ============================================================
# STEP 8: META-LEARNER CALIBRATION
# ============================================================

def meta_learner_calibration(final_results, anchor_mask, anchor_indices, anchor_pseudo_soh,
                             cycles):
    """
    Meta-Learner: Ridge Regression + Isotonic Regression.

    1. Ridge Regression: trained on anchor points
       - Inputs: all model predictions at anchor points
       - Target: anchor pseudo SOH values
       - Predicts calibrated SOH for all cycles

    2. Isotonic Regression: enforces monotonic decreasing SOH
       - Applied to Ridge output
    """
    print(f"\n{'=' * 70}")
    print(f"[STEP 8] META-LEARNER CALIBRATION")
    print(f"{'=' * 70}")

    n_cycles = len(cycles)
    model_names = sorted(final_results.keys())

    if len(model_names) == 0:
        print("  ERROR: No model predictions available for meta-learner")
        return None, None

    # Build prediction matrix: (n_cycles, n_models)
    pred_matrix = np.column_stack([final_results[name] for name in model_names])
    print(f"  Prediction matrix shape: {pred_matrix.shape}")
    print(f"  Models: {model_names}")

    # --- Get anchor training data ---
    valid_anchor_idx = [idx for idx in anchor_indices if not np.isnan(anchor_pseudo_soh[idx])]

    if len(valid_anchor_idx) < 2:
        print(f"  WARNING: Only {len(valid_anchor_idx)} valid anchor points. "
              f"Need >= 2 for Ridge regression.")
        print(f"  Falling back to weighted ensemble mean...")

        # Fallback: weighted ensemble (ML=2x, DL=1x)
        calibrated = np.zeros(n_cycles)
        for i in range(n_cycles):
            vals = []
            wts = []
            for name in model_names:
                v = final_results[name][i]
                if not np.isnan(v) and 70.0 <= v <= 105.0:
                    vals.append(v)
                    wts.append(2.0 if name.startswith("ML_") else 1.0)
            if vals:
                calibrated[i] = np.average(vals, weights=wts)
            else:
                calibrated[i] = EXPECTED_SOH
        # Monotonic
        for i in range(1, n_cycles):
            if calibrated[i] > calibrated[i - 1]:
                calibrated[i] = calibrated[i - 1]
        return calibrated, {"method": "weighted_ensemble"}

    # --- Ridge Regression on anchor points ---
    X_anchor = pred_matrix[valid_anchor_idx]
    y_anchor = np.array([anchor_pseudo_soh[idx] for idx in valid_anchor_idx])

    print(f"\n  Ridge Regression:")
    print(f"    Training on {len(valid_anchor_idx)} anchor points")
    print(f"    Input features: {len(model_names)} model predictions")

    # Use Ridge with regularization (handles multicollinearity between models)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_anchor, y_anchor)

    # Print Ridge coefficients (model weights)
    print(f"\n    Model weights (Ridge coefficients):")
    for name, coef in zip(model_names, ridge.coef_):
        print(f"      {name}: {coef:+.4f}")
    print(f"    Intercept: {ridge.intercept_:.4f}")

    # Ridge training fit
    y_anchor_pred = ridge.predict(X_anchor)
    ridge_mae = np.mean(np.abs(y_anchor - y_anchor_pred))
    print(f"    Training MAE (anchor): {ridge_mae:.4f}%")

    # Predict calibrated SOH for all cycles
    ridge_soh = ridge.predict(pred_matrix)
    print(f"    Ridge output range: {ridge_soh.min():.2f}% - {ridge_soh.max():.2f}%")
    print(f"    Ridge last cycle: {ridge_soh[-1]:.2f}%")

    # --- Isotonic Regression (enforce monotonic decreasing) ---
    print(f"\n  Isotonic Regression (monotonic decreasing):")

    # IsotonicRegression with increasing=False enforces decreasing
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")

    # Fit on all cycles (x = cycle index, y = ridge SOH)
    # The isotonic regression will enforce the decreasing constraint
    cycle_indices = np.arange(n_cycles).astype(float)
    calibrated_soh = iso.fit_transform(cycle_indices, ridge_soh)

    # Clip to reasonable range
    calibrated_soh = np.clip(calibrated_soh, 50.0, 105.0)

    print(f"    Isotonic output range: {calibrated_soh.min():.2f}% - {calibrated_soh.max():.2f}%")
    print(f"    Isotonic last cycle: {calibrated_soh[-1]:.2f}%")
    print(f"    Last cycle error: {abs(EXPECTED_SOH - calibrated_soh[-1]):.2f}%")

    meta_info = {
        "method": "ridge_isotonic",
        "ridge_model": ridge,
        "iso_model": iso,
        "model_names": model_names,
        "ridge_coefs": dict(zip(model_names, ridge.coef_)),
        "ridge_intercept": ridge.intercept_,
        "ridge_mae": ridge_mae,
    }

    return calibrated_soh, meta_info


# ============================================================
# STEP 9: RESULTS & VISUALIZATION
# ============================================================

def build_comparison_table(final_results, calibrated_soh, zs_df, layer2_df, cycles):
    """Build 3-layer comparison table: Layer 1 vs Layer 2 vs Layer 3."""
    print(f"\n{'=' * 70}")
    print(f"[STEP 9] THREE-LAYER COMPARISON: Zero-Shot vs Fine-Tuned vs Domain-Adapted")
    print(f"{'=' * 70}")

    comparison = []

    for model_name in sorted(final_results.keys()):
        l3_last = final_results[model_name][-1]
        l3_err = abs(EXPECTED_SOH - l3_last)

        # Layer 1 (zero-shot)
        l1_last = np.nan
        l1_err = np.nan
        if zs_df is not None and model_name in zs_df.columns:
            l1_last = zs_df[model_name].values[-1]
            l1_err = abs(EXPECTED_SOH - l1_last)

        # Layer 2 (fine-tuned)
        l2_last = np.nan
        l2_err = np.nan
        if layer2_df is not None and model_name in layer2_df.columns:
            l2_last = layer2_df[model_name].values[-1]
            l2_err = abs(EXPECTED_SOH - l2_last)

        # Improvements
        l1_to_l3 = l1_err - l3_err if not np.isnan(l1_err) else np.nan
        l2_to_l3 = l2_err - l3_err if not np.isnan(l2_err) else np.nan

        comparison.append({
            "Model": model_name,
            "Type": "ML" if model_name.startswith("ML_") else "DL",
            "L1_ZeroShot_SOH": l1_last,
            "L1_AbsErr": l1_err,
            "L2_FineTuned_SOH": l2_last,
            "L2_AbsErr": l2_err,
            "L3_DomainAdapt_SOH": l3_last,
            "L3_AbsErr": l3_err,
            "L1_to_L3_Improv": l1_to_l3,
            "L2_to_L3_Improv": l2_to_l3,
        })

    # Add Meta-Learner row
    if calibrated_soh is not None:
        meta_last = calibrated_soh[-1]
        meta_err = abs(EXPECTED_SOH - meta_last)
        comparison.append({
            "Model": "META_LEARNER",
            "Type": "META",
            "L1_ZeroShot_SOH": np.nan,
            "L1_AbsErr": np.nan,
            "L2_FineTuned_SOH": np.nan,
            "L2_AbsErr": np.nan,
            "L3_DomainAdapt_SOH": meta_last,
            "L3_AbsErr": meta_err,
            "L1_to_L3_Improv": np.nan,
            "L2_to_L3_Improv": np.nan,
        })

    # Sort by L3 absolute error
    comparison.sort(key=lambda x: x["L3_AbsErr"])

    # Print comparison table
    print(f"\n  {'Rank':<5} {'Model':<28} {'L1 SOH':>8} {'L1|E|':>7} "
          f"{'L2 SOH':>8} {'L2|E|':>7} "
          f"{'L3 SOH':>8} {'L3|E|':>7} {'L1->L3':>8} {'L2->L3':>8}")
    print(f"  {'-' * 106}")

    for i, c in enumerate(comparison):
        l1_str = f"{c['L1_ZeroShot_SOH']:.2f}" if not np.isnan(c['L1_ZeroShot_SOH']) else "N/A"
        l1e_str = f"{c['L1_AbsErr']:.2f}" if not np.isnan(c['L1_AbsErr']) else "N/A"
        l2_str = f"{c['L2_FineTuned_SOH']:.2f}" if not np.isnan(c['L2_FineTuned_SOH']) else "N/A"
        l2e_str = f"{c['L2_AbsErr']:.2f}" if not np.isnan(c['L2_AbsErr']) else "N/A"
        l1l3_str = f"{c['L1_to_L3_Improv']:+.2f}" if not np.isnan(c['L1_to_L3_Improv']) else "N/A"
        l2l3_str = f"{c['L2_to_L3_Improv']:+.2f}" if not np.isnan(c['L2_to_L3_Improv']) else "N/A"

        print(f"  {i + 1:<5} {c['Model']:<28} {l1_str:>8} {l1e_str:>7} "
              f"{l2_str:>8} {l2e_str:>7} "
              f"{c['L3_DomainAdapt_SOH']:>8.2f} {c['L3_AbsErr']:>7.2f} "
              f"{l1l3_str:>8} {l2l3_str:>8}")

    # Summary statistics
    model_rows = [c for c in comparison if c["Model"] != "META_LEARNER"]
    valid_l2l3 = [c["L2_to_L3_Improv"] for c in model_rows if not np.isnan(c["L2_to_L3_Improv"])]
    valid_l1l3 = [c["L1_to_L3_Improv"] for c in model_rows if not np.isnan(c["L1_to_L3_Improv"])]

    print(f"\n  --- Summary (excluding Meta-Learner) ---")
    if valid_l1l3:
        improved_l1l3 = sum(1 for v in valid_l1l3 if v > 0)
        print(f"  L1->L3 improved: {improved_l1l3}/{len(valid_l1l3)}")
        print(f"  L1->L3 avg improvement: {np.mean(valid_l1l3):+.3f}%")
    if valid_l2l3:
        improved_l2l3 = sum(1 for v in valid_l2l3 if v > 0)
        print(f"  L2->L3 improved: {improved_l2l3}/{len(valid_l2l3)}")
        print(f"  L2->L3 avg improvement: {np.mean(valid_l2l3):+.3f}%")

    if calibrated_soh is not None:
        print(f"\n  Meta-Learner (calibrated): {calibrated_soh[-1]:.2f}% "
              f"(|err|={abs(EXPECTED_SOH - calibrated_soh[-1]):.2f}%)")

    return comparison


def save_results(final_results, calibrated_soh, comparison, round_history,
                 cycles, pseudo_labels, confidence, anchor_mask, meta_info):
    """Save all domain adaptation results to CSV files."""
    print(f"\n[SAVING RESULTS]")

    # 1. Per-cycle predictions (all models + meta-learner)
    pred_df = pd.DataFrame({"cycle": cycles})
    pred_df["is_anchor"] = anchor_mask.astype(int)
    pred_df["pseudo_label"] = pseudo_labels
    pred_df["confidence"] = confidence
    for name in sorted(final_results.keys()):
        pred_df[name] = final_results[name]
    if calibrated_soh is not None:
        pred_df["META_LEARNER"] = calibrated_soh
    pred_path = os.path.join(OUTPUT_DIR, "domain_adapted_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  [CSV] domain_adapted_predictions.csv ({len(pred_df)} rows)")

    # 2. Comparison summary
    comp_df = pd.DataFrame(comparison)
    comp_path = os.path.join(OUTPUT_DIR, "Result_DomainAdaptation_Comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  [CSV] Result_DomainAdaptation_Comparison.csv ({len(comp_df)} rows)")

    # 3. Self-training convergence history
    if round_history:
        hist_df = pd.DataFrame(round_history)
        hist_path = os.path.join(OUTPUT_DIR, "self_training_convergence.csv")
        hist_df.to_csv(hist_path, index=False)
        print(f"  [CSV] self_training_convergence.csv ({len(hist_df)} rows)")

    # 4. Meta-learner details
    if meta_info is not None and meta_info.get("method") == "ridge_isotonic":
        meta_df = pd.DataFrame({
            "model": list(meta_info["ridge_coefs"].keys()),
            "ridge_coefficient": list(meta_info["ridge_coefs"].values()),
        })
        meta_df.loc[len(meta_df)] = {"model": "intercept",
                                      "ridge_coefficient": meta_info["ridge_intercept"]}
        meta_path = os.path.join(OUTPUT_DIR, "meta_learner_coefficients.csv")
        meta_df.to_csv(meta_path, index=False)
        print(f"  [CSV] meta_learner_coefficients.csv ({len(meta_df)} rows)")

    return pred_path, comp_path


def plot_all(final_results, calibrated_soh, comparison, round_history,
             all_round_predictions, zs_df, layer2_df,
             cycles, pseudo_labels, confidence, anchor_mask, anchor_indices,
             anchor_pseudo_soh, meta_info):
    """Generate all Layer 3 visualizations."""
    print(f"\n[VISUALIZATIONS]")

    n_anchors = int(anchor_mask.sum())
    anchor_cycles_arr = cycles[anchor_indices] if len(anchor_indices) > 0 else np.array([])
    valid_anchor_idx = [idx for idx in anchor_indices if not np.isnan(anchor_pseudo_soh[idx])]

    # Color palettes
    ml_colors = ["#6BAED6", "#2171B5", "#74C476", "#238B45", "#41AB5D"]
    dl_colors = ["#FC9272", "#EF3B2C", "#FB6A4A", "#CB181D", "#A50F15"]

    # ----------------------------------------------------------------
    # PLOT 1: All Domain-Adapted Models + Meta-Learner SOH Timeline
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    ml_names = sorted([n for n in final_results if n.startswith("ML_")])
    dl_names = sorted([n for n in final_results if n.startswith("DL_")])

    for i, name in enumerate(ml_names):
        ax.plot(cycles, final_results[name],
                label=f"{name} ({final_results[name][-1]:.1f}%)",
                color=ml_colors[i % len(ml_colors)], linewidth=1.5, marker="s", markersize=2,
                alpha=0.7)
    for i, name in enumerate(dl_names):
        ax.plot(cycles, final_results[name],
                label=f"{name} ({final_results[name][-1]:.1f}%)",
                color=dl_colors[i % len(dl_colors)], linewidth=1.5, marker="o", markersize=2,
                alpha=0.7)

    # Meta-learner (thick line)
    if calibrated_soh is not None:
        ax.plot(cycles, calibrated_soh,
                label=f"META_LEARNER ({calibrated_soh[-1]:.1f}%)",
                color="black", linewidth=3, marker="D", markersize=4, zorder=8)

    # Pseudo labels (dashed)
    ax.plot(cycles, pseudo_labels,
            label=f"Pseudo Labels ({pseudo_labels[-1]:.1f}%)",
            color="purple", linewidth=1.5, linestyle=":", alpha=0.6)

    # Anchor pseudo SOH
    if valid_anchor_idx:
        ax.scatter(cycles[valid_anchor_idx],
                   anchor_pseudo_soh[valid_anchor_idx],
                   color="gold", edgecolors="black", s=120, zorder=10,
                   label=f"Anchor Pseudo SOH ({n_anchors} pts)", marker="*")

    ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=2.5,
               label=f"Expected SOH ({EXPECTED_SOH:.0f}%)")

    for ac in anchor_cycles_arr:
        ax.axvline(x=ac, color="gold", alpha=0.2, linewidth=1, linestyle=":")

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("SOH (%)", fontsize=12)
    ax.set_title("Battery B Phase 2 - Layer 3: Domain Adaptation (Self-Training + Meta-Learner)\n"
                 f"All Models + Calibrated Meta-Learner ({n_anchors} anchor points)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "domain_adapted_all_models_soh.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [PLOT] domain_adapted_all_models_soh.png")

    # ----------------------------------------------------------------
    # PLOT 2: Three-Layer Bar Comparison (Layer 1 vs 2 vs 3)
    # ----------------------------------------------------------------
    model_rows = [c for c in comparison if c["Model"] != "META_LEARNER"]
    models_with_all = [c for c in model_rows
                       if not np.isnan(c["L1_ZeroShot_SOH"])
                       and not np.isnan(c["L2_FineTuned_SOH"])]

    if models_with_all:
        fig, ax = plt.subplots(figsize=(18, 8))

        n_models = len(models_with_all)
        x = np.arange(n_models)
        width = 0.25

        l1_vals = [c["L1_ZeroShot_SOH"] for c in models_with_all]
        l2_vals = [c["L2_FineTuned_SOH"] for c in models_with_all]
        l3_vals = [c["L3_DomainAdapt_SOH"] for c in models_with_all]
        names = [c["Model"] for c in models_with_all]

        bars1 = ax.bar(x - width, l1_vals, width, color="#BDBDBD",
                       edgecolor="black", linewidth=0.5, label="Layer 1: Zero-Shot")
        bars2 = ax.bar(x, l2_vals, width, color="#6BAED6",
                       edgecolor="black", linewidth=0.5, label="Layer 2: Fine-Tuned")
        bars3 = ax.bar(x + width, l3_vals, width, color="#238B45",
                       edgecolor="black", linewidth=0.5, label="Layer 3: Domain-Adapted")

        # Add improvement annotations on L3 bars
        for i, c in enumerate(models_with_all):
            l2l3 = c["L2_to_L3_Improv"]
            if not np.isnan(l2l3):
                color = "green" if l2l3 > 0 else "red"
                ax.annotate(f"{l2l3:+.1f}%", xy=(x[i] + width, l3_vals[i]),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=7, fontweight="bold", color=color)

        ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=2.5,
                   label=f"Expected SOH ({EXPECTED_SOH}%)")

        # Add Meta-Learner as separate bar at the end
        if calibrated_soh is not None:
            meta_x = n_models
            ax.bar(meta_x, calibrated_soh[-1], width * 2, color="black",
                   edgecolor="gold", linewidth=2, label=f"Meta-Learner ({calibrated_soh[-1]:.1f}%)")
            names.append("META_LEARNER")
            x_all = np.append(x, meta_x)
        else:
            x_all = x

        ax.set_xticks(x_all)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("SOH (%)", fontsize=11)
        ax.set_title("Battery B Phase 2 - Three-Layer Comparison\n"
                     "Layer 1 (Zero-Shot) vs Layer 2 (Fine-Tuned) vs Layer 3 (Domain-Adapted)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "three_layer_comparison_bar.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] three_layer_comparison_bar.png")

    # ----------------------------------------------------------------
    # PLOT 3: Self-Training Convergence
    # ----------------------------------------------------------------
    if round_history and len(round_history) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        rounds = [r["round"] for r in round_history]

        # 3a: Anchor MAE
        ax = axes[0]
        anchor_maes = [r["anchor_mae"] for r in round_history]
        ax.plot(rounds, anchor_maes, "o-", color="#2171B5", linewidth=2, markersize=8)
        ax.set_xlabel("Self-Training Round", fontsize=11)
        ax.set_ylabel("Anchor MAE (%)", fontsize=11)
        ax.set_title("Anchor Point MAE Convergence", fontsize=12, fontweight="bold")
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)

        # 3b: Label Drift
        ax = axes[1]
        drifts = [r["label_drift"] for r in round_history]
        ax.plot(rounds, drifts, "s-", color="#E74C3C", linewidth=2, markersize=8)
        ax.set_xlabel("Self-Training Round", fontsize=11)
        ax.set_ylabel("Mean Label Drift (%)", fontsize=11)
        ax.set_title("Pseudo Label Drift per Round", fontsize=12, fontweight="bold")
        ax.set_xticks(rounds)
        ax.grid(True, alpha=0.3)

        # 3c: Last Cycle Pseudo Label
        ax = axes[2]
        lasts = [r["pseudo_last"] for r in round_history]
        ax.plot(rounds, lasts, "D-", color="#238B45", linewidth=2, markersize=8)
        ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=1.5,
                   label=f"Expected ({EXPECTED_SOH}%)")
        ax.set_xlabel("Self-Training Round", fontsize=11)
        ax.set_ylabel("Last Cycle Pseudo SOH (%)", fontsize=11)
        ax.set_title("Last Cycle SOH per Round", fontsize=12, fontweight="bold")
        ax.set_xticks(rounds)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        fig.suptitle("Self-Training Convergence Metrics", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "self_training_convergence.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] self_training_convergence.png")

    # ----------------------------------------------------------------
    # PLOT 4: Meta-Learner Coefficients Bar Chart
    # ----------------------------------------------------------------
    if meta_info is not None and meta_info.get("method") == "ridge_isotonic":
        fig, ax = plt.subplots(figsize=(14, 6))

        model_names_meta = list(meta_info["ridge_coefs"].keys())
        coefs = list(meta_info["ridge_coefs"].values())

        colors_bar = []
        for name in model_names_meta:
            if name.startswith("ML_"):
                colors_bar.append("#2171B5")
            else:
                colors_bar.append("#E74C3C")

        bars = ax.bar(range(len(model_names_meta)), coefs, color=colors_bar,
                      edgecolor="black", linewidth=0.5)

        # Add value labels
        for i, (coef, name) in enumerate(zip(coefs, model_names_meta)):
            ax.text(i, coef + (0.01 if coef >= 0 else -0.03),
                    f"{coef:.3f}", ha="center", fontsize=8, fontweight="bold")

        ax.axhline(y=0, color="black", linewidth=1)
        ax.set_xticks(range(len(model_names_meta)))
        short_names = [n.replace("ML_", "").replace("DL_", "") for n in model_names_meta]
        ax.set_xticklabels(short_names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Ridge Coefficient (Model Weight)", fontsize=11)
        ax.set_title(f"Meta-Learner: Ridge Regression Model Weights\n"
                     f"Intercept = {meta_info['ridge_intercept']:.4f}, "
                     f"Anchor MAE = {meta_info['ridge_mae']:.4f}%",
                     fontsize=12, fontweight="bold")

        legend_elements = [
            Patch(facecolor="#2171B5", edgecolor="black", label="ML Models"),
            Patch(facecolor="#E74C3C", edgecolor="black", label="DL Models"),
        ]
        ax.legend(handles=legend_elements, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "meta_learner_coefficients.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] meta_learner_coefficients.png")

    # ----------------------------------------------------------------
    # PLOT 5: Per-Model Three-Layer Timeline (subplots)
    # ----------------------------------------------------------------
    model_rows = [c for c in comparison if c["Model"] != "META_LEARNER"]
    models_for_subplot = [c for c in model_rows
                          if not np.isnan(c.get("L1_ZeroShot_SOH", np.nan))
                          and not np.isnan(c.get("L2_FineTuned_SOH", np.nan))]

    if models_for_subplot and zs_df is not None and layer2_df is not None:
        n_models = len(models_for_subplot)
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, c in enumerate(models_for_subplot):
            ax = axes[i]
            model_name = c["Model"]

            # Layer 1 (zero-shot)
            if model_name in zs_df.columns:
                ax.plot(cycles[:len(zs_df)], zs_df[model_name].values,
                        color="#BDBDBD", linewidth=1, linestyle="--",
                        label="L1: Zero-Shot", alpha=0.7)

            # Layer 2 (fine-tuned)
            if model_name in layer2_df.columns:
                ax.plot(cycles[:len(layer2_df)], layer2_df[model_name].values,
                        color="#6BAED6", linewidth=1.5, linestyle="-.",
                        label="L2: Fine-Tuned", alpha=0.8)

            # Layer 3 (domain-adapted)
            if model_name in final_results:
                ax.plot(cycles, final_results[model_name],
                        color="#238B45", linewidth=2, label="L3: Domain-Adapted")

            # Anchor points
            if valid_anchor_idx:
                ax.scatter(cycles[valid_anchor_idx],
                           anchor_pseudo_soh[valid_anchor_idx],
                           color="gold", edgecolors="black", s=60, zorder=10, marker="*")

            ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=1.5, alpha=0.7)

            l2l3 = c.get("L2_to_L3_Improv", np.nan)
            l2l3_str = f"{l2l3:+.2f}" if not np.isnan(l2l3) else "N/A"
            ax.set_title(f"{model_name}\nL2->L3: {l2l3_str}%",
                         fontsize=9, fontweight="bold")
            ax.set_xlabel("Cycle", fontsize=8)
            ax.set_ylabel("SOH (%)", fontsize=8)
            ax.legend(fontsize=6, loc="best")
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(len(models_for_subplot), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Three-Layer Per-Model Comparison\n"
                     "Layer 1 (Zero-Shot) vs Layer 2 (Fine-Tuned) vs Layer 3 (Domain-Adapted)",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "three_layer_per_model_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] three_layer_per_model_comparison.png")

    # ----------------------------------------------------------------
    # PLOT 6: Confidence Map + Pseudo Labels
    # ----------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top: Pseudo labels with confidence coloring
    scatter = ax1.scatter(cycles, pseudo_labels, c=confidence, cmap="RdYlGn",
                          s=40, edgecolors="black", linewidths=0.3, vmin=0, vmax=1)
    plt.colorbar(scatter, ax=ax1, label="Confidence")

    if valid_anchor_idx:
        ax1.scatter(cycles[valid_anchor_idx],
                    anchor_pseudo_soh[valid_anchor_idx],
                    color="gold", edgecolors="black", s=150, zorder=10,
                    label=f"Anchor Points ({n_anchors})", marker="*")

    if calibrated_soh is not None:
        ax1.plot(cycles, calibrated_soh, color="black", linewidth=2.5,
                 label=f"Meta-Learner ({calibrated_soh[-1]:.1f}%)")

    ax1.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=2,
                label=f"Expected SOH ({EXPECTED_SOH}%)")
    ax1.set_ylabel("SOH (%)", fontsize=12)
    ax1.set_title("Domain-Adapted Pseudo Labels (color = confidence)\n"
                  "Green = high confidence, Red = low confidence",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(True, alpha=0.3)

    # Bottom: Confidence over cycles
    ax2.bar(cycles, confidence, color=plt.cm.RdYlGn(confidence), edgecolor="none", width=0.8)
    ax2.axhline(y=1.0, color="gold", linestyle="--", linewidth=1.5,
                label="Anchor (conf=1.0)")
    ax2.axhline(y=0.5, color="gray", linestyle=":", linewidth=1,
                label="Medium confidence")
    ax2.set_xlabel("Cycle", fontsize=12)
    ax2.set_ylabel("Confidence", fontsize=12)
    ax2.set_title("Per-Cycle Confidence Score", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confidence_map.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [PLOT] confidence_map.png")

    # ----------------------------------------------------------------
    # PLOT 7: L2->L3 Improvement Waterfall
    # ----------------------------------------------------------------
    model_rows = [c for c in comparison
                  if c["Model"] != "META_LEARNER"
                  and not np.isnan(c.get("L2_to_L3_Improv", np.nan))]

    if model_rows:
        fig, ax = plt.subplots(figsize=(14, 6))

        names = [c["Model"].replace("ML_", "").replace("DL_", "") for c in model_rows]
        improvements = [c["L2_to_L3_Improv"] for c in model_rows]
        colors_wf = ["#2CA02C" if imp > 0 else "#D62728" for imp in improvements]

        bars = ax.bar(range(len(names)), improvements, color=colors_wf,
                      edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="black", linewidth=1)

        for i, (imp, name) in enumerate(zip(improvements, names)):
            ax.text(i, imp + (0.2 if imp >= 0 else -0.4),
                    f"{imp:+.2f}%", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Improvement in |Error| (%)", fontsize=11)
        ax.set_title("Layer 2 -> Layer 3 Improvement (Positive = Better)\n"
                     "Domain Adaptation improvement over Fine-Tuning",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "l2_to_l3_improvement_waterfall.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] l2_to_l3_improvement_waterfall.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    total_t0 = time.time()

    print("=" * 70)
    print("Battery B Phase 2 - Layer 3: Domain Adaptation")
    print("Self-Training + Meta-Learner")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Expected SOH: ~{EXPECTED_SOH}%")
    print(f"  Self-training rounds: {SELF_TRAIN_ROUNDS}")
    print(f"  ML anchor weight: {ANCHOR_WEIGHT_ML}x, pseudo weight: {PSEUDO_WEIGHT_ML}x * confidence")
    print(f"  DL fine-tune LR: {DL_FINETUNE_LR}, epochs: {DL_FINETUNE_EPOCHS}")
    print(f"  Convergence threshold: {CONVERGENCE_THRESHOLD}% anchor MAE improvement")
    print(f"  Skip models: {SKIP_FINETUNE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # STEP 1-4: Data Loading & Feature Extraction
    # ================================================================
    print(f"\n{'=' * 70}")
    print("[STEPS 1-4] Data Loading & Feature Extraction")
    print(f"{'=' * 70}")

    raw_df = load_battery_b(MAT_PATH)
    bounds = detect_cycles(raw_df)
    cycle_df = extract_features(raw_df, bounds)

    # Remove short last cycle (same as Layer 2)
    if "cycle_duration" in cycle_df.columns:
        dur = cycle_df["cycle_duration"].values
        if dur[-1] < np.median(dur) * 0.5:
            cycle_df = cycle_df.iloc[:-1].reset_index(drop=True)
            bounds = bounds[:len(cycle_df)]
            print(f"  Last cycle removed (too short). Remaining: {len(cycle_df)}")

    cycles = cycle_df["cycle"].values
    n_cycles = len(cycle_df)

    # Keep a copy without trends for DL models
    cycle_df_no_trends = cycle_df.copy()

    # Add trend features (for ML models)
    cycle_df_with_trends = add_trends(cycle_df)

    # Detect anchor points
    anchor_mask, anchor_indices = detect_anchor_points(cycle_df, bounds, raw_df)

    # Generate polynomial pseudo SOH for anchor points
    anchor_pseudo_soh = generate_pseudo_soh_polynomial(cycle_df, anchor_mask, anchor_indices)

    if anchor_pseudo_soh is None or np.all(np.isnan(anchor_pseudo_soh)):
        print("\n[ERROR] No valid anchor pseudo SOH labels generated.")
        exit(1)

    # ================================================================
    # STEP 5: Load Layer 2 Results
    # ================================================================
    layer2_df = load_layer2_results()

    if layer2_df is None:
        print("\n[ERROR] Layer 2 results are required for Domain Adaptation.")
        print("Run Battery_B_Phase2_FineTuning.py first.")
        exit(1)

    # ================================================================
    # STEP 6: Ensemble Pseudo Label Generation
    # ================================================================
    pseudo_labels, confidence = generate_ensemble_pseudo_labels(
        layer2_df, anchor_mask, anchor_pseudo_soh
    )

    # ================================================================
    # STEP 7: Self-Training Loop
    # ================================================================
    final_results, pseudo_labels, confidence, round_history, all_round_predictions = \
        self_training_loop(
            cycle_df_with_trends, cycle_df_no_trends,
            anchor_mask, anchor_indices, anchor_pseudo_soh,
            pseudo_labels, confidence
        )

    if not final_results:
        print("\n[ERROR] No models produced domain-adapted results!")
        exit(1)

    # ================================================================
    # STEP 8: Meta-Learner Calibration
    # ================================================================
    calibrated_soh, meta_info = meta_learner_calibration(
        final_results, anchor_mask, anchor_indices, anchor_pseudo_soh, cycles
    )

    # ================================================================
    # STEP 9: Results & Visualization
    # ================================================================
    print(f"\n{'=' * 70}")
    print("[STEP 9] Results Comparison & Visualization")
    print(f"{'=' * 70}")

    # Load Layer 1 (zero-shot) for comparison
    zs_df = load_zeroshot_results()

    # Build comparison table
    comparison = build_comparison_table(final_results, calibrated_soh, zs_df, layer2_df, cycles)

    # Save results
    save_results(final_results, calibrated_soh, comparison, round_history,
                 cycles, pseudo_labels, confidence, anchor_mask, meta_info)

    # Generate plots
    plot_all(final_results, calibrated_soh, comparison, round_history,
             all_round_predictions, zs_df, layer2_df,
             cycles, pseudo_labels, confidence, anchor_mask, anchor_indices,
             anchor_pseudo_soh, meta_info)

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    total_elapsed = time.time() - total_t0

    print(f"\n{'=' * 70}")
    print(f"DONE - Layer 3: Domain Adaptation")
    print(f"{'=' * 70}")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Cycles: {n_cycles}, Anchor points: {int(anchor_mask.sum())}")
    print(f"  Self-training rounds completed: {len(round_history)}")
    print(f"  Models: {len(final_results)} "
          f"(ML: {sum(1 for k in final_results if k.startswith('ML_'))}, "
          f"DL: {sum(1 for k in final_results if k.startswith('DL_'))})")

    if calibrated_soh is not None:
        print(f"  Meta-Learner calibrated SOH (last): {calibrated_soh[-1]:.2f}% "
              f"(|err|={abs(EXPECTED_SOH - calibrated_soh[-1]):.2f}%)")

    # Best individual model
    model_rows = [c for c in comparison if c["Model"] != "META_LEARNER"]
    if model_rows:
        best = min(model_rows, key=lambda x: x["L3_AbsErr"])
        print(f"  Best individual model: {best['Model']} "
              f"(SOH={best['L3_DomainAdapt_SOH']:.2f}%, |err|={best['L3_AbsErr']:.2f}%)")

    print(f"{'=' * 70}")
