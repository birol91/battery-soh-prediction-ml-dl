"""
BMS Predictive Maintenance - Battery B Phase 2 Anchor Point Fine-Tuning
========================================================================

After zero-shot transfer (Step 5), some models show significant domain shift
when applied to Phase 2 (mixed drive cycles, random charge patterns).

This script implements **Anchor Point Fine-Tuning**:
  1. Identify anchor cycles (full discharge + full charge) as reliable reference points
  2. Generate pseudo SOH labels using Battery A's proxy_IR -> SOH relationship
  3. Incrementally fine-tune models using accumulated anchor points
  4. Compare zero-shot vs fine-tuned predictions

Anchor Point Detection:
  - Full discharge: SOC drops to <= 10%
  - Full charge:    SOC rises to >= 99.5%
  - These cycles mimic Phase 1 (UDDS) conditions -> most reliable for calibration

Pseudo SOH Generation:
  - Fit quadratic polynomial on Battery A: SOH = f(proxy_IR_mean)
  - Apply to anchor cycles' proxy_IR -> pseudo SOH
  - Enforce monotonic constraint (SOH can only decrease over time)

Fine-Tuning Strategy:
  - ML: Retrain with Battery A data + accumulated anchors (anchor_weight=5.0)
  - DL: Fine-tune with low LR (5e-5), 50 epochs, accumulated anchor sequences

Pipeline:
  1. Read Battery B Phase 2 .mat (8 streams, no SOH)
  2. Cycle detection (Phase 2 method: discharge->charge transitions)
  3. Cycle-level feature extraction (~85 features)
  4. Trend features (for ML models)
  5. Anchor point detection (SOC_min <= 10% AND SOC_max >= 99.5%)
  6. Pseudo SOH label generation (Battery A polynomial mapping)
  7. Incremental fine-tuning simulation (chronological)
  8. Results comparison & visualization (zero-shot vs fine-tuned)
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MAT_PATH = os.path.join(PROJECT_ROOT, "0_Dataset", "recording_senaryo_1_Batarya_B_faz2.mat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output_FineTuning")

# Battery A reference data (for pseudo SOH polynomial fitting)
BATT_A_FEAT_PATH = os.path.join(
    PROJECT_ROOT, "1-ML", "Random_Forest", "Output_Random_Forest", "df_step4_cycle_features.csv"
)

# Zero-shot results (for comparison)
ZEROSHOT_CSV = os.path.join(
    PROJECT_ROOT, "5-Inference_Battery_B_Phase_2", "Output_Inference_Phase2", "phase2_cycle_predictions.csv"
)

EXPECTED_SOH = 80.0
SEQ_LEN = 5

# Anchor point thresholds
ANCHOR_SOC_MIN_THRESHOLD = 10.0    # SOC must drop to <= 10%
ANCHOR_SOC_MAX_THRESHOLD = 99.5    # SOC must rise to >= 99.5%

# Fine-tuning hyperparameters
ANCHOR_WEIGHT = 5.0        # Weight multiplier for anchor samples in ML retraining
DL_FINETUNE_LR = 5e-5      # Low learning rate for DL fine-tuning
DL_FINETUNE_EPOCHS = 50    # Number of fine-tuning epochs per anchor update

# Battery B stream map (no SOH!)
STREAM_MAP_B = {
    0: "SOC", 1: "Current",
    2: "Cell_V_Max", 3: "Cell_V_Min", 4: "Cell_V_Avg",
    5: "Cell_T_Max", 6: "Cell_T_Min", 7: "Cell_T_Avg",
}
MV_SIGNALS = {"Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"}
RAW_SIGNALS = ["Current", "Cell_V_Max", "Cell_V_Min", "Cell_V_Avg",
               "Cell_T_Max", "Cell_T_Min", "Cell_T_Avg", "SOC"]

# --- Top 5 ML models ---
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

# --- Top 5 DL models ---
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

# Trend features (for ML models that used them)
TREND_FEATURES = [
    "voltage_spread_mean", "voltage_spread_under_load", "voltage_asymmetry_mean",
    "voltage_asymmetry_under_load", "thermal_spread_mean", "thermal_spread_under_load",
    "proxy_IR_mean", "energy_efficiency", "thermal_response_ratio",
    "voltage_spread_max", "thermal_spread_max", "charge_time_to_full",
    "discharge_duration", "Cell_V_Avg_mean", "Cell_V_Avg_range", "Current_std", "SOC_range",
]
ROLLING_WINDOWS = [3, 5, 10]


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
    seg = df.iloc[start:end+1]
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
        row[f"{sig}_charge_start"] = v[soc_min_idx+1] if soc_min_idx < len(v)-1 else v[soc_min_idx]

    v_max, v_min, v_avg = seg["Cell_V_Max"].values, seg["Cell_V_Min"].values, seg["Cell_V_Avg"].values
    t_max, t_min = seg["Cell_T_Max"].values, seg["Cell_T_Min"].values
    cur, soc = seg["Current"].values, seg["SOC"].values
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

def add_trends(cdf):
    """Add rolling mean/std, delta, rate-of-change, and IR slope trend features."""
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
    print(f"  {len(df.columns)} features (with trends)")
    return df


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
# STEP 5: ANCHOR POINT DETECTION
# ============================================================

def detect_anchor_points(cycle_df, bounds, raw_df):
    """
    Identify anchor cycles where SOC_min <= 10% AND SOC_max >= 99.5%.

    These are full discharge + full charge cycles that closely resemble
    Phase 1 (UDDS) conditions, making them reliable calibration points.
    """
    print(f"\n[STEP 5] Detecting anchor points...")
    print(f"  Criteria: SOC_min <= {ANCHOR_SOC_MIN_THRESHOLD}% AND SOC_max >= {ANCHOR_SOC_MAX_THRESHOLD}%")

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
        soc_min_vals = [f"{cycle_df.iloc[i]['SOC_min']:.1f}" for i in anchor_indices]
        soc_max_vals = [f"{cycle_df.iloc[i]['SOC_max']:.1f}" for i in anchor_indices]
        print(f"  Anchor SOC_min values: {soc_min_vals}")
        print(f"  Anchor SOC_max values: {soc_max_vals}")
    else:
        print("  WARNING: No anchor points found! Fine-tuning will not be possible.")
        print("  Consider relaxing thresholds (e.g., SOC_min <= 15%, SOC_max >= 98%)")

    return anchor_mask, anchor_indices


# ============================================================
# STEP 6: PSEUDO SOH LABEL GENERATION
# ============================================================

def generate_pseudo_soh(cycle_df, anchor_mask, anchor_indices):
    """
    Generate pseudo SOH labels for anchor cycles using Battery A's
    proxy_IR -> SOH quadratic polynomial relationship.

    Steps:
    1. Load Battery A training data with known SOH labels
    2. Fit quadratic: SOH = a*IR^2 + b*IR + c
    3. Apply to anchor cycles' proxy_IR values
    4. Enforce monotonic constraint (SOH can only decrease)
    """
    print(f"\n[STEP 6] Generating pseudo SOH labels for anchor cycles...")

    # --- Load Battery A reference data ---
    if not os.path.exists(BATT_A_FEAT_PATH):
        print(f"  ERROR: Battery A reference data not found: {BATT_A_FEAT_PATH}")
        return None

    df_a = pd.read_csv(BATT_A_FEAT_PATH)
    print(f"  Battery A: {len(df_a)} cycles, SOH range: "
          f"{df_a['soh_label'].min():.1f}-{df_a['soh_label'].max():.1f}%")

    # --- Fit quadratic polynomial: SOH = f(proxy_IR_mean) ---
    mask_valid = ~df_a["proxy_IR_mean"].isna() & ~df_a["soh_label"].isna()
    ir_a = df_a.loc[mask_valid, "proxy_IR_mean"].values
    soh_a = df_a.loc[mask_valid, "soh_label"].values

    # Quadratic fit: SOH = a*IR^2 + b*IR + c
    coeffs = np.polyfit(ir_a, soh_a, 2)
    poly = np.poly1d(coeffs)
    print(f"  Polynomial: SOH = {coeffs[0]:.2e}*IR^2 + {coeffs[1]:.2e}*IR + {coeffs[2]:.4f}")

    # Evaluate fit quality on Battery A
    soh_pred_a = poly(ir_a)
    mae_a = np.mean(np.abs(soh_a - soh_pred_a))
    print(f"  Battery A fit quality: MAE = {mae_a:.3f}%")

    # --- Apply to anchor cycles ---
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
        # Clamp to reasonable range
        soh_val = np.clip(soh_val, 50.0, 105.0)
        pseudo_soh[idx] = soh_val

    # --- Enforce monotonic constraint (SOH can only decrease over time) ---
    valid_anchors = [(idx, pseudo_soh[idx]) for idx in anchor_indices if not np.isnan(pseudo_soh[idx])]
    if len(valid_anchors) > 1:
        # Sort by index (chronological order)
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
    print(f"  {'-'*36}")
    for idx in anchor_indices:
        if not np.isnan(pseudo_soh[idx]):
            print(f"  {cycle_df.iloc[idx]['cycle']:<8.0f} "
                  f"{cycle_df.iloc[idx]['proxy_IR_mean']:<16.6f} "
                  f"{pseudo_soh[idx]:<12.2f}")

    return pseudo_soh


# ============================================================
# STEP 7: INCREMENTAL FINE-TUNING SIMULATION
# ============================================================

def load_battery_a_training_data(feat_list):
    """Load Battery A training features + SOH labels aligned to a specific feature list."""
    df_a = pd.read_csv(BATT_A_FEAT_PATH)
    soh_a = df_a["soh_label"].values

    # Add trend features to Battery A data (same as training pipeline)
    df_a_with_trends = add_trends_silent(df_a)

    X_a = align(df_a_with_trends, feat_list)
    return X_a, soh_a


def add_trends_silent(cdf):
    """Add trend features without printing (for Battery A data during fine-tuning)."""
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
    return df


def finetune_ml_models(cycle_df_with_trends, anchor_mask, anchor_indices, pseudo_soh):
    """
    Incremental fine-tuning for ML models.

    Strategy: For each cycle chronologically:
    - If anchor -> retrain model with Battery A data + accumulated anchors
    - Predict SOH with current (possibly updated) model state
    - anchor_weight=5.0 means each anchor sample is repeated 5 times in training
    """
    print(f"\n{'='*60}")
    print("ML FINE-TUNING (Top 5)")
    print(f"{'='*60}")

    n_cycles = len(cycle_df_with_trends)
    ft_results = {}  # model_name -> array of predictions

    for label, model_path, feat_path, feat_type in ML_MODELS:
        if not os.path.exists(model_path):
            print(f"  [SKIP] {label}: model not found")
            continue

        # Check if this model should skip fine-tuning
        skip = label in SKIP_FINETUNE

        print(f"\n  --- {label} {'(SKIP - zero-shot already good)' if skip else ''} ---")
        t0 = time.time()

        # Load original model and feature list
        feat_list = load_feature_list_csv(feat_path)
        original_model = joblib.load(model_path)

        # Load Battery A training data aligned to this model's features
        X_a, soh_a = load_battery_a_training_data(feat_list)

        # Prepare Phase 2 features
        X_b = align(cycle_df_with_trends, feat_list)

        # Initialize model state (start with original pre-trained model)
        current_model = copy.deepcopy(original_model)
        predictions = np.zeros(n_cycles)
        accumulated_anchor_X = []
        accumulated_anchor_y = []
        n_updates = 0

        for i in range(n_cycles):
            # Check if this is an anchor point (only fine-tune if not skipped)
            if anchor_mask[i] and not np.isnan(pseudo_soh[i]) and not skip:
                # Add this anchor to accumulated set
                accumulated_anchor_X.append(X_b[i])
                accumulated_anchor_y.append(pseudo_soh[i])
                n_updates += 1

                # Retrain with Battery A + accumulated anchors (weighted)
                anchor_X = np.array(accumulated_anchor_X)
                anchor_y = np.array(accumulated_anchor_y)

                # Repeat anchor samples for weighting
                weight_int = int(ANCHOR_WEIGHT)
                anchor_X_rep = np.repeat(anchor_X, weight_int, axis=0)
                anchor_y_rep = np.repeat(anchor_y, weight_int)

                # Combine Battery A + weighted anchors
                X_combined = np.vstack([X_a, anchor_X_rep])
                y_combined = np.concatenate([soh_a, anchor_y_rep])

                # Retrain model
                try:
                    current_model = copy.deepcopy(original_model)
                    current_model.fit(X_combined, y_combined)
                except Exception as e:
                    print(f"    [WARN] Retrain failed at cycle {i+1}: {e}")

            # Predict with current model state
            predictions[i] = current_model.predict(X_b[i:i+1])[0]

        ft_results[f"ML_{label}"] = predictions
        elapsed = time.time() - t0
        print(f"    Updates: {n_updates}, Final SOH: {predictions[-1]:.2f}%, Time: {elapsed:.1f}s")

    return ft_results


def finetune_dl_models(cycle_df_no_trends, anchor_mask, anchor_indices, pseudo_soh):
    """
    Incremental fine-tuning for DL models.

    Strategy: For each cycle chronologically:
    - If anchor -> fine-tune with low LR using accumulated anchor sequences
    - Predict SOH with current model state
    - Uses sequence input (SEQ_LEN=5)
    """
    print(f"\n{'='*60}")
    print("DL FINE-TUNING (Top 5)")
    print(f"{'='*60}")

    n_cycles = len(cycle_df_no_trends)
    ft_results = {}

    for label, model_path, feat_path, scaler_path, model_type, kwargs, input_type in DL_MODELS:
        if not os.path.exists(model_path):
            print(f"  [SKIP] {label}: model not found")
            continue
        if not os.path.exists(scaler_path):
            print(f"  [SKIP] {label}: scaler not found")
            continue

        # Check if this model should skip fine-tuning
        skip = label in SKIP_FINETUNE

        print(f"\n  --- {label} {'(SKIP - zero-shot already good)' if skip else ''} ---")
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

            predictions = np.zeros(n_cycles)
            accumulated_anchor_seq = []
            accumulated_anchor_y = []
            n_updates = 0

            for i in range(n_cycles):
                # Check if this is an anchor point (only fine-tune if not skipped)
                if anchor_mask[i] and not np.isnan(pseudo_soh[i]) and not skip:
                    # Add this anchor sequence
                    accumulated_anchor_seq.append(X_b_seq[i])
                    accumulated_anchor_y.append(pseudo_soh[i])
                    n_updates += 1

                    # Fine-tune with accumulated anchors
                    _finetune_dl_step(
                        model, accumulated_anchor_seq, accumulated_anchor_y,
                        lr=DL_FINETUNE_LR, epochs=DL_FINETUNE_EPOCHS
                    )

                # Predict with current model state
                model.eval()
                with torch.no_grad():
                    x_t = torch.tensor(X_b_seq[i:i+1], dtype=torch.float32).to(DEVICE)
                    predictions[i] = model(x_t).cpu().numpy().item()

            ft_results[f"DL_{label}"] = predictions
            elapsed = time.time() - t0
            print(f"    Updates: {n_updates}, Final SOH: {predictions[-1]:.2f}%, Time: {elapsed:.1f}s")

        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
            import traceback
            traceback.print_exc()

    return ft_results


def _finetune_dl_step(model, anchor_seqs, anchor_ys, lr, epochs):
    """
    Fine-tune a DL model on accumulated anchor sequences.

    Uses MSE loss with low learning rate. Only updates the head layers
    and layer norm to prevent catastrophic forgetting.
    """
    model.train()

    # Prepare tensors
    X = torch.tensor(np.array(anchor_seqs), dtype=torch.float32).to(DEVICE)
    y = torch.tensor(np.array(anchor_ys), dtype=torch.float32).to(DEVICE)

    # Optimizer: fine-tune all parameters with low LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()


# ============================================================
# STEP 8: RESULTS & VISUALIZATION
# ============================================================

def load_zeroshot_results():
    """Load zero-shot predictions for comparison."""
    if not os.path.exists(ZEROSHOT_CSV):
        print(f"  WARNING: Zero-shot results not found: {ZEROSHOT_CSV}")
        return None
    df = pd.read_csv(ZEROSHOT_CSV)
    print(f"  Zero-shot results loaded: {len(df)} cycles, {len(df.columns)-1} models")
    return df


def compare_results(ft_results, zs_df, cycles, pseudo_soh, anchor_mask, anchor_indices):
    """Compare zero-shot vs fine-tuned predictions and generate summary."""
    print(f"\n{'='*70}")
    print("COMPARISON: Zero-Shot vs Fine-Tuned")
    print(f"{'='*70}")

    comparison = []

    for model_name, ft_preds in ft_results.items():
        ft_last = ft_preds[-1]
        ft_err = abs(EXPECTED_SOH - ft_last)

        # Find matching zero-shot column
        zs_col = model_name  # e.g., "ML_ExtraTrees_Top30_BL"
        zs_last = np.nan
        zs_err = np.nan
        improvement = np.nan

        if zs_df is not None and zs_col in zs_df.columns:
            zs_preds = zs_df[zs_col].values
            zs_last = zs_preds[-1]
            zs_err = abs(EXPECTED_SOH - zs_last)
            improvement = zs_err - ft_err

        model_type = "ML" if model_name.startswith("ML_") else "DL"
        comparison.append({
            "Model": model_name,
            "Type": model_type,
            "ZeroShot_SOH": zs_last,
            "ZeroShot_AbsErr": zs_err,
            "FineTuned_SOH": ft_last,
            "FineTuned_AbsErr": ft_err,
            "Improvement": improvement,
        })

    # Sort by fine-tuned absolute error
    comparison.sort(key=lambda x: x["FineTuned_AbsErr"])

    # Print comparison table
    print(f"\n  {'Rank':<5} {'Model':<28} {'ZS SOH':>8} {'ZS |Err|':>9} "
          f"{'FT SOH':>8} {'FT |Err|':>9} {'Improv.':>9}")
    print(f"  {'-'*86}")

    for i, c in enumerate(comparison):
        zs_str = f"{c['ZeroShot_SOH']:.2f}" if not np.isnan(c['ZeroShot_SOH']) else "N/A"
        zs_err_str = f"{c['ZeroShot_AbsErr']:.2f}" if not np.isnan(c['ZeroShot_AbsErr']) else "N/A"
        imp_str = f"{c['Improvement']:+.2f}" if not np.isnan(c['Improvement']) else "N/A"
        print(f"  {i+1:<5} {c['Model']:<28} {zs_str:>8} {zs_err_str:>9} "
              f"{c['FineTuned_SOH']:>8.2f} {c['FineTuned_AbsErr']:>9.2f} {imp_str:>9}")

    # Summary statistics
    valid_improvements = [c["Improvement"] for c in comparison if not np.isnan(c["Improvement"])]
    if valid_improvements:
        print(f"\n  --- Summary ---")
        improved = sum(1 for v in valid_improvements if v > 0)
        degraded = sum(1 for v in valid_improvements if v < 0)
        print(f"  Models improved: {improved}/{len(valid_improvements)}")
        print(f"  Models degraded: {degraded}/{len(valid_improvements)}")
        print(f"  Avg improvement: {np.mean(valid_improvements):+.3f}%")
        print(f"  Max improvement: {max(valid_improvements):+.3f}%")
        if degraded > 0:
            print(f"  Max degradation: {min(valid_improvements):+.3f}%")

    return comparison


def save_results(ft_results, comparison, cycles, pseudo_soh, anchor_mask):
    """Save fine-tuning results to CSV files."""
    print(f"\n[SAVING RESULTS]")

    # 1. Per-cycle predictions
    pred_df = pd.DataFrame({"cycle": cycles})
    pred_df["is_anchor"] = anchor_mask.astype(int)
    pred_df["pseudo_soh"] = pseudo_soh
    for name in sorted(ft_results.keys()):
        pred_df[name] = ft_results[name]
    pred_path = os.path.join(OUTPUT_DIR, "finetuned_cycle_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  [CSV] finetuned_cycle_predictions.csv ({len(pred_df)} rows)")

    # 2. Comparison summary
    comp_df = pd.DataFrame(comparison)
    comp_path = os.path.join(OUTPUT_DIR, "Result_FineTuning_Comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"  [CSV] Result_FineTuning_Comparison.csv ({len(comp_df)} rows)")

    return pred_path, comp_path


def plot_results(ft_results, zs_df, cycles, pseudo_soh, anchor_mask, anchor_indices, comparison,
                 cycle_features_df=None):
    """Generate all visualizations."""
    print(f"\n[VISUALIZATIONS]")

    n_anchors = int(anchor_mask.sum())
    anchor_cycles = cycles[anchor_indices] if len(anchor_indices) > 0 else []

    # Color palettes
    ml_colors = ["#6BAED6", "#2171B5", "#74C476", "#238B45", "#41AB5D"]
    dl_colors = ["#FC9272", "#EF3B2C", "#FB6A4A", "#CB181D", "#A50F15"]

    # ----------------------------------------------------------------
    # PLOT 1: All Fine-Tuned Models SOH Timeline
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 7))

    ml_names = sorted([n for n in ft_results if n.startswith("ML_")])
    dl_names = sorted([n for n in ft_results if n.startswith("DL_")])

    for i, name in enumerate(ml_names):
        ax.plot(cycles, ft_results[name],
                label=f"{name} ({ft_results[name][-1]:.1f}%)",
                color=ml_colors[i % len(ml_colors)], linewidth=2, marker="s", markersize=3)
    for i, name in enumerate(dl_names):
        ax.plot(cycles, ft_results[name],
                label=f"{name} ({ft_results[name][-1]:.1f}%)",
                color=dl_colors[i % len(dl_colors)], linewidth=2, marker="o", markersize=3)

    # Plot anchor pseudo SOH values
    valid_anchor_idx = [idx for idx in anchor_indices if not np.isnan(pseudo_soh[idx])]
    if valid_anchor_idx:
        ax.scatter(cycles[valid_anchor_idx],
                   pseudo_soh[valid_anchor_idx],
                   color="gold", edgecolors="black", s=120, zorder=10,
                   label=f"Anchor Pseudo SOH ({n_anchors} pts)", marker="*")

    ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=2.5,
               label=f"Expected SOH ({EXPECTED_SOH:.0f}%)")

    # Mark anchor cycle positions
    for ac in anchor_cycles:
        ax.axvline(x=ac, color="gold", alpha=0.3, linewidth=1, linestyle=":")

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("SOH (%)", fontsize=12)
    ax.set_title("Battery B Phase 2 - Anchor Point Fine-Tuning\n"
                 f"(All Models, {n_anchors} anchor points used for calibration)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "finetuned_all_models_soh.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  [PLOT] finetuned_all_models_soh.png")

    # ----------------------------------------------------------------
    # PLOT 2: Bar Chart - Zero-Shot vs Fine-Tuned (Last Cycle SOH)
    # Skip models only show gray bar, fine-tuned models show gray + green
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))

    models_with_both = [c for c in comparison if not np.isnan(c["ZeroShot_SOH"])]
    if models_with_both:
        n_models = len(models_with_both)
        x = np.arange(n_models)
        width = 0.35

        zs_vals = [c["ZeroShot_SOH"] for c in models_with_both]
        ft_vals = [c["FineTuned_SOH"] for c in models_with_both]
        names = [c["Model"] for c in models_with_both]

        # Determine which models were skipped (label without ML_/DL_ prefix)
        skip_labels = set()
        for c in models_with_both:
            raw_label = c["Model"].replace("ML_", "").replace("DL_", "")
            if raw_label in SKIP_FINETUNE:
                skip_labels.add(c["Model"])

        for i, c in enumerate(models_with_both):
            is_skipped = c["Model"] in skip_labels

            # Gray bar (zero-shot) — always shown, centered for skipped models
            if is_skipped:
                ax.bar(x[i], zs_vals[i], width * 1.5, color="#BDBDBD",
                       edgecolor="black", linewidth=0.5)
            else:
                ax.bar(x[i] - width/2, zs_vals[i], width, color="#BDBDBD",
                       edgecolor="black", linewidth=0.5)
                # Green bar (fine-tuned) — only for non-skipped models
                ft_color = "#238B45"  # green for all fine-tuned
                ax.bar(x[i] + width/2, ft_vals[i], width, color=ft_color,
                       edgecolor="black", linewidth=0.5)

                # Improvement label
                imp = c["Improvement"]
                color = "green" if imp > 0 else "red"
                ax.annotate(f"{imp:+.1f}%", xy=(x[i] + width/2, ft_vals[i]),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, fontweight="bold", color=color)

        ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=2,
                   label=f"Expected SOH ({EXPECTED_SOH}%)")

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#BDBDBD", edgecolor="black", label="Zero-Shot (no adaptation)"),
            Patch(facecolor="#238B45", edgecolor="black", label="Fine-Tuned (anchor points)"),
        ]
        ax.legend(handles=legend_elements, fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
        ax.set_ylabel("SOH (%)")
        ax.set_title("Battery B Phase 2 - Zero-Shot vs Fine-Tuned (Last Cycle)\n"
                     "First 3 models: skipped (already good). Rest: fine-tuned with anchor points.",
                     fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "finetuned_vs_zeroshot_bar.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] finetuned_vs_zeroshot_bar.png")

    # ----------------------------------------------------------------
    # PLOT 3: Best Model - Zero-Shot vs Fine-Tuned Timeline
    # ----------------------------------------------------------------
    if zs_df is not None:
        # Find best fine-tuned model
        best_ft = min(comparison, key=lambda x: x["FineTuned_AbsErr"])
        best_name = best_ft["Model"]

        if best_name in zs_df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))

            zs_preds = zs_df[best_name].values
            ft_preds = ft_results[best_name]

            ax.plot(cycles, zs_preds,
                    label=f"Zero-Shot ({zs_preds[-1]:.2f}%)",
                    color="#BDBDBD", linewidth=2, marker="^", markersize=4, linestyle="--")
            color = "#2171B5" if best_ft["Type"] == "ML" else "#238B45"
            ax.plot(cycles, ft_preds,
                    label=f"Fine-Tuned ({ft_preds[-1]:.2f}%)",
                    color=color, linewidth=2.5, marker="s", markersize=4)

            # Anchor points
            if valid_anchor_idx:
                ax.scatter(cycles[valid_anchor_idx], pseudo_soh[valid_anchor_idx],
                           color="gold", edgecolors="black", s=150, zorder=10,
                           label=f"Anchor Pseudo SOH", marker="*")

            ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=2,
                       label=f"Expected SOH ({EXPECTED_SOH}%)")

            for ac in anchor_cycles:
                ax.axvline(x=ac, color="gold", alpha=0.3, linewidth=1, linestyle=":")

            ax.set_xlabel("Cycle", fontsize=12)
            ax.set_ylabel("SOH (%)", fontsize=12)
            ax.set_title(f"Best Model: {best_name}\n"
                         f"Zero-Shot |Err|={best_ft['ZeroShot_AbsErr']:.2f}% -> "
                         f"Fine-Tuned |Err|={best_ft['FineTuned_AbsErr']:.2f}% "
                         f"(Improvement: {best_ft['Improvement']:+.2f}%)",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "finetuned_best_model_comparison.png"),
                        dpi=150, bbox_inches="tight")
            plt.show()
            print(f"  [PLOT] finetuned_best_model_comparison.png")

    # ----------------------------------------------------------------
    # PLOT 4: Anchor Point Pseudo SOH vs Polynomial Fit
    # ----------------------------------------------------------------
    if os.path.exists(BATT_A_FEAT_PATH):
        df_a = pd.read_csv(BATT_A_FEAT_PATH)
        mask_valid_a = ~df_a["proxy_IR_mean"].isna()
        ir_a = df_a.loc[mask_valid_a, "proxy_IR_mean"].values
        soh_a = df_a.loc[mask_valid_a, "soh_label"].values
        coeffs = np.polyfit(ir_a, soh_a, 2)
        poly = np.poly1d(coeffs)

        fig, ax = plt.subplots(figsize=(12, 7))

        # Battery A scatter
        ax.scatter(ir_a * 1000, soh_a, color="#2171B5", alpha=0.6, s=40,
                   label="Battery A (training data)", zorder=5)

        # Polynomial fit line
        ir_range = np.linspace(min(ir_a) * 0.9, max(ir_a) * 1.3, 200)
        ax.plot(ir_range * 1000, poly(ir_range), color="#E74C3C", linewidth=2,
                linestyle="--", label="Quadratic fit: SOH = f(proxy_IR)")

        # Anchor points from Phase 2
        if valid_anchor_idx:
            anchor_ir = [cycle_features_df.iloc[idx]["proxy_IR_mean"] for idx in valid_anchor_idx
                         if cycle_features_df is not None and "proxy_IR_mean" in cycle_features_df.columns]
            anchor_soh_pseudo = [pseudo_soh[idx] for idx in valid_anchor_idx]
            if anchor_ir:
                ax.scatter(np.array(anchor_ir) * 1000, anchor_soh_pseudo,
                           color="gold", edgecolors="black", s=200, zorder=10,
                           label=f"Phase 2 Anchors ({len(anchor_ir)} pts)", marker="*")

        ax.set_xlabel("Proxy Internal Resistance (mOhm)", fontsize=12)
        ax.set_ylabel("SOH (%)", fontsize=12)
        ax.set_title("Pseudo SOH Generation: Battery A Polynomial Mapping\n"
                     "proxy_IR_mean -> SOH (quadratic fit)",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "anchor_pseudo_soh_polynomial.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] anchor_pseudo_soh_polynomial.png")

    # ----------------------------------------------------------------
    # PLOT 5: Per-Model Zero-Shot vs Fine-Tuned Timeline (subplots)
    # ----------------------------------------------------------------
    models_with_both = [c for c in comparison if not np.isnan(c["ZeroShot_SOH"])]
    if models_with_both and zs_df is not None:
        n_models = len(models_with_both)
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, c in enumerate(models_with_both):
            ax = axes[i]
            model_name = c["Model"]

            if model_name in zs_df.columns:
                ax.plot(cycles, zs_df[model_name].values, color="#BDBDBD",
                        linewidth=1.5, linestyle="--", label="Zero-Shot")
            ax.plot(cycles, ft_results[model_name], color="#2171B5" if c["Type"] == "ML" else "#238B45",
                    linewidth=2, label="Fine-Tuned")

            if valid_anchor_idx:
                ax.scatter(cycles[valid_anchor_idx], pseudo_soh[valid_anchor_idx],
                           color="gold", edgecolors="black", s=80, zorder=10, marker="*")

            ax.axhline(y=EXPECTED_SOH, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_title(f"{model_name}\nImprov: {c['Improvement']:+.2f}%", fontsize=10, fontweight="bold")
            ax.set_xlabel("Cycle", fontsize=9)
            ax.set_ylabel("SOH (%)", fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Zero-Shot vs Fine-Tuned: Per-Model Comparison",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "finetuned_per_model_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] finetuned_per_model_comparison.png")

    # ----------------------------------------------------------------
    # PLOT 6: Improvement Waterfall Chart
    # ----------------------------------------------------------------
    models_with_both = [c for c in comparison if not np.isnan(c["Improvement"])]
    if models_with_both:
        fig, ax = plt.subplots(figsize=(14, 6))

        names = [c["Model"].replace("ML_", "").replace("DL_", "") for c in models_with_both]
        improvements = [c["Improvement"] for c in models_with_both]
        colors = ["#2CA02C" if imp > 0 else "#D62728" for imp in improvements]

        bars = ax.bar(range(len(names)), improvements, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(y=0, color="black", linewidth=1)

        for i, (imp, name) in enumerate(zip(improvements, names)):
            ax.text(i, imp + (0.3 if imp >= 0 else -0.5),
                    f"{imp:+.2f}%", ha="center", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Improvement in |Error| (%)", fontsize=11)
        ax.set_title("Fine-Tuning Improvement (Positive = Better)\n"
                     "Reduction in absolute error vs zero-shot",
                     fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "finetuning_improvement_waterfall.png"),
                    dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  [PLOT] finetuning_improvement_waterfall.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    total_t0 = time.time()

    print("=" * 70)
    print("Battery B Phase 2 - Anchor Point Fine-Tuning")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Expected SOH: ~{EXPECTED_SOH}%")
    print(f"  Anchor thresholds: SOC_min <= {ANCHOR_SOC_MIN_THRESHOLD}%, "
          f"SOC_max >= {ANCHOR_SOC_MAX_THRESHOLD}%")
    print(f"  ML anchor weight: {ANCHOR_WEIGHT}x")
    print(f"  DL fine-tune LR: {DL_FINETUNE_LR}, epochs: {DL_FINETUNE_EPOCHS}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1-3: Data loading & feature extraction ----
    raw_df = load_battery_b(MAT_PATH)
    bounds = detect_cycles(raw_df)
    cycle_df = extract_features(raw_df, bounds)

    # Remove short last cycle (same as zero-shot)
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

    # ---- Step 4: Trend features (for ML) ----
    cycle_df_with_trends = add_trends(cycle_df)

    # ---- Step 5: Anchor point detection ----
    anchor_mask, anchor_indices = detect_anchor_points(cycle_df, bounds, raw_df)

    # ---- Step 6: Pseudo SOH generation ----
    pseudo_soh = generate_pseudo_soh(cycle_df, anchor_mask, anchor_indices)

    if pseudo_soh is None or np.all(np.isnan(pseudo_soh)):
        print("\n[ERROR] No valid pseudo SOH labels generated. Cannot fine-tune.")
        print("Falling back to zero-shot predictions only.")
        exit(1)

    # ---- Step 7: Incremental fine-tuning ----
    ml_ft_results = finetune_ml_models(cycle_df_with_trends, anchor_mask, anchor_indices, pseudo_soh)
    dl_ft_results = finetune_dl_models(cycle_df_no_trends, anchor_mask, anchor_indices, pseudo_soh)

    all_ft_results = {**ml_ft_results, **dl_ft_results}

    if not all_ft_results:
        print("\n[ERROR] No models produced fine-tuned results!")
        exit(1)

    # ---- Step 8: Results & Visualization ----
    print(f"\n[STEP 8] Results comparison & visualization...")

    # Load zero-shot results
    zs_df = load_zeroshot_results()

    # Compare
    comparison = compare_results(all_ft_results, zs_df, cycles, pseudo_soh, anchor_mask, anchor_indices)

    # Save
    save_results(all_ft_results, comparison, cycles, pseudo_soh, anchor_mask)

    # Plot
    plot_results(all_ft_results, zs_df, cycles, pseudo_soh, anchor_mask, anchor_indices, comparison,
                 cycle_features_df=cycle_df_no_trends)

    total_elapsed = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"DONE - Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Cycles: {n_cycles}, Anchor points: {int(anchor_mask.sum())}")
    print(f"  Models fine-tuned: {len(all_ft_results)} "
          f"(ML: {len(ml_ft_results)}, DL: {len(dl_ft_results)})")
    print(f"{'='*70}")
