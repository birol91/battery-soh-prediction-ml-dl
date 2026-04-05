"""
BMS Predictive Maintenance - Battery B Inference (Phase 1)
===========================================================

Uses top 5 ML and top 5 DL models trained on Battery A to predict
SOH on Battery B (unseen data).

Top 5 ML: ExtraTrees_Top30_BL, ExtraTrees_Top30_TN, XGBoost_Top30_BL,
          XGBoost_Top30_TN, RF_Top30_TN
Top 5 DL: GRU_Top30_TN, GRU_TN, GRU_Top30_BL, LSTM_Top30_TN, Transformer_TN

Battery B: UDDS drive cycle, no SOH stream (8 signals only).
Ground Truth: SOH = 88% (Simulink shutdown point).

Pipeline:
  1. Read Battery B .mat (8 streams, no SOH)
  2. mV -> V conversion
  3. Cycle boundary detection
  4. Cycle-level feature extraction (~85 features)
  5. Trend features (for ML models only)
  6. ML Inference (5 models)
  7. DL Inference (5 models with scaling + sequences)
  8. Results table + visualizations
"""

import math
import os
import time
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MAT_PATH = os.path.join(PROJECT_ROOT, "0_Dataset", "recording_senaryo_1_Battery_B_SOH88.mat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output_Inference")

GROUND_TRUTH_SOH = 88.0
SEQ_LEN = 5

# Battery B stream map (no SOH!)
STREAM_MAP_B = {
    0: "SOC", 1: "Current",
    2: "Cell_V_Max", 3: "Cell_V_Min", 4: "Cell_V_Avg",
    5: "Cell_T_Max", 6: "Cell_T_Min", 7: "Cell_T_Avg",
}
MV_SIGNALS = {"Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"}
RAW_SIGNALS = ["Current", "Cell_V_Max", "Cell_V_Min", "Cell_V_Avg",
               "Cell_T_Max", "Cell_T_Min", "Cell_T_Avg", "SOC"]

# --- Top 5 ML models (model_path, feature_cols_path, label) ---
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
# (label, model_path, feature_cols_path, scaler_path, model_class, model_kwargs, input_type)
# model_kwargs come from df_step9_best_params.csv for tuned, baseline defaults for baseline
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
# DATA LOADING & FEATURE EXTRACTION (same as training pipeline)
# ============================================================

def load_battery_b(mat_path):
    print(f"\n[STEP 1] Reading Battery B .mat...")
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


def detect_cycles(df, soc_high=99.0, min_dur=5000):
    print(f"\n[STEP 2] Detecting cycle boundaries...")
    soc = df["SOC"].values
    above = soc >= soc_high
    trans = np.diff(above.astype(int))
    enter = np.where(trans == 1)[0]
    exit_h = np.where(trans == -1)[0]
    peaks = []
    if soc[0] >= soc_high and len(exit_h) > 0:
        peaks.append(exit_h[0])
    for i in range(len(enter)):
        ea = exit_h[exit_h > enter[i]]
        if len(ea) > 0:
            peaks.append(ea[0])
    filtered = []
    for p in peaks:
        if not filtered or (p - filtered[-1]) >= min_dur:
            filtered.append(p)
    bounds = [(filtered[i], filtered[i+1]) for i in range(len(filtered)-1)]
    if filtered and (len(df) - filtered[-1]) > min_dur:
        bounds.append((filtered[-1], len(df)-1))
    print(f"  {len(bounds)} cycles detected")
    return bounds


def compute_cycle_features(df, start, end, cycle_num):
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

    v_sp = v_max - v_min
    row["voltage_spread_mean"] = np.mean(v_sp)
    row["voltage_spread_max"] = np.max(v_sp)
    row["voltage_spread_std"] = np.std(v_sp)

    vts = v_max - v_min
    mm = vts > 0.001
    if np.any(mm):
        va = (v_max[mm]-v_avg[mm])/(v_avg[mm]-v_min[mm]+1e-2)
        row["voltage_asymmetry_mean"] = np.mean(va)
    else:
        row["voltage_asymmetry_mean"] = 1.0

    am = abs_cur > 0.5
    row["voltage_spread_under_load"] = np.mean(v_sp[am]) if np.any(am) else np.mean(v_sp)
    amm = am & (vts > 0.001)
    if np.any(amm):
        row["voltage_asymmetry_under_load"] = np.mean((v_max[amm]-v_avg[amm])/(v_avg[amm]-v_min[amm]+1e-2))
    else:
        row["voltage_asymmetry_under_load"] = 1.0

    t_sp = t_max - t_min
    row["thermal_spread_mean"] = np.mean(t_sp)
    row["thermal_spread_max"] = np.max(t_sp)
    row["thermal_spread_std"] = np.std(t_sp)
    row["thermal_spread_under_load"] = np.mean(t_sp[am]) if np.any(am) else np.mean(t_sp)

    di, dv = np.diff(cur), np.diff(v_avg)
    ldm = np.abs(di) > 2.0
    if np.sum(ldm) > 5:
        pir = np.abs(dv[ldm]/di[ldm])
        pir = pir[pir < 1.0]
        if len(pir) > 0:
            row["proxy_IR_mean"], row["proxy_IR_median"], row["proxy_IR_std"] = np.mean(pir), np.median(pir), np.std(pir)
        else:
            row["proxy_IR_mean"] = row["proxy_IR_median"] = row["proxy_IR_std"] = np.nan
    else:
        row["proxy_IR_mean"] = row["proxy_IR_median"] = row["proxy_IR_std"] = np.nan

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

    sam = soc[soc_min_idx:]
    for thr, name in [(50, "50"), (80, "80"), (95, "95"), (99.5, "full")]:
        idx = np.where(sam >= thr)[0]
        row[f"charge_time_to_{name}"] = idx[0] if len(idx) > 0 else np.nan
    row["discharge_duration"] = soc_min_idx
    row["soc_range"] = np.max(soc) - np.min(soc)

    W = 60
    if len(t_sp) > W:
        ts_s = pd.Series(t_sp).rolling(W, min_periods=W).mean().values
        ac_s = pd.Series(abs_cur).rolling(W, min_periods=W).mean().values
        dts, dai = np.diff(ts_s), np.diff(ac_s)
        vm = (~np.isnan(dts)) & (~np.isnan(dai)) & (np.abs(dai) > 0.5)
        if np.sum(vm) > 3:
            r = dts[vm]/dai[vm]
            r = r[np.abs(r) < 10]
            row["thermal_response_ratio"] = np.mean(r) if len(r) > 0 else np.nan
        else:
            row["thermal_response_ratio"] = np.nan
    else:
        row["thermal_response_ratio"] = np.nan

    row["cycle_duration"] = end - start + 1
    row["active_duration"] = int((abs_cur > 0.5).sum())
    row["rest_duration"] = (end - start + 1) - row["active_duration"]
    return row


def extract_features(df, bounds):
    print(f"\n[STEP 3] Extracting cycle features...")
    rows = [compute_cycle_features(df, s, e, i+1) for i, (s, e) in enumerate(bounds)]
    cdf = pd.DataFrame(rows)
    cdf = cdf.ffill().bfill()
    for c in cdf.columns:
        if cdf[c].isna().any():
            cdf[c] = cdf[c].fillna(cdf[c].median())
    print(f"  {len(cdf)} cycles, {len(cdf.columns)} features")
    return cdf


def add_trends(cdf):
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
            ws = max(0, i-9)
            wd = pir[ws:i+1]
            if len(wd) >= 3:
                sl[i] = np.polyfit(np.arange(len(wd)), wd, 1)[0]
        df["proxy_IR_trend_slope_10"] = sl
    print(f"  {len(df.columns)} features (with trends)")
    return df


# ============================================================
# DL MODEL DEFINITIONS
# ============================================================

class SOH_GRU(nn.Module):
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_dim, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        self.ln = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
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
        self.head = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
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
        pe[:, 1::2] = torch.cos(pos * div[:d_model//2]) if d_model % 2 == 0 else torch.cos(pos * div[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class SOH_Transformer(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pe = PositionalEncoding(d_model)
        el = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(el, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
    def forward(self, x):
        h = self.pe(self.proj(x))
        h = self.dropout(self.ln(self.enc(h)[:, -1, :]))
        return self.head(h).squeeze(-1)


def create_dl_model(model_type, n_features, kwargs):
    if model_type == "GRU":
        return SOH_GRU(n_features, **kwargs)
    elif model_type == "LSTM":
        return SOH_LSTM(n_features, **kwargs)
    elif model_type == "Transformer":
        return SOH_Transformer(n_features, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def make_sequences(X, seq_len):
    n, nf = X.shape
    seqs = np.zeros((n, seq_len, nf), dtype=np.float32)
    for i in range(n):
        if i < seq_len:
            pad = seq_len - (i + 1)
            seqs[i, pad:, :] = X[:i+1, :]
        else:
            seqs[i] = X[i-seq_len+1:i+1, :]
    return seqs


# ============================================================
# INFERENCE FUNCTIONS
# ============================================================

def load_feature_list_csv(csv_path):
    """Load feature names from df_step7_selected_features.csv"""
    df = pd.read_csv(csv_path, nrows=0)
    return [c for c in df.columns if c not in ["cycle", "soh_label"]]


def load_feature_list_txt(txt_path):
    with open(txt_path) as f:
        return [l.strip() for l in f if l.strip()]


def align(features_df, feat_list):
    df = features_df.copy()
    for c in feat_list:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feat_list].values.astype(np.float32)
    return np.nan_to_num(X, nan=0.0)


def run_ml_inference(features_df_with_trends):
    print(f"\n{'='*60}")
    print("ML INFERENCE (Top 5)")
    print(f"{'='*60}")
    results = {}
    for label, model_path, feat_path, feat_type in ML_MODELS:
        if not os.path.exists(model_path):
            print(f"  [SKIP] {label}: model not found")
            continue
        feat_list = load_feature_list_csv(feat_path)
        X = align(features_df_with_trends, feat_list)
        model = joblib.load(model_path)
        preds = model.predict(X)
        results[f"ML_{label}"] = preds
        err = GROUND_TRUTH_SOH - preds[-1]
        print(f"  [OK] {label}: last cycle SOH={preds[-1]:.2f}% (error={err:+.2f}%)")
    return results


def run_dl_inference(features_df_no_trends):
    print(f"\n{'='*60}")
    print("DL INFERENCE (Top 5)")
    print(f"{'='*60}")
    results = {}
    for label, model_path, feat_path, scaler_path, model_type, kwargs, input_type in DL_MODELS:
        if not os.path.exists(model_path):
            print(f"  [SKIP] {label}: model not found")
            continue
        if not os.path.exists(scaler_path):
            print(f"  [SKIP] {label}: scaler not found")
            continue
        try:
            feat_list = load_feature_list_txt(feat_path)
            X = align(features_df_no_trends, feat_list)
            n_features = len(feat_list)

            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X).astype(np.float32)

            X_seq = make_sequences(X_scaled, SEQ_LEN)

            model = create_dl_model(model_type, n_features, kwargs)
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            model.eval()
            model = model.to(DEVICE)

            with torch.no_grad():
                x_t = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
                preds = model(x_t).cpu().numpy()

            results[f"DL_{label}"] = preds
            err = GROUND_TRUTH_SOH - preds[-1]
            print(f"  [OK] {label}: last cycle SOH={preds[-1]:.2f}% (error={err:+.2f}%)")
        except Exception as e:
            print(f"  [ERROR] {label}: {e}")
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Battery B Inference - Top 5 ML + Top 5 DL")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Ground Truth: SOH = {GROUND_TRUTH_SOH}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Feature extraction ---
    df = load_battery_b(MAT_PATH)
    bounds = detect_cycles(df)
    cycle_df = extract_features(df, bounds)

    # Remove short last cycle
    if "cycle_duration" in cycle_df.columns:
        dur = cycle_df["cycle_duration"].values
        if dur[-1] < np.median(dur) * 0.5:
            cycle_df = cycle_df.iloc[:-1].reset_index(drop=True)
            print(f"  Last cycle removed (too short). Remaining: {len(cycle_df)}")

    cycles = cycle_df["cycle"].values

    # --- ML needs trend features ---
    cycle_df_with_trends = add_trends(cycle_df)

    # --- Run inference ---
    ml_results = run_ml_inference(cycle_df_with_trends)
    dl_results = run_dl_inference(cycle_df)  # DL uses no trends

    all_results = {**ml_results, **dl_results}

    if not all_results:
        print("\n[ERROR] No models produced results!")
        exit(1)

    # --- Results table ---
    print(f"\n{'='*70}")
    print("RESULTS - Battery B SOH Prediction")
    print(f"{'='*70}")
    print(f"  Ground Truth: SOH = {GROUND_TRUTH_SOH}%")
    print(f"\n  {'Rank':<6} {'Model':<30} {'Last Cycle SOH':<18} {'Error':<12} {'|Error|':<10}")
    print(f"  {'-'*76}")

    summary = []
    for name, preds in all_results.items():
        err = GROUND_TRUTH_SOH - preds[-1]
        summary.append({"Model": name, "SOH_Pred": preds[-1], "Error": err, "Abs_Error": abs(err),
                         "Type": "ML" if name.startswith("ML_") else "DL"})
    summary.sort(key=lambda x: x["Abs_Error"])

    for i, s in enumerate(summary):
        print(f"  {i+1:<6} {s['Model']:<30} {s['SOH_Pred']:<18.2f} {s['Error']:<+12.2f} {s['Abs_Error']:<10.2f}")

    # --- Save CSVs ---
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "Result_Inference.csv"), index=False)
    print(f"\n[CSV] Result_Inference.csv saved")

    pred_df = pd.DataFrame({"cycle": cycles})
    for name in sorted(all_results.keys()):
        pred_df[name] = all_results[name]
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "cycle_predictions.csv"), index=False)
    print(f"[CSV] cycle_predictions.csv saved")

    # --- Visualizations ---

    # 1. All models SOH prediction
    fig, ax = plt.subplots(figsize=(16, 7))
    ml_colors = ["#6BAED6", "#2171B5", "#74C476", "#238B45", "#41AB5D"]
    dl_colors = ["#FC9272", "#EF3B2C", "#FB6A4A", "#CB181D", "#A50F15"]
    ml_names = sorted([n for n in all_results if n.startswith("ML_")])
    dl_names = sorted([n for n in all_results if n.startswith("DL_")])

    for i, name in enumerate(ml_names):
        ax.plot(cycles, all_results[name], label=f"{name} ({all_results[name][-1]:.1f}%)",
                color=ml_colors[i % len(ml_colors)], linewidth=2, marker="s", markersize=3)
    for i, name in enumerate(dl_names):
        ax.plot(cycles, all_results[name], label=f"{name} ({all_results[name][-1]:.1f}%)",
                color=dl_colors[i % len(dl_colors)], linewidth=2, marker="o", markersize=3)

    ax.axhline(y=GROUND_TRUTH_SOH, color="red", linestyle="--", linewidth=2.5,
               label=f"Ground Truth (SOH={GROUND_TRUTH_SOH:.0f}%)")
    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("SOH (%)", fontsize=12)
    ax.set_title("Battery B - SOH Prediction (Top 5 ML + Top 5 DL)\n"
                 "Models trained on Battery A, inference on Battery B (unseen)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "all_models_soh.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] all_models_soh.png")

    # 2. Bar chart - last cycle SOH
    fig, ax = plt.subplots(figsize=(14, 7))
    bar_colors = ["#2171B5" if s["Type"] == "ML" else "#238B45" for s in summary]
    bars = ax.bar(range(len(summary)), [s["SOH_Pred"] for s in summary],
                  color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=GROUND_TRUTH_SOH, color="red", linestyle="--", linewidth=2,
               label=f"Ground Truth ({GROUND_TRUTH_SOH}%)")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([s["Model"] for s in summary], rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("SOH (%)")
    ax.set_title("Battery B - Last Cycle SOH Prediction\n(Blue=ML, Green=DL, Red line=Ground Truth)",
                 fontweight="bold")
    for i, s in enumerate(summary):
        ax.text(i, s["SOH_Pred"] + 0.3, f"{s['Error']:+.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "last_cycle_bar.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] last_cycle_bar.png")

    # 3. Best ML vs Best DL
    best_ml = next((s for s in summary if s["Type"] == "ML"), None)
    best_dl = next((s for s in summary if s["Type"] == "DL"), None)
    if best_ml and best_dl:
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cycles, all_results[best_ml["Model"]], label=f"{best_ml['Model']} ({best_ml['SOH_Pred']:.2f}%)",
                color="#2171B5", linewidth=2.5, marker="s", markersize=4)
        ax.plot(cycles, all_results[best_dl["Model"]], label=f"{best_dl['Model']} ({best_dl['SOH_Pred']:.2f}%)",
                color="#238B45", linewidth=2.5, marker="o", markersize=4)
        ax.axhline(y=GROUND_TRUTH_SOH, color="red", linestyle="--", linewidth=2,
                   label=f"Ground Truth ({GROUND_TRUTH_SOH}%)")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("SOH (%)")
        ax.set_title(f"Battery B - Best ML vs Best DL\n"
                     f"ML: {best_ml['Model']} | DL: {best_dl['Model']}",
                     fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "best_ml_vs_dl.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[PLOT] best_ml_vs_dl.png")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("INFERENCE COMPLETED")
    print(f"{'='*70}")
    print(f"  Best overall: {summary[0]['Model']} (SOH={summary[0]['SOH_Pred']:.2f}%, error={summary[0]['Error']:+.2f}%)")
    if best_ml:
        print(f"  Best ML:      {best_ml['Model']} (SOH={best_ml['SOH_Pred']:.2f}%, error={best_ml['Error']:+.2f}%)")
    if best_dl:
        print(f"  Best DL:      {best_dl['Model']} (SOH={best_dl['SOH_Pred']:.2f}%, error={best_dl['Error']:+.2f}%)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*70}")
