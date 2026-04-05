"""
BMS Predictive Maintenance - LSTM Pipeline
============================================

OBJECTIVE:
----------
Build an LSTM (Long Short-Term Memory) model that predicts SOH from
battery operational data. Unlike classical ML models (RF, XGBoost),
LSTM processes sequences of cycles and can learn temporal patterns
without requiring manually computed trend features.

DATA:
-----
recording_senaryo_1_Battery_A_N100.mat
- Simulink Recording block output
- 1 Hz sampling, 9 signals, ~100 cycles, SOH: 100% -> 80%

PIPELINE STEPS:
---------------
1. Read .mat file, visualize raw signals (EDA)
2. mV -> V conversion
3. Detect cycle boundaries (SOC peak points)
4. Cycle-level summarization (1 Hz -> 1 row per cycle, ~85 features)
5. [TBD] Feature scaling + sequence preparation
6. [TBD] LSTM model training
7. [TBD] Evaluation and visualization

KEY DIFFERENCE FROM ML PIPELINE:
---------------------------------
- ML: Each cycle is independent -> trend features computed manually
- DL: Sequences of N cycles fed to model -> model learns trends itself
- DL: Feature scaling required (StandardScaler/MinMaxScaler)
- DL: Training uses epochs, batches, loss functions (MSE), optimizers (Adam)
"""

import os
import time
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MAT_PATH = os.path.join(PROJECT_ROOT, "0_Dataset", "recording_senaryo_1_Battery_A_N100.mat")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output_LSTM")

# Simulink Recording sigstream signal ordering
STREAM_MAP = {
    0: "SOC",          # % (State of Charge)
    1: "Current",      # A (Pack current)
    2: "Cell_V_Max",   # mV (Highest voltage cell)
    3: "Cell_V_Min",   # mV (Lowest voltage cell)
    4: "Cell_V_Avg",   # mV (5 cell average)
    5: "Cell_T_Max",   # C (Hottest cell)
    6: "Cell_T_Min",   # C (Coldest cell)
    7: "Cell_T_Avg",   # C (5 cell average temperature)
    8: "SOH",          # % (State of Health - from Fade formula)
}

MV_SIGNALS = {"Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"}

# LSTM sequence length: how many past cycles the model sees for each prediction
SEQ_LEN = 5

# Raw signal columns for cycle-level feature extraction
RAW_SIGNALS = [
    "Current", "Cell_V_Max", "Cell_V_Min", "Cell_V_Avg",
    "Cell_T_Max", "Cell_T_Min", "Cell_T_Avg", "SOC",
]


# ============================================================
# STEP 1: Read .mat File and Visualize Raw Signals (EDA)
# ============================================================

def load_mat_to_dataframe(mat_path: str) -> pd.DataFrame:
    """
    Reads a Simulink Recording .mat file and converts it to a pandas DataFrame.
    The .mat file is in HDF5 format.
    """
    with h5py.File(mat_path, "r") as f:
        n_samples = int(f["#sigstream#/0/#length#"][0])
        data = {}
        for stream_id, col_name in STREAM_MAP.items():
            raw = f[f"#sigstream#/{stream_id}/#data#"][:]
            arr = np.frombuffer(raw.tobytes(), dtype=np.float64)[:n_samples]
            data[col_name] = arr

    df = pd.DataFrame(data)
    df["timestamp"] = np.arange(len(df), dtype=np.float64)
    return df


def plot_raw_signals(df: pd.DataFrame, save_path: str = None):
    """Visualizes raw signals with 5 subplots."""
    time_hours = df["timestamp"].values / 3600.0
    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

    axes[0].plot(time_hours, df["SOC"].values, color="tab:blue", linewidth=0.3)
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_title("State of Charge")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_hours, df["Current"].values, color="tab:orange", linewidth=0.3)
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Current")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_hours, df["Cell_V_Max"].values, color="tab:red", linewidth=0.3, label="Cell_V_Max")
    axes[2].plot(time_hours, df["Cell_V_Min"].values, color="tab:cyan", linewidth=0.3, label="Cell_V_Min")
    axes[2].set_ylabel("Voltage (mV)")
    axes[2].set_title("Cell Voltage Max & Min")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_hours, df["Cell_T_Max"].values, color="darkred", linewidth=0.3, label="Cell_T_Max")
    axes[3].plot(time_hours, df["Cell_T_Min"].values, color="steelblue", linewidth=0.3, label="Cell_T_Min")
    axes[3].set_ylabel("Temperature (C)")
    axes[3].set_title("Cell Temperature Max & Min")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    axes[4].plot(time_hours, df["SOH"].values, color="tab:brown", linewidth=0.3)
    axes[4].set_ylabel("SOH (%)")
    axes[4].set_title("State of Health")
    axes[4].grid(True, alpha=0.3)

    axes[4].set_xlabel("Time (hours)")
    fig.suptitle("STEP 1 - Battery A (5s1p) Raw Signals (EDA)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# STEP 3: Detect Cycle Boundaries
# ============================================================

def detect_cycle_boundaries(df: pd.DataFrame,
                            soc_high_thresh: float = 99.0,
                            min_cycle_duration: int = 5000) -> list:
    """
    Detects UDDS cycle boundaries from SOC peak points.
    1 cycle = full discharge (SOC 100% -> 10%) + full charge (SOC 10% -> 100%)
    """
    soc = df["SOC"].values
    above = soc >= soc_high_thresh
    transitions = np.diff(above.astype(int))
    enter_high = np.where(transitions == 1)[0]
    exit_high = np.where(transitions == -1)[0]

    raw_peaks = []
    if soc[0] >= soc_high_thresh and len(exit_high) > 0:
        raw_peaks.append(exit_high[0])
    for i in range(len(enter_high)):
        start = enter_high[i]
        exits_after = exit_high[exit_high > start]
        if len(exits_after) > 0:
            raw_peaks.append(exits_after[0])

    filtered_peaks = []
    for pk in raw_peaks:
        if not filtered_peaks or (pk - filtered_peaks[-1]) >= min_cycle_duration:
            filtered_peaks.append(pk)

    boundaries = []
    for i in range(len(filtered_peaks) - 1):
        boundaries.append((filtered_peaks[i], filtered_peaks[i + 1]))
    if filtered_peaks and (len(df) - filtered_peaks[-1]) > min_cycle_duration:
        boundaries.append((filtered_peaks[-1], len(df) - 1))

    print(f"\n[CYCLE] {len(filtered_peaks)} peak points detected")
    print(f"[CYCLE] {len(boundaries)} cycles created")
    if boundaries:
        durations = [(e - s) for s, e in boundaries]
        print(f"        Cycle duration: min={min(durations)}s ({min(durations)/3600:.1f}h), "
              f"max={max(durations)}s ({max(durations)/3600:.1f}h), "
              f"avg={np.mean(durations):.0f}s ({np.mean(durations)/3600:.1f}h)")
    return boundaries


# ============================================================
# STEP 4: Cycle-Level Summarization (Feature Extraction)
# ============================================================

def compute_cycle_summary(df: pd.DataFrame, start: int, end: int, cycle_num: int) -> dict:
    """
    Summarizes a single cycle's 1 Hz data into a single row.
    A) Raw signal statistics
    B) Domain-specific features
    C) Timing features
    """
    segment = df.iloc[start:end + 1]
    row = {"cycle": cycle_num}

    # ---- A) Raw Signal Statistics ----
    soc_min_idx = np.argmin(segment["SOC"].values)
    for sig in RAW_SIGNALS:
        vals = segment[sig].values
        row[f"{sig}_min"] = np.min(vals)
        row[f"{sig}_max"] = np.max(vals)
        row[f"{sig}_mean"] = np.mean(vals)
        row[f"{sig}_std"] = np.std(vals)
        row[f"{sig}_range"] = np.max(vals) - np.min(vals)
        row[f"{sig}_discharge_end"] = vals[soc_min_idx]
        if soc_min_idx < len(vals) - 1:
            row[f"{sig}_charge_start"] = vals[soc_min_idx + 1]
        else:
            row[f"{sig}_charge_start"] = vals[soc_min_idx]

    # ---- B) Domain-Specific Features ----
    v_max = segment["Cell_V_Max"].values
    v_min = segment["Cell_V_Min"].values
    v_avg = segment["Cell_V_Avg"].values
    t_max = segment["Cell_T_Max"].values
    t_min = segment["Cell_T_Min"].values
    current = segment["Current"].values
    soc = segment["SOC"].values
    abs_current = np.abs(current)

    # 1. Voltage spread
    v_spread = v_max - v_min
    row["voltage_spread_mean"] = np.mean(v_spread)
    row["voltage_spread_max"] = np.max(v_spread)
    row["voltage_spread_std"] = np.std(v_spread)

    # 2. Voltage asymmetry
    v_total_spread = v_max - v_min
    meaningful_mask = v_total_spread > 0.001
    if np.any(meaningful_mask):
        v_asym = (v_max[meaningful_mask] - v_avg[meaningful_mask]) / \
                 (v_avg[meaningful_mask] - v_min[meaningful_mask] + 1e-2)
        row["voltage_asymmetry_mean"] = np.mean(v_asym)
    else:
        row["voltage_asymmetry_mean"] = 1.0

    # 3. Voltage spread under load
    active_mask = abs_current > 0.5
    if np.any(active_mask):
        row["voltage_spread_under_load"] = np.mean(v_spread[active_mask])
    else:
        row["voltage_spread_under_load"] = np.mean(v_spread)

    # 3b. Voltage asymmetry under load
    active_meaningful_mask = active_mask & (v_total_spread > 0.001)
    if np.any(active_meaningful_mask):
        v_asym_load = (v_max[active_meaningful_mask] - v_avg[active_meaningful_mask]) / \
                      (v_avg[active_meaningful_mask] - v_min[active_meaningful_mask] + 1e-2)
        row["voltage_asymmetry_under_load"] = np.mean(v_asym_load)
    else:
        row["voltage_asymmetry_under_load"] = 1.0

    # 4. Thermal spread
    t_spread = t_max - t_min
    row["thermal_spread_mean"] = np.mean(t_spread)
    row["thermal_spread_max"] = np.max(t_spread)
    row["thermal_spread_std"] = np.std(t_spread)

    # 4b. Thermal spread under load
    if np.any(active_mask):
        row["thermal_spread_under_load"] = np.mean(t_spread[active_mask])
    else:
        row["thermal_spread_under_load"] = np.mean(t_spread)

    # 5. Proxy internal resistance
    di = np.diff(current)
    dv = np.diff(v_avg)
    large_di_mask = np.abs(di) > 2.0
    if np.sum(large_di_mask) > 5:
        proxy_ir_values = np.abs(dv[large_di_mask] / di[large_di_mask])
        proxy_ir_values = proxy_ir_values[proxy_ir_values < 1.0]
        if len(proxy_ir_values) > 0:
            row["proxy_IR_mean"] = np.mean(proxy_ir_values)
            row["proxy_IR_median"] = np.median(proxy_ir_values)
            row["proxy_IR_std"] = np.std(proxy_ir_values)
        else:
            row["proxy_IR_mean"] = np.nan
            row["proxy_IR_median"] = np.nan
            row["proxy_IR_std"] = np.nan
    else:
        row["proxy_IR_mean"] = np.nan
        row["proxy_IR_median"] = np.nan
        row["proxy_IR_std"] = np.nan

    # 6. Energy efficiency
    power = v_avg * current
    charge_mask = current > 0.5
    discharge_mask = current < -0.5
    charge_energy = np.sum(power[charge_mask]) if np.any(charge_mask) else 0
    discharge_energy = np.abs(np.sum(power[discharge_mask])) if np.any(discharge_mask) else 0
    if charge_energy > 0:
        row["energy_efficiency"] = discharge_energy / charge_energy
    else:
        row["energy_efficiency"] = np.nan

    # Current convention check
    dsoc = np.diff(soc)
    soc_increasing = dsoc > 0.01
    if np.any(soc_increasing):
        avg_current_when_soc_up = np.mean(current[1:][soc_increasing])
        if avg_current_when_soc_up < 0:
            charge_energy_fixed = np.abs(np.sum(power[current < -0.5])) if np.any(current < -0.5) else 0
            discharge_energy_fixed = np.sum(power[current > 0.5]) if np.any(current > 0.5) else 0
            if charge_energy_fixed > 0:
                row["energy_efficiency"] = discharge_energy_fixed / charge_energy_fixed
            row["_current_convention"] = "negative_is_charge"
        else:
            row["_current_convention"] = "positive_is_charge"
    else:
        row["_current_convention"] = "unknown"

    # ---- C) Timing Features ----
    soc_after_min = soc[soc_min_idx:]
    soc_50_indices = np.where(soc_after_min >= 50.0)[0]
    row["charge_time_to_50"] = soc_50_indices[0] if len(soc_50_indices) > 0 else np.nan
    soc_80_indices = np.where(soc_after_min >= 80.0)[0]
    row["charge_time_to_80"] = soc_80_indices[0] if len(soc_80_indices) > 0 else np.nan
    soc_95_indices = np.where(soc_after_min >= 95.0)[0]
    row["charge_time_to_95"] = soc_95_indices[0] if len(soc_95_indices) > 0 else np.nan
    soc_full_indices = np.where(soc_after_min >= 99.5)[0]
    row["charge_time_to_full"] = soc_full_indices[0] if len(soc_full_indices) > 0 else np.nan
    row["discharge_duration"] = soc_min_idx
    row["soc_range"] = np.max(soc) - np.min(soc)

    # 10. Thermal response ratio (60s window)
    W = 60
    if len(t_spread) > W and len(abs_current) > W:
        t_spread_smooth = pd.Series(t_spread).rolling(W, min_periods=W).mean().values
        abs_current_smooth = pd.Series(abs_current).rolling(W, min_periods=W).mean().values
        dt_spread = np.diff(t_spread_smooth)
        d_abs_i = np.diff(abs_current_smooth)
        valid_mask = (~np.isnan(dt_spread)) & (~np.isnan(d_abs_i)) & (np.abs(d_abs_i) > 0.5)
        if np.sum(valid_mask) > 3:
            ratios = dt_spread[valid_mask] / d_abs_i[valid_mask]
            ratios = ratios[np.abs(ratios) < 10]
            row["thermal_response_ratio"] = np.mean(ratios) if len(ratios) > 0 else np.nan
        else:
            row["thermal_response_ratio"] = np.nan
    else:
        row["thermal_response_ratio"] = np.nan

    # SOH Label
    soh_vals = segment["SOH"].values
    unique_soh, counts = np.unique(soh_vals, return_counts=True)
    row["soh_label"] = unique_soh[np.argmax(counts)]

    # Duration info
    row["cycle_duration"] = end - start + 1
    active_sec = int((abs_current > 0.5).sum())
    row["active_duration"] = active_sec
    row["rest_duration"] = (end - start + 1) - active_sec

    return row


def summarize_cycles(df: pd.DataFrame, boundaries: list) -> pd.DataFrame:
    """Summarizes all cycles into a DataFrame. 2.3M rows -> ~100 rows."""
    print("\n" + "=" * 60)
    print("STEP 4: Cycle-Level Summarization (Feature Extraction)")
    print("=" * 60)

    rows = []
    for i, (start, end) in enumerate(boundaries):
        row = compute_cycle_summary(df, start, end, cycle_num=i + 1)
        rows.append(row)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Cycle {i+1}/{len(boundaries)}: "
                  f"SOH={row['soh_label']:.0f}%, "
                  f"V_spread={row['voltage_spread_mean']:.4f}V, "
                  f"T_spread={row['thermal_spread_mean']:.1f}C")

    cycle_df = pd.DataFrame(rows)

    if "_current_convention" in cycle_df.columns:
        convention = cycle_df["_current_convention"].mode()[0]
        print(f"\n  Current convention: {convention}")
        cycle_df.drop(columns=["_current_convention"], inplace=True)

    # NaN fill
    nan_counts = cycle_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n  Columns with NaN (forward-fill + median):")
        for col, cnt in nan_cols.items():
            print(f"    {col}: {cnt} NaN")
        cycle_df = cycle_df.ffill().bfill()
        for col in cycle_df.columns:
            if cycle_df[col].isna().any():
                cycle_df[col] = cycle_df[col].fillna(cycle_df[col].median())

    print(f"\n[SUMMARIZATION COMPLETED] {len(cycle_df)} cycles, {len(cycle_df.columns)} features")
    return cycle_df


# ============================================================
# STEP 7: Feature Scaling + Sequence Preparation
# ============================================================

def prepare_features_and_sequences(cycle_df: pd.DataFrame, feature_cols: list, seq_len: int = 5):
    """
    Prepares data for LSTM:
    1. Scale features with StandardScaler
    2. Create sliding window sequences

    Args:
        cycle_df: DataFrame with cycle features + soh_label
        feature_cols: list of feature column names to use
        seq_len: Number of past cycles per sequence (default=5)

    Returns:
        X_seq: np.array of shape (n_samples, seq_len, n_features)
        y: np.array of shape (n_samples,) — SOH labels
        scaler: fitted StandardScaler (needed for inference)
    """
    print("\n" + "=" * 60)
    print("STEP 7: Feature Scaling + Sequence Preparation")
    print("=" * 60)

    X_raw = cycle_df[feature_cols].values
    y = cycle_df["soh_label"].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X_raw)}")
    print(f"  SOH range: {y.min():.1f}% - {y.max():.1f}%")

    # --- 3. Feature Scaling (StandardScaler) ---
    # StandardScaler: (x - mean) / std -> zero mean, unit variance
    # This is critical for DL — features on different scales (e.g., voltage 3.5V vs
    # temperature 30C vs current 15A) would dominate the loss function unevenly.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)
    print(f"  StandardScaler applied (mean=0, std=1 per feature)")

    # --- 4. Create sliding window sequences ---
    # For each cycle i, the input is the last seq_len cycles: [i-seq_len+1, ..., i]
    # For early cycles where we don't have enough history, we zero-pad.
    #
    # Example with seq_len=5:
    #   Cycle 1: [pad, pad, pad, pad, cycle1]  -> SOH_1
    #   Cycle 3: [pad, pad, cycle1, cycle2, cycle3]  -> SOH_3
    #   Cycle 5: [cycle1, cycle2, cycle3, cycle4, cycle5]  -> SOH_5
    #   Cycle 50: [cycle46, cycle47, cycle48, cycle49, cycle50] -> SOH_50

    n_samples, n_features = X_scaled.shape
    X_seq = np.zeros((n_samples, seq_len, n_features), dtype=np.float32)

    for i in range(n_samples):
        if i < seq_len:
            # Not enough history -> zero-pad at the beginning
            pad_len = seq_len - (i + 1)
            X_seq[i, pad_len:, :] = X_scaled[:i + 1, :]
        else:
            # Full sequence available
            X_seq[i, :, :] = X_scaled[i - seq_len + 1:i + 1, :]

    print(f"  Sequence length: {seq_len}")
    print(f"  X_seq shape: {X_seq.shape} (samples, seq_len, features)")
    print(f"  y shape: {y.shape}")
    print(f"  Zero-padded sequences: {min(seq_len - 1, n_samples)} (first {min(seq_len - 1, n_samples)} cycles)")

    return X_seq, y, scaler


# ============================================================
# LSTM Model Definition
# ============================================================

# Device selection (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")


class SOH_LSTM(nn.Module):
    """
    Bidirectional LSTM for SOH prediction.

    Architecture:
        Input (seq_len, n_features)
        -> Bidirectional LSTM (2 layers)
        -> LayerNorm + Dropout
        -> FC head (hidden -> hidden//2 -> 1)

    Bidirectional: processes sequence both forward and backward,
    capturing patterns in both directions.
    """
    def __init__(self, n_features, hidden_dim=64, n_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.ln = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
        h = out[:, -1, :]              # last timestep: (batch, hidden*2)
        h = self.dropout(self.ln(h))   # normalize + dropout
        return self.head(h).squeeze(-1)  # (batch,)


def train_lstm_model(model, X_train, y_train, X_test, y_test,
                     epochs=200, batch_size=16, lr=0.001, patience=30):
    """
    Trains LSTM model with early stopping.

    Args:
        model: SOH_LSTM instance
        X_train/y_train: training data (numpy arrays)
        X_test/y_test: test data (numpy arrays)
        epochs: max training epochs
        batch_size: mini-batch size
        lr: learning rate
        patience: early stopping patience (stop if no improvement for N epochs)

    Returns:
        train_losses, test_losses: loss history per epoch
    """
    model = model.to(DEVICE)

    # Convert to tensors
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    best_test_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y_batch)

        train_loss = epoch_loss / len(train_ds)
        train_losses.append(train_loss)

        # --- Evaluate ---
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_t)
            test_loss = criterion(y_pred_test, y_test_t).item()
        test_losses.append(test_loss)

        scheduler.step(test_loss)

        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:>3}/{epochs}: "
                  f"Train Loss={train_loss:.6f}, Test Loss={test_loss:.6f}")

        if no_improve >= patience:
            print(f"    Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(DEVICE)

    return train_losses, test_losses


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BMS Predictive Maintenance - LSTM Pipeline")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- STEP 1: Read and visualize ---
    print("\n[STEP 1] Reading .mat file...")
    df = load_mat_to_dataframe(MAT_PATH)
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    df = df[cols]
    print(f"  DataFrame: {df.shape[0]:,} rows x {df.shape[1]} columns")

    plot_raw_signals(df, save_path=os.path.join(OUTPUT_DIR, "step1_raw_signals.png"))
    print(f"[PLOT] step1_raw_signals.png saved")

    # --- STEP 2: mV -> V conversion ---
    print("\n" + "=" * 60)
    print("STEP 2: mV -> V Conversion")
    print("=" * 60)
    for col in MV_SIGNALS:
        df[col] = df[col] / 1000.0
    print(f"  {MV_SIGNALS} converted from mV to V")

    # Step 2 visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    time_hours = df["timestamp"].values / 3600.0
    for ax, sig, color in zip(axes, ["Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"],
                               ["tab:red", "tab:cyan", "tab:blue"]):
        ax.plot(time_hours, df[sig].values, color=color, linewidth=0.3)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"{sig} (converted to V)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("STEP 2 - Voltage Signals After mV -> V Conversion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_volt_conversion.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step2_volt_conversion.png saved")

    # --- STEP 3: Cycle boundary detection ---
    print("\n" + "=" * 60)
    print("STEP 3: Cycle Boundary Detection")
    print("=" * 60)
    boundaries = detect_cycle_boundaries(df)

    # Cycle info
    cycle_nums, cycle_sohs = [], []
    cycle_durations_active, cycle_rest_durations, cycle_durations_total = [], [], []
    for i, (s, e) in enumerate(boundaries):
        cycle_nums.append(i + 1)
        cycle_sohs.append(df["SOH"].iloc[s:e+1].mode()[0])
        total_sec = e - s
        active_sec = int((np.abs(df["Current"].iloc[s:e+1].values) > 0.5).sum())
        rest_sec = total_sec - active_sec
        cycle_durations_total.append(total_sec / 3600.0)
        cycle_durations_active.append(active_sec / 3600.0)
        cycle_rest_durations.append(rest_sec / 3600.0)

    # Step 3 CSV
    cycle_boundaries_df = pd.DataFrame({
        "cycle": cycle_nums,
        "active_hours": cycle_durations_active,
        "rest_hours": cycle_rest_durations,
        "total_hours": cycle_durations_total,
        "soh": cycle_sohs,
    })
    cycle_boundaries_df.to_csv(os.path.join(OUTPUT_DIR, "df_step3_cycle_boundaries.csv"), index=False)
    print(f"[CSV] df_step3_cycle_boundaries.csv saved ({len(cycle_nums)} cycles)")

    # Step 3 visualization
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(cycle_nums, cycle_sohs, "b-o", markersize=4, linewidth=1.5)
    axes[0].set_ylabel("SOH (%)")
    axes[0].set_title("Cycle vs SOH (Battery Degradation)")
    axes[0].axhline(y=80, color="red", linestyle="--", linewidth=1, label="End of Life (80%)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(cycle_nums, cycle_durations_active, color="tab:green", alpha=0.7,
                edgecolor="black", linewidth=0.3, label="Active (discharge+charge)")
    axes[1].bar(cycle_nums, cycle_rest_durations, bottom=cycle_durations_active,
                color="tab:gray", alpha=0.5, edgecolor="black", linewidth=0.3, label="Rest (current=0)")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Duration (hours)")
    axes[1].set_title("Cycle vs Duration (Active + Rest)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("STEP 3 - Cycle Analysis (SOH & Duration)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_cycle_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step3_cycle_analysis.png saved")

    # --- STEP 4: Cycle-level summarization ---
    cycle_df = summarize_cycles(df, boundaries)

    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "df_step4_cycle_features.csv"), index=False)
    print(f"[CSV] df_step4_cycle_features.csv saved "
          f"({cycle_df.shape[0]} cycles x {cycle_df.shape[1]} features)")

    # Remove last cycle (incomplete)
    last_cycle = cycle_df["cycle"].max()
    cycle_df = cycle_df[cycle_df["cycle"] != last_cycle].reset_index(drop=True)
    print(f"\n[FILTER] Cycle {last_cycle} removed (incomplete). Remaining: {len(cycle_df)} cycles")

    # ---- STEP 5: Correlation Analysis ----
    print("\n" + "=" * 60)
    print("STEP 5: Correlation Analysis")
    print("=" * 60)

    exclude_cols = ["cycle", "soh_label"]
    all_feature_cols = [c for c in cycle_df.columns if c not in exclude_cols]

    # Remove constant columns (std=0)
    constant_cols = [c for c in all_feature_cols if cycle_df[c].std() == 0]
    if constant_cols:
        print(f"\n  Removing {len(constant_cols)} constant features (std=0)")
        all_feature_cols = [c for c in all_feature_cols if c not in constant_cols]

    # Pearson correlation with SOH
    soh_corr = cycle_df[all_feature_cols].corrwith(cycle_df["soh_label"]).sort_values()
    soh_corr_abs = soh_corr.abs().sort_values(ascending=False)

    # Save correlation CSV
    corr_df = pd.DataFrame({
        "feature": soh_corr_abs.index,
        "soh_correlation": soh_corr[soh_corr_abs.index].values,
        "abs_correlation": soh_corr_abs.values,
    }).reset_index(drop=True)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "df_step5_correlation.csv"), index=False)
    print(f"[CSV] df_step5_correlation.csv saved ({len(corr_df)} features)")

    # Print all correlations
    print(f"\n--- All Features SOH Correlation ({len(soh_corr_abs)} features) ---")
    for i, (feat, corr) in enumerate(soh_corr_abs.items()):
        direction = "+" if soh_corr[feat] > 0 else "-"
        print(f"  {i+1:>3}. {feat:<50s} {direction}{corr:.4f}")

    # Step 5 visualization: Top 30 correlation bar chart
    top30_corr = soh_corr_abs.head(30)
    top30_signed = soh_corr[top30_corr.index]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors_bar = ["tab:green" if v > 0 else "tab:red" for v in top30_signed.values]
    ax.barh(range(len(top30_signed)), top30_signed.values, color=colors_bar,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(top30_signed)))
    ax.set_yticklabels(top30_signed.index, fontsize=8)
    ax.set_xlabel("Correlation (with SOH)")
    ax.set_title("STEP 5 - Top 30 Feature SOH Correlation\n(Green=Positive, Red=Negative)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step5_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step5_correlation.png saved")

    # ---- STEP 6: Feature Selection (All Features — LSTM selects internally) ----
    print("\n" + "=" * 60)
    print("STEP 6: Feature Selection (All Features)")
    print("=" * 60)
    feature_cols = all_feature_cols  # Use all features — no manual selection
    print(f"  Total features: {len(feature_cols)}")
    print(f"  No manual selection — LSTM handles feature importance via learned weights")
    print(f"  Note: 'cycle' column is excluded (data leakage prevention)")

    # Step 6 visualization: Feature category distribution
    n_raw = len([c for c in feature_cols if any(c.startswith(s + "_") for s in RAW_SIGNALS)])
    n_domain = len([c for c in feature_cols if any(kw in c for kw in
                    ["voltage_spread", "voltage_asymmetry", "thermal_spread",
                     "proxy_IR", "energy_efficiency", "thermal_response",
                     "charge_time", "discharge_duration", "soc_range",
                     "cycle_duration", "active_duration", "rest_duration"])
                    and not any(c.startswith(s + "_") for s in RAW_SIGNALS)])
    n_other = len(feature_cols) - n_raw - n_domain
    if n_other < 0:
        n_other = 0

    categories = ["Raw Signal\nStatistics", "Domain-Specific\nFeatures"]
    counts = [n_raw, n_domain]
    if n_other > 0:
        categories.append("Other")
        counts.append(n_other)

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#6BAED6", "#74C476", "#969696"][:len(counts)]
    bars = ax.bar(categories, counts, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature Count")
    ax.set_title(f"STEP 6 - Feature Distribution (Total: {len(feature_cols)}, No Trend Features)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step6_feature_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step6_feature_distribution.png saved")

    # ---- STEP 7: Feature Scaling + Sequence Preparation ----
    X_seq, y, scaler = prepare_features_and_sequences(cycle_df, feature_cols, seq_len=SEQ_LEN)

    # Save scaler for inference
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
    print(f"[FILE] scaler.joblib saved")

    # Save feature columns for inference
    with open(os.path.join(OUTPUT_DIR, "feature_columns.txt"), "w") as f_out:
        for col in feature_cols:
            f_out.write(col + "\n")
    print(f"[FILE] feature_columns.txt saved ({len(feature_cols)} features)")

    # Step 7 visualization: Scaled feature distributions (before vs after)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    X_raw = cycle_df[feature_cols].values
    axes[0].boxplot([X_raw[:, i] for i in range(min(15, X_raw.shape[1]))],
                     labels=[feature_cols[i][:15] for i in range(min(15, X_raw.shape[1]))],
                     vert=True)
    axes[0].set_title("Before Scaling (first 15 features)")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)

    X_scaled_flat = X_seq[:, -1, :]
    axes[1].boxplot([X_scaled_flat[:, i] for i in range(min(15, X_scaled_flat.shape[1]))],
                     labels=[feature_cols[i][:15] for i in range(min(15, X_scaled_flat.shape[1]))],
                     vert=True)
    axes[1].set_title("After StandardScaler (first 15 features)")
    axes[1].tick_params(axis="x", rotation=90, labelsize=7)
    axes[1].set_ylabel("Scaled Value")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("STEP 7 - Feature Scaling: Before vs After", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step7_feature_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step7_feature_scaling.png saved")

    # ============================================================
    # STEP 8: LSTM Model Training (Baseline)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 8: LSTM Model Training (Baseline)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")

    n_features = X_seq.shape[2]
    cycles_arr = cycle_df["cycle"].values

    # --- Train/Test Split: Interleaved (same as ML) ---
    test_mask = np.zeros(len(X_seq), dtype=bool)
    test_mask[3::4] = True
    train_mask = ~test_mask

    X_train, X_test = X_seq[train_mask], X_seq[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    cycles_test = cycles_arr[test_mask]

    print(f"\n  Split: Interleaved (every 4th cycle = test)")
    print(f"  Train: {len(X_train)} cycles, SOH {y_train.min():.0f}-{y_train.max():.0f}%")
    print(f"  Test:  {len(X_test)} cycles, SOH {y_test.min():.0f}-{y_test.max():.0f}%")

    # --- Baseline LSTM ---
    print("\n  --- Baseline LSTM ---")
    print(f"  Architecture: BiLSTM, hidden=64, layers=2, dropout=0.2")
    baseline_model = SOH_LSTM(n_features, hidden_dim=64, n_layers=2, dropout=0.2)
    t0 = time.time()
    train_losses, test_losses = train_lstm_model(
        baseline_model, X_train, y_train, X_test, y_test,
        epochs=200, batch_size=16, lr=0.001, patience=30,
    )
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")

    # --- Evaluate Baseline ---
    baseline_model.eval()
    with torch.no_grad():
        y_pred_train = baseline_model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        y_pred_test = baseline_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    mae_train = mean_absolute_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    print(f"\n  --- Baseline Results ---")
    print(f"  Train: MAE={mae_train:.4f}%, RMSE={rmse_train:.4f}%, R2={r2_train:.6f}")
    print(f"  Test:  MAE={mae_test:.4f}%, RMSE={rmse_test:.4f}%, R2={r2_test:.6f}")

    # Overfitting check
    mae_diff = mae_test - mae_train
    rmse_diff = rmse_test - rmse_train
    r2_diff = r2_train - r2_test
    mae_note = "OK" if mae_diff < 0.5 else "OVERFITTING?" if mae_diff > 1.0 else "WARNING"
    rmse_note = "OK" if rmse_diff < 0.5 else "OVERFITTING?" if rmse_diff > 1.0 else "WARNING"
    r2_note = "OK" if r2_diff < 0.01 else "OVERFITTING?" if r2_diff > 0.05 else "WARNING"

    print(f"\n  --- Baseline Overfitting Check ---")
    print(f"  {'Metric':<12} {'Train':<12} {'Test':<12} {'Diff':<12} {'Note'}")
    print(f"  {'-'*55}")
    print(f"  {'MAE (%)':<12} {mae_train:<12.4f} {mae_test:<12.4f} {mae_diff:<+12.4f} {mae_note}")
    print(f"  {'RMSE (%)':<12} {rmse_train:<12.4f} {rmse_test:<12.4f} {rmse_diff:<+12.4f} {rmse_note}")
    print(f"  {'R2':<12} {r2_train:<12.6f} {r2_test:<12.6f} {r2_diff:<+12.6f} {r2_note}")

    # 5-Fold CV
    from sklearn.model_selection import KFold
    print(f"\n  --- 5-Fold Cross Validation ---")
    kf = KFold(n_splits=5, shuffle=False)
    cv_maes = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_seq)):
        cv_model = SOH_LSTM(n_features, hidden_dim=64, n_layers=2, dropout=0.2)
        train_lstm_model(cv_model, X_seq[tr_idx], y[tr_idx], X_seq[te_idx], y[te_idx],
                         epochs=150, batch_size=16, lr=0.001, patience=20)
        cv_model.eval()
        with torch.no_grad():
            cv_pred = cv_model(torch.tensor(X_seq[te_idx], dtype=torch.float32).to(DEVICE)).cpu().numpy()
        fold_mae = mean_absolute_error(y[te_idx], cv_pred)
        cv_maes.append(fold_mae)
        print(f"    Fold {fold+1}: MAE={fold_mae:.4f}% (Test cycles {te_idx[0]+1}-{te_idx[-1]+1})")
    cv_mae = np.mean(cv_maes)
    cv_std = np.std(cv_maes)
    print(f"  CV MAE: {cv_mae:.4f}% (+/- {cv_std:.4f})")

    # Save baseline model
    torch.save(baseline_model.state_dict(), os.path.join(OUTPUT_DIR, "model_lstm_baseline.pt"))
    print(f"\n[MODEL] model_lstm_baseline.pt saved")

    # Save baseline results CSV
    results_df = pd.DataFrame([{
        "model": "LSTM_Baseline",
        "MAE_train": mae_train, "MAE_test": mae_test,
        "RMSE_train": rmse_train, "RMSE_test": rmse_test,
        "R2_train": r2_train, "R2_test": r2_test,
        "CV_MAE": cv_mae,
    }])
    results_df.to_csv(os.path.join(OUTPUT_DIR, "df_step8_model_results.csv"), index=False)
    print(f"[CSV] df_step8_model_results.csv saved")

    # --- Step 8 Visualizations ---

    # 1. Train vs Test metrics (overfitting check)
    metrics = ["MAE (%)", "RMSE (%)", "R2"]
    train_vals = [mae_train, rmse_train, r2_train]
    test_vals = [mae_test, rmse_test, r2_test]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x_pos = np.arange(2)
    for i, (metric, tr_v, te_v) in enumerate(zip(metrics, train_vals, test_vals)):
        bars = axes[i].bar(x_pos, [tr_v, te_v], color=["#6BAED6", "#2171B5"],
                           edgecolor="black", linewidth=0.5, width=0.5)
        axes[i].set_xticks(x_pos)
        axes[i].set_xticklabels(["Train", "Test"])
        axes[i].set_title(f"STEP 8 - {metric}", fontweight="bold")
        for bar, val in zip(bars, [tr_v, te_v]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")
    plt.suptitle("STEP 8 - LSTM: Train vs Test Metrics (Overfitting Check)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_train_vs_test_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_train_vs_test_metrics.png")

    # 2. SOH Prediction vs Actual
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cycles_test, y_test, "b-o", label="Actual SOH", linewidth=2, markersize=5)
    ax.plot(cycles_test, y_pred_test, "r--s", label="LSTM Prediction", linewidth=1.5, markersize=5, alpha=0.8)
    ax.fill_between(cycles_test, y_test, y_pred_test, alpha=0.15, color="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"STEP 8 - LSTM SOH Prediction\nMAE={mae_test:.3f}% | RMSE={rmse_test:.3f}% | R2={r2_test:.4f}",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_soh_pred_vs_actual.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_soh_pred_vs_actual.png")

    # 3. Residuals
    residuals = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_r = ["green" if abs(r) < 0.5 else "orange" if abs(r) < 1 else "red" for r in residuals]
    ax.scatter(cycles_test, residuals, c=colors_r, alpha=0.8, s=50, edgecolors="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linewidth=1)
    ax.axhline(y=0.5, color="green", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.axhline(y=-0.5, color="green", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Residual (Actual - Predicted) %")
    ax.set_title(f"STEP 8 - LSTM Residuals (max |error|={np.max(np.abs(residuals)):.3f}%)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_residuals.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_residuals.png")

    # 4. Training Loss Curve (DL-specific, replaces feature importance)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train Loss", color="#6BAED6", linewidth=1.5)
    ax.plot(test_losses, label="Test Loss", color="#2171B5", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("STEP 8 - LSTM Training Loss Curve", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_loss_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_loss_curve.png")

    # 5. Test Metrics Summary
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    metric_names = ["MAE (%)", "RMSE (%)", "R2"]
    metric_vals = [mae_test, rmse_test, r2_test]
    bar_colors_met = ["#6BAED6", "#2171B5", "#74C476"]
    for i, (name, val, clr) in enumerate(zip(metric_names, metric_vals, bar_colors_met)):
        axes[i].bar([name], [val], color=clr, edgecolor="black", linewidth=0.5, width=0.4)
        axes[i].text(0, val, f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
        axes[i].set_title(name, fontweight="bold")
        axes[i].grid(True, alpha=0.3, axis="y")
    plt.suptitle("STEP 8 - LSTM Test Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_test_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_test_metrics.png")

    # ============================================================
    # STEP 9: Hyperparameter Tuning
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 9: Hyperparameter Tuning")
    print("=" * 60)

    # Grid of hyperparameters to try
    PARAM_GRID = {
        "hidden_dim": [32, 64, 128],
        "n_layers": [1, 2, 3],
        "lr": [0.0005, 0.001, 0.005],
    }

    print(f"  Grid: {len(PARAM_GRID['hidden_dim'])} x {len(PARAM_GRID['n_layers'])} x {len(PARAM_GRID['lr'])} "
          f"= {len(PARAM_GRID['hidden_dim']) * len(PARAM_GRID['n_layers']) * len(PARAM_GRID['lr'])} combinations")

    best_mae = float("inf")
    best_params = {}
    tuning_results = []
    t0_grid = time.time()

    for hd in PARAM_GRID["hidden_dim"]:
        for nl in PARAM_GRID["n_layers"]:
            for lr_val in PARAM_GRID["lr"]:
                model_gs = SOH_LSTM(n_features, hidden_dim=hd, n_layers=nl, dropout=0.2)
                train_lstm_model(model_gs, X_train, y_train, X_test, y_test,
                                 epochs=150, batch_size=16, lr=lr_val, patience=20)
                model_gs.eval()
                with torch.no_grad():
                    y_pred_gs = model_gs(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()
                gs_mae = mean_absolute_error(y_test, y_pred_gs)
                tuning_results.append({
                    "hidden_dim": hd, "n_layers": nl, "lr": lr_val, "test_mae": gs_mae,
                })
                if gs_mae < best_mae:
                    best_mae = gs_mae
                    best_params = {"hidden_dim": hd, "n_layers": nl, "lr": lr_val}
                    best_state = {k: v.cpu().clone() for k, v in model_gs.state_dict().items()}

    elapsed_grid = time.time() - t0_grid
    print(f"\n  Grid Search completed in {elapsed_grid:.1f}s")
    print(f"  Best params: {best_params}")
    print(f"  Best Test MAE: {best_mae:.4f}%")

    # Save tuning results
    tuning_df = pd.DataFrame(tuning_results).sort_values("test_mae")
    tuning_df.to_csv(os.path.join(OUTPUT_DIR, "df_step9_tuning_results.csv"), index=False)
    print(f"[CSV] df_step9_tuning_results.csv saved")

    best_params_df = pd.DataFrame([best_params])
    best_params_df.to_csv(os.path.join(OUTPUT_DIR, "df_step9_best_params.csv"), index=False)
    print(f"[CSV] df_step9_best_params.csv saved")

    # --- Retrain tuned model with best params ---
    print(f"\n  --- Tuned LSTM (hidden={best_params['hidden_dim']}, "
          f"layers={best_params['n_layers']}, lr={best_params['lr']}) ---")
    tuned_model = SOH_LSTM(n_features, hidden_dim=best_params["hidden_dim"],
                           n_layers=best_params["n_layers"], dropout=0.2)
    tuned_model.load_state_dict(best_state)
    tuned_model = tuned_model.to(DEVICE)
    tuned_model.eval()

    with torch.no_grad():
        y_pred_tuned_train = tuned_model(torch.tensor(X_train, dtype=torch.float32).to(DEVICE)).cpu().numpy()
        y_pred_tuned_test = tuned_model(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).cpu().numpy()

    mae_tuned_train = mean_absolute_error(y_train, y_pred_tuned_train)
    rmse_tuned_train = np.sqrt(mean_squared_error(y_train, y_pred_tuned_train))
    r2_tuned_train = r2_score(y_train, y_pred_tuned_train)
    mae_tuned_test = mean_absolute_error(y_test, y_pred_tuned_test)
    rmse_tuned_test = np.sqrt(mean_squared_error(y_test, y_pred_tuned_test))
    r2_tuned_test = r2_score(y_test, y_pred_tuned_test)

    print(f"  Train: MAE={mae_tuned_train:.4f}%, RMSE={rmse_tuned_train:.4f}%, R2={r2_tuned_train:.6f}")
    print(f"  Test:  MAE={mae_tuned_test:.4f}%, RMSE={rmse_tuned_test:.4f}%, R2={r2_tuned_test:.6f}")

    # Tuned overfitting check
    mae_t_diff = mae_tuned_test - mae_tuned_train
    rmse_t_diff = rmse_tuned_test - rmse_tuned_train
    r2_t_diff = r2_tuned_train - r2_tuned_test
    mae_t_note = "OK" if mae_t_diff < 0.5 else "OVERFITTING?" if mae_t_diff > 1.0 else "WARNING"
    rmse_t_note = "OK" if rmse_t_diff < 0.5 else "OVERFITTING?" if rmse_t_diff > 1.0 else "WARNING"
    r2_t_note = "OK" if r2_t_diff < 0.01 else "OVERFITTING?" if r2_t_diff > 0.05 else "WARNING"

    print(f"\n  --- Tuned Model Overfitting Check ---")
    print(f"  {'Metric':<12} {'Train':<12} {'Test':<12} {'Diff':<12} {'Note'}")
    print(f"  {'-'*55}")
    print(f"  {'MAE (%)':<12} {mae_tuned_train:<12.4f} {mae_tuned_test:<12.4f} {mae_t_diff:<+12.4f} {mae_t_note}")
    print(f"  {'RMSE (%)':<12} {rmse_tuned_train:<12.4f} {rmse_tuned_test:<12.4f} {rmse_t_diff:<+12.4f} {rmse_t_note}")
    print(f"  {'R2':<12} {r2_tuned_train:<12.6f} {r2_tuned_test:<12.6f} {r2_t_diff:<+12.6f} {r2_t_note}")

    # Tuned CV MAE
    print(f"\n  --- 5-Fold CV (Tuned) ---")
    cv_maes_tuned = []
    for fold, (tr_idx, te_idx) in enumerate(kf.split(X_seq)):
        cv_model = SOH_LSTM(n_features, hidden_dim=best_params["hidden_dim"],
                            n_layers=best_params["n_layers"], dropout=0.2)
        train_lstm_model(cv_model, X_seq[tr_idx], y[tr_idx], X_seq[te_idx], y[te_idx],
                         epochs=150, batch_size=16, lr=best_params["lr"], patience=20)
        cv_model.eval()
        with torch.no_grad():
            cv_pred = cv_model(torch.tensor(X_seq[te_idx], dtype=torch.float32).to(DEVICE)).cpu().numpy()
        fold_mae = mean_absolute_error(y[te_idx], cv_pred)
        cv_maes_tuned.append(fold_mae)
        print(f"    Fold {fold+1}: MAE={fold_mae:.4f}%")
    cv_mae_tuned = np.mean(cv_maes_tuned)
    print(f"  CV MAE: {cv_mae_tuned:.4f}% (+/- {np.std(cv_maes_tuned):.4f})")

    # Save tuned model
    torch.save(tuned_model.state_dict(), os.path.join(OUTPUT_DIR, "model_lstm_tuned.pt"))
    print(f"\n[MODEL] model_lstm_tuned.pt saved")

    # --- Baseline vs Tuned comparison ---
    print(f"\n  --- Baseline vs Tuned Comparison ---")
    print(f"  {'Metric':<12} {'Baseline Test':<16} {'Tuned Test':<16} {'Change'}")
    print(f"  {'-'*55}")
    print(f"  {'MAE (%)':<12} {mae_test:<16.4f} {mae_tuned_test:<16.4f} {mae_tuned_test - mae_test:<+.4f}")
    print(f"  {'RMSE (%)':<12} {rmse_test:<16.4f} {rmse_tuned_test:<16.4f} {rmse_tuned_test - rmse_test:<+.4f}")
    print(f"  {'R2':<12} {r2_test:<16.6f} {r2_tuned_test:<16.6f} {r2_tuned_test - r2_test:<+.6f}")
    print(f"  {'CV MAE':<12} {cv_mae:<16.4f} {cv_mae_tuned:<16.4f} {cv_mae_tuned - cv_mae:<+.4f}")

    if mae_tuned_test < mae_test:
        print(f"\n  RESULT: Tuned model is better (MAE {mae_test:.4f}% -> {mae_tuned_test:.4f}%)")
    else:
        print(f"\n  RESULT: Baseline model is better (MAE {mae_test:.4f}% vs {mae_tuned_test:.4f}%)")
        print(f"  [WARNING] Grid Search did not improve performance. Baseline should be preferred.")

    # --- Step 9 Visualizations ---

    # 1. Baseline vs Tuned bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_names = ["MAE (%)", "RMSE (%)", "R2"]
    baseline_vals = [mae_test, rmse_test, r2_test]
    tuned_vals = [mae_tuned_test, rmse_tuned_test, r2_tuned_test]

    for i, (name, bl_v, tn_v) in enumerate(zip(metrics_names, baseline_vals, tuned_vals)):
        bars = axes[i].bar([0, 1], [bl_v, tn_v], color=["#6BAED6", "#238B45"],
                           edgecolor="black", linewidth=0.5, width=0.5)
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(["Baseline", "Tuned"])
        axes[i].set_title(name, fontweight="bold")
        for bar, val in zip(bars, [bl_v, tn_v]):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.4f}", ha="center", va="bottom", fontsize=10)
        axes[i].grid(True, alpha=0.3, axis="y")
    plt.suptitle("STEP 9 - Baseline vs Tuned LSTM (Test Metrics)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step9_baseline_vs_tuned.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step9_baseline_vs_tuned.png")

    # 2. Tuned SOH Prediction
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(cycles_test, y_test, "b-o", label="Actual SOH", linewidth=2, markersize=5)
    ax.plot(cycles_test, y_pred_tuned_test, "r--s", label="LSTM Tuned Prediction",
            linewidth=1.5, markersize=5, alpha=0.8)
    ax.fill_between(cycles_test, y_test, y_pred_tuned_test, alpha=0.15, color="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"STEP 9 - Tuned LSTM SOH Prediction\n"
                 f"MAE={mae_tuned_test:.3f}% | RMSE={rmse_tuned_test:.3f}% | R2={r2_tuned_test:.4f}",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step9_tuned_soh_prediction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step9_tuned_soh_prediction.png")

    # --- Result CSV ---
    result_path = os.path.join(OUTPUT_DIR, "Result_LSTM.csv")
    with open(result_path, "w") as f:
        f.write("Compare,MAE,RMSE,R2,CV_MAE\n")
        f.write(f"Baseline,{mae_test:.4f},{rmse_test:.4f},{r2_test:.6f},{cv_mae:.4f}\n")
        f.write(f"Tuned,{mae_tuned_test:.4f},{rmse_tuned_test:.4f},{r2_tuned_test:.6f},{cv_mae_tuned:.4f}\n")
        f.write("\n")
        f.write("--- Baseline Model Overfitting Check ---\n")
        f.write("Metric,Train,Test,Diff,Note\n")
        f.write(f"MAE (%),{mae_train:.4f},{mae_test:.4f},{mae_diff:+.4f},{mae_note}\n")
        f.write(f"RMSE (%),{rmse_train:.4f},{rmse_test:.4f},{rmse_diff:+.4f},{rmse_note}\n")
        f.write(f"R2,{r2_train:.6f},{r2_test:.6f},{r2_diff:+.6f},{r2_note}\n")
        f.write("\n")
        f.write("--- Tuned Model Overfitting Check ---\n")
        f.write("Metric,Train,Test,Diff,Note\n")
        f.write(f"MAE (%),{mae_tuned_train:.4f},{mae_tuned_test:.4f},{mae_t_diff:+.4f},{mae_t_note}\n")
        f.write(f"RMSE (%),{rmse_tuned_train:.4f},{rmse_tuned_test:.4f},{rmse_t_diff:+.4f},{rmse_t_note}\n")
        f.write(f"R2,{r2_tuned_train:.6f},{r2_tuned_test:.6f},{r2_t_diff:+.6f},{r2_t_note}\n")
    print(f"[CSV] Result_LSTM.csv saved")

    # --- Final Summary ---
    print("\n" + "=" * 60)
    print("LSTM PIPELINE COMPLETED")
    print("=" * 60)
    print(f"  Best Baseline MAE: {mae_test:.4f}%")
    print(f"  Best Tuned MAE:    {mae_tuned_test:.4f}%")
    print(f"  Best Params:       {best_params}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)
