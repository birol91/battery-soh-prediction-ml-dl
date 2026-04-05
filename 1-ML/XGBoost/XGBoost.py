"""
BMS Predictive Maintenance - Phase 1 Pipeline (v1 Birol)
========================================================

PROBLEM DEFINITION:
-------------------
We have a Li-ion battery pack (LG 18650HG2 NMC cells) modeled in Simulink
with a 5s1p (5 series cells, 1 parallel) configuration.

The battery is repeatedly charged/discharged using the UDDS (Urban Dynamometer
Driving Schedule) drive cycle. With each cycle, the battery ages slightly
(capacity decreases, internal resistance increases).

After 100 cycles, SOH (State of Health) drops from 100% to ~80%.
SOH = 80% is considered "End of Life".

OBJECTIVE:
----------
Build ML models that predict SOH from the battery's operational data
(voltage, current, temperature, SOC).

The model will only use signals visible to the BMS:
- Cell_V_Max, Cell_V_Min, Cell_V_Avg (min/max/avg voltage of 5 cells)
- Cell_T_Max, Cell_T_Min, Cell_T_Avg (min/max/avg temperature of 5 cells)
- Current (pack current)
- SOC (State of Charge)

SOH label is calculated from the Simscape Fade formula (ground truth).
This formula is used instead of RPT (Reference Performance Test) in real world.

DATA:
-----
recording_senaryo_1_Battery_A_N100.mat
- Simulink Recording block output
- 1 Hz sampling frequency
- 9 signals (SOC, Current, Cell_V_Max/Min/Avg, Cell_T_Max/Min/Avg, SOH)
- ~100 cycles, SOH: 100% -> 80%

PIPELINE STEPS:
---------------
1. Read .mat file, visualize raw signals (EDA)
2. mV -> V conversion
3. Detect cycle boundaries (SOC peak points)
4. Cycle-level summarization (1 Hz -> 1 row per cycle)
5. Add trend features (rolling, delta, rate_of_change)
6. Train ML model (XGBoost)
7. Visualize and report results
"""

import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# ============================================================
# STEP 1: Read .mat File and Visualize Raw Signals (EDA)
# ============================================================

# Simulink Recording sigstream signal ordering:
# Each signal corresponds to a stream index
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

# Voltage signals come in mV, will stay as mV
MV_SIGNALS = {"Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"}

# Path to .mat file
MAT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "0_Dataset", "recording_senaryo_1_Battery_A_N100.mat")


def load_mat_to_dataframe(mat_path: str) -> pd.DataFrame:
    """
    Reads a Simulink Recording .mat file and converts it to a pandas DataFrame.

    The .mat file is in HDF5 format. Inside, each signal is stored as a
    separate stream under #sigstream#.

    Returns:
        DataFrame: 9 signals + timestamp column
    """
    with h5py.File(mat_path, "r") as f:
        # Read total number of samples
        n_samples = int(f["#sigstream#/0/#length#"][0])

        data = {}
        for stream_id, col_name in STREAM_MAP.items():
            # Read raw data
            raw = f[f"#sigstream#/{stream_id}/#data#"][:]
            arr = np.frombuffer(raw.tobytes(), dtype=np.float64)[:n_samples]

            data[col_name] = arr

    df = pd.DataFrame(data)

    # 1 Hz sampling -> each row is 1 second
    df["timestamp"] = np.arange(len(df), dtype=np.float64)

    return df


# ============================================================
# STEP 1 (continued): Visualize Raw Signals
# ============================================================

def plot_raw_signals(df: pd.DataFrame, save_path: str = None):
    """Visualizes raw signals with 5 subplots."""
    time_hours = df["timestamp"].values / 3600.0

    fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)

    # 1. SOC
    axes[0].plot(time_hours, df["SOC"].values, color="tab:blue", linewidth=0.3)
    axes[0].set_ylabel("SOC (%)")
    axes[0].set_title("State of Charge")
    axes[0].grid(True, alpha=0.3)

    # 2. Current
    axes[1].plot(time_hours, df["Current"].values, color="tab:orange", linewidth=0.3)
    axes[1].set_ylabel("Current (A)")
    axes[1].set_title("Current")
    axes[1].grid(True, alpha=0.3)

    # 3. Cell V Max and Cell V Min
    axes[2].plot(time_hours, df["Cell_V_Max"].values, color="tab:red", linewidth=0.3, label="Cell_V_Max")
    axes[2].plot(time_hours, df["Cell_V_Min"].values, color="tab:cyan", linewidth=0.3, label="Cell_V_Min")
    axes[2].set_ylabel("Voltage (mV)")
    axes[2].set_title("Cell Voltage Max & Min")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # 4. Cell T Max and Cell T Min
    axes[3].plot(time_hours, df["Cell_T_Max"].values, color="darkred", linewidth=0.3, label="Cell_T_Max")
    axes[3].plot(time_hours, df["Cell_T_Min"].values, color="steelblue", linewidth=0.3, label="Cell_T_Min")
    axes[3].set_ylabel("Temperature (C)")
    axes[3].set_title("Cell Temperature Max & Min")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    # 5. SOH
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

    Simulation starts at SOC=100%. Cycle boundaries are determined at SOC
    peak points (after full charge ~100%).
    From one peak point to the next peak point = 1 full cycle.

    Args:
        df: Raw data DataFrame (must have SOC column)
        soc_high_thresh: Threshold for "fully charged" SOC (%)
        min_cycle_duration: Minimum duration between two peaks (seconds)

    Returns:
        [(start_idx, end_idx), ...] list
    """
    soc = df["SOC"].values

    # Find entry/exit points of regions where SOC crosses above the threshold
    above = soc >= soc_high_thresh
    transitions = np.diff(above.astype(int))
    enter_high = np.where(transitions == 1)[0]   # crossing above threshold (charge completed)
    exit_high = np.where(transitions == -1)[0]    # dropping below threshold (discharge starting)

    # If data starts at SOC=100%, the first "high" region starts without a transition
    # In this case we can use the first exit_high point
    # But a more robust way: find the start of each high region

    # Find the peak point of SOC in each high region
    # High region = consecutive regions where SOC >= soc_high_thresh
    # Peak point = end of the high region (just before discharge starts)
    raw_peaks = []

    # If data starts high, the first exit_high is a peak
    if soc[0] >= soc_high_thresh and len(exit_high) > 0:
        raw_peaks.append(exit_high[0])

    # Subsequent peaks: exit_high after each enter_high
    for i in range(len(enter_high)):
        start = enter_high[i]
        exits_after = exit_high[exit_high > start]
        if len(exits_after) > 0:
            peak_idx = exits_after[0]
            raw_peaks.append(peak_idx)

    # Filter out peaks that are too close together
    filtered_peaks = []
    for pk in raw_peaks:
        if not filtered_peaks or (pk - filtered_peaks[-1]) >= min_cycle_duration:
            filtered_peaks.append(pk)

    # Create cycle boundaries: peak to peak
    # Each cycle: SOC starts at ~100% -> drops to 10% -> rises back to 100%
    boundaries = []
    for i in range(len(filtered_peaks) - 1):
        boundaries.append((filtered_peaks[i], filtered_peaks[i + 1]))

    # Last segment: from last peak to end of data (may be incomplete cycle)
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

# Raw signal columns for intra-cycle statistics extraction
RAW_SIGNALS = [
    "Current", "Cell_V_Max", "Cell_V_Min", "Cell_V_Avg",
    "Cell_T_Max", "Cell_T_Min", "Cell_T_Avg", "SOC",
]


def compute_cycle_summary(df: pd.DataFrame, start: int, end: int, cycle_num: int) -> dict:
    """
    Summarizes a single cycle's 1 Hz data into a single row.

    A) Raw signal statistics (mean, std, min, max, range, discharge_end, charge_start)
    B) Domain-specific features (voltage_spread, thermal_spread, proxy_IR, energy_efficiency)
    C) Timing features (charge_time, discharge_duration, soc_range)
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
        # End-of-discharge value (at the moment SOC is minimum)
        row[f"{sig}_discharge_end"] = vals[soc_min_idx]
        # Start-of-charge value (first value after SOC minimum)
        if soc_min_idx < len(vals) - 1:
            row[f"{sig}_charge_start"] = vals[soc_min_idx + 1]
        else:
            row[f"{sig}_charge_start"] = vals[soc_min_idx]

    # ---- B) Domain-Specific Feature'lar ----

    v_max = segment["Cell_V_Max"].values
    v_min = segment["Cell_V_Min"].values
    v_avg = segment["Cell_V_Avg"].values
    t_max = segment["Cell_T_Max"].values
    t_min = segment["Cell_T_Min"].values
    current = segment["Current"].values
    soc = segment["SOC"].values
    abs_current = np.abs(current)

    # 1. Voltage spread (cell imbalance)
    #    As battery ages, cells become more imbalanced -> spread increases
    v_spread = v_max - v_min
    row["voltage_spread_mean"] = np.mean(v_spread)
    row["voltage_spread_max"] = np.max(v_spread)
    row["voltage_spread_std"] = np.std(v_spread)

    # 2. Voltage asymmetry: (V_Max - V_Avg) / (V_Avg - V_Min + eps)
    #    1.0 = symmetric, >1 = one cell is high, <1 = one cell is low
    #    Only moments with meaningful spread (V_Max - V_Min > 1mV) are used
    v_total_spread = v_max - v_min
    meaningful_mask = v_total_spread > 0.001  # moments with spread above 1mV
    if np.any(meaningful_mask):
        v_asym = (v_max[meaningful_mask] - v_avg[meaningful_mask]) / \
                 (v_avg[meaningful_mask] - v_min[meaningful_mask] + 1e-2)
        row["voltage_asymmetry_mean"] = np.mean(v_asym)
    else:
        row["voltage_asymmetry_mean"] = 1.0  # cells are equal -> symmetric

    # 3. Voltage spread under load (active moments: |Current| > 0.5A)
    #    Removes rest moments, calculates spread only under load
    active_mask = abs_current > 0.5
    if np.any(active_mask):
        row["voltage_spread_under_load"] = np.mean(v_spread[active_mask])
    else:
        row["voltage_spread_under_load"] = np.mean(v_spread)

    # 3b. Voltage asymmetry under load (active moments: |Current| > 0.5A)
    #     Asymmetry only at moments under load with meaningful spread
    active_meaningful_mask = active_mask & (v_total_spread > 0.001)
    if np.any(active_meaningful_mask):
        v_asym_load = (v_max[active_meaningful_mask] - v_avg[active_meaningful_mask]) / \
                      (v_avg[active_meaningful_mask] - v_min[active_meaningful_mask] + 1e-2)
        row["voltage_asymmetry_under_load"] = np.mean(v_asym_load)
    else:
        row["voltage_asymmetry_under_load"] = 1.0

    # 4. Thermal spread (thermal imbalance)
    #    As battery ages, internal resistance increases -> heat increases -> spread increases
    t_spread = t_max - t_min
    row["thermal_spread_mean"] = np.mean(t_spread)
    row["thermal_spread_max"] = np.max(t_spread)
    row["thermal_spread_std"] = np.std(t_spread)

    # 4b. Thermal spread under load (active moments: |Current| > 0.5A)
    #     Removes rest moments, calculates thermal spread only under load
    if np.any(active_mask):
        row["thermal_spread_under_load"] = np.mean(t_spread[active_mask])
    else:
        row["thermal_spread_under_load"] = np.mean(t_spread)

    # 5. Proxy internal resistance: delta_V / delta_I
    #    Internal resistance estimate from voltage response during large current changes
    #    As battery ages, internal resistance increases -> this value rises
    di = np.diff(current)
    dv = np.diff(v_avg)
    large_di_mask = np.abs(di) > 2.0  # at least 2A current change
    if np.sum(large_di_mask) > 5:
        proxy_ir_values = np.abs(dv[large_di_mask] / di[large_di_mask])
        proxy_ir_values = proxy_ir_values[proxy_ir_values < 1.0]  # filter out above 1 Ohm
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

    # 6. Energy efficiency: discharge energy / charge energy
    #    As battery ages, losses increase -> efficiency decreases
    power = v_avg * current  # V * A = W
    charge_mask = current > 0.5
    discharge_mask = current < -0.5

    charge_energy = np.sum(power[charge_mask]) if np.any(charge_mask) else 0
    discharge_energy = np.abs(np.sum(power[discharge_mask])) if np.any(discharge_mask) else 0

    if charge_energy > 0:
        row["energy_efficiency"] = discharge_energy / charge_energy
    else:
        row["energy_efficiency"] = np.nan

    # Current convention check: check current direction when SOC is increasing
    dsoc = np.diff(soc)
    soc_increasing = dsoc > 0.01
    if np.any(soc_increasing):
        avg_current_when_soc_up = np.mean(current[1:][soc_increasing])
        if avg_current_when_soc_up < 0:
            # Convention reversed: negative = charge, positive = discharge
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

    # 7. Charge time: Durations in charge phase (seconds)
    #    As battery ages, charge time increases
    soc_after_min = soc[soc_min_idx:]

    soc_50_indices = np.where(soc_after_min >= 50.0)[0]
    row["charge_time_to_50"] = soc_50_indices[0] if len(soc_50_indices) > 0 else np.nan

    soc_80_indices = np.where(soc_after_min >= 80.0)[0]
    row["charge_time_to_80"] = soc_80_indices[0] if len(soc_80_indices) > 0 else np.nan

    soc_95_indices = np.where(soc_after_min >= 95.0)[0]
    row["charge_time_to_95"] = soc_95_indices[0] if len(soc_95_indices) > 0 else np.nan

    soc_full_indices = np.where(soc_after_min >= 99.5)[0]
    row["charge_time_to_full"] = soc_full_indices[0] if len(soc_full_indices) > 0 else np.nan

    # 8. Discharge duration: from cycle start to SOC minimum (seconds)
    row["discharge_duration"] = soc_min_idx

    # 9. SOC range (SOC window used in this cycle)
    row["soc_range"] = np.max(soc) - np.min(soc)

    # 10. Thermal response ratio: delta_T_spread / delta_|Current|
    #     Calculated over 60-second window averages
    #     Thermal effect of current change -> increases with aging
    W = 60  # window size (seconds)
    if len(t_spread) > W and len(abs_current) > W:
        # Smoothed signals with 60s window
        t_spread_smooth = pd.Series(t_spread).rolling(W, min_periods=W).mean().values
        abs_current_smooth = pd.Series(abs_current).rolling(W, min_periods=W).mean().values
        # Difference of smoothed signals
        dt_spread = np.diff(t_spread_smooth)
        d_abs_i = np.diff(abs_current_smooth)
        # Non-NaN moments with meaningful current change
        valid_mask = (~np.isnan(dt_spread)) & (~np.isnan(d_abs_i)) & (np.abs(d_abs_i) > 0.5)
        if np.sum(valid_mask) > 3:
            ratios = dt_spread[valid_mask] / d_abs_i[valid_mask]
            ratios = ratios[np.abs(ratios) < 10]
            row["thermal_response_ratio"] = np.mean(ratios) if len(ratios) > 0 else np.nan
        else:
            row["thermal_response_ratio"] = np.nan
    else:
        row["thermal_response_ratio"] = np.nan

    # --- SOH Label ---
    soh_vals = segment["SOH"].values
    unique_soh, counts = np.unique(soh_vals, return_counts=True)
    row["soh_label"] = unique_soh[np.argmax(counts)]

    # Cycle duration information
    row["cycle_duration"] = end - start + 1
    active_sec = int((abs_current > 0.5).sum())
    row["active_duration"] = active_sec
    row["rest_duration"] = (end - start + 1) - active_sec

    return row


def summarize_cycles(df: pd.DataFrame, boundaries: list) -> pd.DataFrame:
    """
    Summarizes all cycles and creates a DataFrame.
    2.3M rows of 1 Hz data -> ~100 rows of cycle summary
    """
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

    # Remove _current_convention debug column
    if "_current_convention" in cycle_df.columns:
        convention = cycle_df["_current_convention"].mode()[0]
        print(f"\n  Current convention: {convention}")
        cycle_df.drop(columns=["_current_convention"], inplace=True)

    # NaN check and fill (forward-fill + median)
    nan_counts = cycle_df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols) > 0:
        print(f"\n  Columns with NaN (will be filled with forward-fill + median):")
        for col, cnt in nan_cols.items():
            print(f"    {col}: {cnt} NaN")
        cycle_df = cycle_df.ffill().bfill()
        for col in cycle_df.columns:
            if cycle_df[col].isna().any():
                cycle_df[col] = cycle_df[col].fillna(cycle_df[col].median())

    print(f"\n[SUMMARIZATION COMPLETED] {len(cycle_df)} cycles, {len(cycle_df.columns)} features")

    return cycle_df



# Base features for trend computation
TREND_FEATURES = [
    # Step 3 Domain-Specific Features (9 total)
    "voltage_spread_mean",
    "voltage_spread_under_load",
    "voltage_asymmetry_mean",
    "voltage_asymmetry_under_load",
    "thermal_spread_mean",
    "thermal_spread_under_load",
    "proxy_IR_mean",
    "energy_efficiency",
    "thermal_response_ratio",
    # Additional features
    "voltage_spread_max",
    "thermal_spread_max",
    "charge_time_to_full",
    "discharge_duration",
    "Cell_V_Avg_mean",
    "Cell_V_Avg_range",
    "Current_std",
    "SOC_range",
]

ROLLING_WINDOWS = [3, 5, 10]


def add_trend_features(cycle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds cycle-over-cycle trend features.

    For each TREND_FEATURES column:
    - Rolling mean and std (windows: 3, 5, 10)
    - Delta (difference from previous cycle)
    - Rate of change (change rate over 2 cycles)
    """
    print("\n" + "=" * 60)
    print("STEP 5: Trend Features")
    print("=" * 60)

    df = cycle_df.copy()
    n_before = len(df.columns)

    for feat in TREND_FEATURES:
        if feat not in df.columns:
            print(f"  [WARNING] {feat} column not found, skipping.")
            continue

        vals = df[feat]

        # Rolling mean and std
        for w in ROLLING_WINDOWS:
            df[f"{feat}_rolling_mean_{w}"] = vals.rolling(window=w, min_periods=1).mean()
            df[f"{feat}_rolling_std_{w}"] = vals.rolling(window=w, min_periods=1).std().fillna(0)

        # Delta: difference from previous cycle
        df[f"{feat}_delta"] = vals.diff().fillna(0)

        # Rate of change: average change over 2 cycles
        df[f"{feat}_roc"] = vals.diff(periods=2).fillna(0) / 2.0

    # Additional trend feature: proxy_IR_trend_slope_10
    # Linear slope of proxy_IR_mean over last 10 cycles (internal resistance increase rate)
    if "proxy_IR_mean" in df.columns:
        proxy_ir = df["proxy_IR_mean"].values
        slopes = np.zeros(len(proxy_ir))
        for i in range(len(proxy_ir)):
            window_start = max(0, i - 9)
            window_data = proxy_ir[window_start : i + 1]
            if len(window_data) >= 3:
                x = np.arange(len(window_data))
                coeffs = np.polyfit(x, window_data, 1)
                slopes[i] = coeffs[0]
        df["proxy_IR_trend_slope_10"] = slopes

    # NOTE: cycle_normalized (cycle/max) is NOT added!
    # Cycle number is unknown in the real world, feeding it to the model would be data leakage.

    n_after = len(df.columns)
    print(f"\n  {n_before} feature -> {n_after} feature (+{n_after - n_before} trend feature)")

    return df











# ============================================================
# MAIN EXECUTION
# ============================================================


if __name__ == "__main__":
    print("=" * 60)
    print("BMS Predictive Maintenance - Faz 1 Pipeline")
    print("=" * 60)

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Output_XGBoost")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = load_mat_to_dataframe(MAT_PATH)

    # Make timestamp the first column
    cols = ["timestamp"] + [c for c in df.columns if c != "timestamp"]
    df = df[cols]

    print(f"\nDataFrame size: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # STEP 1 CSV: Raw data (optional)
    #df.to_csv(os.path.join(OUTPUT_DIR, "df_step1_raw_data.csv"), index=False)
    print(f"[CSV] df_step1_raw_data.csv saved ({df.shape[0]:,} rows)")

    # Visualize raw signals (in mV)
    plot_raw_signals(df, save_path=os.path.join(OUTPUT_DIR, "step1_raw_signals.png"))

    # ---- STEP 2: mV -> V Conversion ----
    print("\n" + "=" * 60)
    print("STEP 2: mV -> V Conversion")
    print("=" * 60)
    for col in MV_SIGNALS:
        df[col] = df[col] / 1000.0
    print(f"  {MV_SIGNALS} columns converted mV -> V")

    # STEP 2 CSV: Data with V conversion applied
    #df.to_csv(os.path.join(OUTPUT_DIR, "df_step2_volt_conversion.csv"), index=False)
    print(f"[CSV] df_step2_volt_conversion.csv saved ({df.shape[0]:,} rows)")

    # STEP 2 Visual: mV vs V comparison (before/after conversion)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    time_hours = df["timestamp"].values / 3600.0
    v_signals = ["Cell_V_Max", "Cell_V_Min", "Cell_V_Avg"]
    colors = ["tab:red", "tab:cyan", "tab:blue"]
    for ax, sig, color in zip(axes, v_signals, colors):
        ax.plot(time_hours, df[sig].values, color=color, linewidth=0.3)
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel("Voltage (V)")
        ax.set_title(f"{sig} (converted to V)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("STEP 2 - Voltage Signals After mV -> V Conversion", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step2_volt_conversion.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- STEP 3: Detect cycle boundaries ----
    boundaries = detect_cycle_boundaries(df)

    # Collect cycle information
    cycle_nums = []
    cycle_sohs = []
    cycle_durations_total = []   # total duration (including rest)
    cycle_durations_active = []  # active duration (excluding rest, |Current| > 0.5A)
    cycle_rest_durations = []    # rest duration
    for i, (s, e) in enumerate(boundaries):
        cycle_nums.append(i + 1)
        cycle_sohs.append(df["SOH"].iloc[s:e+1].mode()[0])
        total_sec = e - s
        active_sec = int((np.abs(df["Current"].iloc[s:e+1].values) > 0.5).sum())
        rest_sec = total_sec - active_sec
        cycle_durations_total.append(total_sec / 3600.0)
        cycle_durations_active.append(active_sec / 3600.0)
        cycle_rest_durations.append(rest_sec / 3600.0)

    # STEP 3 CSV: Cycle boundaries table
    cycle_boundaries_df = pd.DataFrame({
        "cycle": cycle_nums,
        "active_hours": cycle_durations_active,
        "rest_hours": cycle_rest_durations,
        "total_hours": cycle_durations_total,
        "soh": cycle_sohs,
    })
    cycle_boundaries_df.to_csv(os.path.join(OUTPUT_DIR, "df_step3_cycle_boundaries.csv"), index=False)
    print(f"[CSV] df_step3_cycle_boundaries.csv saved ({len(cycle_nums)} cycles)")

    # Cycle vs SOH and Cycle vs Duration plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 1. Cycle vs SOH
    axes[0].plot(cycle_nums, cycle_sohs, "b-o", markersize=4, linewidth=1.5)
    axes[0].set_ylabel("SOH (%)")
    axes[0].set_title("Cycle vs SOH (Battery Aging)")
    axes[0].axhline(y=80, color="red", linestyle="--", linewidth=1, label="End of Life (%80)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Cycle vs Duration (active duration and rest duration separately)
    axes[1].bar(cycle_nums, cycle_durations_active, color="tab:green", alpha=0.7,
                edgecolor="black", linewidth=0.3, label="Active (discharge+charge)")
    axes[1].bar(cycle_nums, cycle_rest_durations, bottom=cycle_durations_active,
                color="tab:gray", alpha=0.5, edgecolor="black", linewidth=0.3, label="Rest (current=0)")
    axes[1].set_xlabel("Cycle")
    axes[1].set_ylabel("Duration (hours)")
    axes[1].set_title("Cycle vs Duration (Active + Rest)")
    axes[1].axhline(y=np.mean(cycle_durations_active), color="red", linestyle="--",
                    linewidth=1, label=f"Active Avg: {np.mean(cycle_durations_active):.1f}h")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("STEP 3 - Cycle Analysis (SOH & Duration)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step3_cycle_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- STEP 4: Cycle-level summarization ----
    cycle_df = summarize_cycles(df, boundaries)

    # STEP 4 CSV: Cycle-level features
    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "df_step4_cycle_features.csv"), index=False)
    print(f"[CSV] df_step4_cycle_features.csv saved "
          f"({cycle_df.shape[0]} cycles x {cycle_df.shape[1]} features)")

    # Remove last cycle (incomplete cycle)
    last_cycle = cycle_df["cycle"].max()
    cycle_df = cycle_df[cycle_df["cycle"] != last_cycle].reset_index(drop=True)
    print(f"\n[FILTER] Cycle {last_cycle} removed (incomplete). Remaining: {len(cycle_df)} cycles")
    
    
    
    # ---- STEP 5: Add trend features ----
    cycle_df = add_trend_features(cycle_df)

    # STEP 5 CSV: Final data with trend features added
    cycle_df.to_csv(os.path.join(OUTPUT_DIR, "df_step5_trend_features.csv"), index=False)
    print(f"[CSV] df_step5_trend_features.csv saved "
          f"({cycle_df.shape[0]} cycles x {cycle_df.shape[1]} features)")

    # ---- Domain-Specific Feature Plots ----
    cycles = cycle_df["cycle"].values
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    # 1. Voltage Spread
    axes[0, 0].plot(cycles, cycle_df["voltage_spread_mean"], "b-o", markersize=3, linewidth=1)
    axes[0, 0].fill_between(cycles,
                            cycle_df["voltage_spread_mean"] - cycle_df["voltage_spread_std"],
                            cycle_df["voltage_spread_mean"] + cycle_df["voltage_spread_std"],
                            alpha=0.2)
    axes[0, 0].set_ylabel("Voltage Spread (V)")
    axes[0, 0].set_title("1. Voltage Spread Mean (Cell Imbalance)")
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Voltage Spread Under Load
    axes[0, 1].plot(cycles, cycle_df["voltage_spread_under_load"], "orange", marker="o", markersize=3, linewidth=1)
    axes[0, 1].set_ylabel("V Spread (V)")
    axes[0, 1].set_title("2. Voltage Spread Under Load Mean")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Voltage Asymmetry
    axes[0, 2].plot(cycles, cycle_df["voltage_asymmetry_mean"], "tab:purple", marker="o", markersize=3, linewidth=1)
    axes[0, 2].set_ylabel("Asymmetry")
    axes[0, 2].set_title("3. Voltage Asymmetry Mean (Voltage Symmetry)")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Voltage Asymmetry Under Load
    axes[1, 0].plot(cycles, cycle_df["voltage_asymmetry_under_load"], "tab:cyan", marker="o", markersize=3, linewidth=1)
    axes[1, 0].set_ylabel("Asymmetry")
    axes[1, 0].set_title("4. Voltage Asymmetry Under Load Mean")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Thermal Spread
    axes[1, 1].plot(cycles, cycle_df["thermal_spread_mean"], "r-o", markersize=3, linewidth=1)
    axes[1, 1].fill_between(cycles,
                            cycle_df["thermal_spread_mean"] - cycle_df["thermal_spread_std"],
                            cycle_df["thermal_spread_mean"] + cycle_df["thermal_spread_std"],
                            alpha=0.2, color="red")
    axes[1, 1].set_ylabel("Thermal Spread (C)")
    axes[1, 1].set_title("5. Thermal Spread Mean (Thermal Imbalance)")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Thermal Spread Under Load
    axes[1, 2].plot(cycles, cycle_df["thermal_spread_under_load"], "tab:red", marker="o", markersize=3, linewidth=1)
    axes[1, 2].set_ylabel("Thermal Spread (C)")
    axes[1, 2].set_title("6. Thermal Spread Under Load Mean")
    axes[1, 2].grid(True, alpha=0.3)

    # 7. Proxy Internal Resistance
    axes[2, 0].plot(cycles, cycle_df["proxy_IR_mean"], "g-o", markersize=3, linewidth=1)
    axes[2, 0].set_ylabel("Proxy IR (Ohm)")
    axes[2, 0].set_title("7. Proxy Internal Resistance Mean")
    axes[2, 0].grid(True, alpha=0.3)

    # 8. Energy Efficiency
    axes[2, 1].plot(cycles, cycle_df["energy_efficiency"], "m-o", markersize=3, linewidth=1)
    axes[2, 1].set_ylabel("Efficiency")
    axes[2, 1].set_xlabel("Cycle")
    axes[2, 1].set_title("8. Energy Efficiency (Discharge/Charge)")
    axes[2, 1].grid(True, alpha=0.3)

    # 9. Thermal Response Ratio
    axes[2, 2].plot(cycles, cycle_df["thermal_response_ratio"], "tab:brown", marker="o", markersize=3, linewidth=1)
    axes[2, 2].set_ylabel("Ratio")
    axes[2, 2].set_xlabel("Cycle")
    axes[2, 2].set_title("9. Thermal Response Ratio (dT/dI)")
    axes[2, 2].grid(True, alpha=0.3)

    fig.suptitle("STEP 4 - Domain-Specific Features vs Cycle", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step4_domain_features.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- STEP 5 Plots: Trend Features ----
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))

    # 1. Voltage Spread - Rolling Mean
    axes[0, 0].plot(cycles, cycle_df["voltage_spread_mean"], "b-", linewidth=0.8, alpha=0.4, label="Raw")
    axes[0, 0].plot(cycles, cycle_df["voltage_spread_mean_rolling_mean_3"], "r-", linewidth=1.2, label="Rolling 3")
    axes[0, 0].plot(cycles, cycle_df["voltage_spread_mean_rolling_mean_5"], "g-", linewidth=1.2, label="Rolling 5")
    axes[0, 0].plot(cycles, cycle_df["voltage_spread_mean_rolling_mean_10"], "k-", linewidth=1.5, label="Rolling 10")
    axes[0, 0].set_ylabel("Voltage Spread (V)")
    axes[0, 0].set_title("1. Voltage Spread - Rolling Mean")
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Voltage Spread Under Load - Rolling Mean
    axes[0, 1].plot(cycles, cycle_df["voltage_spread_under_load"], "b-", linewidth=0.8, alpha=0.4, label="Raw")
    axes[0, 1].plot(cycles, cycle_df["voltage_spread_under_load_rolling_mean_3"], "r-", linewidth=1.2, label="Rolling 3")
    axes[0, 1].plot(cycles, cycle_df["voltage_spread_under_load_rolling_mean_5"], "g-", linewidth=1.2, label="Rolling 5")
    axes[0, 1].plot(cycles, cycle_df["voltage_spread_under_load_rolling_mean_10"], "k-", linewidth=1.5, label="Rolling 10")
    axes[0, 1].set_ylabel("V Spread (V)")
    axes[0, 1].set_title("2. V Spread Under Load - Rolling Mean")
    axes[0, 1].legend(fontsize=7)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Voltage Asymmetry Under Load - Rolling Mean
    axes[0, 2].plot(cycles, cycle_df["voltage_asymmetry_under_load"], "b-", linewidth=0.8, alpha=0.4, label="Raw")
    axes[0, 2].plot(cycles, cycle_df["voltage_asymmetry_under_load_rolling_mean_3"], "r-", linewidth=1.2, label="Rolling 3")
    axes[0, 2].plot(cycles, cycle_df["voltage_asymmetry_under_load_rolling_mean_5"], "g-", linewidth=1.2, label="Rolling 5")
    axes[0, 2].plot(cycles, cycle_df["voltage_asymmetry_under_load_rolling_mean_10"], "k-", linewidth=1.5, label="Rolling 10")
    axes[0, 2].set_ylabel("Asymmetry")
    axes[0, 2].set_title("3. V Asymmetry Under Load - Rolling Mean")
    axes[0, 2].legend(fontsize=7)
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Thermal Spread Under Load - Rolling Mean
    axes[1, 0].plot(cycles, cycle_df["thermal_spread_under_load"], "b-", linewidth=0.8, alpha=0.4, label="Raw")
    axes[1, 0].plot(cycles, cycle_df["thermal_spread_under_load_rolling_mean_3"], "r-", linewidth=1.2, label="Rolling 3")
    axes[1, 0].plot(cycles, cycle_df["thermal_spread_under_load_rolling_mean_5"], "g-", linewidth=1.2, label="Rolling 5")
    axes[1, 0].plot(cycles, cycle_df["thermal_spread_under_load_rolling_mean_10"], "k-", linewidth=1.5, label="Rolling 10")
    axes[1, 0].set_ylabel("Thermal Spread (C)")
    axes[1, 0].set_title("4. T Spread Under Load - Rolling Mean")
    axes[1, 0].legend(fontsize=7)
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Proxy IR - Rolling Mean
    axes[1, 1].plot(cycles, cycle_df["proxy_IR_mean"], "b-", linewidth=0.8, alpha=0.4, label="Raw")
    axes[1, 1].plot(cycles, cycle_df["proxy_IR_mean_rolling_mean_3"], "r-", linewidth=1.2, label="Rolling 3")
    axes[1, 1].plot(cycles, cycle_df["proxy_IR_mean_rolling_mean_5"], "g-", linewidth=1.2, label="Rolling 5")
    axes[1, 1].plot(cycles, cycle_df["proxy_IR_mean_rolling_mean_10"], "k-", linewidth=1.5, label="Rolling 10")
    axes[1, 1].set_ylabel("Proxy IR (Ohm)")
    axes[1, 1].set_title("5. Proxy IR - Rolling Mean")
    axes[1, 1].legend(fontsize=7)
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Proxy IR Trend Slope (linear slope over last 10 cycles)
    axes[1, 2].plot(cycles, cycle_df["proxy_IR_trend_slope_10"], "g-o", markersize=3, linewidth=1)
    axes[1, 2].axhline(y=0, color="red", linewidth=0.5, linestyle="--")
    axes[1, 2].set_ylabel("Slope (Ohm/cycle)")
    axes[1, 2].set_title("6. Proxy IR Trend Slope (Last 10 Cycles)")
    axes[1, 2].grid(True, alpha=0.3)

    # 7. Energy Efficiency - Delta (cycle-to-cycle change)
    axes[2, 0].bar(cycles, cycle_df["energy_efficiency_delta"], color="tab:orange", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[2, 0].axhline(y=0, color="black", linewidth=0.5)
    axes[2, 0].set_ylabel("Delta")
    axes[2, 0].set_xlabel("Cycle")
    axes[2, 0].set_title("7. Energy Efficiency - Delta")
    axes[2, 0].grid(True, alpha=0.3)

    # 8. Charge Time to Full - Rate of Change
    axes[2, 1].plot(cycles, cycle_df["charge_time_to_full_roc"], "tab:purple", marker="o", markersize=2, linewidth=1)
    axes[2, 1].axhline(y=0, color="black", linewidth=0.5)
    axes[2, 1].set_ylabel("ROC (s/cycle)")
    axes[2, 1].set_xlabel("Cycle")
    axes[2, 1].set_title("8. Charge Time to Full - Rate of Change")
    axes[2, 1].grid(True, alpha=0.3)

    # 9. Thermal Spread - Rolling Std (stability)
    axes[2, 2].plot(cycles, cycle_df["thermal_spread_mean_rolling_std_3"], "r-", linewidth=1.2, label="Std 3")
    axes[2, 2].plot(cycles, cycle_df["thermal_spread_mean_rolling_std_5"], "g-", linewidth=1.2, label="Std 5")
    axes[2, 2].plot(cycles, cycle_df["thermal_spread_mean_rolling_std_10"], "k-", linewidth=1.5, label="Std 10")
    axes[2, 2].set_ylabel("Rolling Std")
    axes[2, 2].set_xlabel("Cycle")
    axes[2, 2].set_title("9. Thermal Spread - Rolling Std (Fluctuation)")
    axes[2, 2].legend(fontsize=7)
    axes[2, 2].grid(True, alpha=0.3)

    fig.suptitle("STEP 5 - Trend Features", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step5_trend_features.png"), dpi=150, bbox_inches="tight")
    plt.close()


    # ---- STEP 6: Feature Selection ----
    print("\n" + "=" * 60)
    print("STEP 6: Feature Selection")
    print("=" * 60)

    # All feature columns except cycle and soh_label
    exclude_cols = ["cycle", "soh_label"]
    feature_cols = [c for c in cycle_df.columns if c not in exclude_cols]

    # --- 0) Remove constant features (std=0, correlation cannot be computed) ---
    constant_cols = [c for c in feature_cols if cycle_df[c].std() == 0]
    if constant_cols:
        print(f"\n  Constant features (std=0, removing): {len(constant_cols)}")
        for c in constant_cols:
            print(f"    {c} = {cycle_df[c].iloc[0]}")
        feature_cols = [c for c in feature_cols if c not in constant_cols]

    # --- 1) Correlation with SOH (Feature vs Target) ---
    soh_corr = cycle_df[feature_cols].corrwith(cycle_df["soh_label"]).sort_values()
    soh_corr_abs = soh_corr.abs().sort_values(ascending=False)

    # Save correlation results to CSV
    corr_df = pd.DataFrame({
        "feature": soh_corr.index,
        "soh_correlation": soh_corr.values,
        "abs_correlation": soh_corr.abs().values,
    }).sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    corr_df.to_csv(os.path.join(OUTPUT_DIR, "df_step6_correlation.csv"), index=False)
    print(f"[CSV] df_step6_correlation.csv saved ({len(corr_df)} features)")

    print(f"\n--- All Features SOH Correlation ({len(soh_corr_abs)} features) ---")
    for i, (feat, corr) in enumerate(soh_corr_abs.items()):
        direction = "+" if soh_corr[feat] > 0 else "-"
        print(f"  {i+1:>3}. {feat:<50s} {direction}{corr:.4f}")

    # STEP 6 Visual: Top 30 feature correlation bar chart
    top30_corr = soh_corr_abs.head(30)
    top30_signed = soh_corr[top30_corr.index]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors_bar = ["tab:green" if v > 0 else "tab:red" for v in top30_signed.values]
    ax.barh(range(len(top30_signed)), top30_signed.values, color=colors_bar,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(top30_signed)))
    ax.set_yticklabels(top30_signed.index, fontsize=8)
    ax.set_xlabel("Correlation (with SOH)")
    ax.set_title("STEP 6 - Top 30 Feature SOH Correlation\n(Green=Positive, Red=Negative)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step6_correlation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ============================================================
    # STEP 7: All Features (Feature Selection Left to Model)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 7: All Features Will Be Used (XGB selects on its own)")
    print("=" * 60)
    print(f"\n  Total feature count: {len(feature_cols)}")
    print(f"  No feature selection applied — XGBoost makes its own selection via feature_importance")

    # STEP 7 Visual: Feature count comparison
    n_raw = 8 * 7  # 8 sinyal x 7 istatistik = 56
    n_domain = len([c for c in feature_cols if any(kw in c for kw in
                    ["voltage_spread", "voltage_asymmetry", "thermal_spread",
                     "proxy_IR", "energy_efficiency", "thermal_response",
                     "charge_time", "discharge_duration", "soc_range",
                     "cycle_duration", "active_duration", "rest_duration"])
                    and "rolling" not in c and "delta" not in c and "roc" not in c])
    n_trend = len([c for c in feature_cols if any(kw in c for kw in
                   ["rolling", "delta", "roc", "slope"])])
    n_other = len(feature_cols) - n_raw - n_domain - n_trend
    if n_other < 0:
        n_other = 0

    categories = ["Raw Signal\nStatistics", "Domain-Specific\nFeatures",
                   "Trend\nFeatures", "Other"]
    counts = [n_raw, n_domain, n_trend, n_other]
    # Remove zero-count categories
    non_zero = [(c, n) for c, n in zip(categories, counts) if n > 0]
    categories = [x[0] for x in non_zero]
    counts = [x[1] for x in non_zero]

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_colors = ["#6BAED6", "#74C476", "#FD8D3C", "#969696"][:len(counts)]
    bars = ax.bar(categories, counts, color=bar_colors, edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(cnt), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature Count")
    ax.set_title(f"STEP 7 - Feature Distribution Fed to Model (Total: {len(feature_cols)})",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step7_feature_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()




    # ============================================================
    # STEP 8: Model Training (XGBoost)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 8: Model Training (XGBoost)")
    print("=" * 60)

    # Separate Feature (X) and Target (y) — all features are used
    X = cycle_df[feature_cols].values
    y = cycle_df["soh_label"].values
    cycles_arr = cycle_df["cycle"].values

    print(f"\n  Feature count: {X.shape[1]}")
    print(f"  Sample count:  {X.shape[0]}")
    print(f"  SOH range:     {y.min():.1f} - {y.max():.1f}")

    # --- Train / Test Split: Interleaved (her 4. cycle test) ---
    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[3::4] = True
    train_mask = ~test_mask

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    cycles_test = cycles_arr[test_mask]

    print(f"\n  Split: Interleaved (every 4th cycle is test)")
    print(f"  Train: {len(X_train)} cycle, SOH {y_train.min():.0f}-{y_train.max():.0f}%")
    print(f"  Test:  {len(X_test)} cycle, SOH {y_test.min():.0f}-{y_test.max():.0f}%")

    # --- Sample Weight: more weight to low SOH region ---
    sample_weight = 1.0 + (y_train.max() - y_train) / (y_train.max() - y_train.min() + 1e-9)
    sample_weight = sample_weight / sample_weight.mean()

    # --- XGBoost ---
    print("\n  --- XGBoost Regressor ---")
    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
    y_pred_rf_test = xgb_model.predict(X_test)
    y_pred_rf_train = xgb_model.predict(X_train)

    # Train metrics
    mae_train = mean_absolute_error(y_train, y_pred_rf_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
    r2_train = r2_score(y_train, y_pred_rf_train)

    # Test metrics
    mae_test = mean_absolute_error(y_test, y_pred_rf_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
    r2_test = r2_score(y_test, y_pred_rf_test)

    # 5-Fold Cross Validation
    cv_scores_rf = cross_val_score(
        XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                     reg_lambda=1.0, random_state=42, verbosity=0),
        X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    cv_mae = -cv_scores_rf.mean()
    cv_std = cv_scores_rf.std()

    print(f"\n    {'Metric':<12} {'Train':<12} {'Test':<12} {'Diff':<12} {'Note'}")
    print(f"    {'-'*60}")
    mae_diff = mae_test - mae_train
    rmse_diff = rmse_test - rmse_train
    r2_diff = r2_train - r2_test
    mae_yorum = "OK" if mae_diff < 0.5 else "OVERFITTING?" if mae_diff > 1.0 else "WARNING"
    rmse_yorum = "OK" if rmse_diff < 0.5 else "OVERFITTING?" if rmse_diff > 1.0 else "WARNING"
    r2_yorum = "OK" if r2_diff < 0.01 else "OVERFITTING?" if r2_diff > 0.05 else "WARNING"
    print(f"    {'MAE (%)':<12} {mae_train:<12.4f} {mae_test:<12.4f} {mae_diff:<+12.4f} {mae_yorum}")
    print(f"    {'RMSE (%)':<12} {rmse_train:<12.4f} {rmse_test:<12.4f} {rmse_diff:<+12.4f} {rmse_yorum}")
    print(f"    {'R2':<12} {r2_train:<12.6f} {r2_test:<12.6f} {r2_diff:<+12.6f} {r2_yorum}")
    print(f"\n    5-Fold CV MAE: {cv_mae:.4f} (+/- {cv_std:.4f})")

    # ============================================================
    # STEP 8 Visualization
    # ============================================================

    # --- 1. Train vs Test Metrics Bar Chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["MAE (%)", "RMSE (%)", "R2"]
    train_vals = [mae_train, rmse_train, r2_train]
    test_vals = [mae_test, rmse_test, r2_test]

    for ax, metric, tr, te in zip(axes, metrics, train_vals, test_vals):
        x = np.arange(2)
        bars = ax.bar(x, [tr, te], color=["tab:blue", "tab:orange"],
                      edgecolor="black", linewidth=0.5, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Train", "Test"])
        ax.set_title(f"STEP 8 - {metric}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, [tr, te]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.4f}", ha="center", va="bottom", fontweight="bold")
        if metric == "R2":
            ax.set_ylim(min(0.95, min(tr, te) - 0.01), 1.001)

    plt.suptitle("STEP 8 - XGBoost: Train vs Test Metrics (Overfitting Check)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_train_vs_test_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_train_vs_test_metrics.png")

    # --- 2. SOH Prediction vs Actual (Test) ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cycles_test, y_test, "b-o", label="Actual SOH", linewidth=2, markersize=5)
    ax.plot(cycles_test, y_pred_rf_test, "r--s", label="XGB Prediction",
            linewidth=1.5, markersize=5, alpha=0.8)
    ax.fill_between(cycles_test, y_test, y_pred_rf_test, alpha=0.15, color="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"STEP 8 - XGBoost SOH Prediction\n"
                 f"MAE={mae_test:.4f}%  |  RMSE={rmse_test:.4f}%  |  R2={r2_test:.6f}",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_soh_pred_vs_actual.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_soh_pred_vs_actual.png")

    # --- 3. Residuals (Test) ---
    residuals = y_test - y_pred_rf_test
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if abs(r) < 0.5 else "orange" if abs(r) < 1 else "red"
              for r in residuals]
    ax.scatter(cycles_test, residuals, c=colors, alpha=0.8, s=50,
               edgecolors="black", linewidth=0.5)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax.axhline(y=0.5, color="green", linestyle="--", linewidth=0.6, alpha=0.5, label="±0.5%")
    ax.axhline(y=-0.5, color="green", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.axhline(y=1.0, color="orange", linestyle="--", linewidth=0.6, alpha=0.5, label="±1.0%")
    ax.axhline(y=-1.0, color="orange", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Residual (Actual - Prediction) %")
    ax.set_title(f"STEP 8 - XGBoost Residuals (max |error|={np.max(np.abs(residuals)):.3f}%)",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_residuals.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_residuals.png")

    # --- 4. Feature Importance (Top 30) ---
    importances = xgb_model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    top_n = min(30, len(feat_imp))
    feat_names = [f[0] for f in feat_imp[:top_n]][::-1]
    feat_vals = [f[1] for f in feat_imp[:top_n]][::-1]

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(feat_names)))
    ax.barh(range(len(feat_names)), feat_vals, color=colors,
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels(feat_names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"STEP 8 - XGBoost Feature Importance (Top {top_n} / {len(feature_cols)})",
                 fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_feature_importance.png")

    # --- 5. Test Metrics Summary ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # MAE
    axes[0].bar(["Test MAE"], [mae_test], color="#6BAED6", edgecolor="black",
                linewidth=0.5, width=0.4)
    axes[0].set_ylabel("MAE (%)")
    axes[0].set_title("Mean Absolute Error", fontweight="bold")
    axes[0].text(0, mae_test, f"{mae_test:.4f}%", ha="center", va="bottom",
                 fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    # RMSE
    axes[1].bar(["Test RMSE"], [rmse_test], color="#2171B5", edgecolor="black",
                linewidth=0.5, width=0.4)
    axes[1].set_ylabel("RMSE (%)")
    axes[1].set_title("Root Mean Squared Error", fontweight="bold")
    axes[1].text(0, rmse_test, f"{rmse_test:.4f}%", ha="center", va="bottom",
                 fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    # R2
    axes[2].bar(["Test R2"], [r2_test], color="#74C476", edgecolor="black",
                linewidth=0.5, width=0.4)
    axes[2].set_ylabel("R2")
    axes[2].set_title("R-Squared", fontweight="bold")
    axes[2].set_ylim(min(0.95, r2_test - 0.01), 1.001)
    axes[2].text(0, r2_test, f"{r2_test:.6f}", ha="center", va="bottom",
                 fontsize=14, fontweight="bold")
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("STEP 8 - XGBoost Test Results",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step8_test_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step8_test_metrics.png")

    # --- Save results to CSV ---
    results_df = pd.DataFrame([{
        "model": "XGBoost",
        "MAE_train": mae_train, "MAE_test": mae_test,
        "RMSE_train": rmse_train, "RMSE_test": rmse_test,
        "R2_train": r2_train, "R2_test": r2_test,
        "CV_MAE": cv_mae, "CV_STD": cv_std,
    }])
    results_df.to_csv(os.path.join(OUTPUT_DIR, "df_step8_model_results.csv"), index=False)
    print(f"[CSV] df_step8_model_results.csv saved")

    # Save model weights
    joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "model_xgb_baseline.joblib"))
    print(f"[MODEL] model_xgb_baseline.joblib saved")

    # ============================================================
    # STEP 9: Hyperparameter Tuning (Grid Search)
    # ============================================================
    print("\n" + "=" * 60)
    print("STEP 9: Hyperparameter Tuning (GridSearchCV, 5-Fold)")
    print("=" * 60)

    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n  Total combinations: {total_combos}")
    print(f"  Total fits with 5-Fold CV: {total_combos * 5}")

    gs = GridSearchCV(
        XGBRegressor(random_state=42, verbosity=0),
        param_grid,
        cv=5,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train, sample_weight=sample_weight)

    print(f"\n  Best parameters: {gs.best_params_}")
    print(f"  Best CV MAE: {-gs.best_score_:.4f}%")

    # Predict with best model
    xgb_tuned = gs.best_estimator_
    y_pred_tuned_test = xgb_tuned.predict(X_test)
    y_pred_tuned_train = xgb_tuned.predict(X_train)

    mae_tuned_train = mean_absolute_error(y_train, y_pred_tuned_train)
    rmse_tuned_train = np.sqrt(mean_squared_error(y_train, y_pred_tuned_train))
    r2_tuned_train = r2_score(y_train, y_pred_tuned_train)

    mae_tuned_test = mean_absolute_error(y_test, y_pred_tuned_test)
    rmse_tuned_test = np.sqrt(mean_squared_error(y_test, y_pred_tuned_test))
    r2_tuned_test = r2_score(y_test, y_pred_tuned_test)

    # 5-Fold CV (tuned model) — detailed fold analysis
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=False)

    print(f"\n  --- 5-Fold CV Detayli Analiz (Tuned Model) ---")
    print(f"  {'Fold':<6} {'Test MAE':<12} {'Test RMSE':<12} {'Test R2':<12} {'Train Cycles':<20} {'Test Cycles'}")
    print(f"  {'-'*75}")

    cv_test_maes = []
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_cv_train, X_cv_test = X[train_idx], X[test_idx]
        y_cv_train, y_cv_test = y[train_idx], y[test_idx]

        rf_cv = XGBRegressor(**gs.best_params_, random_state=42, verbosity=0)
        rf_cv.fit(X_cv_train, y_cv_train)

        y_cv_pred_test = rf_cv.predict(X_cv_test)

        cv_te_mae = mean_absolute_error(y_cv_test, y_cv_pred_test)
        cv_te_rmse = np.sqrt(mean_squared_error(y_cv_test, y_cv_pred_test))
        cv_te_r2 = r2_score(y_cv_test, y_cv_pred_test)
        cv_test_maes.append(cv_te_mae)

        train_range = f"{train_idx[0]+1}-{train_idx[-1]+1}"
        test_range = f"{test_idx[0]+1}-{test_idx[-1]+1}"

        print(f"  {fold_i+1:<6} {cv_te_mae:<12.4f} {cv_te_rmse:<12.4f} {cv_te_r2:<12.6f} {train_range:<20} {test_range}")

    cv_mae_tuned = np.mean(cv_test_maes)
    cv_std_tuned = np.std(cv_test_maes)
    print(f"\n  CV MAE Average: {cv_mae_tuned:.4f} (+/- {cv_std_tuned:.4f})")

    print(f"\n  --- Baseline vs Tuned Comparison ---")
    print(f"  {'Metric':<12} {'Baseline Test':<15} {'Tuned Test':<15} {'Change'}")
    print(f"  {'-'*55}")
    print(f"  {'MAE (%)':<12} {mae_test:<15.4f} {mae_tuned_test:<15.4f} {mae_tuned_test - mae_test:<+.4f}")
    print(f"  {'RMSE (%)':<12} {rmse_test:<15.4f} {rmse_tuned_test:<15.4f} {rmse_tuned_test - rmse_test:<+.4f}")
    print(f"  {'R2':<12} {r2_test:<15.6f} {r2_tuned_test:<15.6f} {r2_tuned_test - r2_test:<+.6f}")
    print(f"  {'CV MAE':<12} {cv_mae:<15.4f} {cv_mae_tuned:<15.4f} {cv_mae_tuned - cv_mae:<+.4f}")

    # Overfitting check
    mae_t_diff = mae_tuned_test - mae_tuned_train
    rmse_t_diff = rmse_tuned_test - rmse_tuned_train
    r2_t_diff = r2_tuned_train - r2_tuned_test
    mae_t_yorum = "OK" if abs(mae_t_diff) < 0.5 else "OVERFITTING?" if mae_t_diff > 1.0 else "WARNING"
    rmse_t_yorum = "OK" if abs(rmse_t_diff) < 0.5 else "OVERFITTING?" if rmse_t_diff > 1.0 else "WARNING"
    r2_t_yorum = "OK" if abs(r2_t_diff) < 0.01 else "OVERFITTING?" if r2_t_diff > 0.05 else "WARNING"
    print(f"\n  --- Tuned Model Overfitting Check ---")
    print(f"  {'Metric':<12} {'Train':<12} {'Test':<12} {'Diff':<12} {'Note'}")
    print(f"  {'-'*55}")
    print(f"  {'MAE (%)':<12} {mae_tuned_train:<12.4f} {mae_tuned_test:<12.4f} {mae_t_diff:<+12.4f} {mae_t_yorum}")
    print(f"  {'RMSE (%)':<12} {rmse_tuned_train:<12.4f} {rmse_tuned_test:<12.4f} {rmse_t_diff:<+12.4f} {rmse_t_yorum}")
    print(f"  {'R2':<12} {r2_tuned_train:<12.6f} {r2_tuned_test:<12.6f} {r2_t_diff:<+12.6f} {r2_t_yorum}")

    # --- Visual 1: Baseline vs Tuned Bar Chart ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_names = ["MAE (%)", "RMSE (%)", "R2"]
    baseline_vals = [mae_test, rmse_test, r2_test]
    tuned_vals = [mae_tuned_test, rmse_tuned_test, r2_tuned_test]
    bar_colors = [["#6BAED6", "#2171B5"], ["#6BAED6", "#2171B5"], ["#74C476", "#238B45"]]

    for ax, metric, bv, tv, cols in zip(axes, metrics_names, baseline_vals, tuned_vals, bar_colors):
        x = np.arange(2)
        bars = ax.bar(x, [bv, tv], color=cols, edgecolor="black", linewidth=0.5, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["Baseline", "Tuned"])
        ax.set_title(f"{metric}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, [bv, tv]):
            fmt = f"{val:.4f}" if metric != "R2" else f"{val:.6f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontweight="bold", fontsize=11)
        if metric == "R2":
            ax.set_ylim(min(0.95, min(bv, tv) - 0.01), 1.001)

    plt.suptitle("STEP 9 - Baseline vs Tuned XGBoost (Test Metrics)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step9_baseline_vs_tuned.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step9_baseline_vs_tuned.png")

    # --- Visual 2: Tuned SOH Prediction vs Actual ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(cycles_test, y_test, "b-o", label="Actual SOH", linewidth=2, markersize=5)
    ax.plot(cycles_test, y_pred_tuned_test, "r--s", label="XGB Tuned Prediction",
            linewidth=1.5, markersize=5, alpha=0.8)
    ax.fill_between(cycles_test, y_test, y_pred_tuned_test, alpha=0.15, color="red")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH (%)")
    ax.set_title(f"STEP 9 - Tuned XGBoost SOH Prediction\n"
                 f"MAE={mae_tuned_test:.4f}%  |  RMSE={rmse_tuned_test:.4f}%  |  R2={r2_tuned_test:.6f}",
                 fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "step9_tuned_soh_prediction.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] step9_tuned_soh_prediction.png")

    # --- Save results to CSV ---
    tuned_results_df = pd.DataFrame([{
        "model": "XGB_Baseline",
        "MAE_test": mae_test, "RMSE_test": rmse_test, "R2_test": r2_test,
        "CV_MAE": cv_mae,
    }, {
        "model": "XGB_Tuned",
        "MAE_test": mae_tuned_test, "RMSE_test": rmse_tuned_test, "R2_test": r2_tuned_test,
        "CV_MAE": cv_mae_tuned,
    }])
    tuned_results_df.to_csv(os.path.join(OUTPUT_DIR, "df_step9_tuning_results.csv"), index=False)
    print(f"[CSV] df_step9_tuning_results.csv saved")

    # Save best params
    best_params_df = pd.DataFrame([gs.best_params_])
    best_params_df.to_csv(os.path.join(OUTPUT_DIR, "df_step9_best_params.csv"), index=False)
    print(f"[CSV] df_step9_best_params.csv saved")

    # Save tuned model
    joblib.dump(xgb_tuned, os.path.join(OUTPUT_DIR, "model_xgb_tuned.joblib"))
    print(f"[MODEL] model_xgb_tuned.joblib saved")

    # Save feature columns (for inference)
    with open(os.path.join(OUTPUT_DIR, "feature_columns.txt"), "w") as f_out:
        for col in feature_cols:
            f_out.write(col + "\n")
    print(f"[FILE] feature_columns.txt ({len(feature_cols)} features)")

    if mae_tuned_test < mae_test:
        print(f"\n  RESULT: Tuned model is better (MAE {mae_test:.4f}% -> {mae_tuned_test:.4f}%)")
    else:
        print(f"\n  RESULT: Baseline model is better (MAE {mae_test:.4f}% vs {mae_tuned_test:.4f}%)")
        print(f"  [WARNING] Grid Search did not improve. Baseline model should be used.")

    # --- Result CSV: Collect all results in a single file ---
    result_path = os.path.join(OUTPUT_DIR, "Result_XGBoost.csv")
    with open(result_path, "w") as f:
        # Baseline vs Tuned Comparison
        f.write("Compare,MAE,RMSE,R2\n")
        f.write(f"Baseline,{mae_test:.4f},{rmse_test:.4f},{r2_test:.6f}\n")
        f.write(f"Tuned,{mae_tuned_test:.4f},{rmse_tuned_test:.4f},{r2_tuned_test:.6f}\n")
        f.write("\n")
        # Baseline Model Overfitting Check
        f.write("--- Baseline Model Overfitting Check ---\n")
        f.write("Metric,Train,Test,Diff,Note\n")
        f.write(f"MAE (%),{mae_train:.4f},{mae_test:.4f},{mae_diff:+.4f},{mae_yorum}\n")
        f.write(f"RMSE (%),{rmse_train:.4f},{rmse_test:.4f},{rmse_diff:+.4f},{rmse_yorum}\n")
        f.write(f"R2,{r2_train:.6f},{r2_test:.6f},{r2_diff:+.6f},{r2_yorum}\n")
        f.write("\n")
        # Tuned Model Overfitting Check
        f.write("--- Tuned Model Overfitting Check ---\n")
        f.write("Metric,Train,Test,Diff,Note\n")
        f.write(f"MAE (%),{mae_tuned_train:.4f},{mae_tuned_test:.4f},{mae_t_diff:+.4f},{mae_t_yorum}\n")
        f.write(f"RMSE (%),{rmse_tuned_train:.4f},{rmse_tuned_test:.4f},{rmse_t_diff:+.4f},{rmse_t_yorum}\n")
        f.write(f"R2,{r2_tuned_train:.6f},{r2_tuned_test:.6f},{r2_t_diff:+.6f},{r2_t_yorum}\n")
    print(f"[CSV] Result_XGBoost.csv saved")
