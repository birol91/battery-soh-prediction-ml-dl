"""
DL Model Comparison — Merge all Result CSVs
=============================================
Reads each DL model's Result CSV from their Output folders
and writes a unified DL_Comparison.csv file.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model name -> Result CSV path
RESULT_FILES = {
    "LSTM": os.path.join(SCRIPT_DIR, "LSTM", "Output_LSTM", "Result_LSTM.csv"),
    "LSTM_Top30": os.path.join(SCRIPT_DIR, "LSTM", "Output_LSTM_Top30", "Result_LSTM_Top30.csv"),
    "GRU": os.path.join(SCRIPT_DIR, "GRU", "Output_GRU", "Result_GRU.csv"),
    "GRU_Top30": os.path.join(SCRIPT_DIR, "GRU", "Output_GRU_Top30", "Result_GRU_Top30.csv"),
    "CNN": os.path.join(SCRIPT_DIR, "CNN", "Output_CNN", "Result_CNN.csv"),
    "CNN_Top30": os.path.join(SCRIPT_DIR, "CNN", "Output_CNN_Top30", "Result_CNN_Top30.csv"),
    "MLP": os.path.join(SCRIPT_DIR, "MLP", "Output_MLP", "Result_MLP.csv"),
    "MLP_Top30": os.path.join(SCRIPT_DIR, "MLP", "Output_MLP_Top30", "Result_MLP_Top30.csv"),
    "Transformer": os.path.join(SCRIPT_DIR, "Transformer", "Output_Transformer", "Result_Transformer.csv"),
    "Transformer_Top30": os.path.join(SCRIPT_DIR, "Transformer", "Output_Transformer_Top30", "Result_Transformer_Top30.csv"),
}

OUTPUT_PATH = os.path.join(SCRIPT_DIR, "DL_Comparison.csv")


def parse_result_csv(path: str) -> dict:
    """Read Baseline and Tuned MAE, RMSE, R2 (and CV_MAE if present) from a Result CSV."""
    result = {"Baseline": {}, "Tuned": {}}
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[0] in ("Baseline", "Tuned"):
                result[row[0]] = {
                    "MAE": row[1],
                    "RMSE": row[2],
                    "R2": row[3],
                    "CV_MAE": row[4] if len(row) >= 5 else "",
                }
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("DL Model Comparison")
    print("=" * 60)

    rows = []
    for model_name, result_path in RESULT_FILES.items():
        if not os.path.exists(result_path):
            print(f"  [WARNING] {model_name}: Result CSV not found — {result_path}")
            continue

        data = parse_result_csv(result_path)
        for version in ("Baseline", "Tuned"):
            if data[version]:
                row_data = {
                    "Model": f"{model_name}_{version}",
                    "MAE": data[version]["MAE"],
                    "RMSE": data[version]["RMSE"],
                    "R2": data[version]["R2"],
                }
                if data[version].get("CV_MAE"):
                    row_data["CV_MAE"] = data[version]["CV_MAE"]
                rows.append(row_data)
                cv_str = f", CV_MAE={data[version]['CV_MAE']}" if data[version].get("CV_MAE") else ""
                print(f"  [OK] {model_name}_{version}: MAE={data[version]['MAE']}, "
                      f"RMSE={data[version]['RMSE']}, R2={data[version]['R2']}{cv_str}")

    if rows:
        # Sort by MAE (lower = better)
        rows.sort(key=lambda x: float(x["MAE"]))

        # Determine fieldnames (CV_MAE may or may not exist)
        has_cv = any("CV_MAE" in r for r in rows)
        fieldnames = ["Model", "MAE", "RMSE", "R2"]
        if has_cv:
            fieldnames.append("CV_MAE")

        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[CSV] DL_Comparison.csv saved ({len(rows)} models)")
        print(f"  Path: {OUTPUT_PATH}")

        # Print as table
        header = f"  {'Model':<35} {'MAE':<10} {'RMSE':<10} {'R2':<12}"
        if has_cv:
            header += f" {'CV_MAE':<10}"
        print(f"\n{header}")
        print(f"  {'-'*75}")
        for row in rows:
            line = f"  {row['Model']:<35} {row['MAE']:<10} {row['RMSE']:<10} {row['R2']:<12}"
            if has_cv and "CV_MAE" in row:
                line += f" {row['CV_MAE']:<10}"
            print(line)

        # --- Visual: MAE, RMSE, R2 comparison (3 subplots) ---
        model_names = [r["Model"] for r in rows]
        maes = [float(r["MAE"]) for r in rows]
        rmses = [float(r["RMSE"]) for r in rows]
        r2s = [float(r["R2"]) for r in rows]

        # Colors: Baseline = light blue, Tuned = dark green
        bar_colors = ["#238B45" if "Tuned" in m else "#6BAED6" for m in model_names]

        fig, axes = plt.subplots(1, 3, figsize=(22, max(6, len(model_names) * 0.5)))

        # 1. MAE
        bars = axes[0].barh(model_names[::-1], maes[::-1], color=bar_colors[::-1],
                            edgecolor="black", linewidth=0.3)
        axes[0].set_xlabel("MAE (%)")
        axes[0].set_title("Mean Absolute Error (lower = better)", fontweight="bold")
        for bar, val in zip(bars, maes[::-1]):
            axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=9)
        axes[0].grid(True, alpha=0.3, axis="x")

        # 2. RMSE
        bars = axes[1].barh(model_names[::-1], rmses[::-1], color=bar_colors[::-1],
                            edgecolor="black", linewidth=0.3)
        axes[1].set_xlabel("RMSE (%)")
        axes[1].set_title("Root Mean Squared Error (lower = better)", fontweight="bold")
        for bar, val in zip(bars, rmses[::-1]):
            axes[1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=9)
        axes[1].grid(True, alpha=0.3, axis="x")

        # 3. R2
        bars = axes[2].barh(model_names[::-1], r2s[::-1], color=bar_colors[::-1],
                            edgecolor="black", linewidth=0.3)
        axes[2].set_xlabel("R2")
        axes[2].set_title("R-Squared (higher = better)", fontweight="bold")
        r2_min = min(r2s)
        axes[2].set_xlim(min(0.85, r2_min - 0.02), 1.001)
        for bar, val in zip(bars, r2s[::-1]):
            axes[2].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                         f"{val:.4f}", va="center", fontsize=9)
        axes[2].grid(True, alpha=0.3, axis="x")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#6BAED6", edgecolor="black", label="Baseline"),
            Patch(facecolor="#238B45", edgecolor="black", label="Tuned"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("DL Model Comparison - SOH Estimation", fontsize=15,
                     fontweight="bold", y=1.06)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "DL_Comparison.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[PLOT] DL_Comparison.png")

    else:
        print("\n  [WARNING] No Result CSVs found. Run the DL model pipelines first.")
