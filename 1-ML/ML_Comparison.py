"""
ML Model Comparison — Merge all Result CSVs
=============================================
Reads each model's Result CSV from their Output folders
and writes a unified ML_Comparison.csv file.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model name -> Result CSV path
RESULT_FILES = {
    "Random_Forest": os.path.join(SCRIPT_DIR, "Random_Forest", "Output_Random_Forest", "Result_Random_Forest.csv"),
    "Random_Forest_Top30": os.path.join(SCRIPT_DIR, "Random_Forest", "Output_Random_Forest_Top30", "Result_Random_Forest_Top30.csv"),
    "LightGBM": os.path.join(SCRIPT_DIR, "LightGBM", "Output_LightGBM", "Result_LightGBM.csv"),
    "LightGBM_Top30": os.path.join(SCRIPT_DIR, "LightGBM", "Output_LightGBM_Top30", "Result_LightGBM_Top30.csv"),
    "XGBoost": os.path.join(SCRIPT_DIR, "XGBoost", "Output_XGBoost", "Result_XGBoost.csv"),
    "XGBoost_Top30": os.path.join(SCRIPT_DIR, "XGBoost", "Output_XGBoost_Top30", "Result_XGBoost_Top30.csv"),
    "GradientBoosting": os.path.join(SCRIPT_DIR, "GradientBoosting", "Output_GradientBoosting", "Result_GradientBoosting.csv"),
    "GradientBoosting_Top30": os.path.join(SCRIPT_DIR, "GradientBoosting", "Output_GradientBoosting_Top30", "Result_GradientBoosting_Top30.csv"),
    "ExtraTrees": os.path.join(SCRIPT_DIR, "ExtraTrees", "Output_ExtraTrees", "Result_ExtraTrees.csv"),
    "ExtraTrees_Top30": os.path.join(SCRIPT_DIR, "ExtraTrees", "Output_ExtraTrees_Top30", "Result_ExtraTrees_Top30.csv"),
    "CatBoost": os.path.join(SCRIPT_DIR, "Catboost", "Output_CatBoost", "Result_CatBoost.csv"),
    "CatBoost_Top30": os.path.join(SCRIPT_DIR, "Catboost", "Output_CatBoost_Top30", "Result_CatBoost_Top30.csv"),
}

OUTPUT_PATH = os.path.join(SCRIPT_DIR, "ML_Comparison.csv")


def parse_result_csv(path: str) -> dict:
    """Read Baseline and Tuned MAE, RMSE, R2 values from a Result CSV."""
    result = {"Baseline": {}, "Tuned": {}}
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[0] in ("Baseline", "Tuned"):
                result[row[0]] = {
                    "MAE": row[1],
                    "RMSE": row[2],
                    "R2": row[3],
                }
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("ML Model Comparison")
    print("=" * 60)

    rows = []
    for model_name, result_path in RESULT_FILES.items():
        if not os.path.exists(result_path):
            print(f"  [WARNING] {model_name}: Result CSV not found — {result_path}")
            continue

        data = parse_result_csv(result_path)
        for version in ("Baseline", "Tuned"):
            if data[version]:
                rows.append({
                    "Model": f"{model_name}_{version}",
                    "MAE": data[version]["MAE"],
                    "RMSE": data[version]["RMSE"],
                    "R2": data[version]["R2"],
                })
                print(f"  [OK] {model_name}_{version}: MAE={data[version]['MAE']}, "
                      f"RMSE={data[version]['RMSE']}, R2={data[version]['R2']}")

    if rows:
        # Sort by MAE (lower = better)
        rows.sort(key=lambda x: float(x["MAE"]))

        with open(OUTPUT_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Model", "MAE", "RMSE", "R2"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"\n[CSV] ML_Comparison.csv saved ({len(rows)} models)")
        print(f"  Path: {OUTPUT_PATH}")

        # Print as table
        print(f"\n  {'Model':<35} {'MAE':<10} {'RMSE':<10} {'R2':<12}")
        print(f"  {'-'*67}")
        for row in rows:
            print(f"  {row['Model']:<35} {row['MAE']:<10} {row['RMSE']:<10} {row['R2']:<12}")

        # --- Visual: MAE, RMSE, R2 comparison (3 subplots) ---
        model_names = [r["Model"] for r in rows]
        maes = [float(r["MAE"]) for r in rows]
        rmses = [float(r["RMSE"]) for r in rows]
        r2s = [float(r["R2"]) for r in rows]

        # Colors: Baseline = light blue, Tuned = dark blue
        bar_colors = ["#2171B5" if "Tuned" in m else "#6BAED6" for m in model_names]

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
        axes[2].set_xlim(min(0.9, min(r2s) - 0.02), 1.001)
        for bar, val in zip(bars, r2s[::-1]):
            axes[2].text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                         f"{val:.4f}", va="center", fontsize=9)
        axes[2].grid(True, alpha=0.3, axis="x")

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#6BAED6", edgecolor="black", label="Baseline"),
            Patch(facecolor="#2171B5", edgecolor="black", label="Tuned"),
        ]
        fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=11,
                   bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("ML Model Comparison - SOH Estimation", fontsize=15,
                     fontweight="bold", y=1.06)
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, "ML_Comparison.png"), dpi=150, bbox_inches="tight")
        plt.show()
        print(f"[PLOT] ML_Comparison.png")

    else:
        print("\n  [WARNING] No Result CSVs found. Run the model pipelines first.")
