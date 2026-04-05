"""
ML vs DL - Top 5 Model Comparison
===================================
Reads the best 5 ML and best 5 DL models from their respective
Comparison CSVs and creates a unified comparison.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

ML_CSV = os.path.join(PROJECT_ROOT, "1-ML", "ML_Comparison.csv")
DL_CSV = os.path.join(PROJECT_ROOT, "2-DL", "DL_Comparison.csv")
OUTPUT_DIR = SCRIPT_DIR


def read_top_n(csv_path, n=5, category=""):
    """Read top N models sorted by MAE from a comparison CSV."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["Category"] = category
            rows.append(row)
    rows.sort(key=lambda x: float(x["MAE"]))
    return rows[:n]


if __name__ == "__main__":
    print("=" * 70)
    print("ML vs DL - Top 5 Model Comparison")
    print("=" * 70)

    # --- Read top 5 from each ---
    if not os.path.exists(ML_CSV):
        print(f"[ERROR] ML Comparison CSV not found: {ML_CSV}")
        print("  Run 1-ML/ML_Comparison.py first.")
        exit(1)
    if not os.path.exists(DL_CSV):
        print(f"[ERROR] DL Comparison CSV not found: {DL_CSV}")
        print("  Run 2-DL/DL_Comparison.py first.")
        exit(1)

    ml_top5 = read_top_n(ML_CSV, 5, "ML")
    dl_top5 = read_top_n(DL_CSV, 5, "DL")

    all_models = ml_top5 + dl_top5
    all_models.sort(key=lambda x: float(x["MAE"]))

    # --- Print Table ---
    print(f"\n{'Rank':<6} {'Category':<6} {'Model':<35} {'MAE (%)':<10} {'RMSE (%)':<10} {'R2':<12}")
    print("-" * 79)
    for i, m in enumerate(all_models):
        print(f"{i+1:<6} {m['Category']:<6} {m['Model']:<35} {m['MAE']:<10} {m['RMSE']:<10} {m['R2']:<12}")

    # --- Save CSV ---
    csv_path = os.path.join(OUTPUT_DIR, "ML_vs_DL_Top5.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Rank", "Category", "Model", "MAE", "RMSE", "R2"])
        writer.writeheader()
        for i, m in enumerate(all_models):
            writer.writerow({
                "Rank": i + 1,
                "Category": m["Category"],
                "Model": m["Model"],
                "MAE": m["MAE"],
                "RMSE": m["RMSE"],
                "R2": m["R2"],
            })
    print(f"\n[CSV] ML_vs_DL_Top5.csv saved")

    # --- Extract values ---
    names = [m["Model"] for m in all_models]
    maes = [float(m["MAE"]) for m in all_models]
    rmses = [float(m["RMSE"]) for m in all_models]
    r2s = [float(m["R2"]) for m in all_models]
    categories = [m["Category"] for m in all_models]

    # Colors: ML = blue tones, DL = green tones
    color_map = {"ML": "#2171B5", "DL": "#238B45"}
    bar_colors = [color_map[c] for c in categories]

    # ================================================================
    # PLOT 1: MAE, RMSE, R2 (3 subplots)
    # ================================================================
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # MAE
    bars = axes[0].barh(names[::-1], maes[::-1], color=bar_colors[::-1],
                        edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("MAE (%)")
    axes[0].set_title("Mean Absolute Error (lower = better)", fontweight="bold")
    for bar, val in zip(bars, maes[::-1]):
        axes[0].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)
    axes[0].grid(True, alpha=0.3, axis="x")

    # RMSE
    bars = axes[1].barh(names[::-1], rmses[::-1], color=bar_colors[::-1],
                        edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("RMSE (%)")
    axes[1].set_title("Root Mean Squared Error (lower = better)", fontweight="bold")
    for bar, val in zip(bars, rmses[::-1]):
        axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)
    axes[1].grid(True, alpha=0.3, axis="x")

    # R2
    bars = axes[2].barh(names[::-1], r2s[::-1], color=bar_colors[::-1],
                        edgecolor="black", linewidth=0.3)
    axes[2].set_xlabel("R2")
    axes[2].set_title("R-Squared (higher = better)", fontweight="bold")
    r2_min = min(r2s)
    axes[2].set_xlim(min(0.99, r2_min - 0.002), 1.001)
    for bar, val in zip(bars, r2s[::-1]):
        axes[2].text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9)
    axes[2].grid(True, alpha=0.3, axis="x")

    legend_elements = [
        Patch(facecolor="#2171B5", edgecolor="black", label="ML (Classical)"),
        Patch(facecolor="#238B45", edgecolor="black", label="DL (Deep Learning)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2, fontsize=12,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("ML vs DL - Top 5 Models Comparison (SOH Estimation)",
                 fontsize=16, fontweight="bold", y=1.06)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ML_vs_DL_Top5_Metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] ML_vs_DL_Top5_Metrics.png saved")

    # ================================================================
    # PLOT 2: MAE Bar Chart (side by side ML vs DL)
    # ================================================================
    fig, ax = plt.subplots(figsize=(14, 7))

    ml_names = [m["Model"] for m in ml_top5]
    ml_maes = [float(m["MAE"]) for m in ml_top5]
    dl_names = [m["Model"] for m in dl_top5]
    dl_maes = [float(m["MAE"]) for m in dl_top5]

    x = np.arange(5)
    width = 0.35

    bars_ml = ax.bar(x - width/2, ml_maes, width, label="ML (Classical)",
                      color="#6BAED6", edgecolor="black", linewidth=0.5)
    bars_dl = ax.bar(x + width/2, dl_maes, width, label="DL (Deep Learning)",
                      color="#74C476", edgecolor="black", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars_ml, ml_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(bars_dl, dl_maes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("MAE (%)", fontsize=12)
    ax.set_title("ML vs DL - Top 5 MAE Comparison (lower = better)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{i+1}" for i in range(5)])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    # Add model names as annotations
    for i, (ml_n, dl_n) in enumerate(zip(ml_names, dl_names)):
        ax.annotate(ml_n, xy=(i - width/2, 0), fontsize=6, ha="center",
                    va="top", rotation=45, color="#2171B5")
        ax.annotate(dl_n, xy=(i + width/2, 0), fontsize=6, ha="center",
                    va="top", rotation=45, color="#238B45")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ML_vs_DL_Top5_MAE.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] ML_vs_DL_Top5_MAE.png saved")

    # ================================================================
    # PLOT 3: Summary Card
    # ================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    best_ml = ml_top5[0]
    best_dl = dl_top5[0]

    summary_text = (
        f"BEST ML MODEL\n"
        f"{'='*40}\n"
        f"Model:  {best_ml['Model']}\n"
        f"MAE:    {best_ml['MAE']}%\n"
        f"RMSE:   {best_ml['RMSE']}%\n"
        f"R2:     {best_ml['R2']}\n"
        f"\n"
        f"BEST DL MODEL\n"
        f"{'='*40}\n"
        f"Model:  {best_dl['Model']}\n"
        f"MAE:    {best_dl['MAE']}%\n"
        f"RMSE:   {best_dl['RMSE']}%\n"
        f"R2:     {best_dl['R2']}\n"
        f"\n"
        f"WINNER: ML ({best_ml['Model']})\n"
        f"ML is {float(best_dl['MAE'])/float(best_ml['MAE']):.1f}x better than DL on this dataset\n"
        f"(100 cycles, single battery)"
    )

    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
            fontsize=12, verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=1", facecolor="#F0F4FF", edgecolor="#2171B5", linewidth=2))

    ax.set_title("ML vs DL - Final Verdict (Battery A, SOH Estimation)",
                 fontsize=14, fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "ML_vs_DL_Summary.png"),
                dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[PLOT] ML_vs_DL_Summary.png saved")

    # --- Final Print ---
    print(f"\n{'='*70}")
    print("ML vs DL COMPARISON COMPLETED")
    print(f"{'='*70}")
    print(f"  Best ML: {best_ml['Model']} (MAE={best_ml['MAE']}%)")
    print(f"  Best DL: {best_dl['Model']} (MAE={best_dl['MAE']}%)")
    print(f"  Winner:  ML ({float(best_dl['MAE'])/float(best_ml['MAE']):.1f}x better)")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"{'='*70}")
