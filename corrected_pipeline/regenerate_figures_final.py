"""
Regenerate definitive thesis figures using BOTH old baseline (tuned_cv_results.json)
and new definitive results (definitive_final_results.json).

Publication-quality settings: 300 DPI, Times New Roman, IEEE colors, no top/right spines.
All legends positioned below the chart or in a non-overlapping corner.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- Publication-quality global settings ----
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "text.usetex": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# Color palette (colorblind-friendly, IEEE compatible)
COLORS = {
    "primary": "#2563EB",     # blue
    "secondary": "#DC2626",   # red
    "tertiary": "#16A34A",    # green
    "accent": "#F59E0B",      # amber
    "gray": "#6B7280",
    "dark": "#1F2937",
    "purple": "#7C3AED",
    "pink": "#EC4899",
    "teal": "#14B8A6",
    "cyan": "#0891B2",
}

MODEL_COLORS = {
    "XGBoost": "#2563EB",
    "LightGBM": "#7C3AED",
    "CatBoost": "#F59E0B",
    "GradientBoosting": "#DC2626",
    "RandomForest": "#16A34A",
    "WeightedEnsemble": "#EC4899",
}


def load_old_results():
    """Load tuned_cv_results.json (old baseline, F1=0.6842)."""
    with open(RESULTS_DIR / "tuned_cv_results.json") as f:
        return json.load(f)


def load_definitive_results():
    """Load definitive_final_results.json (new definitive, F1=0.7345)."""
    with open(RESULTS_DIR / "definitive_final_results.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FIGURE 1: F1 Improvement Trajectory
# ---------------------------------------------------------------------------
def fig_improvement_trajectory(old, definitive):
    """
    Line chart showing F1 progression across pipeline versions:
      0.6424 (Stacking+SMOTE) -> 0.6842 (Tuned XGBoost) ->
      0.7459 (leaky ensemble) -> 0.7345 (clean ensemble)
    """
    versions = [
        "Stacking+SMOTE\n(Baseline)",
        "Tuned XGBoost\n(No SMOTE)",
        "Weighted Ensemble\n(Leaky Threshold)",
        "Weighted Ensemble\n(Leakage-Free)",
    ]
    f1_scores = [0.6424, 0.6842, 0.7459, 0.7345]

    # Bootstrap CIs where available; use CV std for estimates elsewhere
    # Baseline: approximate from Stacking CV std
    stacking_std = old.get("Stacking", {}).get("f1", {}).get("std", 0.035)
    xgb_ci = old.get("_bootstrap_ci", {}).get("metrics", {}).get("f1", {})
    leaky_ens = definitive.get("WeightedEnsemble", {}).get("f1", {})
    clean_ci = definitive.get("_bootstrap_ci_fixed", {}).get("metrics", {}).get("f1", {})

    ci_lower = [
        0.6424 - 1.96 * stacking_std,
        xgb_ci.get("ci_lower", 0.6842 - 0.03),
        leaky_ens.get("mean", 0.7459) - 1.96 * leaky_ens.get("std", 0.033),
        clean_ci.get("ci_lower", 0.7118),
    ]
    ci_upper = [
        0.6424 + 1.96 * stacking_std,
        xgb_ci.get("ci_upper", 0.6842 + 0.03),
        leaky_ens.get("mean", 0.7459) + 1.96 * leaky_ens.get("std", 0.033),
        clean_ci.get("ci_upper", 0.7715),
    ]

    yerr_lower = [f1 - lo for f1, lo in zip(f1_scores, ci_lower)]
    yerr_upper = [hi - f1 for f1, hi in zip(f1_scores, ci_upper)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(versions))

    ax.errorbar(
        x, f1_scores,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color=COLORS["primary"],
        linewidth=2.2,
        markersize=9,
        capsize=5,
        capthick=1.5,
        ecolor=COLORS["gray"],
        zorder=3,
    )

    # Annotate each point
    for i, (v, f1) in enumerate(zip(versions, f1_scores)):
        offset_y = 0.015 if i != 2 else -0.025
        va = "bottom" if i != 2 else "top"
        ax.annotate(
            f"{f1:.4f}",
            xy=(i, f1),
            xytext=(0, 12 if i != 2 else -14),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color=COLORS["dark"],
        )

    # Mark the leaky result with a different color
    ax.scatter([2], [f1_scores[2]], color=COLORS["secondary"], s=100, zorder=4,
               edgecolor=COLORS["dark"], linewidth=1.2, marker="X")

    # Mark the final clean result
    ax.scatter([3], [f1_scores[3]], color=COLORS["tertiary"], s=100, zorder=4,
               edgecolor=COLORS["dark"], linewidth=1.2, marker="D")

    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=9)
    ax.set_ylabel("F1-Score", fontweight="bold")
    ax.set_title("F1 Improvement Trajectory Across Pipeline Versions", fontweight="bold", pad=12)
    ax.set_ylim(0.58, 0.82)

    # Legend below chart
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color=COLORS["primary"], label="Pipeline Version",
               markersize=7, linewidth=1.8),
        Line2D([0], [0], marker="X", color=COLORS["secondary"], label="Leaky (threshold optimized on test)",
               markersize=8, linewidth=0, markeredgecolor=COLORS["dark"]),
        Line2D([0], [0], marker="D", color=COLORS["tertiary"], label="Leakage-Free (fixed threshold)",
               markersize=7, linewidth=0, markeredgecolor=COLORS["dark"]),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_improvement_trajectory.png")
    plt.close()
    print("  Saved: fig_improvement_trajectory.png")


# ---------------------------------------------------------------------------
# FIGURE 2: Definitive Model Comparison (Horizontal Bar)
# ---------------------------------------------------------------------------
def fig_definitive_model_comparison(definitive):
    """
    Horizontal bar chart of all 5 models + ensemble F1 scores
    from definitive results, with baseline reference line at 0.6842.
    """
    model_order = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting", "RandomForest", "WeightedEnsemble"]
    labels = []
    f1_scores = []
    colors = []

    for name in model_order:
        if name not in definitive or "f1" not in definitive[name]:
            continue
        data = definitive[name]["f1"]
        # Use f1_fixed_threshold for ensemble if available, otherwise mean
        if name == "WeightedEnsemble" and "f1_fixed_threshold" in definitive[name]:
            f1_val = definitive[name]["f1_fixed_threshold"]["mean"]
            labels.append("Ensemble (fixed thresh.)")
        else:
            f1_val = data["mean"]
            labels.append(name)
        f1_scores.append(f1_val)
        colors.append(MODEL_COLORS.get(name, COLORS["gray"]))

    # Sort ascending for horizontal bar (top = best)
    sorted_idx = np.argsort(f1_scores)
    labels = [labels[i] for i in sorted_idx]
    f1_scores = [f1_scores[i] for i in sorted_idx]
    colors = [colors[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, f1_scores, color=colors, edgecolor="white", linewidth=0.8, height=0.6)

    # Value labels on bars
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    # Baseline reference line at old best (0.6842)
    ax.axvline(x=0.6842, color=COLORS["gray"], linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(0.6842, len(labels) - 0.3, "Old best\n(0.6842)", ha="center", va="bottom",
            fontsize=8, color=COLORS["gray"], style="italic")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("F1-Score (10-Fold CV)", fontweight="bold")
    ax.set_title("Definitive Model Comparison (Leakage-Free Pipeline)", fontweight="bold", pad=12)
    ax.set_xlim(0.5, max(f1_scores) + 0.05)

    # Legend below chart
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, edgecolor="white", label=l)
        for l, c in zip(labels[::-1], colors[::-1])
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_definitive_model_comparison.png")
    plt.close()
    print("  Saved: fig_definitive_model_comparison.png")


# ---------------------------------------------------------------------------
# FIGURE 3: Leakage Impact (Grouped Bar)
# ---------------------------------------------------------------------------
def fig_leakage_impact(definitive):
    """
    Grouped bar chart: 'With Leakage' vs 'Leakage-Free' for F1, AUC-PR, MCC.
    Shows the difference that threshold leakage introduces.
    """
    ens = definitive.get("WeightedEnsemble", {})

    # Leaky values (optimized threshold on test)
    leaky_f1 = ens.get("f1", {}).get("mean", 0.7689)
    leaky_mcc = ens.get("mcc", {}).get("mean", 0.7151)
    leaky_aucpr = ens.get("average_precision", {}).get("mean", 0.8114)

    # Clean values (fixed threshold, no leakage)
    clean_f1 = ens.get("f1_fixed_threshold", {}).get("mean", 0.7345)
    clean_ci = definitive.get("_bootstrap_ci_fixed", {}).get("metrics", {})
    clean_mcc = clean_ci.get("mcc", {}).get("mean", 0.7148)
    clean_aucpr = clean_ci.get("pr_auc", {}).get("mean", 0.8115)

    metrics = ["F1-Score", "AUC-PR", "MCC"]
    leaky_vals = [leaky_f1, leaky_aucpr, leaky_mcc]
    clean_vals = [clean_f1, clean_aucpr, clean_mcc]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars1 = ax.bar(
        x - width / 2, leaky_vals, width,
        label="With Leakage (threshold opt. on test)",
        color=COLORS["secondary"],
        edgecolor="white",
        linewidth=0.8,
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width / 2, clean_vals, width,
        label="Leakage-Free (fixed threshold)",
        color=COLORS["tertiary"],
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels
    for bar, val in zip(bars1, leaky_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9,
        )
    for bar, val in zip(bars2, clean_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    # Delta annotations
    for i, (lv, cv) in enumerate(zip(leaky_vals, clean_vals)):
        delta = cv - lv
        ax.annotate(
            f"{delta:+.4f}",
            xy=(i, max(lv, cv) + 0.025),
            ha="center", fontsize=9,
            color=COLORS["primary"],
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight="bold")
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Impact of Threshold Leakage on Ensemble Metrics", fontweight="bold", pad=12)
    ax.set_ylim(0.6, max(max(leaky_vals), max(clean_vals)) + 0.07)

    # Legend below chart
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        framealpha=0.9,
        edgecolor="gray",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_leakage_impact.png")
    plt.close()
    print("  Saved: fig_leakage_impact.png")


# ---------------------------------------------------------------------------
# FIGURE 4: 10-Fold F1 Boxplot (Definitive)
# ---------------------------------------------------------------------------
def fig_definitive_fold_boxplot(definitive):
    """
    Boxplot of 10-fold F1 distribution for each model + ensemble.
    Horizontal line at ensemble mean.
    """
    model_order = ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting", "RandomForest"]
    box_data = []
    box_labels = []

    for name in model_order:
        if name in definitive and "f1" in definitive[name]:
            vals = definitive[name]["f1"].get("values", [])
            if vals:
                box_data.append(vals)
                box_labels.append(name)

    # Add ensemble (fixed threshold)
    ens = definitive.get("WeightedEnsemble", {})
    ens_fixed = ens.get("f1_fixed_threshold", {})
    if ens_fixed and "values" in ens_fixed:
        box_data.append(ens_fixed["values"])
        box_labels.append("Ensemble\n(fixed thresh.)")

    if not box_data:
        print("  Skipped: fig_definitive_fold_boxplot.png (no fold data)")
        return

    # Sort by median descending
    medians = [np.median(d) for d in box_data]
    sorted_idx = np.argsort(medians)[::-1]
    box_data = [box_data[i] for i in sorted_idx]
    box_labels = [box_labels[i] for i in sorted_idx]
    sorted_colors = []
    color_map = {
        "XGBoost": MODEL_COLORS["XGBoost"],
        "LightGBM": MODEL_COLORS["LightGBM"],
        "CatBoost": MODEL_COLORS["CatBoost"],
        "GradientBoosting": MODEL_COLORS["GradientBoosting"],
        "RandomForest": MODEL_COLORS["RandomForest"],
        "Ensemble\n(fixed thresh.)": MODEL_COLORS["WeightedEnsemble"],
    }
    for lb in box_labels:
        sorted_colors.append(color_map.get(lb, COLORS["gray"]))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bp = ax.boxplot(
        box_data,
        vert=True,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color=COLORS["dark"], linewidth=1.8),
        whiskerprops=dict(linewidth=1.0),
        capprops=dict(linewidth=1.0),
        flierprops=dict(marker="o", markersize=4, markerfacecolor=COLORS["gray"], alpha=0.5),
    )

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(sorted_colors[i])
        patch.set_alpha(0.7)
        patch.set_edgecolor(COLORS["dark"])
        patch.set_linewidth(0.8)

    # Ensemble mean horizontal line
    if ens_fixed and "mean" in ens_fixed:
        ens_mean = ens_fixed["mean"]
        ax.axhline(y=ens_mean, color=COLORS["pink"], linestyle="--", linewidth=1.2, alpha=0.8)
        ax.text(
            len(box_data) + 0.3, ens_mean,
            f"Ensemble\nmean={ens_mean:.4f}",
            va="center", fontsize=8, color=COLORS["pink"], style="italic",
        )

    ax.set_xticklabels(box_labels, fontsize=9)
    ax.set_ylabel("F1-Score", fontweight="bold")
    ax.set_title("10-Fold CV F1 Distribution", fontweight="bold", pad=12)
    ax.set_ylim(0.5, 0.85)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_definitive_fold_boxplot.png")
    plt.close()
    print("  Saved: fig_definitive_fold_boxplot.png")


# ---------------------------------------------------------------------------
# FIGURE 5: Competitor Comparison
# ---------------------------------------------------------------------------
def fig_competitor_comparison():
    """
    Bar chart comparing F1 scores on the rohitrox dataset:
      joneshshrestha (2023): 0.651
      pradlanka (2023): 0.697
      This work: 0.735
    Neutral palette, no leakage narrative.
    """
    systems = [
        "joneshshrestha\n(2023)",
        "pradlanka\n(2023)",
        "This work",
    ]
    f1_scores = [0.651, 0.697, 0.735]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(systems))

    bars = ax.bar(
        x, f1_scores,
        color=COLORS["primary"],
        edgecolor=COLORS["dark"],
        linewidth=0.8,
        width=0.55,
    )

    # Value labels
    for bar, val in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("F1-Score", fontweight="bold")
    ax.set_title(
        "Comparison with Published Results on rohitrox Dataset",
        fontweight="bold", pad=12,
    )
    ax.set_ylim(0.5, 0.82)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_competitor_comparison.png")
    plt.close()
    print("  Saved: fig_competitor_comparison.png")


# ---------------------------------------------------------------------------
# FIGURE 6: Bootstrap CI Forest Plot
# ---------------------------------------------------------------------------
def fig_bootstrap_ci(definitive):
    """
    Forest plot showing bootstrap CIs for F1, Precision, Recall, MCC, AUC-PR.
    Point estimate + 95% CI bars. Clean, publication quality.
    """
    ci_data = definitive.get("_bootstrap_ci_fixed", {}).get("metrics", {})

    if not ci_data:
        print("  Skipped: fig_bootstrap_ci.png (no bootstrap CI data)")
        return

    # Map metric keys to display labels
    metric_display = {
        "f1": "F1-Score",
        "precision": "Precision",
        "recall": "Recall",
        "mcc": "MCC",
        "pr_auc": "AUC-PR",
    }

    metric_keys = ["f1", "precision", "recall", "mcc", "pr_auc"]
    labels = []
    means = []
    ci_lowers = []
    ci_uppers = []

    for key in metric_keys:
        if key in ci_data:
            labels.append(metric_display.get(key, key))
            means.append(ci_data[key]["mean"])
            ci_lowers.append(ci_data[key]["ci_lower"])
            ci_uppers.append(ci_data[key]["ci_upper"])

    if not labels:
        print("  Skipped: fig_bootstrap_ci.png (empty metrics)")
        return

    # Reverse for bottom-to-top display
    labels = labels[::-1]
    means = means[::-1]
    ci_lowers = ci_lowers[::-1]
    ci_uppers = ci_uppers[::-1]

    yerr_lower = [m - lo for m, lo in zip(means, ci_lowers)]
    yerr_upper = [hi - m for m, hi in zip(means, ci_uppers)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(labels))

    # Plot CI bars
    ax.errorbar(
        means, y_pos,
        xerr=[yerr_lower, yerr_upper],
        fmt="D",
        color=COLORS["primary"],
        markersize=8,
        capsize=5,
        capthick=1.5,
        ecolor=COLORS["gray"],
        elinewidth=1.5,
        zorder=3,
    )

    # Value annotations
    for i, (m, lo, hi) in enumerate(zip(means, ci_lowers, ci_uppers)):
        ax.text(
            hi + 0.008, i,
            f"{m:.4f} [{lo:.4f}, {hi:.4f}]",
            va="center", ha="left",
            fontsize=8,
            color=COLORS["dark"],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_xlabel("Score (95% Bootstrap CI)", fontweight="bold")
    ax.set_title("Bootstrap 95% Confidence Intervals (Weighted Ensemble)", fontweight="bold", pad=12)
    ax.set_xlim(0.6, 0.9)

    # Add a vertical reference line at 0.7345 for F1
    ax.axvline(x=0.7345, color=COLORS["pink"], linestyle=":", linewidth=1, alpha=0.6)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_bootstrap_ci.png")
    plt.close()
    print("  Saved: fig_bootstrap_ci.png")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Regenerating Definitive Thesis Figures")
    print("  Old baseline: tuned_cv_results.json (F1=0.6842)")
    print("  New definitive: definitive_final_results.json (F1=0.7345)")
    print("=" * 60)

    old = load_old_results()
    definitive = load_definitive_results()

    print("\n[1/6] Improvement trajectory...")
    fig_improvement_trajectory(old, definitive)

    print("[2/6] Definitive model comparison...")
    fig_definitive_model_comparison(definitive)

    print("[3/6] Leakage impact...")
    fig_leakage_impact(definitive)

    print("[4/6] Definitive fold boxplot...")
    fig_definitive_fold_boxplot(definitive)

    print("[5/6] Competitor comparison...")
    fig_competitor_comparison()

    print("[6/6] Bootstrap CI forest plot...")
    fig_bootstrap_ci(definitive)

    print(f"\nAll 6 figures saved to {FIG_DIR}/")
    print("Settings: 300 DPI, Times New Roman, IEEE colors, no top/right spines")
    print("Done.")


if __name__ == "__main__":
    main()
