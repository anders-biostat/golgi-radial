"""Visualization routines for temporal organelle dispersion and compaction dynamics."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d


def plot_timecourse(
    csv_path: str,
    save_path: str,
    analysis_mode: str = "centrosome", # "centrosome" or "golgi"
    metric: str = "std_dev",           # "std_dev" or "70pct_quantile"
    frame_interval_min: float = 5.0,
):
    """Generates publication-level line plots for spatial standard deviation or 70% quantile radius."""
    analysis_mode_clean = analysis_mode.lower()
    if analysis_mode_clean not in ["centrosome", "golgi"]:
        raise ValueError("`analysis_mode` must be either 'centrosome' or 'golgi'")

    if metric not in ["std_dev", "70pct_quantile"]:
        raise ValueError("`metric` must be either 'std_dev' or '70pct_quantile'")

    merged_df = pd.read_csv(csv_path)
    merged_df["minutes"] = merged_df["frame"] * frame_interval_min

    # Ensure required metric column exists
    if metric == "std_dev":
        if "std_dev" not in merged_df.columns:
            if "variance" in merged_df.columns:
                merged_df["std_dev"] = np.sqrt(merged_df["variance"])
            else:
                raise KeyError("Neither 'std_dev' nor 'variance' column was found in the dataset.")
    elif metric == "70pct_quantile" and "70pct_quantile" not in merged_df.columns:
        raise KeyError("'70pct_quantile' column not found in the dataset.")

    # Calculate 'N' unique cells for legend labels
    merged_df["unique_cell_id"] = merged_df["filename"].astype(str) + "_" + merged_df["cell_idx"].astype(str)
    cell_counts = merged_df.groupby("cond")["unique_cell_id"].nunique()
    label_mapping = {cond: f"{cond} (N={count})" for cond, count in cell_counts.items()}

    # Replicate-wise aggregation
    plot_df = (
        merged_df.groupby(["cond", "experiment", "minutes"])[metric]
        .mean()
        .reset_index()
    )

    # Time 0 baseline normalization per replicate
    baseline_df = (
        plot_df.loc[plot_df.groupby(["cond", "experiment"])["minutes"].idxmin()]
        [["cond", "experiment", metric]]
        .rename(columns={metric: "baseline_val"})
    )

    plot_df = pd.merge(plot_df, baseline_df, on=["cond", "experiment"], how="left")
    plot_df["fold_change"] = plot_df[metric] / plot_df["baseline_val"]
    plot_df["cond_label"] = plot_df["cond"].map(label_mapping)

    # Mild Gaussian Kernel Smoothing per replicate
    plot_df["kernel_smoothed_fc"] = (
        plot_df.groupby(["cond_label", "experiment"])["fold_change"]
        .transform(lambda x: gaussian_filter1d(x, sigma=1.0))
    )

    # Color palette
    base_colors = {
        "WT": "#1f77b4", 
        "NINKO": "#ff7f0e", 
        "250KO": "#2ca02c", 
        "CROCCKO": "#9467bd"
    }
    condition_colors = {label_mapping[cond]: color for cond, color in base_colors.items() if cond in label_mapping}

    # Plot styling
    sns.set_theme(context="paper", style="ticks", font_scale=1.3)
    plt.figure(figsize=(8, 6))

    sns.lineplot(
        data=plot_df,
        x="minutes",
        y="kernel_smoothed_fc",
        hue="cond_label",
        palette=condition_colors,
        linewidth=2.5,
        alpha=1.0,
        errorbar=("ci", 95),
        err_kws={"alpha": 0.15}
    )

    # Metric-specific axes and titles
    if metric == "std_dev":
        ylabel = "Relative Spatial Std. Dev. (Fold Change)"
        title = f"Golgi Dispersion Analysis Mode: {analysis_mode.capitalize()}"
    else:
        ylabel = "Relative 70% Radius (Fold Change)"
        title = f"Golgi Compaction Dynamics ({analysis_mode.capitalize()} Mode)"

    plt.xlabel("Time (minutes)", fontweight="bold", labelpad=10)
    plt.ylabel(ylabel, fontweight="bold", labelpad=10)
    plt.xlim(0, max(plot_df["minutes"]))
    plt.legend(title=None, frameon=False, loc="lower left")
    plt.title(title, fontsize=16, fontweight="bold", pad=20)

    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Publication plot saved → {save_path}")

def plot_outlier_check(df_raw, df_clean, save_path):
    """QC plot comparing metric distributions before and after IQR filtering."""
    cols = [c for c in ("cell_area", "std_dev") if c in df_raw.columns]
    if not cols:
        return

    sns.set_theme(context="paper", style="ticks")
    fig, axes = plt.subplots(1, len(cols), figsize=(6 * len(cols), 4), squeeze=False)

    for ax, col in zip(axes[0], cols):
        sns.histplot(df_raw[col].dropna(), ax=ax, color="grey",
                     label=f"Before (n={len(df_raw)})", stat="density", alpha=0.5)
        sns.histplot(df_clean[col].dropna(), ax=ax, color="crimson",
                     label=f"After (n={len(df_clean)})", stat="density", alpha=0.5)
        ax.set_title(col)
        ax.legend(frameon=False)

    sns.despine(fig=fig)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Outlier QC plot saved → {save_path}")