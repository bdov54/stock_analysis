import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from utils import logger



def plot_stock_detail(yearly_df: pd.DataFrame, compid: str):
    sub = yearly_df[yearly_df["CompID"] == compid].copy()

    if sub.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No data for {compid}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    sub = sub.sort_values("Year")

    metrics = [
        ("ROE", "ROE"),
        ("ROIC", "ROIC"),
        ("EBIT_margin", "EBIT Margin"),
        ("CFO_NI", "CFO / Net Income"),
        ("REV_CAGR_3Y", "Revenue CAGR 3Y"),
        ("EPS_CAGR_3Y", "EPS CAGR 3Y"),
        ("D_E", "Debt / Equity"),
        ("Current_Ratio", "Current Ratio"),
    ]

    available_metrics = [(col, label) for col, label in metrics if col in sub.columns]

    if len(available_metrics) == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"No metric columns for {compid}", ha="center", va="center")
        ax.set_axis_off()
        return fig

    n = len(available_metrics)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows))
    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    else:
        axes = axes.reshape(nrows, ncols)

    axes_flat = [ax for row in axes for ax in row]

    for i, (col, label) in enumerate(available_metrics):
        ax = axes_flat[i]
        s = pd.to_numeric(sub[col], errors="coerce")

        ax.plot(sub["Year"], s, marker="o", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Year")
        ax.grid(True, alpha=0.3)

    # hide unused axes
    for j in range(len(available_metrics), len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(f"Stock Detail: {compid}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def save_outputs(
    ranked_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    artifacts_dir: str,
    charts_dir: str,
    cluster_artifacts: dict | None = None,
):
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # 1) Save main tables
    ranked_path = os.path.join(artifacts_dir, "ranked_results.csv")
    portfolio_path = os.path.join(artifacts_dir, "portfolio.csv")

    ranked_df.to_csv(ranked_path, index=False)
    portfolio_df.to_csv(portfolio_path, index=False)

    # 2) Save thresholds if available
    thresholds = ranked_df.attrs.get("thresholds", None)
    if thresholds is not None:
        thresholds_path = os.path.join(artifacts_dir, "thresholds.json")
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(thresholds, f, ensure_ascii=False, indent=2)

    # 3) Save cluster meta if provided
    if cluster_artifacts is not None:
        cluster_meta_path = os.path.join(artifacts_dir, "cluster_summary.json")
        meta = {
            "silhouette": None if pd.isna(cluster_artifacts.get("silhouette")) else float(cluster_artifacts.get("silhouette")),
            "n_clusters": cluster_artifacts.get("n_clusters"),
            "feature_cols_used": cluster_artifacts.get("feature_cols_used"),
            "cluster_sizes": cluster_artifacts.get("cluster_sizes"),
            "scaler_type": cluster_artifacts.get("scaler_type"),
            "clip_quantiles": cluster_artifacts.get("clip_quantiles"),
        }
        with open(cluster_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    # 4) Simple charts
    if "TOTAL_SCORE" in ranked_df.columns:
        plt.figure(figsize=(10, 4))
        ranked_df["TOTAL_SCORE"].dropna().hist(bins=25)
        plt.title("Distribution of TOTAL_SCORE")
        plt.xlabel("TOTAL_SCORE")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "total_score_distribution.png"), dpi=180)
        plt.close()

    if "PASS_COUNT" in ranked_df.columns:
        plt.figure(figsize=(10, 4))
        ranked_df["PASS_COUNT"].dropna().hist(bins=20)
        plt.title("Distribution of PASS_COUNT")
        plt.xlabel("PASS_COUNT")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "pass_count_distribution.png"), dpi=180)
        plt.close()

    if "cluster" in ranked_df.columns:
        vc = ranked_df["cluster"].value_counts(dropna=False).sort_index()
        plt.figure(figsize=(8, 4))
        vc.plot(kind="bar")
        plt.title("Cluster Sizes")
        plt.xlabel("Cluster")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "cluster_sizes.png"), dpi=180)
        plt.close()

    logger.info("Saved outputs to %s", artifacts_dir)