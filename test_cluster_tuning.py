import os
os.environ["OMP_NUM_THREADS"] = "1"

from config import AppConfig
from data_loader import load_master_dataset
from feature_engineering import build_yearly_features, build_company_features
from screening import apply_hard_filter, apply_scoring
from clustering import search_best_kmeans, run_kmeans


def main():
    # =========================
    # CONFIG
    # =========================
    cfg = AppConfig(
        file_path="data/Greece.xlsx",
        use_clustering=False,  # test clustering riêng
    )

    # =========================
    # LOAD DATA
    # =========================
    master_df, company_info, meta = load_master_dataset(
        file_path=cfg.file_path,
        year_min=cfg.year_min,
        year_max=cfg.year_max,
    )

    print("\n=== LOAD META ===")
    print(meta)

    # =========================
    # FEATURE ENGINEERING
    # =========================
    yearly_df = build_yearly_features(master_df, company_info)

    feature_df = build_company_features(
        yearly_df=yearly_df,
        year_max=cfg.year_max,
        window_years=cfg.window_years,
        slope_years=cfg.slope_years,
    )

    # =========================
    # HARD FILTER
    # =========================
    filtered_df = apply_hard_filter(feature_df, cfg.hard_filter_rules)

    # =========================
    # SCORING
    # =========================
    ranked_df = apply_scoring(
        filtered_df,
        mode=cfg.mode,
        target_keep=cfg.target_keep,
        metric_rules=cfg.metric_rules,
        pillar_metrics=cfg.pillar_metrics,
        scoring_weights=cfg.scoring_weights,
        manual_thresholds=cfg.manual_thresholds,
    )

    print("\n=== BASE DATA ===")
    print("yearly_df:", yearly_df.shape)
    print("feature_df:", feature_df.shape)
    print("filtered_df:", filtered_df.shape)
    print("ranked_df:", ranked_df.shape)

    # =========================
    # STEP 1 + STEP 2 + STEP 3 + STEP 4
    # - feature set đã giảm trong config.py
    # - thử K=2..5
    # - clip outlier 1%–99%
    # - Standard vs Robust scaler
    # - ép min cluster size >= 5
    # =========================
    print("\n=== TRY STANDARD SCALER + CLIP 1%-99% ===")
    res_std = search_best_kmeans(
        df=ranked_df,
        feature_cols=cfg.cluster_features,
        k_values=range(2, 6),
        scaler_type="standard",
        clip_quantiles=(0.01, 0.99),
        min_cluster_size=5,
        apply_caps=True,
        apply_skew_transform=True,
    )
    print(res_std["summary_df"])
    print("Best K:", res_std["best_k"])
    print("Best silhouette:", res_std["best_silhouette"])

    print("\n=== TRY ROBUST SCALER + CLIP 1%-99% ===")
    res_rob = search_best_kmeans(
        df=ranked_df,
        feature_cols=cfg.cluster_features,
        k_values=range(2, 6),
        scaler_type="robust",
        clip_quantiles=(0.01, 0.99),
        min_cluster_size=5,
        apply_caps=True,
        apply_skew_transform=True,
    )
    print(res_rob["summary_df"])
    print("Best K:", res_rob["best_k"])
    print("Best silhouette:", res_rob["best_silhouette"])

    # =========================
    # CHOOSE BEST SETUP
    # =========================
    std_sil = res_std["best_silhouette"]
    rob_sil = res_rob["best_silhouette"]

    if rob_sil is not None and std_sil is not None:
        best_scaler = "robust" if rob_sil > std_sil else "standard"
    elif rob_sil is not None:
        best_scaler = "robust"
    else:
        best_scaler = "standard"

    best_k = res_rob["best_k"] if best_scaler == "robust" else res_std["best_k"]

    print(f"\n=== FINAL FIT: scaler={best_scaler}, k={best_k} ===")

    clustered_df, cluster_artifacts = run_kmeans(
        df=ranked_df,
        feature_cols=cfg.cluster_features,
        n_clusters=best_k,
        scaler_type=best_scaler,
        clip_quantiles=(0.01, 0.99),
        apply_caps=True,
        apply_skew_transform=True,
    )

    print(clustered_df[["CompID", "cluster"]].head(20))
    print("Final silhouette:", cluster_artifacts["silhouette"])

    # =========================
    # CLUSTER DIAGNOSTICS
    # =========================
    print("\n=== CLUSTER SIZE ===")
    print(clustered_df["cluster"].value_counts(dropna=False).sort_index())

    cols_to_check = [
        "TOTAL_SCORE",
        "MED_ROE",
        "MED_EBIT_margin",
        "MED_REV_CAGR_3Y",
        "MED_EPS_CAGR_3Y",
        "MED_CFO_NI",
        "MED_D_E",
        "MED_NetDebt_EBITDA",
        "MED_Current_Ratio",
        "MED_PPE_Assets",
    ]
    available_cols = [c for c in cols_to_check if c in clustered_df.columns]

    print("\n=== CLUSTER PROFILE ===")
    print(clustered_df.groupby("cluster")[available_cols].mean())

    print("\n=== SMALL / DISTINCT CLUSTER MEMBERS ===")
    cluster_sizes = clustered_df["cluster"].value_counts()
    small_clusters = cluster_sizes[cluster_sizes <= 5].index.tolist()

    if small_clusters:
        cols_to_show = [
            "CompID",
            "cluster",
            "TOTAL_SCORE",
            "MED_ROE",
            "MED_EBIT_margin",
            "MED_REV_CAGR_3Y",
            "MED_EPS_CAGR_3Y",
            "MED_CFO_NI",
            "MED_D_E",
            "MED_NetDebt_EBITDA",
            "MED_Current_Ratio",
            "MED_PPE_Assets",
        ]
        cols_to_show = [c for c in cols_to_show if c in clustered_df.columns]
        print(clustered_df.loc[clustered_df["cluster"].isin(small_clusters), cols_to_show])
    else:
        print("Không có cụm quá nhỏ.")

    print("\n=== TOP 15 RANKED ===")
    top_cols = ["CompID", "TOTAL_SCORE", "PASS_COUNT"]
    if "cluster" in clustered_df.columns:
        top_cols.append("cluster")
    top_cols = [c for c in top_cols if c in clustered_df.columns]
    print(clustered_df[top_cols].head(15))


if __name__ == "__main__":
    main()