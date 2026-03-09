import numpy as np

from data_loader import load_master_dataset
from feature_engineering import build_yearly_features, build_company_features
from screening import apply_hard_filter, apply_scoring, build_portfolio
from clustering import run_kmeans, save_cluster_artifacts
from reporting import save_outputs
from utils import logger


def run_pipeline(config):
    config.ensure_dirs()

    logger.info("Running pipeline with config: %s", config.__dict__)

    # 1) Load
    master_df, company_info, meta = load_master_dataset(
        file_path=config.file_path,
        year_min=config.year_min,
        year_max=config.year_max,
    )

    # 2) Feature engineering
    yearly_df = build_yearly_features(master_df, company_info)
    feature_df = build_company_features(
        yearly_df=yearly_df,
        year_max=config.year_max,
        window_years=config.window_years,
        slope_years=config.slope_years,
    )
   
# Gắn lại metadata công ty sau khi aggregate company-level features
    if company_info is not None and not company_info.empty:
        company_meta_cols = [
            "CompID",
            "Company Common Name",
            "TRBC Industry Name",
            "GICS Sub-Industry Name",
            "Country of Exchange",
            "Organization Founded Year",
            "Date Became Public",
        ]
    company_meta_cols = [c for c in company_meta_cols if c in company_info.columns]

    company_meta = company_info[company_meta_cols].drop_duplicates(subset=["CompID"])
    feature_df = feature_df.merge(company_meta, on="CompID", how="left")
    
    # 3) Round 1: Hard filter
    round1_df = apply_hard_filter(
        feature_df=feature_df,
        hard_filter_rules=config.hard_filter_rules,
    )

    # 4) Scoring on round 1
    ranked_df = apply_scoring(
        df=round1_df,
        mode=config.mode,
        target_keep=config.target_keep,
        metric_rules=config.metric_rules,
        pillar_metrics=config.pillar_metrics,
        scoring_weights=config.scoring_weights,
        manual_thresholds=config.manual_thresholds,
    )

    # 5) Round 2: decide who passes scoring
    ranked_df["ROUND2_PASS"] = (
        (ranked_df["PASS_COUNT"] >= 5) |
        (ranked_df["TOTAL_SCORE"] > 0)
    )

    round2_df = ranked_df[ranked_df["ROUND2_PASS"] == True].copy()

    # 6) Cluster on round 1, not round 2
    clustered_round1_df, cluster_artifacts = run_kmeans(
        df=round1_df,
        feature_cols=config.cluster_features,
        n_clusters=getattr(config, "cluster_k_fixed", 2),
        scaler_type=getattr(config, "cluster_scaler_type", "robust"),
        clip_quantiles=getattr(config, "cluster_clip_quantiles", (0.01, 0.99)),
        apply_caps=True,
        apply_skew_transform=True,
    )

    # 7) Map cluster from round 1 to all ranked companies
    cluster_map = clustered_round1_df.set_index("CompID")["cluster"].to_dict()
    ranked_df["cluster"] = ranked_df["CompID"].map(cluster_map)
    round2_df["cluster"] = round2_df["CompID"].map(cluster_map)

    # 8) Build portfolio from round 2 only, diversified by cluster
    portfolio_df = build_portfolio(
        ranked_df=round2_df,
        portfolio_size=config.portfolio_size,
    )

    # 9) Save
    save_cluster_artifacts(cluster_artifacts, config.artifacts_dir)

    save_outputs(
        ranked_df=ranked_df,
        portfolio_df=portfolio_df,
        artifacts_dir=config.artifacts_dir,
        charts_dir=config.charts_dir,
        cluster_artifacts=cluster_artifacts,
    )

    logger.info(
        "Pipeline completed | round1=%s | round2=%s | portfolio=%s",
        len(round1_df), len(round2_df), len(portfolio_df)
    )

    return {
        "meta": meta,
        "master_df": master_df,
        "yearly_df": yearly_df,
        "feature_df": feature_df,
        "round1_df": round1_df,
        "ranked_df": ranked_df,
        "round2_df": round2_df,
        "portfolio_df": portfolio_df,
        "cluster_artifacts": cluster_artifacts,
        "clustered_round1_df": clustered_round1_df,
    }