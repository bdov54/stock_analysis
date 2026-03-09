import numpy as np
import pandas as pd

from utils import logger


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _recommend_threshold(series: pd.Series, direction: str, target_keep: float = 0.25) -> float:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return np.nan

    if direction == "high":
        return float(s.quantile(1 - target_keep))
    elif direction == "low":
        return float(s.quantile(target_keep))
    elif direction == "abs_low":
        return float(s.abs().quantile(target_keep))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def _winsorize_series(series: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = s.dropna()
    if len(valid) == 0:
        return s
    lower = valid.quantile(lower_q)
    upper = valid.quantile(upper_q)
    return s.clip(lower=lower, upper=upper)


def _zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mu = s.mean()
    sigma = s.std(ddof=0)
    if pd.isna(sigma) or sigma == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sigma


def apply_hard_filter(feature_df: pd.DataFrame, hard_filter_rules: dict) -> pd.DataFrame:
    df = feature_df.copy()

    require_positive_equity = hard_filter_rules.get("require_positive_equity", True)
    min_coverage = hard_filter_rules.get("min_coverage", 0.60)
    min_cfo_ni = hard_filter_rules.get("min_cfo_ni", 0.50)
    max_de_ratio = hard_filter_rules.get("max_de_ratio", 2.50)
    max_netdebt_ebitda = hard_filter_rules.get("max_netdebt_ebitda", 4.00)
    min_roe = hard_filter_rules.get("min_roe", 0.00)

    # Coverage
    coverage_cols = [c for c in df.columns if c.startswith("COV_")]
    if coverage_cols:
        df["HF_min_coverage"] = df[coverage_cols].mean(axis=1) >= min_coverage
    else:
        df["HF_min_coverage"] = True

    # Positive equity proxy
    # D/E only exists when equity > 0 in feature engineering, so use it as a proxy
    if require_positive_equity:
        if "MED_D_E" in df.columns:
            df["HF_positive_equity"] = df["MED_D_E"].notna()
        else:
            df["HF_positive_equity"] = True
    else:
        df["HF_positive_equity"] = True

    # CFO/NI
    if "MED_CFO_NI" in df.columns:
        df["HF_cfo_ni"] = df["MED_CFO_NI"] >= min_cfo_ni
    else:
        df["HF_cfo_ni"] = True

    # D/E
    if "MED_D_E" in df.columns:
        df["HF_de"] = df["MED_D_E"] <= max_de_ratio
    else:
        df["HF_de"] = True

    # NetDebt/EBITDA
    if "MED_NetDebt_EBITDA" in df.columns:
        s = pd.to_numeric(df["MED_NetDebt_EBITDA"], errors="coerce")
        df["HF_netdebt_ebitda"] = s.isna() | (s <= max_netdebt_ebitda)
    else:
        df["HF_netdebt_ebitda"] = True

    # ROE
    if "MED_ROE" in df.columns:
        df["HF_roe"] = df["MED_ROE"] >= min_roe
    else:
        df["HF_roe"] = True

    hard_cols = [c for c in df.columns if c.startswith("HF_")]
    df["HARD_FILTER_PASS"] = df[hard_cols].all(axis=1)

    filtered = df[df["HARD_FILTER_PASS"]].copy()

    logger.info(
        "Hard filter kept %s / %s companies.",
        len(filtered),
        len(df),
    )
    return filtered


def apply_scoring(
    df: pd.DataFrame,
    mode: str,
    target_keep: float,
    metric_rules: dict,
    pillar_metrics: dict,
    scoring_weights: dict,
    manual_thresholds: dict | None = None,
) -> pd.DataFrame:
    ranked = df.copy()
    manual_thresholds = manual_thresholds or {}

    threshold_rows = []

    # 1) Threshold screening
    pass_cols = []
    for metric, direction in metric_rules.items():
        if metric not in ranked.columns:
            ranked[f"PASS_{metric}"] = False
            pass_cols.append(f"PASS_{metric}")
            threshold_rows.append({
                "metric": metric,
                "direction": direction,
                "threshold": np.nan,
                "mode": mode,
            })
            continue

        s = _safe_series(ranked, metric)

        if mode == "manual" and metric in manual_thresholds:
            thr = manual_thresholds[metric]
        else:
            thr = _recommend_threshold(s, direction, target_keep=target_keep)

        if direction == "high":
            ranked[f"PASS_{metric}"] = s >= thr
        elif direction == "low":
            ranked[f"PASS_{metric}"] = s <= thr
        elif direction == "abs_low":
            ranked[f"PASS_{metric}"] = s.abs() <= thr
        else:
            ranked[f"PASS_{metric}"] = False

        pass_cols.append(f"PASS_{metric}")
        threshold_rows.append({
            "metric": metric,
            "direction": direction,
            "threshold": thr,
            "mode": mode,
        })

    ranked["PASS_COUNT"] = ranked[pass_cols].sum(axis=1)

    # 2) Winsorize + z-score for scoring
    score_metric_values = {}
    for metric, direction in metric_rules.items():
        if metric not in ranked.columns:
            ranked[f"ZS_{metric}"] = np.nan
            continue

        s = _winsorize_series(ranked[metric], 0.01, 0.99)
        z = _zscore(s)

        # For "low" and "abs_low", lower values are better -> reverse sign
        if direction == "low":
            z = -z
        elif direction == "abs_low":
            z = -_zscore(s.abs())

        ranked[f"ZS_{metric}"] = z
        score_metric_values[metric] = z

    # 3) Pillar scores
    for pillar, metrics in pillar_metrics.items():
        z_cols = [f"ZS_{m}" for m in metrics if f"ZS_{m}" in ranked.columns]
        if z_cols:
            ranked[f"SCORE_{pillar.upper()}"] = ranked[z_cols].mean(axis=1, skipna=True)
        else:
            ranked[f"SCORE_{pillar.upper()}"] = np.nan

    # 4) Total score
    total_score = pd.Series(0.0, index=ranked.index)
    total_weight = 0.0

    for pillar, weight in scoring_weights.items():
        col = f"SCORE_{pillar.upper()}"
        if col in ranked.columns:
            s = pd.to_numeric(ranked[col], errors="coerce").fillna(0.0)
            total_score = total_score + weight * s
            total_weight += weight

    if total_weight > 0:
        ranked["TOTAL_SCORE"] = total_score / total_weight
    else:
        ranked["TOTAL_SCORE"] = total_score

    ranked = ranked.sort_values(["TOTAL_SCORE", "PASS_COUNT"], ascending=[False, False]).reset_index(drop=True)

    threshold_df = pd.DataFrame(threshold_rows)
    ranked.attrs["thresholds"] = threshold_df.to_dict(orient="records")

    logger.info(
        "Scoring completed for %s companies. Top TOTAL_SCORE=%.4f",
        len(ranked),
        ranked["TOTAL_SCORE"].max() if len(ranked) > 0 else np.nan,
    )

    return ranked


def build_portfolio(
    ranked_df: pd.DataFrame,
    portfolio_size: int = 7,
) -> pd.DataFrame:
    df = ranked_df.copy()
    if len(df) == 0:
        return df.copy()

    df = df.sort_values("TOTAL_SCORE", ascending=False).reset_index(drop=True)

    # Nếu không có cluster hoặc chỉ có 1 cluster thì fallback top N
    if "cluster" not in df.columns or df["cluster"].nunique() < 2:
        return df.head(portfolio_size).copy()

    unique_clusters = sorted(df["cluster"].dropna().unique().tolist())

    # Lấy ít nhất 1 mã từ mỗi cluster
    picks = []
    used_ids = set()

    for cl in unique_clusters:
        sub = df[df["cluster"] == cl]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        picks.append(row)
        used_ids.add(row["CompID"])

    portfolio = pd.DataFrame(picks).reset_index(drop=True)

    # Fill phần còn lại theo score, nhưng tránh dồn quá nhiều vào 1 cluster
    remaining = df[~df["CompID"].isin(used_ids)].copy()
    max_per_cluster = max(1, portfolio_size // max(2, len(unique_clusters)))

    cluster_counts = portfolio["cluster"].value_counts().to_dict()

    more_rows = []
    for _, row in remaining.iterrows():
        cl = row["cluster"]
        current_count = cluster_counts.get(cl, 0)

        if current_count >= max_per_cluster:
            continue

        more_rows.append(row)
        cluster_counts[cl] = current_count + 1

        if len(portfolio) + len(more_rows) >= portfolio_size:
            break

    if more_rows:
        portfolio = pd.concat([portfolio, pd.DataFrame(more_rows)], ignore_index=True)

    # Nếu vẫn thiếu thì fill bằng top score còn lại
    if len(portfolio) < portfolio_size:
        used_ids = set(portfolio["CompID"].tolist())
        fallback = df[~df["CompID"].isin(used_ids)].head(portfolio_size - len(portfolio))
        portfolio = pd.concat([portfolio, fallback], ignore_index=True)

    return portfolio.head(portfolio_size).reset_index(drop=True)