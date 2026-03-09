import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from utils import logger


def _get_scaler(scaler_type: str = "standard"):
    scaler_type = (scaler_type or "standard").lower()
    if scaler_type == "robust":
        return RobustScaler()
    return StandardScaler()


def _clip_outliers(df: pd.DataFrame, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        s = pd.to_numeric(out[col], errors="coerce")
        valid = s.dropna()
        if len(valid) == 0:
            continue
        low = valid.quantile(lower_q)
        high = valid.quantile(upper_q)
        out[col] = s.clip(lower=low, upper=high)
    return out


def _cap_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Cap theo trực giác tài chính để giảm singleton/outlier clusters
    if "MED_Current_Ratio" in out.columns:
        out["MED_Current_Ratio"] = pd.to_numeric(out["MED_Current_Ratio"], errors="coerce").clip(lower=0, upper=10)

    if "MED_EBIT_margin" in out.columns:
        out["MED_EBIT_margin"] = pd.to_numeric(out["MED_EBIT_margin"], errors="coerce").clip(lower=-0.5, upper=0.6)

    if "MED_PPE_Assets" in out.columns:
        out["MED_PPE_Assets"] = pd.to_numeric(out["MED_PPE_Assets"], errors="coerce").clip(lower=0, upper=0.9)

    if "MED_D_E" in out.columns:
        out["MED_D_E"] = pd.to_numeric(out["MED_D_E"], errors="coerce").clip(lower=0, upper=10)

    if "MED_NetDebt_EBITDA" in out.columns:
        out["MED_NetDebt_EBITDA"] = pd.to_numeric(out["MED_NetDebt_EBITDA"], errors="coerce").clip(lower=0, upper=10)

    if "MED_CFO_NI" in out.columns:
        out["MED_CFO_NI"] = pd.to_numeric(out["MED_CFO_NI"], errors="coerce").clip(lower=-5, upper=5)

    return out


def _transform_skewed_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # log1p cho các biến lệch mạnh, không áp cho biến có thể âm
    for col in ["MED_Current_Ratio", "MED_NetDebt_EBITDA", "MED_Firm_Age"]:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            out[col] = np.log1p(np.maximum(s, 0))

    return out


def prepare_cluster_matrix(
    df: pd.DataFrame,
    feature_cols,
    clip_quantiles=(0.01, 0.99),
    apply_caps: bool = True,
    apply_skew_transform: bool = True,
):
    # 1) dedup tên feature được yêu cầu
    use_cols = [c for c in feature_cols if c in df.columns]
    use_cols = list(dict.fromkeys(use_cols))

    if len(use_cols) == 0:
        raise ValueError("Không có cluster feature nào tồn tại trong DataFrame.")

    # 2) lấy matrix và xử lý duplicate column names trong df
    x = df.loc[:, use_cols].copy()

    # Nếu DataFrame có duplicate columns, giữ lần xuất hiện đầu tiên
    if x.columns.duplicated().any():
        dup_cols = x.columns[x.columns.duplicated()].tolist()
        logger.warning("Duplicate cluster columns detected and removed: %s", dup_cols)
        x = x.loc[:, ~x.columns.duplicated()]

    # update lại use_cols theo x thực tế
    use_cols = list(x.columns)

    x = x.replace([np.inf, -np.inf], np.nan)

    # 3) Clip outlier theo quantile
    if clip_quantiles is not None:
        x = _clip_outliers(x, lower_q=clip_quantiles[0], upper_q=clip_quantiles[1])

    # 4) Cap cứng một số biến tài chính méo mạnh
    if apply_caps:
        x = _cap_specific_features(x)

    # 5) Impute median
    imputer = SimpleImputer(strategy="median")
    x_imputed = imputer.fit_transform(x)

    # 6) Dựng DataFrame từ shape thực tế
    x_prepared = pd.DataFrame(
        x_imputed,
        columns=use_cols,
        index=x.index,
    )

    # 7) Transform skewed features
    if apply_skew_transform:
        x_prepared = _transform_skewed_features(x_prepared)

    return x_prepared, use_cols, imputer


def search_best_kmeans(
    df: pd.DataFrame,
    feature_cols,
    k_values=range(2, 6),
    scaler_type: str = "standard",
    clip_quantiles=(0.01, 0.99),
    random_state: int = 42,
    min_cluster_size: int = 5,
    apply_caps: bool = True,
    apply_skew_transform: bool = True,
):
    x_prepared, use_cols, imputer = prepare_cluster_matrix(
        df=df,
        feature_cols=feature_cols,
        clip_quantiles=clip_quantiles,
        apply_caps=apply_caps,
        apply_skew_transform=apply_skew_transform,
    )

    scaler = _get_scaler(scaler_type)
    x_scaled = scaler.fit_transform(x_prepared)

    rows = []
    best_model = None
    best_labels = None
    best_k = None
    best_sil = -1

    for k in k_values:
        if len(x_prepared) <= k:
            continue

        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(x_scaled)

        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        min_size = int(cluster_sizes.min())
        max_size = int(cluster_sizes.max())

        if len(np.unique(labels)) < 2:
            sil = np.nan
        else:
            sil = silhouette_score(x_scaled, labels)

        valid = min_size >= min_cluster_size

        rows.append({
            "k": k,
            "silhouette": sil,
            "min_cluster_size": min_size,
            "max_cluster_size": max_size,
            "valid": valid,
            "n_features": len(use_cols),
            "scaler": scaler_type,
            "clip_quantiles": clip_quantiles,
            "apply_caps": apply_caps,
            "apply_skew_transform": apply_skew_transform,
        })

        if valid and pd.notna(sil) and sil > best_sil:
            best_sil = sil
            best_model = model
            best_labels = labels
            best_k = k

    summary_df = pd.DataFrame(rows).sort_values(
        ["valid", "silhouette"], ascending=[False, False]
    ).reset_index(drop=True)

    logger.info(
        "Cluster search done | scaler=%s | best_k=%s | best_silhouette=%.4f",
        scaler_type,
        best_k,
        best_sil if pd.notna(best_sil) else float("nan"),
    )

    return {
        "summary_df": summary_df,
        "best_k": best_k,
        "best_silhouette": best_sil,
        "best_model": best_model,
        "best_labels": best_labels,
        "prepared_df": x_prepared,
        "feature_cols_used": use_cols,
        "imputer": imputer,
        "scaler": scaler,
    }


def run_kmeans(
    df: pd.DataFrame,
    feature_cols,
    n_clusters: int = 3,
    scaler_type: str = "standard",
    clip_quantiles=(0.01, 0.99),
    random_state: int = 42,
    apply_caps: bool = True,
    apply_skew_transform: bool = True,
):
    x_prepared, use_cols, imputer = prepare_cluster_matrix(
        df=df,
        feature_cols=feature_cols,
        clip_quantiles=clip_quantiles,
        apply_caps=apply_caps,
        apply_skew_transform=apply_skew_transform,
    )

    scaler = _get_scaler(scaler_type)
    x_scaled = scaler.fit_transform(x_prepared)

    model = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
    labels = model.fit_predict(x_scaled)

    sil = silhouette_score(x_scaled, labels) if len(np.unique(labels)) >= 2 else np.nan
    cluster_sizes = pd.Series(labels).value_counts().sort_index()

    out = df.copy()
    out["cluster"] = labels

    logger.info(
        "KMeans fitted with %s clusters. Silhouette=%.4f | scaler=%s | features=%s | min_cluster=%s | max_cluster=%s",
        n_clusters,
        sil if pd.notna(sil) else float("nan"),
        scaler_type,
        len(use_cols),
        int(cluster_sizes.min()) if len(cluster_sizes) else None,
        int(cluster_sizes.max()) if len(cluster_sizes) else None,
    )

    artifacts = {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "feature_cols_used": use_cols,
        "silhouette": sil,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes.to_dict(),
        "scaler_type": scaler_type,
        "clip_quantiles": clip_quantiles,
        "apply_caps": apply_caps,
        "apply_skew_transform": apply_skew_transform,
    }

    return out, artifacts


def save_cluster_artifacts(artifacts: dict, artifacts_dir: str):
    os.makedirs(artifacts_dir, exist_ok=True)

    model_path = os.path.join(artifacts_dir, "cluster_model.pkl")
    scaler_path = os.path.join(artifacts_dir, "cluster_scaler.pkl")
    imputer_path = os.path.join(artifacts_dir, "cluster_imputer.pkl")
    meta_path = os.path.join(artifacts_dir, "cluster_meta.json")

    joblib.dump(artifacts["model"], model_path)
    joblib.dump(artifacts["scaler"], scaler_path)
    joblib.dump(artifacts["imputer"], imputer_path)

    meta = {
        "feature_cols_used": artifacts["feature_cols_used"],
        "silhouette": None if pd.isna(artifacts["silhouette"]) else float(artifacts["silhouette"]),
        "n_clusters": int(artifacts["n_clusters"]),
        "cluster_sizes": artifacts.get("cluster_sizes", {}),
        "scaler_type": artifacts["scaler_type"],
        "clip_quantiles": list(artifacts["clip_quantiles"]) if artifacts["clip_quantiles"] is not None else None,
        "apply_caps": artifacts.get("apply_caps", True),
        "apply_skew_transform": artifacts.get("apply_skew_transform", True),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info("Saved cluster artifacts to %s", artifacts_dir)