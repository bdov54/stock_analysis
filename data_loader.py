import re
from typing import Any

import pandas as pd

from config import SELECTED_FIELDS
from utils import logger


def clean_excel_text(x: Any):
    if pd.isna(x):
        return x
    s = str(x)
    for ch in ["\u00a0", "\u200b", "\ufeff", "\u200e", "\u200f"]:
        s = s.replace(ch, " ")
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"^\s*['\u2019]+", "", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s


def make_unique_columns(cols):
    seen = {}
    out = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__dup{seen[c]}")
    return out


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = make_unique_columns([clean_excel_text(c) for c in df.columns])
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].map(clean_excel_text)
    return df


def parse_year(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if dt.notna().mean() > 0.2:
        return dt.dt.year.astype("Int64")

    y = pd.to_numeric(series, errors="coerce")
    if y.notna().mean() > 0.2:
        return y.astype("Int64")

    s2 = series.astype(str).map(clean_excel_text)
    y2 = s2.str.extract(r"(\d{4})", expand=False)
    y2 = pd.to_numeric(y2, errors="coerce")
    return y2.astype("Int64")


def norm_sheet_name(name: str) -> str:
    return clean_excel_text(name).upper().replace(" ", "")


def prepare_sheet(raw_df: pd.DataFrame, sheet_code: str, year_min: int, year_max: int) -> pd.DataFrame:
    df = clean_df(raw_df)

    cols_norm = {clean_excel_text(c).lower(): c for c in df.columns}
    comp_col = cols_norm.get("compid")
    year_col = cols_norm.get("year")
    if comp_col is None or year_col is None:
        raise ValueError(f"[{sheet_code}] missing CompID/Year")

    df = df.rename(columns={comp_col: "CompID", year_col: "Year"})
    df["CompID"] = df["CompID"].astype(str).map(clean_excel_text)
    df["Year"] = parse_year(df["Year"])
    df = df[df["Year"].between(year_min, year_max, inclusive="both")]

    keep = ["CompID", "Year"]
    for c in SELECTED_FIELDS[sheet_code]:
        if c in df.columns:
            keep.append(c)
    df = df[keep].copy()

    for c in df.columns:
        if c not in ("CompID", "Year"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["CompID", "Year"]).drop_duplicates(["CompID", "Year"], keep="last")
    rename = {c: f"{sheet_code}__{c}" for c in df.columns if c not in ("CompID", "Year")}
    return df.rename(columns=rename)


def outer_merge(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    out = None
    for d in dfs:
        if d is None or d.empty:
            continue
        out = d if out is None else out.merge(d, on=["CompID", "Year"], how="outer")
    return out if out is not None else pd.DataFrame()


def load_company_info(all_sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    comp_sheet = None
    for name in all_sheets.keys():
        if norm_sheet_name(name) == "COMP":
            comp_sheet = all_sheets[name]
            break

    if comp_sheet is None:
        logger.warning("COMP sheet not found. Industry insights will be limited.")
        return pd.DataFrame(columns=["CompID"])

    comp = clean_df(comp_sheet)
    rename_map = {}
    if "Code" in comp.columns:
        rename_map["Code"] = "CompID"
    comp = comp.rename(columns=rename_map)

    cols = [
        "CompID",
        "Company Common Name",
        "GICS Sub-Industry Name",
        "TRBC Industry Name",
        "Organization Founded Year",
        "Date Became Public",
    ]
    cols = [c for c in cols if c in comp.columns]
    comp = comp[cols].copy()
    comp["CompID"] = comp["CompID"].astype(str).map(clean_excel_text)
    comp = comp.drop_duplicates(subset=["CompID"])
    return comp


def load_master_dataset(file_path: str, year_min: int, year_max: int):
    logger.info("Loading Excel file: %s", file_path)
    all_sheets = pd.read_excel(file_path, sheet_name=None, engine="openpyxl")
    company_info = load_company_info(all_sheets)

    norm_map = {norm_sheet_name(k): k for k in all_sheets.keys()}
    prepared = []
    used = []
    skipped = []

    for sheet_code in SELECTED_FIELDS.keys():
        key = norm_sheet_name(sheet_code)
        if key not in norm_map:
            skipped.append((sheet_code, "sheet not found"))
            continue

        real_name = norm_map[key]
        try:
            df = prepare_sheet(all_sheets[real_name], sheet_code, year_min, year_max)
            prepared.append(df)
            used.append(sheet_code)
            logger.info("Loaded %s <- %s | rows=%s", sheet_code, real_name, len(df))
        except Exception as e:
            skipped.append((sheet_code, str(e)))
            logger.warning("Skipped %s: %s", sheet_code, e)

    master = outer_merge(prepared)
    if master.empty:
        raise ValueError("Master dataset is empty after loading and merging sheets.")

    master = master.sort_values(["CompID", "Year"]).reset_index(drop=True)
    meta = {
        "used_sheets": used,
        "skipped_sheets": skipped,
        "companies": int(master["CompID"].nunique()),
        "rows": int(len(master)),
        "year_min": int(master["Year"].min()),
        "year_max": int(master["Year"].max()),
    }
    return master, company_info, meta
