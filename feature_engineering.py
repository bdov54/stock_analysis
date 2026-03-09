import numpy as np
import pandas as pd

from utils import logger


def safe_div(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    out = a / b
    return out.replace([np.inf, -np.inf], np.nan)


def avg_prev(df: pd.DataFrame, col: str) -> pd.Series:
    return (pd.to_numeric(df[col], errors="coerce") + pd.to_numeric(df.groupby("CompID")[col].shift(1), errors="coerce")) / 2


def cagr(df: pd.DataFrame, col: str, n: int = 3) -> pd.Series:
    x = pd.to_numeric(df[col], errors="coerce")
    x0 = pd.to_numeric(df.groupby("CompID")[col].shift(n), errors="coerce")
    out = (x / x0) ** (1 / n) - 1
    return out.where((x > 0) & (x0 > 0))


def slope_by_group(sub: pd.DataFrame, ycol: str) -> float:
    sub = sub.dropna(subset=[ycol, "Year"])
    if len(sub) < 3:
        return np.nan

    yy = pd.to_numeric(sub[ycol], errors="coerce").values
    tt = pd.to_numeric(sub["Year"], errors="coerce").astype(int).values
    tt = tt - tt.min()

    if np.all(np.isnan(yy)):
        return np.nan

    return float(np.polyfit(tt, yy, 1)[0])


def first_existing(df: pd.DataFrame, candidates) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def has_col(df: pd.DataFrame, col: str | None) -> bool:
    return col is not None and col in df.columns


def get_series(df: pd.DataFrame, col: str | None, default=np.nan) -> pd.Series:
    if has_col(df, col):
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def build_yearly_features(master_df: pd.DataFrame, company_info: pd.DataFrame | None = None) -> pd.DataFrame:
    df = master_df.copy()

    if company_info is not None and not company_info.empty:
        df = df.merge(company_info, on="CompID", how="left")

    df = df.sort_values(["CompID", "Year"]).copy()

    # ==========
    # Column mapping based on your actual Excel
    # ==========
    rev = first_existing(df, [
        "IS1__Revenue from Business Activities - Total",
        "IS4__Revenue from Business Activities - Total",
    ])
    revenue_alt = first_existing(df, [
        "IS1__Revenue from Goods & Services",
        "IS1__Sales of Goods & Services - Net - Unclassified",
    ])
    if rev is None:
        rev = revenue_alt

    ebit = first_existing(df, [
        "IS4__Earnings before Interest & Taxes (EBIT)",
    ])
    ebitda = first_existing(df, [
        "IS4__Earnings before Interest Taxes Depreciation & Amortization",
    ])
    gross_profit = first_existing(df, [
        "IS2__Gross Profit - Industrials/Property - Total",
    ])
    ni = first_existing(df, [
        "IS2__Net Income after Tax",
        "IS2__Net Income after Minority Interest",
    ])
    pretax = first_existing(df, [
        "IS2__Income before Taxes",
    ])
    tax = first_existing(df, [
        "IS2__Income Taxes",
    ])
    cfo = first_existing(df, [
        "CF1__Net Cash Flow from Operating Activities",
        "CF1__Net Cash Flow from Operating Activities.1",
    ])
    fcf = first_existing(df, [
        "CF3__Free Cash Flow",
        "CF3__Free Cash Flow to Equity",
    ])
    assets = first_existing(df, [
        "BS3__Total Assets",
        "BS6__Total Liabilities & Equity",
    ])
    debt_total = first_existing(df, [
        "BS8__Debt - Total",
    ])
    net_debt = first_existing(df, [
        "BS8__Net Debt",
    ])
    cash = first_existing(df, [
        "BS8__Cash & Cash Equivalents - Total",
        "BS1__Cash & Cash Equivalents - Total",
    ])
    shares = first_existing(df, [
        "BS7__Common Shares - Outstanding - Total",
    ])
    eps = first_existing(df, [
        "IS3__EPS - Basic - excl Extraordinary Items, Common - Total",
        "IS3__EPS - Basic - excl Extraordinary Items - Normalized - Total",
    ])
    current_assets = first_existing(df, [
        "BS2__Total Current Assets",
    ])
    current_liabilities = first_existing(df, [
        "BS4__Total Current Liabilities",
    ])
    inventory = first_existing(df, [
        "BS1__Inventories - Total",
    ])
    ppe = first_existing(df, [
        "BS2__Property Plant & Equipment - Net - Total",
    ])
    employees = first_existing(df, [
        "IS1__Employees - Full-Time/Full-Time Equivalents - Period End",
        "IS1__Employees - Full-Time/Full-Time Equivalents - Current Date",
    ])

    equity = first_existing(df, [
        "BS6__Common Equity Attributable to Parent Shareholders",
        "BS6__Total Shareholders' Equity incl Minority Intr & Hybrid Debt",
        "BS6__Common Equity - Total",
        "BS8__Shareholders Equity - Common",
    ])
    if equity is None:
        raise ValueError("Không tìm thấy cột equity để tính ROE/ROIC.")

    # ==========
    # Base numeric series
    # ==========
    s_rev = get_series(df, rev)
    s_ebit = get_series(df, ebit)
    s_ebitda = get_series(df, ebitda)
    s_gross_profit = get_series(df, gross_profit)
    s_ni = get_series(df, ni)
    s_pretax = get_series(df, pretax)
    s_tax = get_series(df, tax)
    s_cfo = get_series(df, cfo)
    s_fcf = get_series(df, fcf)
    s_assets = get_series(df, assets)
    s_debt_total = get_series(df, debt_total)
    s_net_debt = get_series(df, net_debt)
    s_cash = get_series(df, cash)
    s_shares = get_series(df, shares)
    s_eps = get_series(df, eps)
    s_current_assets = get_series(df, current_assets)
    s_current_liabilities = get_series(df, current_liabilities)
    s_inventory = get_series(df, inventory)
    s_ppe = get_series(df, ppe)
    s_employees = get_series(df, employees)
    s_equity = get_series(df, equity)

    # ==========
    # Averages
    # ==========
    df["avg_equity"] = avg_prev(df.assign(_equity=s_equity), "_equity")
    df["avg_assets"] = avg_prev(df.assign(_assets=s_assets), "_assets")

    # ==========
    # Profitability / Efficiency
    # ==========
    df["ROE"] = np.where(df["avg_equity"] > 0, safe_div(s_ni, df["avg_equity"]), np.nan)
    df["ROA"] = np.where(df["avg_assets"] > 0, safe_div(s_ni, df["avg_assets"]), np.nan)

    if rev is not None:
        df["EBIT_margin"] = np.where(s_rev > 0, safe_div(s_ebit, s_rev), np.nan)
        df["Gross_margin"] = np.where(s_rev > 0, safe_div(s_gross_profit, s_rev), np.nan)
        df["Asset_Turnover"] = np.where(df["avg_assets"] > 0, safe_div(s_rev, df["avg_assets"]), np.nan)
        df["CFO_margin"] = np.where(s_rev > 0, safe_div(s_cfo, s_rev), np.nan)
        df["FCF_margin"] = np.where(s_rev > 0, safe_div(s_fcf, s_rev), np.nan)
        df["Employees_Productivity"] = np.where(s_employees > 0, safe_div(s_rev, s_employees), np.nan)
    else:
        df["EBIT_margin"] = np.nan
        df["Gross_margin"] = np.nan
        df["Asset_Turnover"] = np.nan
        df["CFO_margin"] = np.nan
        df["FCF_margin"] = np.nan
        df["Employees_Productivity"] = np.nan

    # ==========
    # Earnings quality
    # ==========
    df["CFO_NI"] = np.where(s_ni > 0, safe_div(s_cfo, s_ni), np.nan)
    df["Sloan"] = np.where(df["avg_assets"] > 0, safe_div(s_ni - s_cfo, df["avg_assets"]), np.nan)

    prev_shares = pd.to_numeric(df.groupby("CompID")["_tmp"].shift(1), errors="coerce") if False else df.groupby("CompID").apply(
        lambda g: pd.to_numeric(g[shares], errors="coerce").shift(1) if has_col(g, shares) else pd.Series(np.nan, index=g.index)
    ).reset_index(level=0, drop=True)

    df["Share_Dilution"] = np.where(prev_shares > 0, safe_div(s_shares - prev_shares, prev_shares), np.nan)

    # ==========
    # ROIC
    # ==========
    etr = safe_div(s_tax, s_pretax).clip(lower=0, upper=0.40)
    df["NOPAT"] = s_ebit * (1 - etr.fillna(0.25))
    df["InvestedCapital"] = s_debt_total + s_equity - s_cash
    df["avg_InvestedCapital"] = avg_prev(df.assign(_inv_cap=df["InvestedCapital"]), "_inv_cap")
    df["ROIC"] = np.where(df["avg_InvestedCapital"] > 0, safe_div(df["NOPAT"], df["avg_InvestedCapital"]), np.nan)

    # ==========
    # Growth
    # ==========
    if rev is not None:
        df["REV_CAGR_3Y"] = cagr(df.assign(_rev=s_rev), "_rev", n=3)
    else:
        df["REV_CAGR_3Y"] = np.nan

    if eps is not None:
        df["EPS_CAGR_3Y"] = cagr(df.assign(_eps=s_eps), "_eps", n=3)
    else:
        df["EPS_CAGR_3Y"] = np.nan

    # ==========
    # Safety / Structure
    # ==========
    df["D_E"] = np.where(s_equity > 0, safe_div(s_debt_total, s_equity), np.nan)
    df["NetDebt_EBITDA"] = np.where(s_ebitda > 0, safe_div(np.maximum(s_net_debt, 0), s_ebitda), np.nan)
    df["Current_Ratio"] = np.where(s_current_liabilities > 0, safe_div(s_current_assets, s_current_liabilities), np.nan)
    df["Cash_Assets"] = np.where(s_assets > 0, safe_div(s_cash, s_assets), np.nan)
    df["Inventory_Assets"] = np.where(s_assets > 0, safe_div(s_inventory, s_assets), np.nan)
    df["PPE_Assets"] = np.where(s_assets > 0, safe_div(s_ppe, s_assets), np.nan)

    # ==========
    # Firm profile
    # ==========
    if "Organization Founded Year" in df.columns:
        founded = pd.to_numeric(df["Organization Founded Year"], errors="coerce")
        df["Firm_Age"] = df["Year"] - founded
    else:
        df["Firm_Age"] = np.nan

    if "Date Became Public" in df.columns:
        listed = pd.to_datetime(df["Date Became Public"], errors="coerce").dt.year
        df["Years_Listed"] = df["Year"] - listed
    else:
        df["Years_Listed"] = np.nan

    logger.info("Built yearly features for %s company-year rows.", len(df))
    return df


def build_company_features(yearly_df: pd.DataFrame, year_max: int, window_years: int = 5, slope_years: int = 3) -> pd.DataFrame:
    start_year = year_max - window_years + 1
    panel = yearly_df[yearly_df["Year"].between(start_year, year_max)].copy()

    metric_cols = [
        "ROE",
        "ROA",
        "ROIC",
        "EBIT_margin",
        "Gross_margin",
        "Asset_Turnover",
        "CFO_NI",
        "CFO_margin",
        "FCF_margin",
        "Sloan",
        "REV_CAGR_3Y",
        "EPS_CAGR_3Y",
        "D_E",
        "NetDebt_EBITDA",
        "Current_Ratio",
        "Cash_Assets",
        "Inventory_Assets",
        "PPE_Assets",
        "Share_Dilution",
        "Firm_Age",
        "Years_Listed",
    ]
    available = [c for c in metric_cols if c in panel.columns]

    feature_df = panel.groupby("CompID")[available].median().add_prefix("MED_").reset_index()

    if "Share_Dilution" in panel.columns:
        dilution = (
            panel.groupby("CompID", as_index=False)["Share_Dilution"]
            .mean()
            .rename(columns={"Share_Dilution": "MEAN_Share_Dilution_5Y"})
        )
        feature_df = feature_df.merge(dilution, on="CompID", how="left")
    else:
        feature_df["MEAN_Share_Dilution_5Y"] = np.nan

    coverage = (
        panel.groupby("CompID")[available]
        .agg(lambda s: float(pd.to_numeric(s, errors="coerce").notna().mean()))
        .add_prefix("COV_")
        .reset_index()
    )
    feature_df = feature_df.merge(coverage, on="CompID", how="left")

    last_slope_window = yearly_df[yearly_df["Year"].between(year_max - slope_years + 1, year_max)].copy()

    trend_rows = []
    for compid, sub in last_slope_window.groupby("CompID"):
        trend_rows.append({
            "CompID": compid,
            "SLOPE_EBIT_margin_3Y": slope_by_group(sub, "EBIT_margin"),
            "SLOPE_ROE_3Y": slope_by_group(sub, "ROE"),
            "SLOPE_ROIC_3Y": slope_by_group(sub, "ROIC"),
        })

    trend_df = pd.DataFrame(trend_rows)
    feature_df = feature_df.merge(trend_df, on="CompID", how="left")

    logger.info("Built company-level features for %s companies.", feature_df["CompID"].nunique())
    return feature_df