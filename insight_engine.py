import pandas as pd

from industry_insights import INDUSTRY_INSIGHTS
from macro_insights import MACRO_INSIGHTS


def normalize_industry(raw_value: str | None) -> str:
    if raw_value is None or pd.isna(raw_value):
        return "Other"

    s = str(raw_value).lower()
    if any(k in s for k in ["bank", "banks"]):
        return "Banks"
    if any(k in s for k in ["software", "technology", "it services", "computer", "digital"]):
        return "Technology"
    if any(k in s for k in ["retail", "specialty stores", "consumer discretionary"]):
        return "Consumer Retail"
    if any(k in s for k in ["food", "dairy", "beverages", "consumer staples"]):
        return "Consumer Staples"
    if any(k in s for k in ["shipping", "marine", "maritime"]):
        return "Shipping"
    if any(k in s for k in ["port", "logistics", "transportation infrastructure"]):
        return "Ports & Logistics"
    if any(k in s for k in ["renewable", "utilities", "electric", "energy"]):
        return "Renewable Energy"
    if any(k in s for k in ["financial", "exchange", "securities", "asset management", "insurance"]):
        return "Financial Services"
    if any(k in s for k in ["health", "medical", "pharma", "biotech"]):
        return "Healthcare"
    if any(k in s for k in ["industrial", "machinery", "manufacturing", "construction"]):
        return "Industrials"
    return "Other"


def enrich_with_insights(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()

    industry_source = None
    for c in ["GICS Sub-Industry Name", "TRBC Industry Name"]:
        if c in work.columns:
            industry_source = c
            break

    if industry_source is None:
        work["industry_bucket"] = "Other"
    else:
        work["industry_bucket"] = work[industry_source].map(normalize_industry)

    work["industry_outlook"] = work["industry_bucket"].map(lambda x: INDUSTRY_INSIGHTS.get(x, INDUSTRY_INSIGHTS["Other"])["outlook"])
    work["industry_drivers"] = work["industry_bucket"].map(lambda x: " | ".join(INDUSTRY_INSIGHTS.get(x, INDUSTRY_INSIGHTS["Other"])["drivers"]))
    work["industry_risks"] = work["industry_bucket"].map(lambda x: " | ".join(INDUSTRY_INSIGHTS.get(x, INDUSTRY_INSIGHTS["Other"])["risks"]))
    work["macro_context"] = MACRO_INSIGHTS["summary"]
    return work


def get_macro_insights() -> dict:
    return MACRO_INSIGHTS
