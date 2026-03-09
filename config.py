import os
from dataclasses import dataclass, field
from typing import Dict, List


DEFAULT_SELECTED_FIELDS = {
    "COMP": [
        "Code",
        "Company Common Name",
        "GICS Sub-Industry Name",
        "Organization Founded Year",
        "Date Became Public",
        "TRBC Industry Name",
        "Country of Exchange",
    ],

    "BS1": [
        "Cash & Cash Equivalents - Total",
        "Trade Accounts & Trade Notes Receivable - Net",
        "Inventories - Total",
    ],

    "BS2": [
        "Total Current Assets",
        "Property Plant & Equipment - Net - Total",
    ],

    "BS3": [
        "Total Assets",
    ],

    "BS4": [
        "Short-Term Debt & Current Portion of Long-Term Debt",
        "Total Current Liabilities",
    ],

    "BS5": [
        "Debt - Long-Term - Total",
    ],

    "BS6": [
        "Total Liabilities",
        "Common Equity Attributable to Parent Shareholders",
        "Common Equity - Total",
        "Total Shareholders' Equity incl Minority Intr & Hybrid Debt",
        "Retained Earnings - Total",
        "Total Liabilities & Equity",
    ],

    "BS7": [
        "Common Shares - Outstanding - Total",
        "Common Shares - Issued - Total",
    ],

    "BS8": [
        "Net Debt",
        "Debt - Total",
        "Cash & Cash Equivalents - Total",
        "Net Operating Assets",
        "Shareholders Equity - Common",
        "Tangible Total Equity",
    ],

    "BS9": [
        "Working Capital",
    ],

    "IS1": [
        "Revenue from Business Activities - Total",
        "Revenue from Goods & Services",
        "Employees - Full-Time/Full-Time Equivalents - Period End",
    ],

    "IS2": [
        "Gross Profit - Industrials/Property - Total",
        "Income before Taxes",
        "Income Taxes",
        "Net Income after Tax",
        "Net Income after Minority Interest",
    ],

    "IS3": [
        "EPS - Basic - excl Extraordinary Items, Common - Total",
        "EPS - Basic - excl Extraordinary Items - Normalized - Total",
        "DPS - Common - Net - Issue - By Announcement Date",
    ],

    "IS4": [
        "Revenue from Business Activities - Total",
        "Earnings before Interest & Taxes (EBIT)",
        "Earnings before Interest Taxes Depreciation & Amortization",
        "Operating Expenses - Total",
        "Depreciation Depletion & Amortization - Total",
    ],

    "CF1": [
        "Net Cash Flow from Operating Activities",
        "Capital Expenditures - Total",
        "Dividends - Common - Cash Paid",
    ],

    "CF2": [
        "Net Cash Flow from Investing Activities",
        "Capital Expenditures - Total",
    ],

    "CF3": [
        "Free Cash Flow",
        "Free Cash Flow to Equity",
        "Free Cash Flow Net of Dividends",
        "Dividends Provided/Paid - Common",
    ],
}


DEFAULT_METRIC_RULES = {
    # hiệu quả
    "MED_ROE": "high",
    "MED_EBIT_margin": "high",
    "MED_Gross_margin": "high",
    "MED_Asset_Turnover": "high",

    # tăng trưởng
    "MED_REV_CAGR_3Y": "high",
    "MED_EPS_CAGR_3Y": "high",
    "SLOPE_EBIT_margin_3Y": "high",

    # chất lượng lợi nhuận
    "MED_CFO_NI": "high",
    "MED_CFO_margin": "high",
    "MED_FCF_margin": "high",
    "MED_Sloan": "abs_low",

    # an toàn tài chính
    "MED_D_E": "low",
    "MED_NetDebt_EBITDA": "low",
    "MED_Current_Ratio": "high",
    "MED_Cash_Assets": "high",
    "MEAN_Share_Dilution_5Y": "low",
}


DEFAULT_PILLAR_METRICS = {
    "growth": [
        "MED_REV_CAGR_3Y",
        "MED_EPS_CAGR_3Y",
        "SLOPE_EBIT_margin_3Y",
    ],
    "efficiency": [
        "MED_ROE",
        "MED_EBIT_margin",
        "MED_Gross_margin",
        "MED_Asset_Turnover",
    ],
    "quality": [
        "MED_CFO_NI",
        "MED_CFO_margin",
        "MED_FCF_margin",
        "MED_Sloan",
    ],
    "safety": [
        "MED_D_E",
        "MED_NetDebt_EBITDA",
        "MED_Current_Ratio",
        "MED_Cash_Assets",
    ],
}


DEFAULT_CLUSTER_FEATURES = [
    "MED_ROE",
    "MED_EBIT_margin",
    "MED_ReEV_CAGR_3Y",
    "MED_EPS_CAGR_3Y",
    "MED_CFO_NI",
    "MED_D_E",
    "MED_NetDebt_EBITDA",
    "MED_Current_Ratio",
    "MED_PPE_Assets",
    "MED_Firm_Age",
]

@dataclass
class AppConfig:
    file_path: str = "data/Greece.xlsx"
    artifacts_dir: str = "artifacts"
    charts_dir: str = "artifacts/charts"

    year_min: int = 2018
    year_max: int = 2024
    window_years: int = 5
    slope_years: int = 3
    cluster_scaler_typ: str = "robust"
    cluster_clip_quantiles: tuple = (0.01, 0.99)
    cluster_min_size: int = 5
    cluster_k_fixed: int = 2
    show_cluster_in_ui: bool = False

    mode: str = "data-driven"   # "data-driven" hoặc "manual"
    target_keep: float = 0.25

    use_clustering: bool = True
    n_clusters: int = 4

    portfolio_size: int = 7
    max_per_cluster: int = 2

    hard_filter_rules: Dict = field(default_factory=lambda: {
        "min_coverage": 0.60,
        "min_cfo_ni": 0.50,
        "max_de_ratio": 2.50,
        "max_netdebt_ebitda": 4.00,
        "min_roe": 0.00,
        "require_positive_equity": True,
    })

    scoring_weights: Dict = field(default_factory=lambda: {
        "growth": 0.35,
        "efficiency": 0.30,
        "quality": 0.20,
        "safety": 0.15,
    })

    manual_thresholds: Dict = field(default_factory=dict)

    selected_fields: Dict[str, List[str]] = field(default_factory=lambda: DEFAULT_SELECTED_FIELDS)
    metric_rules: Dict[str, str] = field(default_factory=lambda: DEFAULT_METRIC_RULES)
    pillar_metrics: Dict[str, List[str]] = field(default_factory=lambda: DEFAULT_PILLAR_METRICS)
    cluster_features: List[str] = field(default_factory=lambda: DEFAULT_CLUSTER_FEATURES)
    def ensure_dirs(self):
        os.makedirs(self.artifacts_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)

SELECTED_FIELDS = DEFAULT_SELECTED_FIELDS
METRIC_RULES = DEFAULT_METRIC_RULES
PILLAR_METRICS = DEFAULT_PILLAR_METRICS
CLUSTER_FEATURES = DEFAULT_CLUSTER_FEATURES

