"""
Vector Alpha — Portfolio Learning Lab Configuration
====================================================

All constants, colors, asset metadata, and educational text.
"""

from pathlib import Path

# ============================================================================
# FILE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ============================================================================
# ASSET UNIVERSE
# ============================================================================

ASSET_INFO = {
    "AAPL":  {"name": "Apple",          "sector": "Technology"},
    "ADBE":  {"name": "Adobe",          "sector": "Software"},
    "AMD":   {"name": "AMD",            "sector": "Semiconductors"},
    "AMZN":  {"name": "Amazon",         "sector": "E-Commerce"},
    "CRM":   {"name": "Salesforce",     "sector": "Software"},
    "CSCO":  {"name": "Cisco",          "sector": "Networking"},
    "GOOGL": {"name": "Alphabet",       "sector": "Technology"},
    "INTC":  {"name": "Intel",          "sector": "Semiconductors"},
    "META":  {"name": "Meta",           "sector": "Social Media"},
    "MSFT":  {"name": "Microsoft",      "sector": "Technology"},
    "NFLX":  {"name": "Netflix",        "sector": "Streaming"},
    "NVDA":  {"name": "NVIDIA",         "sector": "Semiconductors"},
    "ORCL":  {"name": "Oracle",         "sector": "Software"},
    "QCOM":  {"name": "Qualcomm",       "sector": "Semiconductors"},
    "TSLA":  {"name": "Tesla",          "sector": "EV / Automotive"},
}

ASSETS = list(ASSET_INFO.keys())

# ============================================================================
# DEFAULT SETTINGS
# ============================================================================

DEFAULT_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
DEFAULT_REBALANCE = "Monthly"
DEFAULT_COST_BPS = 10.0
DATE_RANGE_START = "2020-01-01"
DATE_RANGE_END = "2025-12-31"

REBALANCE_OPTIONS = ["None (Buy & Hold)", "Monthly", "Quarterly", "Yearly"]

REBALANCE_MAP = {
    "None (Buy & Hold)": "none",
    "Monthly": "monthly",
    "Quarterly": "quarterly",
    "Yearly": "yearly",
}

# ============================================================================
# COLORS — Clean educational palette
# ============================================================================

COLOR_PRIMARY = "#2563EB"      # Blue
COLOR_POSITIVE = "#16A34A"     # Green
COLOR_NEGATIVE = "#DC2626"     # Red
COLOR_NEUTRAL = "#6B7280"      # Gray
COLOR_ACCENT = "#F59E0B"       # Amber
COLOR_BACKGROUND = "#F9FAFB"   # Light gray
COLOR_CARD_BG = "#FFFFFF"      # White

ASSET_COLORS = {
    "AAPL":  "#2563EB",
    "ADBE":  "#7C3AED",
    "AMD":   "#DC2626",
    "AMZN":  "#F59E0B",
    "CRM":   "#0891B2",
    "CSCO":  "#059669",
    "GOOGL": "#4F46E5",
    "INTC":  "#0284C7",
    "META":  "#1D4ED8",
    "MSFT":  "#16A34A",
    "NFLX":  "#E11D48",
    "NVDA":  "#65A30D",
    "ORCL":  "#B91C1C",
    "QCOM":  "#7E22CE",
    "TSLA":  "#EA580C",
}

# ============================================================================
# CHART SETTINGS
# ============================================================================

PLOT_HEIGHT = 450
PLOT_FONT_SIZE = 13
TITLE_FONT_SIZE = 16
CHART_TEMPLATE = "plotly_white"
CHART_MARGIN = dict(l=40, r=20, t=50, b=40)

# ============================================================================
# EDUCATIONAL TEXT
# ============================================================================

APP_TITLE = "Vector Alpha"
APP_SUBTITLE = "Interactive Portfolio Lab for Finance Students"
APP_TAGLINE = (
    "Experiment with portfolios and see how risk, rebalancing, "
    "and diversification actually behave in real markets."
)

LANDING_DESCRIPTION = (
    "Build a portfolio, simulate performance, and understand where "
    "returns and risks come from — visually and intuitively."
)

SECTION_HEADERS = {
    "build": "Build Your Portfolio",
    "results": "See What Happened",
    "insights": "What This Means",
    "risk_explorer": "Risk Explorer",
    "return_breakdown": "Return Breakdown",
    "advanced": "Advanced Mode",
}

SECTION_DESCRIPTIONS = {
    "build": "Select assets, set weights, and choose your rebalancing strategy.",
    "results": "Visualize how your portfolio performed over time.",
    "insights": "Plain-English explanations of what drove your results.",
    "risk_explorer": "Understand where your portfolio risk actually comes from.",
    "return_breakdown": "See which assets helped and which ones hurt.",
    "advanced": "Detailed analytics for deeper exploration.",
}

# Concept definitions shown as tooltips or expandable text
CONCEPT_DEFINITIONS = {
    "sharpe_ratio": (
        "The Sharpe Ratio measures return per unit of risk. "
        "A ratio above 1.0 is generally considered good. "
        "Higher means better risk-adjusted performance."
    ),
    "max_drawdown": (
        "Maximum drawdown is the largest peak-to-trough decline. "
        "It tells you the worst loss you would have experienced "
        "if you bought at the peak and sold at the bottom."
    ),
    "volatility": (
        "Volatility measures how much returns fluctuate. "
        "Higher volatility means more uncertainty. "
        "It's the standard deviation of returns, annualized."
    ),
    "cagr": (
        "CAGR is the smoothed annual growth rate. "
        "It shows what your portfolio would have returned each year "
        "if it grew at a steady rate."
    ),
    "rebalancing": (
        "Rebalancing means resetting your portfolio back to target weights. "
        "Without it, winning assets grow to dominate your portfolio, "
        "increasing concentration risk."
    ),
    "diversification": (
        "Diversification means spreading risk across different assets. "
        "True diversification requires low correlation between assets, "
        "not just owning many of them."
    ),
    "risk_contribution": (
        "Risk contribution shows how much each asset adds to total portfolio risk. "
        "An asset can have a small weight but contribute a lot of risk "
        "if it's volatile and correlated with other holdings."
    ),
    "weight_drift": (
        "Weight drift happens because asset prices change at different rates. "
        "If one asset goes up a lot, it becomes a larger share of your portfolio. "
        "This can make your portfolio riskier than intended."
    ),
}

# ============================================================================
# DIVERSIFICATION SCORE PARAMETERS
# ============================================================================

# Weights for composite diversification score (0-10)
DIVSCORE_WEIGHT_HHI = 0.30       # Herfindahl index (weight concentration)
DIVSCORE_WEIGHT_CORR = 0.35      # Average pairwise correlation
DIVSCORE_WEIGHT_RISKCONC = 0.35  # Risk concentration (top-2 share)
