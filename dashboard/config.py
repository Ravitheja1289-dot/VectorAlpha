"""
Vector Alpha Dashboard - Configuration Constants
================================================

All constants, file paths, and configuration in one place for easy maintenance.
"""

from pathlib import Path

# ============================================================================
# FILE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent  # quant-backtesting-engine/
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Parquet files to load
PARQUET_FILES = {
    "prices": DATA_DIR / "prices.parquet",
    "returns": DATA_DIR / "returns.parquet",
    "return_attribution": DATA_DIR / "return_attribution.parquet",
    "risk_attribution": DATA_DIR / "risk_attribution.parquet",
    "risk_metrics": DATA_DIR / "risk_metrics.json",
}

# ============================================================================
# ASSET & PORTFOLIO CONFIGURATION
# ============================================================================

# Asset universe (in order of appearance in Parquet files)
ASSETS = [
    "AAPL", "ADBE", "AMD", "AMZN", "CRM",
    "CSCO", "GOOGL", "INTC", "META", "MSFT",
    "NFLX", "NVDA", "ORCL", "QCOM", "TSLA"
]

PORTFOLIO_TOTAL = "TOTAL"

# Portfolio metadata
PORTFOLIO_DESCRIPTION = "10-asset momentum + mean reversion strategy"
REBALANCING_FREQ = "Weekly"
TRANSACTION_COSTS_BPS = 10  # Basis points
DATE_RANGE = "2020-2025"
RISK_FREE_RATE = 0.02  # 2% annualized risk-free rate

# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================

PAGE_TITLE = "Vector Alpha Research Dashboard"
PAGE_ICON = "chart-line"
LAYOUT = "wide"
SIDEBAR_STATE = "expanded"
INITIAL_SIDEBAR_STATE = "expanded"

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Rolling window options (days)
ROLLING_WINDOWS = [63, 126, 252]  # 3-month, 6-month, 1-year
DEFAULT_ROLLING_WINDOW = 63

# Color scheme
COLOR_POSITIVE = "#1f77b4"  # Blue
COLOR_NEGATIVE = "#d62728"  # Red
COLOR_NEUTRAL = "#7f7f7f"   # Gray
COLOR_ACCENT = "#ff7f0e"    # Orange

# Font & sizing
PLOT_HEIGHT = 600
PLOT_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

# Display metrics (from risk_metrics.json)
DISPLAY_METRICS = {
    "annualized_return_cagr": "CAGR (%)",
    "annualized_volatility": "Volatility (%)",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown (%)",
    "drawdown_duration_days": "Drawdown Duration (days)",
}

# ============================================================================
# SECTION VISIBILITY
# ============================================================================

# Control which sections are visible in dashboard
SHOW_OVERVIEW = True
SHOW_PERFORMANCE = True
SHOW_DRAWDOWN_RISK = True
SHOW_ATTRIBUTION = True
SHOW_SYSTEM_INFO = True

# ============================================================================
# CACHE & PERFORMANCE
# ============================================================================

# Streamlit cache TTL (time-to-live in seconds)
CACHE_TTL_SECONDS = 3600  # 1 hour

# ============================================================================
# VALIDATION RULES
# ============================================================================

# Minimum expected number of rows for each Parquet file
MIN_ROWS = {
    "prices": 250,
    "returns": 250,
    "return_attribution": 250,
    "risk_attribution": 10,  # Asset-level aggregates, not time series
}

# Expected columns (subset that must be present)
REQUIRED_COLUMNS = {
    "prices": ["NVDA", "TSLA"],  # At least these assets
    "returns": ["NVDA"],  # At least one asset
    "return_attribution": ["portfolio_return"],  # Portfolio returns
    "risk_attribution": ["portfolio_volatility"],  # Portfolio risk metrics
}

# ============================================================================
# TEXT CONTENT
# ============================================================================

PROJECT_SUBTITLE = (
    "Institutional-grade portfolio backtesting & attribution analysis. "
    "Read-only research visualization. No live data, optimization, or trading signals."
)

OVERVIEW_DESCRIPTION = f"""
**Vector Alpha** is an institutional-grade multi-asset portfolio backtesting engine.

**Portfolio Details:**
- **Strategy**: {PORTFOLIO_DESCRIPTION}
- **Rebalancing**: {REBALANCING_FREQ}
- **Universe**: {len(ASSETS)} assets (Tech + Enterprise Software)
- **Period**: {DATE_RANGE}
- **Transaction Costs**: {TRANSACTION_COSTS_BPS} bps per trade
- **Design**: Read-only research visualization layer (no live trading)
"""

# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_DATA_NOT_FOUND = (
    f"Data files not found in {DATA_DIR}. "
    "Run `python run_experiment.py` to generate outputs."
)

ERROR_MISSING_FILE = "Missing file: {filename}. Expected at {path}."

ERROR_INDEX_MISMATCH = "Index mismatch between {file1} and {file2}. Cannot align data."

ERROR_EMPTY_DATA = "File {filename} is empty (0 rows). Check data generation."
