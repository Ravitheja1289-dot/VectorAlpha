"""
Vector Alpha Dashboard - Configuration Constants
================================================

All constants, file paths, and configuration in one place for easy maintenance.
"""

from pathlib import Path

# ============================================================================
# FILE PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PARQUET_FILES = {
    "prices": DATA_DIR / "prices.parquet",
    "returns": DATA_DIR / "returns.parquet",
    "return_attribution": DATA_DIR / "return_attribution.parquet",
    "risk_attribution": DATA_DIR / "risk_attribution.parquet",
    "risk_metrics": DATA_DIR / "risk_metrics.json",
    # Multi-strategy files
    "strategy_equity_curves": DATA_DIR / "strategy_equity_curves.parquet",
    "strategy_returns": DATA_DIR / "strategy_returns.parquet",
    "strategy_comparison": DATA_DIR / "strategy_comparison.parquet",
    "strategy_avg_weights": DATA_DIR / "strategy_avg_weights.parquet",
    "stress_test_results": DATA_DIR / "stress_test_results.json",
    "factor_loadings": DATA_DIR / "factor_loadings.parquet",
    "factor_variance": DATA_DIR / "factor_variance.parquet",
}

# ============================================================================
# ASSET & PORTFOLIO CONFIGURATION
# ============================================================================

ASSETS = [
    "AAPL", "ADBE", "AMD", "AMZN", "CRM",
    "CSCO", "GOOGL", "INTC", "META", "MSFT",
    "NFLX", "NVDA", "ORCL", "QCOM", "TSLA"
]

PORTFOLIO_TOTAL = "TOTAL"

PORTFOLIO_DESCRIPTION = "Multi-strategy quantitative portfolio engine"
REBALANCING_FREQ = "Weekly"
TRANSACTION_COSTS_BPS = 10
DATE_RANGE = "2020-2025"
RISK_FREE_RATE = 0.02

# ============================================================================
# STRATEGY CONFIGURATION
# ============================================================================

STRATEGY_NAMES = [
    "Equal Weight", "Momentum", "Mean Reversion", "Multi-Factor",
    "Low Volatility", "Risk Parity", "HRP", "Min Variance",
]

# Distinct color for each strategy
STRATEGY_COLORS = {
    "Equal Weight":   "#1f77b4",
    "Momentum":       "#ff7f0e",
    "Mean Reversion": "#2ca02c",
    "Multi-Factor":   "#d62728",
    "Low Volatility": "#9467bd",
    "Risk Parity":    "#8c564b",
    "HRP":            "#e377c2",
    "Min Variance":   "#17becf",
}

# Distinct color per asset (high contrast)
ASSET_COLORS = {
    "AAPL":  "#e6194b",
    "ADBE":  "#3cb44b",
    "AMD":   "#ffe119",
    "AMZN":  "#4363d8",
    "CRM":   "#f58231",
    "CSCO":  "#911eb4",
    "GOOGL": "#42d4f4",
    "INTC":  "#f032e6",
    "META":  "#bfef45",
    "MSFT":  "#fabebe",
    "NFLX":  "#469990",
    "NVDA":  "#dcbeff",
    "ORCL":  "#9a6324",
    "QCOM":  "#800000",
    "TSLA":  "#000075",
}

# Line styles per asset for additional differentiation
ASSET_DASH = {
    "AAPL": "solid", "ADBE": "dash", "AMD": "dot", "AMZN": "solid",
    "CRM": "dashdot", "CSCO": "dash", "GOOGL": "solid", "INTC": "dot",
    "META": "solid", "MSFT": "dash", "NFLX": "dashdot", "NVDA": "solid",
    "ORCL": "dot", "QCOM": "dash", "TSLA": "solid",
}

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

ROLLING_WINDOWS = [63, 126, 252]
DEFAULT_ROLLING_WINDOW = 63

COLOR_POSITIVE = "#1f77b4"
COLOR_NEGATIVE = "#d62728"
COLOR_NEUTRAL = "#7f7f7f"
COLOR_ACCENT = "#ff7f0e"

PLOT_HEIGHT = 600
PLOT_FONT_SIZE = 12
TITLE_FONT_SIZE = 16

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

DISPLAY_METRICS = {
    "annualized_return_cagr": "CAGR (%)",
    "annualized_volatility": "Volatility (%)",
    "sharpe_ratio": "Sharpe Ratio",
    "max_drawdown": "Max Drawdown (%)",
    "drawdown_duration_days": "Drawdown Duration (days)",
}

ADVANCED_METRICS = {
    "sortino": "Sortino Ratio",
    "calmar": "Calmar Ratio",
    "omega": "Omega Ratio",
    "var_95": "VaR (95%)",
    "cvar_95": "CVaR (95%)",
    "skewness": "Skewness",
    "kurtosis": "Excess Kurtosis",
}

# ============================================================================
# SECTION VISIBILITY
# ============================================================================

SHOW_OVERVIEW = True
SHOW_PERFORMANCE = True
SHOW_DRAWDOWN_RISK = True
SHOW_ATTRIBUTION = True
SHOW_STRATEGIES = True
SHOW_FACTORS = True
SHOW_REALTIME = True
SHOW_SYSTEM_INFO = True

# ============================================================================
# CACHE & PERFORMANCE
# ============================================================================

CACHE_TTL_SECONDS = 3600

# ============================================================================
# VALIDATION RULES
# ============================================================================

MIN_ROWS = {
    "prices": 250,
    "returns": 250,
    "return_attribution": 250,
    "risk_attribution": 10,
}

REQUIRED_COLUMNS = {
    "prices": ["NVDA", "TSLA"],
    "returns": ["NVDA"],
    "return_attribution": ["portfolio_return"],
    "risk_attribution": ["portfolio_volatility"],
}

# ============================================================================
# TEXT CONTENT
# ============================================================================

PROJECT_SUBTITLE = (
    "Institutional-grade multi-strategy portfolio backtesting, attribution & risk analytics. "
    "8 strategies | 15 assets | Real-time monitoring."
)

OVERVIEW_DESCRIPTION = f"""
**Vector Alpha** is an institutional-grade multi-strategy quantitative portfolio engine.

**Portfolio Details:**
- **Strategies**: {len(STRATEGY_NAMES)} active strategies (Momentum, Mean Reversion, Multi-Factor, Low-Vol, Risk Parity, HRP, Min Variance + Equal Weight baseline)
- **Rebalancing**: {REBALANCING_FREQ}
- **Universe**: {len(ASSETS)} assets (Tech + Enterprise Software)
- **Period**: {DATE_RANGE}
- **Transaction Costs**: {TRANSACTION_COSTS_BPS} bps per trade
- **Features**: RSI, Bollinger Bands, Momentum 12-1, Z-Score, Rolling Vol
- **Optimization**: Mean-Variance, Risk Parity, HRP (Lopez de Prado)
- **Risk**: VaR, CVaR, Sortino, Calmar, Omega, Stress Testing, Factor Model
"""

ERROR_DATA_NOT_FOUND = (
    f"Data files not found in {DATA_DIR}. "
    "Run `python run_experiment.py` to generate outputs."
)

ERROR_MISSING_FILE = "Missing file: {filename}. Expected at {path}."
ERROR_INDEX_MISMATCH = "Index mismatch between {file1} and {file2}. Cannot align data."
ERROR_EMPTY_DATA = "File {filename} is empty (0 rows). Check data generation."
