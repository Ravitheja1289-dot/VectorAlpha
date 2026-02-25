"""
Vector Alpha Dashboard - Centralized Data Loader (Enhanced)
==========================================================

All data loading, caching, and validation in one place.
Now supports multi-strategy data, factor model, and stress test results.
"""

import streamlit as st
import pandas as pd
import json
from config import PARQUET_FILES, CACHE_TTL_SECONDS, MIN_ROWS, REQUIRED_COLUMNS


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_prices() -> pd.DataFrame:
    path = PARQUET_FILES["prices"]
    if not path.exists():
        raise FileNotFoundError(f"prices.parquet not found at {path}")
    df = pd.read_parquet(path)
    if df.empty or len(df) < MIN_ROWS["prices"]:
        raise ValueError(f"prices.parquet has insufficient data")
    for col in REQUIRED_COLUMNS["prices"]:
        if col not in df.columns:
            raise ValueError(f"prices.parquet missing column: {col}")
    df['TOTAL'] = df.mean(axis=1)
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_returns() -> pd.DataFrame:
    path = PARQUET_FILES["returns"]
    if not path.exists():
        raise FileNotFoundError(f"returns.parquet not found at {path}")
    df = pd.read_parquet(path)
    if df.empty or len(df) < MIN_ROWS["returns"]:
        raise ValueError(f"returns.parquet has insufficient data")
    df['TOTAL'] = df.mean(axis=1)
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_return_attribution() -> pd.DataFrame:
    path = PARQUET_FILES["return_attribution"]
    if not path.exists():
        raise FileNotFoundError(f"return_attribution.parquet not found at {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("return_attribution.parquet is empty")
    if 'portfolio_return' in df.columns:
        df = df.rename(columns={'portfolio_return': 'TOTAL'})
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_risk_attribution() -> pd.DataFrame:
    path = PARQUET_FILES["risk_attribution"]
    if not path.exists():
        raise FileNotFoundError(f"risk_attribution.parquet not found at {path}")
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError("risk_attribution.parquet is empty")
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_risk_metrics() -> dict:
    path = PARQUET_FILES["risk_metrics"]
    if not path.exists():
        raise FileNotFoundError(f"risk_metrics.json not found at {path}")
    with open(path, 'r') as f:
        metrics = json.load(f)
    required = ["annualized_return_cagr", "annualized_volatility", "sharpe_ratio",
                 "max_drawdown", "drawdown_duration_days"]
    missing = [k for k in required if k not in metrics]
    if missing:
        raise ValueError(f"risk_metrics.json missing keys: {missing}")
    return metrics


# ============================================================================
# Multi-Strategy Data Loaders
# ============================================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_strategy_equity_curves() -> pd.DataFrame:
    path = PARQUET_FILES.get("strategy_equity_curves")
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_strategy_returns() -> pd.DataFrame:
    path = PARQUET_FILES.get("strategy_returns")
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_strategy_comparison() -> pd.DataFrame:
    path = PARQUET_FILES.get("strategy_comparison")
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_strategy_avg_weights() -> pd.DataFrame:
    path = PARQUET_FILES.get("strategy_avg_weights")
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_stress_test_results() -> dict:
    path = PARQUET_FILES.get("stress_test_results")
    if path is None or not path.exists():
        return {}
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_factor_loadings() -> pd.DataFrame:
    path = PARQUET_FILES.get("factor_loadings")
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_factor_variance() -> pd.DataFrame:
    path = PARQUET_FILES.get("factor_variance")
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


# ============================================================================
# MASTER LOADER
# ============================================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_all_data() -> dict:
    """Load and validate ALL data in one call."""

    # Core data (required)
    prices = load_prices()
    returns = load_returns()
    return_attr = load_return_attribution()
    risk_attr = load_risk_attribution()
    risk_metrics = load_risk_metrics()

    # Align core data indices
    common_dates = returns.index.intersection(return_attr.index)
    if len(common_dates) < MIN_ROWS["returns"] - 10:
        raise ValueError(f"Insufficient date overlap: {len(common_dates)} dates")
    returns = returns.loc[common_dates]
    return_attr = return_attr.loc[common_dates]

    # Multi-strategy data (optional)
    strategy_equity = load_strategy_equity_curves()
    strategy_returns = load_strategy_returns()
    strategy_comparison = load_strategy_comparison()
    strategy_weights = load_strategy_avg_weights()
    stress_test = load_stress_test_results()
    factor_loadings = load_factor_loadings()
    factor_variance = load_factor_variance()

    return {
        "prices": prices,
        "returns": returns,
        "return_attribution": return_attr,
        "risk_attribution": risk_attr,
        "risk_metrics": risk_metrics,
        "strategy_equity_curves": strategy_equity,
        "strategy_returns": strategy_returns,
        "strategy_comparison": strategy_comparison,
        "strategy_avg_weights": strategy_weights,
        "stress_test_results": stress_test,
        "factor_loadings": factor_loadings,
        "factor_variance": factor_variance,
    }


def validate_data(data: dict) -> bool:
    required = ["prices", "returns", "return_attribution", "risk_attribution", "risk_metrics"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Missing data keys: {missing}")
    for key in ["prices", "returns", "return_attribution", "risk_attribution"]:
        if data[key].empty:
            raise ValueError(f"{key} is empty")
    return True
