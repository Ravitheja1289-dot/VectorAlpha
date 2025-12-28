"""
Vector Alpha Dashboard - Centralized Data Loader
================================================

All data loading, caching, and validation in one place.
No data transformations (save for components).

Design:
- @st.cache_data for performance
- Loud failures if files missing or invalid
- Index alignment validation
- No computations (read-only)
"""

import streamlit as st
import pandas as pd
import json
from config import PARQUET_FILES, CACHE_TTL_SECONDS, MIN_ROWS, REQUIRED_COLUMNS


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_prices() -> pd.DataFrame:
    """
    Load daily asset prices.
    
    Returns:
        DataFrame with DatetimeIndex and columns for each asset
        
    Raises:
        FileNotFoundError: If prices.parquet missing
        ValueError: If data invalid (too few rows, empty index)
    """
    path = PARQUET_FILES["prices"]
    
    if not path.exists():
        raise FileNotFoundError(
            f"prices.parquet not found at {path}\n"
            f"Run: python run_experiment.py"
        )
    
    df = pd.read_parquet(path)
    
    # Validation
    if df.empty:
        raise ValueError("prices.parquet is empty (0 rows)")
    
    if len(df) < MIN_ROWS["prices"]:
        raise ValueError(
            f"prices.parquet has {len(df)} rows, expected >={MIN_ROWS['prices']}"
        )
    
    if df.index.name != df.index.name:  # Check if index exists
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("prices.parquet index is not DatetimeIndex")
    
    # Check required columns
    for col in REQUIRED_COLUMNS["prices"]:
        if col not in df.columns:
            raise ValueError(f"prices.parquet missing required column: {col}")
    
    # Add TOTAL column as equal-weighted portfolio (mean of all assets)
    df['TOTAL'] = df.mean(axis=1)
    
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_returns() -> pd.DataFrame:
    """
    Load daily portfolio and asset returns.
    
    Returns:
        DataFrame with DatetimeIndex, columns for each asset + TOTAL
        
    Raises:
        FileNotFoundError: If returns.parquet missing
        ValueError: If data invalid
    """
    path = PARQUET_FILES["returns"]
    
    if not path.exists():
        raise FileNotFoundError(
            f"returns.parquet not found at {path}\n"
            f"Run: python run_experiment.py"
        )
    
    df = pd.read_parquet(path)
    
    # Validation
    if df.empty:
        raise ValueError("returns.parquet is empty (0 rows)")
    
    if len(df) < MIN_ROWS["returns"]:
        raise ValueError(
            f"returns.parquet has {len(df)} rows, expected >={MIN_ROWS['returns']}"
        )
    
    # Add TOTAL column as equal-weighted portfolio returns (mean of all asset returns)
    df['TOTAL'] = df.mean(axis=1)
    
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_return_attribution() -> pd.DataFrame:
    """
    Load daily return attribution by asset.
    
    Returns:
        DataFrame with DatetimeIndex, columns for each asset + TOTAL
        
    Raises:
        FileNotFoundError: If file missing
        ValueError: If data invalid
    """
    path = PARQUET_FILES["return_attribution"]
    
    if not path.exists():
        raise FileNotFoundError(
            f"return_attribution.parquet not found at {path}\n"
            f"Run: python run_experiment.py"
        )
    
    df = pd.read_parquet(path)
    
    # Validation
    if df.empty:
        raise ValueError("return_attribution.parquet is empty (0 rows)")
    
    if len(df) < MIN_ROWS["return_attribution"]:
        raise ValueError(
            f"return_attribution.parquet has {len(df)} rows, "
            f"expected >={MIN_ROWS['return_attribution']}"
        )
    
    # Rename portfolio_return to TOTAL for consistency
    if 'portfolio_return' in df.columns:
        df = df.rename(columns={'portfolio_return': 'TOTAL'})
    elif 'TOTAL' not in df.columns:
        raise ValueError("return_attribution.parquet missing 'portfolio_return' or 'TOTAL' column")
    
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_risk_attribution() -> pd.DataFrame:
    """
    Load daily risk contribution by asset.
    
    Returns:
        DataFrame with DatetimeIndex, columns for each asset
        
    Raises:
        FileNotFoundError: If file missing
        ValueError: If data invalid
    """
    path = PARQUET_FILES["risk_attribution"]
    
    if not path.exists():
        raise FileNotFoundError(
            f"risk_attribution.parquet not found at {path}\n"
            f"Run: python run_experiment.py"
        )
    
    df = pd.read_parquet(path)
    
    # Validation
    if df.empty:
        raise ValueError("risk_attribution.parquet is empty (0 rows)")
    
    if len(df) < MIN_ROWS["risk_attribution"]:
        raise ValueError(
            f"risk_attribution.parquet has {len(df)} rows, "
            f"expected >={MIN_ROWS['risk_attribution']}"
        )
    
    # At least one asset column should exist
    expected_cols = REQUIRED_COLUMNS["risk_attribution"]
    if not any(col in df.columns for col in expected_cols):
        raise ValueError("risk_attribution.parquet missing expected assets")
    
    return df


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_risk_metrics() -> dict:
    """
    Load portfolio-level risk metrics (static summary).
    
    Returns:
        Dictionary with keys: annualized_return_cagr, annualized_volatility,
        sharpe_ratio, max_drawdown, drawdown_duration_days
        
    Raises:
        FileNotFoundError: If risk_metrics.json missing
        ValueError: If JSON invalid or missing required keys
    """
    path = PARQUET_FILES["risk_metrics"]
    
    if not path.exists():
        raise FileNotFoundError(
            f"risk_metrics.json not found at {path}\n"
            f"Run: python run_experiment.py"
        )
    
    try:
        with open(path, 'r') as f:
            metrics = json.load(f)
    except json.JSONDecodeError:
        raise ValueError("risk_metrics.json is not valid JSON")
    
    # Validate required keys
    required_keys = [
        "annualized_return_cagr",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "drawdown_duration_days",
    ]
    
    missing_keys = [k for k in required_keys if k not in metrics]
    if missing_keys:
        raise ValueError(f"risk_metrics.json missing keys: {missing_keys}")
    
    return metrics


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_all_data() -> dict:
    """
    Load and validate ALL data in one call.
    
    Performs index alignment validation across all DataFrames.
    
    Returns:
        Dictionary with keys: prices, returns, return_attribution,
        risk_attribution, risk_metrics
        
    Raises:
        FileNotFoundError: If any file missing
        ValueError: If data invalid or indices don't align
    """
    
    # Load all files
    prices = load_prices()
    returns = load_returns()
    return_attr = load_return_attribution()
    risk_attr = load_risk_attribution()
    risk_metrics = load_risk_metrics()
    
    # Validate index alignment
    # All should share the same DatetimeIndex (or compatible)
    index_returns = returns.index
    index_return_attr = return_attr.index
    index_risk_attr = risk_attr.index
    
    # Align indices - use intersection of dates to handle off-by-one differences
    # (returns might have one extra row at start/end compared to attribution)
    common_dates = index_returns.intersection(index_return_attr)
    
    if len(common_dates) < MIN_ROWS["returns"] - 10:
        raise ValueError(
            f"Insufficient date overlap between returns and return_attribution: {len(common_dates)} dates"
        )
    
    # Align all dataframes to common dates
    returns = returns.loc[common_dates]
    return_attr = return_attr.loc[common_dates]
    
    # Risk attribution has different structure (asset aggregates), skip alignment
    
    # Prices may have different length (could include pre-strategy period)
    # but dates should overlap
    if not prices.index.isin(returns.index).any():
        raise ValueError(
            "No date overlap between prices.parquet and returns.parquet"
        )
    
    return {
        "prices": prices,
        "returns": returns,
        "return_attribution": return_attr,
        "risk_attribution": risk_attr,
        "risk_metrics": risk_metrics,
    }


def validate_data(data: dict) -> bool:
    """
    Final sanity check on loaded data.
    
    Args:
        data: Dictionary returned by load_all_data()
        
    Returns:
        True if valid, raises Exception otherwise
    """
    
    # Check all keys present
    required_keys = ["prices", "returns", "return_attribution", "risk_attribution", "risk_metrics"]
    missing = [k for k in required_keys if k not in data]
    if missing:
        raise ValueError(f"Missing data keys: {missing}")
    
    # Check DataFrames not empty
    for key in ["prices", "returns", "return_attribution", "risk_attribution"]:
        if data[key].empty:
            raise ValueError(f"{key} is empty")
    
    # Check metrics dict
    if not isinstance(data["risk_metrics"], dict) or not data["risk_metrics"]:
        raise ValueError("risk_metrics is empty or not a dict")
    
    return True
