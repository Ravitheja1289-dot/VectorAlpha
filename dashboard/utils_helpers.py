"""
Vector Alpha Dashboard - Utility Helpers
========================================

General-purpose utility functions for formatting, validation, and math.
"""

import pandas as pd
import numpy as np


# ============================================================================
# FORMATTING UTILITIES
# ============================================================================

def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as percentage string.
    
    Args:
        value: Decimal (e.g., 0.1234 for 12.34%)
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "12.34%")
    """
    return f"{value * 100:.{decimals}f}%"


def format_bps(value: float, decimals: int = 1) -> str:
    """
    Format a decimal value as basis points.
    
    Args:
        value: Decimal (e.g., 0.001 for 10 bps)
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "10.0 bps")
    """
    return f"{value * 10000:.{decimals}f} bps"


def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format a value as currency ($).
    
    Args:
        value: Dollar amount
        decimals: Number of decimal places
        
    Returns:
        Formatted string (e.g., "$1,234.56")
    """
    return f"${value:,.{decimals}f}"


def format_large_number(value: int) -> str:
    """
    Format large integers with commas.
    
    Args:
        value: Integer
        
    Returns:
        Formatted string (e.g., "1,234,567")
    """
    return f"{value:,}"


def format_date_range(start_date, end_date) -> str:
    """
    Format a date range as readable string.
    
    Args:
        start_date: pd.Timestamp or datetime
        end_date: pd.Timestamp or datetime
        
    Returns:
        Formatted string (e.g., "Jan 1, 2020 – Dec 31, 2025")
    """
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.to_pydatetime()
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.to_pydatetime()
    
    return f"{start_date.strftime('%b %d, %Y')} - {end_date.strftime('%b %d, %Y')}"


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================

def validate_index_alignment(df1: pd.DataFrame, df2: pd.DataFrame, name1: str, name2: str) -> bool:
    """
    Check if two DataFrames have aligned DatetimeIndex.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        name1: Name of first DataFrame (for error message)
        name2: Name of second DataFrame (for error message)
        
    Returns:
        True if aligned, raises ValueError otherwise
        
    Raises:
        ValueError: If indices don't match
    """
    if not df1.index.equals(df2.index):
        raise ValueError(
            f"Index mismatch: {name1} vs {name2}\n"
            f"  {name1}: {df1.index[0]} to {df1.index[-1]} ({len(df1)} rows)\n"
            f"  {name2}: {df2.index[0]} to {df2.index[-1]} ({len(df2)} rows)"
        )
    return True


def validate_column_exists(df: pd.DataFrame, columns: list, name: str) -> bool:
    """
    Check if columns exist in DataFrame.
    
    Args:
        df: DataFrame
        columns: List of column names to check
        name: Name of DataFrame (for error message)
        
    Returns:
        True if all columns exist, raises ValueError otherwise
        
    Raises:
        ValueError: If any column missing
    """
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    return True


# ============================================================================
# STATISTICAL UTILITIES
# ============================================================================

def compute_rolling_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Compute rolling volatility (annualized standard deviation).
    
    Args:
        returns: Daily return series
        window: Rolling window size (days)
        
    Returns:
        Series of rolling volatilities (annualized)
    """
    # Assume 252 trading days per year
    return returns.rolling(window).std() * np.sqrt(252)


def compute_rolling_sharpe(returns: pd.Series, window: int, rf_rate: float = 0.0) -> pd.Series:
    """
    Compute rolling Sharpe ratio.
    
    Args:
        returns: Daily return series
        window: Rolling window size (days)
        rf_rate: Risk-free rate (annualized, default 0%)
        
    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    return (rolling_mean - rf_rate) / rolling_std


def compute_drawdown(returns: pd.Series) -> pd.Series:
    """
    Compute cumulative drawdown from returns.
    
    Args:
        returns: Daily return series
        
    Returns:
        Series of drawdown values (0 to -1)
    """
    cumulative_returns = (1 + returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown


def compute_cagr(returns: pd.Series) -> float:
    """
    Compute annualized return (CAGR) from daily returns.
    
    Args:
        returns: Daily return series
        
    Returns:
        CAGR as decimal (e.g., 0.1234 for 12.34%)
    """
    n_years = len(returns) / 252  # 252 trading days per year
    ending_value = (1 + returns).prod()
    cagr = (ending_value ** (1 / n_years)) - 1
    return cagr


def compute_volatility(returns: pd.Series) -> float:
    """
    Compute annualized volatility from daily returns.
    
    Args:
        returns: Daily return series
        
    Returns:
        Volatility as decimal (e.g., 0.1567 for 15.67%)
    """
    return returns.std() * np.sqrt(252)


def compute_sharpe_ratio(returns: pd.Series, rf_rate: float = 0.0) -> float:
    """
    Compute Sharpe ratio from daily returns.
    
    Args:
        returns: Daily return series
        rf_rate: Risk-free rate (annualized, default 0%)
        
    Returns:
        Sharpe ratio
    """
    excess_return = returns.mean() * 252 - rf_rate
    volatility = returns.std() * np.sqrt(252)
    return excess_return / volatility


# ============================================================================
# DATA TRANSFORMATION UTILITIES
# ============================================================================

def filter_by_date_range(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    """
    Filter DataFrame to date range.
    
    Args:
        df: DataFrame with DatetimeIndex
        start_date: Start date (pd.Timestamp, datetime, or string)
        end_date: End date (pd.Timestamp, datetime, or string)
        
    Returns:
        Filtered DataFrame
    """
    # Convert dates to pandas Timestamp to ensure compatibility with DatetimeIndex
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    
    return df[(df.index >= start_date) & (df.index <= end_date)]


def get_top_contributors(series: pd.Series, n: int = 5) -> pd.Series:
    """
    Get top N contributors (positive) and detractors (negative).
    
    Args:
        series: Series of contributions (can be positive or negative)
        n: Number of top items to return
        
    Returns:
        Sorted series (largest to smallest)
    """
    return series.abs().nlargest(n)


# ============================================================================
# DATETIME UTILITIES
# ============================================================================

def get_date_range_stats(df: pd.DataFrame) -> dict:
    """
    Get statistics about DataFrame's date range.
    
    Args:
        df: DataFrame with DatetimeIndex
        
    Returns:
        Dictionary with start_date, end_date, num_days, num_years
    """
    start_date = df.index.min()
    end_date = df.index.max()
    num_days = len(df)
    num_years = num_days / 252
    
    return {
        "start_date": start_date,
        "end_date": end_date,
        "num_days": num_days,
        "num_years": round(num_years, 2),
        "formatted": format_date_range(start_date, end_date),
    }
