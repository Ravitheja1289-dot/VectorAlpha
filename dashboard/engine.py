"""
Vector Alpha — Compute Engine
==============================

Bridges the dashboard UI to the backend backtesting modules.
Takes user inputs, runs the full pipeline, returns results dict.
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from config import DATA_DIR, DEFAULT_COST_BPS
from backtest.rebalance import get_rebalance_dates
from execution.executor import execute_strategy
from execution.costs import calculate_transaction_costs
from portfolio.portfolio_engine import run_backtest
from risk.metrics import (
    annualized_volatility,
    annualized_return_cagr,
    sharpe_ratio,
    compute_drawdown,
    rolling_volatility,
    rolling_sharpe,
)
from risk.attribution import (
    compute_return_attribution,
    compute_risk_attribution,
)


# ============================================================================
# DATA LOADING (cached)
# ============================================================================

@st.cache_data
def load_prices() -> pd.DataFrame:
    """Load processed price matrix from Parquet."""
    path = DATA_DIR / "prices.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"prices.parquet not found at {path}. "
            "Run `python main.py` first to generate data."
        )
    return pd.read_parquet(path)


@st.cache_data
def load_returns() -> pd.DataFrame:
    """Load processed returns matrix from Parquet."""
    path = DATA_DIR / "returns.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"returns.parquet not found at {path}. "
            "Run `python main.py` first to generate data."
        )
    return pd.read_parquet(path)


# ============================================================================
# MAIN PORTFOLIO LAB ENGINE
# ============================================================================

def run_portfolio_lab(
    assets: list,
    weights: dict,
    start_date: str,
    end_date: str,
    rebalance_freq: str = "monthly",
    cost_bps: float = DEFAULT_COST_BPS,
    show_costs: bool = True,
) -> dict:
    """Run the full portfolio simulation from user inputs.

    Parameters
    ----------
    assets : list
        Selected asset tickers
    weights : dict
        {ticker: weight} where weights sum to 1.0
    start_date, end_date : str
        Date range for simulation
    rebalance_freq : str
        "none", "monthly", "quarterly", "yearly"
    cost_bps : float
        Transaction cost in basis points
    show_costs : bool
        Whether to apply transaction costs

    Returns
    -------
    dict with keys:
        prices, returns, equity, portfolio_returns,
        drawdown_series, max_drawdown, drawdown_duration,
        cagr, volatility, sharpe, daily_weights,
        return_attribution, risk_attribution,
        weight_drift, turnover, transaction_costs,
        rebalance_dates, config
    """
    # Load and filter data
    all_prices = load_prices()
    all_returns = load_returns()

    # Filter to selected assets
    prices = all_prices[assets].copy()
    returns = all_returns[assets].copy()

    # Filter to date range
    prices = prices.loc[start_date:end_date]
    returns = returns.loc[start_date:end_date]

    # Drop any rows with NaN
    prices = prices.dropna()
    returns = returns.dropna()

    if len(prices) < 30:
        raise ValueError("Not enough data points for this date range. Need at least 30 trading days.")

    # Build weight series
    weight_series = pd.Series(weights)
    weight_series = weight_series / weight_series.sum()  # ensure sums to 1

    # Handle no-rebalance (buy & hold) vs rebalanced
    if rebalance_freq == "none":
        result = _run_buy_and_hold(
            prices, returns, weight_series, cost_bps if show_costs else 0.0
        )
    else:
        result = _run_rebalanced(
            prices, returns, weight_series, rebalance_freq,
            cost_bps if show_costs else 0.0
        )

    # Compute risk metrics
    equity = result["equity"]
    port_returns = result["portfolio_returns"]

    cagr = annualized_return_cagr(equity)
    vol = annualized_volatility(port_returns)
    sr = sharpe_ratio(port_returns)
    dd_series, max_dd, dd_duration = compute_drawdown(equity)
    roll_vol = rolling_volatility(port_returns, window=min(63, len(port_returns) // 3))
    roll_sharpe = rolling_sharpe(port_returns, window=min(63, len(port_returns) // 3))

    # Compute attribution
    daily_weights = result["daily_weights"]
    ret_attr = compute_return_attribution(daily_weights, returns)
    risk_attr = compute_risk_attribution(returns, daily_weights)

    # Compute correlation matrix
    correlation = returns.corr()

    # Compute weight drift (how weights change over time)
    weight_drift = daily_weights.copy()

    return {
        # Raw data
        "prices": prices,
        "returns": returns,
        "correlation": correlation,
        # Portfolio results
        "equity": equity,
        "portfolio_returns": port_returns,
        "daily_weights": daily_weights,
        # Risk metrics
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sr,
        "drawdown_series": dd_series,
        "max_drawdown": max_dd,
        "drawdown_duration": dd_duration,
        "rolling_volatility": roll_vol,
        "rolling_sharpe": roll_sharpe,
        # Attribution
        "return_attribution": ret_attr,
        "risk_attribution": risk_attr,
        # Execution details
        "weight_drift": weight_drift,
        "turnover": result.get("turnover"),
        "transaction_costs": result.get("transaction_costs"),
        "total_costs": result.get("total_costs", 0.0),
        "rebalance_dates": result.get("rebalance_dates", []),
        # Config echo
        "config": {
            "assets": assets,
            "weights": weights,
            "start_date": start_date,
            "end_date": end_date,
            "rebalance_freq": rebalance_freq,
            "cost_bps": cost_bps,
            "show_costs": show_costs,
        },
    }


# ============================================================================
# INTERNAL: Buy & Hold (no rebalancing)
# ============================================================================

def _run_buy_and_hold(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    initial_weights: pd.Series,
    cost_bps: float,
) -> dict:
    """Simulate buy-and-hold: invest at initial weights, never rebalance."""
    assets = prices.columns.tolist()

    # Initialize weights on first day, then drift every day
    daily_weights = pd.DataFrame(
        index=prices.index, columns=assets, dtype=float
    )

    current_weights = initial_weights.copy()
    daily_weights.iloc[0] = current_weights

    for i in range(1, len(prices)):
        date = prices.index[i]
        if date in returns.index:
            day_return = returns.loc[date]
            numerator = current_weights * (1.0 + day_return)
            denominator = numerator.sum()
            if abs(denominator) < 1e-12:
                current_weights = initial_weights.copy()
            else:
                current_weights = numerator / denominator
        daily_weights.iloc[i] = current_weights

    # Compute portfolio returns using lagged weights
    lagged = daily_weights.shift(1).dropna(how="all")
    common = lagged.index.intersection(returns.index)
    lagged = lagged.loc[common]
    ret_aligned = returns.loc[common]

    port_returns = (lagged * ret_aligned).sum(axis=1)
    port_returns.name = "portfolio_return"

    # Equity curve
    equity_vals = np.zeros(len(port_returns))
    eq = 1.0
    for i, r in enumerate(port_returns.values):
        eq *= (1.0 + r)
        equity_vals[i] = eq
    equity = pd.Series(equity_vals, index=port_returns.index, name="equity")

    return {
        "equity": equity,
        "portfolio_returns": port_returns,
        "daily_weights": daily_weights.loc[common],
        "turnover": pd.Series(dtype=float),
        "transaction_costs": pd.Series(dtype=float),
        "total_costs": 0.0,
        "rebalance_dates": [],
    }


# ============================================================================
# INTERNAL: Rebalanced portfolio
# ============================================================================

def _run_rebalanced(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    target_weights: pd.Series,
    rebalance_freq: str,
    cost_bps: float,
) -> dict:
    """Simulate a rebalanced portfolio."""
    assets = prices.columns.tolist()

    # Get rebalance dates
    rebalance_dates = get_rebalance_dates(prices.index, rebalance_freq)

    if len(rebalance_dates) < 2:
        raise ValueError(
            f"Only {len(rebalance_dates)} rebalance date(s) for frequency "
            f"'{rebalance_freq}'. Need at least 2. Try a longer time range."
        )

    # Build target weights DataFrame (same weights on every rebalance date)
    target_df = pd.DataFrame(
        {asset: target_weights[asset] for asset in assets},
        index=pd.DatetimeIndex(rebalance_dates),
    )

    # Use the execution engine
    exec_result = execute_strategy(
        prices=prices,
        returns=returns,
        target_weights=target_df,
        rebalance_dates=rebalance_dates,
        cost_bps=cost_bps,
    )

    daily_weights = exec_result["daily_weights"]
    turnover = exec_result["turnover"]
    transaction_costs = exec_result["transaction_costs"]

    # Trim NaN rows before first rebalance — execute_strategy fills
    # the full prices.index but leaves rows as NaN before the first
    # rebalance date, which breaks the length assertion in run_backtest.
    first_valid = daily_weights.first_valid_index()
    if first_valid is not None:
        daily_weights = daily_weights.loc[first_valid:]

    # Run backtest
    bt_result = run_backtest(
        daily_weights=daily_weights,
        returns=returns,
        transaction_costs=transaction_costs,
    )

    net_returns = bt_result["net_returns"]
    equity = bt_result["equity"]
    total_costs = bt_result["daily_costs"].sum()

    # Trim daily_weights to match portfolio returns index
    daily_weights = daily_weights.loc[net_returns.index]

    return {
        "equity": equity,
        "portfolio_returns": net_returns,
        "daily_weights": daily_weights,
        "turnover": turnover,
        "transaction_costs": transaction_costs,
        "total_costs": total_costs,
        "rebalance_dates": rebalance_dates,
    }
