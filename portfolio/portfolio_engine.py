"""
Portfolio Engine

Responsibility:
- Convert daily weights + returns → portfolio PnL
- Apply transaction costs
- Track equity over time

Does NOT contain:
- Strategy logic (that's in strategies/)
- Execution logic (that's in execution/)

Key Formula:
- portfolio_return_t = sum(weight_{t-1}^i * return_t^i) - transaction_cost_t
- equity_t = equity_{t-1} * (1 + portfolio_return_t)

Critical Note:
- Use PREVIOUS day's weights with today's returns
- Transaction costs are subtracted on rebalance dates only
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
import numpy as np

__all__ = ["calculate_portfolio_returns", "calculate_equity_curve"]


def calculate_portfolio_returns(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: pd.Series,
) -> pd.Series:
    """Calculate portfolio returns from weights, asset returns, and costs.
    
    Parameters
    ----------
    daily_weights : pd.DataFrame
        Daily portfolio weights (dates × assets)
        Shape: (n_days, n_assets)
    returns : pd.DataFrame
        Daily asset returns (dates × assets)
        Shape: (n_days, n_assets)
    transaction_costs : pd.Series
        Transaction costs on rebalance dates
        Index: rebalance_dates
        
    Returns
    -------
    pd.Series
        Daily portfolio returns
        
    Notes
    -----
    Portfolio return formula:
        r_portfolio_t = sum(w_{t-1}^i * r_t^i) - tc_t
        
    Where:
        - w_{t-1}^i is weight of asset i at END of day t-1 (after rebalance or drift)
        - r_t^i is return of asset i on day t
        - tc_t is transaction cost on day t (only on rebalance dates, 0 otherwise)
        
    Transaction costs reduce portfolio return on rebalance dates.
    
    Example:
        If turnover = 5% and cost = 10 bps, then tc = 0.05 * 0.001 = 0.05 bps
        This is subtracted from portfolio return on that rebalance date.
    """
    # Align indices
    common_dates = daily_weights.index.intersection(returns.index)
    daily_weights = daily_weights.loc[common_dates]
    returns = returns.loc[common_dates]
    
    # Validation
    assert daily_weights.shape == returns.shape, "Weights and returns must have same shape"
    assert daily_weights.columns.equals(returns.columns), "Weights and returns must have same columns"
    
    # Calculate gross portfolio returns (before costs)
    # r_portfolio_t = sum(w_{t-1}^i * r_t^i)
    # We use current day's weights with current day's returns
    # (weights are already lagged - they represent position entering the day)
    gross_returns = (daily_weights * returns).sum(axis=1)
    
    # Create transaction cost series aligned to daily returns
    # (zero on non-rebalance dates)
    daily_costs = pd.Series(0.0, index=gross_returns.index)
    for date, cost in transaction_costs.items():
        if date in daily_costs.index:
            daily_costs.loc[date] = cost
    
    # Net portfolio returns = gross returns - transaction costs
    portfolio_returns = gross_returns - daily_costs
    
    # Sanity checks
    assert not portfolio_returns.isna().any(), "Portfolio returns contain NaNs"
    assert not portfolio_returns.isin([np.inf, -np.inf]).any(), "Portfolio returns contain infinities"
    
    portfolio_returns.name = 'portfolio_returns'
    return portfolio_returns


def calculate_equity_curve(
    portfolio_returns: pd.Series,
    initial_capital: float = 1.0,
) -> pd.Series:
    """Calculate equity curve from portfolio returns.
    
    Parameters
    ----------
    portfolio_returns : pd.Series
        Daily portfolio returns
    initial_capital : float, default=1.0
        Starting capital (normalized to 1.0)
        
    Returns
    -------
    pd.Series
        Equity curve (cumulative wealth)
        
    Notes
    -----
    Equity formula:
        equity_t = equity_{t-1} * (1 + r_portfolio_t)
        
    Or equivalently:
        equity_t = initial_capital * prod(1 + r_portfolio_s) for s in [0, t]
        
    This is the compounded wealth over time.
    """
    # Cumulative product: (1 + r_0) * (1 + r_1) * ... * (1 + r_t)
    equity = initial_capital * (1.0 + portfolio_returns).cumprod()
    
    # Sanity checks
    assert not equity.isna().any(), "Equity curve contains NaNs"
    assert not equity.isin([np.inf, -np.inf]).any(), "Equity curve contains infinities"
    assert (equity > 0).all(), "Equity curve must remain positive"
    
    equity.name = 'equity'
    return equity


def run_backtest(
    daily_weights: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: pd.Series,
    initial_capital: float = 1.0,
) -> Dict[str, pd.Series]:
    """Run full backtest: weights + returns → equity curve.
    
    Parameters
    ----------
    daily_weights : pd.DataFrame
        Daily portfolio weights (dates × assets)
    returns : pd.DataFrame
        Daily asset returns (dates × assets)
    transaction_costs : pd.Series
        Transaction costs on rebalance dates
    initial_capital : float, default=1.0
        Starting capital
        
    Returns
    -------
    dict
        {
            'portfolio_returns': pd.Series,
            'equity': pd.Series,
        }
    """
    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(
        daily_weights=daily_weights,
        returns=returns,
        transaction_costs=transaction_costs,
    )
    
    # Calculate equity curve
    equity = calculate_equity_curve(
        portfolio_returns=portfolio_returns,
        initial_capital=initial_capital,
    )
    
    return {
        'portfolio_returns': portfolio_returns,
        'equity': equity,
    }
