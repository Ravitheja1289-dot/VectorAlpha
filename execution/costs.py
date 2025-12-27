"""
Transaction Cost Model

Responsibility:
- Compute transaction costs from turnover
- Use linear cost model (bps * turnover)

Default: 10 bps (0.001)
"""
from __future__ import annotations

import pandas as pd
import numpy as np

__all__ = ["calculate_transaction_costs"]


def calculate_transaction_costs(
    turnover: pd.Series,
    cost_bps: float = 10.0,
) -> pd.Series:
    """Calculate transaction costs from turnover.
    
    Parameters
    ----------
    turnover : pd.Series
        Turnover at each rebalance date (sum of absolute weight changes)
    cost_bps : float, default=10.0
        Transaction cost in basis points (10 bps = 0.1%)
        
    Returns
    -------
    pd.Series
        Transaction costs at each rebalance date
        
    Notes
    -----
    Linear cost model:
        cost_t = (cost_bps / 10000) * turnover_t
        
    Example:
        turnover = 0.5 (50% of portfolio traded)
        cost_bps = 10
        cost = 0.001 * 0.5 = 0.0005 (5 bps of portfolio)
    """
    cost_rate = cost_bps / 10000.0
    costs = cost_rate * turnover
    
    # Sanity checks
    assert (costs >= 0).all(), "Transaction costs must be non-negative"
    assert not costs.isna().any(), "Transaction costs contain NaNs"
    assert not costs.isin([np.inf, -np.inf]).any(), "Transaction costs contain infinities"
    
    return costs
