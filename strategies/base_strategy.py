"""
Strategy Base Class (Interface)

Contract:
- Strategy outputs target weights only on rebalance dates
- No daily signals
- No execution logic
- No PnL awareness

This separation is non-negotiable.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import pandas as pd

__all__ = ["Strategy"]


class Strategy(ABC):
    """Base class for portfolio strategies.
    
    Contract
    --------
    - Input: features (dict of DataFrames), rebalance_dates (list of timestamps)
    - Output: target weights (DataFrame with rebalance dates as index, assets as columns)
    
    Rules
    -----
    - Strategy outputs weights ONLY on rebalance dates (not daily)
    - No daily signals
    - No execution logic (slippage, fills, etc.)
    - No PnL awareness (strategy is blind to portfolio state)
    
    Weight Constraints
    ------------------
    - Weights must sum to 1.0 (or 0.0 for no position) at each rebalance date
    - Negative weights allowed if strategy is long-short
    - NaN weights are not allowed
    
    Output Shape
    ------------
    Index: rebalance_dates (subset of trading dates)
    Columns: asset tickers (must match features)
    Values: target weights (floats)
    """
    
    @abstractmethod
    def generate_weights(
        self,
        features: Dict[str, pd.DataFrame],
        rebalance_dates: List[pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate target portfolio weights on rebalance dates.
        
        Parameters
        ----------
        features : Dict[str, pd.DataFrame]
            Feature DataFrames (e.g., {"daily_returns": ..., "vol_20d": ..., "mom_20d": ...})
            Each DataFrame has shape (dates x assets) aligned with prices
        rebalance_dates : List[pd.Timestamp]
            Dates on which to rebalance (typically weekly or monthly)
        
        Returns
        -------
        pd.DataFrame
            Target weights with:
            - Index: rebalance_dates
            - Columns: asset tickers
            - Values: target weights (must sum to 1.0 at each date)
        
        Raises
        ------
        NotImplementedError
            Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement generate_weights()")
