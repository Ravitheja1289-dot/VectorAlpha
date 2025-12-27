"""
Equal Weight Strategy (Baseline)

Behavior:
- On every rebalance date, assign equal weight to all assets
- Sum of weights = 1.0

Why this matters:
- Tests rebalancing logic
- Tests interfaces
- No alpha distractions
- If equal-weight breaks later → your system is broken
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from strategies.base_strategy import Strategy

__all__ = ["EqualWeightStrategy"]


class EqualWeightStrategy(Strategy):
    """Equal weight (1/N) portfolio rebalanced on specified dates.
    
    This is a baseline strategy with no alpha signal.
    Useful for testing infrastructure and as a performance benchmark.
    """
    
    def generate_weights(
        self,
        features: Dict[str, pd.DataFrame],
        rebalance_dates: List[pd.Timestamp],
    ) -> pd.DataFrame:
        """Generate equal weights (1/N) for all assets on each rebalance date.
        
        Parameters
        ----------
        features : Dict[str, pd.DataFrame]
            Feature DataFrames (not used by this strategy, but required by interface)
        rebalance_dates : List[pd.Timestamp]
            Dates on which to rebalance
        
        Returns
        -------
        pd.DataFrame
            Target weights with:
            - Index: rebalance_dates
            - Columns: asset tickers (from features)
            - Values: 1/N for each asset
        """
        # Extract asset list from any feature DataFrame (all should have same columns)
        if not features:
            raise ValueError("features dict is empty")
        
        sample_feature = next(iter(features.values()))
        assets = sample_feature.columns.tolist()
        n_assets = len(assets)
        
        if n_assets == 0:
            raise ValueError("No assets found in features")
        
        # Equal weight for each asset
        equal_weight = 1.0 / n_assets
        
        # Create DataFrame: rebalance_dates × assets, all values = 1/N
        weights = pd.DataFrame(
            data=equal_weight,
            index=rebalance_dates,
            columns=assets,
        )
        
        return weights
