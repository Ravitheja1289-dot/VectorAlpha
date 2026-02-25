"""
Low Volatility Strategy

Behavior:
- Overweight low-volatility assets, underweight high-volatility assets
- Exploits the low-volatility anomaly (low-vol stocks tend to outperform risk-adjusted)

Methodology:
- Compute 60-day rolling volatility for each asset
- Assign weights inversely proportional to volatility
- w_i = (1/vol_i) / sum(1/vol_j)
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import Strategy

__all__ = ["LowVolatilityStrategy"]


class LowVolatilityStrategy(Strategy):
    """Inverse-volatility weighted portfolio.

    Parameters
    ----------
    vol_lookback : int
        Lookback window for volatility estimation (default 60 days).
    """

    def __init__(self, vol_lookback: int = 60):
        self.vol_lookback = vol_lookback

    def generate_weights(
        self,
        features: Dict[str, pd.DataFrame],
        rebalance_dates: List[pd.Timestamp],
    ) -> pd.DataFrame:
        if not features:
            raise ValueError("features dict is empty")

        sample_feature = next(iter(features.values()))
        assets = sample_feature.columns.tolist()
        n_assets = len(assets)

        if n_assets == 0:
            raise ValueError("No assets found in features")

        daily_rets = features.get("daily_returns")
        if daily_rets is None:
            raise ValueError("LowVolatilityStrategy requires 'daily_returns' in features")

        base_weight = 1.0 / n_assets
        weights_list = []

        for date in rebalance_dates:
            loc = daily_rets.index.get_loc(date)
            start = loc - self.vol_lookback

            if start < 0:
                w = pd.Series(base_weight, index=assets, name=date)
                weights_list.append(w)
                continue

            window = daily_rets.iloc[start:loc + 1]
            vol = window.std()

            # Inverse volatility weights
            inv_vol = pd.Series(0.0, index=assets)
            valid = vol > 1e-10
            inv_vol[valid] = 1.0 / vol[valid]

            total = inv_vol.sum()
            if total > 0:
                w = inv_vol / total
            else:
                w = pd.Series(base_weight, index=assets)

            w.name = date
            weights_list.append(w)

        weights = pd.DataFrame(weights_list)
        weights.index.name = "date"
        return weights
