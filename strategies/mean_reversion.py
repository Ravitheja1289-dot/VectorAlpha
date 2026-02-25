"""
Mean Reversion Strategy

Behavior:
- Compute z-score of each asset's price relative to its rolling mean
- Overweight assets that are "cheap" (below mean), underweight "expensive" (above mean)
- Bollinger-band inspired approach

Methodology:
- Rolling 60-day mean and std of returns
- Z-score = (current cumulative return - rolling mean) / rolling std
- Negative z-score => asset is below trend => overweight
- Positive z-score => asset is above trend => underweight
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import Strategy

__all__ = ["MeanReversionStrategy"]


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using z-score of rolling returns.

    Parameters
    ----------
    lookback : int
        Rolling window for mean/std computation (default 60 days).
    z_threshold : float
        Z-score threshold for tilt (default 1.0).
    max_tilt : float
        Maximum tilt multiplier vs equal weight (default 2.0).
    """

    def __init__(
        self,
        lookback: int = 60,
        z_threshold: float = 1.0,
        max_tilt: float = 2.0,
    ):
        self.lookback = lookback
        self.z_threshold = z_threshold
        self.max_tilt = max_tilt

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
            raise ValueError("MeanReversionStrategy requires 'daily_returns' in features")

        # Compute rolling mean and std of daily returns
        rolling_mean = daily_rets.rolling(window=self.lookback).mean()
        rolling_std = daily_rets.rolling(window=self.lookback).std()

        weights_list = []
        base_weight = 1.0 / n_assets

        for date in rebalance_dates:
            if date not in rolling_mean.index:
                w = pd.Series(base_weight, index=assets, name=date)
                weights_list.append(w)
                continue

            r_mean = rolling_mean.loc[date]
            r_std = rolling_std.loc[date]
            r_current = daily_rets.loc[date]

            # Compute z-score (how far current return is from rolling mean)
            z_score = pd.Series(0.0, index=assets)
            valid = r_std > 1e-10
            z_score[valid] = (r_current[valid] - r_mean[valid]) / r_std[valid]

            # Mean reversion: negative z-score => overweight (asset is "cheap")
            # Tilt is proportional to negative z-score
            tilt = -z_score / self.z_threshold  # negative z => positive tilt
            tilt = tilt.clip(-self.max_tilt + 1, self.max_tilt - 1)

            w = pd.Series(base_weight, index=assets) * (1.0 + tilt)

            # Ensure non-negative (long-only)
            w = w.clip(lower=0.0)

            # Renormalize
            total = w.sum()
            if total > 0:
                w = w / total
            else:
                w = pd.Series(base_weight, index=assets)

            w.name = date
            weights_list.append(w)

        weights = pd.DataFrame(weights_list)
        weights.index.name = "date"
        return weights
