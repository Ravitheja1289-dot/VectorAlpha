"""
Cross-Sectional Momentum Strategy

Behavior:
- Rank assets by trailing momentum (12-1 month lookback, skip most recent month)
- Go long top N assets, underweight bottom N
- Rebalance on specified dates

Methodology:
- Classic Jegadeesh & Titman (1993) cross-sectional momentum
- Lookback: 252 days (~12 months), skip last 21 days (~1 month)
- Rank assets by cumulative return over lookback window
- Top quintile gets overweight, bottom quintile gets underweight
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import Strategy

__all__ = ["MomentumStrategy"]


class MomentumStrategy(Strategy):
    """Cross-sectional momentum strategy.

    Parameters
    ----------
    lookback : int
        Momentum lookback window in trading days (default 252 ~ 12 months).
    skip : int
        Skip most recent N days to avoid short-term reversal (default 21 ~ 1 month).
    top_pct : float
        Fraction of assets to overweight (default 0.4 = top 40%).
    long_weight_mult : float
        Multiplier for top-ranked assets relative to equal weight (default 1.5).
    """

    def __init__(
        self,
        lookback: int = 252,
        skip: int = 21,
        top_pct: float = 0.4,
        long_weight_mult: float = 1.5,
    ):
        self.lookback = lookback
        self.skip = skip
        self.top_pct = top_pct
        self.long_weight_mult = long_weight_mult

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

        # We need daily returns to compute cumulative momentum
        daily_rets = features.get("daily_returns")
        if daily_rets is None:
            raise ValueError("MomentumStrategy requires 'daily_returns' in features")

        n_top = max(1, int(n_assets * self.top_pct))
        n_bottom = max(1, int(n_assets * self.top_pct))

        weights_list = []
        for date in rebalance_dates:
            loc = daily_rets.index.get_loc(date)

            # Need enough history
            start_idx = loc - self.lookback - self.skip
            end_idx = loc - self.skip

            if start_idx < 0 or end_idx < 0:
                # Not enough history — fall back to equal weight
                w = pd.Series(1.0 / n_assets, index=assets, name=date)
                weights_list.append(w)
                continue

            # Cumulative return over lookback window (skipping recent days)
            window_rets = daily_rets.iloc[start_idx:end_idx]
            cum_return = (1 + window_rets).prod() - 1  # per-asset cumulative return

            # Rank assets (highest momentum = highest rank)
            ranks = cum_return.rank(ascending=True)

            # Assign weights: overweight top, underweight bottom
            base_weight = 1.0 / n_assets
            w = pd.Series(base_weight, index=assets, name=date)

            top_assets = ranks.nlargest(n_top).index
            bottom_assets = ranks.nsmallest(n_bottom).index

            w[top_assets] = base_weight * self.long_weight_mult
            w[bottom_assets] = base_weight * (2.0 - self.long_weight_mult)

            # Ensure no negative weights (long-only)
            w = w.clip(lower=0.0)

            # Renormalize to sum to 1.0
            w = w / w.sum()

            weights_list.append(w)

        weights = pd.DataFrame(weights_list)
        weights.index.name = "date"
        return weights
