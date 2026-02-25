"""
Multi-Factor Composite Strategy

Behavior:
- Combines momentum, mean-reversion, and low-volatility signals
- Rank-weighted composite score per asset
- Rebalances to overweight top-ranked composite assets

Methodology:
- Factor 1: Momentum (12-1 month cumulative return)
- Factor 2: Mean Reversion (short-term z-score reversal, 20-day)
- Factor 3: Low Volatility (inverse of 60-day rolling vol)
- Each factor is cross-sectionally ranked and normalized to [0, 1]
- Composite score = weighted average of factor ranks
- Weights assigned proportional to composite score
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base_strategy import Strategy

__all__ = ["MultiFactorStrategy"]


class MultiFactorStrategy(Strategy):
    """Multi-factor composite strategy combining momentum, mean-reversion, and low-vol.

    Parameters
    ----------
    momentum_lookback : int
        Lookback for momentum factor (default 252 days).
    momentum_skip : int
        Skip period for momentum (default 21 days).
    reversion_lookback : int
        Lookback for mean-reversion z-score (default 20 days).
    vol_lookback : int
        Lookback for volatility factor (default 60 days).
    factor_weights : dict or None
        Weights for each factor: {"momentum", "reversion", "low_vol"}.
        Default: equal weight (1/3 each).
    concentration : float
        How much to concentrate in top-ranked assets (default 2.0).
        Higher = more concentrated portfolio.
    """

    def __init__(
        self,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        reversion_lookback: int = 20,
        vol_lookback: int = 60,
        factor_weights: dict = None,
        concentration: float = 2.0,
    ):
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.reversion_lookback = reversion_lookback
        self.vol_lookback = vol_lookback
        self.factor_weights = factor_weights or {
            "momentum": 0.40,
            "reversion": 0.25,
            "low_vol": 0.35,
        }
        self.concentration = concentration

    def _compute_momentum_score(
        self, daily_rets: pd.DataFrame, date: pd.Timestamp
    ) -> pd.Series:
        """Compute momentum score for each asset at a given date."""
        loc = daily_rets.index.get_loc(date)
        start = loc - self.momentum_lookback - self.momentum_skip
        end = loc - self.momentum_skip

        if start < 0 or end < 0:
            return None

        window_rets = daily_rets.iloc[start:end]
        cum_return = (1 + window_rets).prod() - 1
        return cum_return

    def _compute_reversion_score(
        self, daily_rets: pd.DataFrame, date: pd.Timestamp
    ) -> pd.Series:
        """Compute mean-reversion z-score for each asset."""
        loc = daily_rets.index.get_loc(date)
        start = loc - self.reversion_lookback

        if start < 0:
            return None

        window = daily_rets.iloc[start:loc + 1]
        rolling_mean = window.mean()
        rolling_std = window.std()

        current = daily_rets.iloc[loc]
        z_score = pd.Series(0.0, index=daily_rets.columns)
        valid = rolling_std > 1e-10
        z_score[valid] = (current[valid] - rolling_mean[valid]) / rolling_std[valid]

        # Negative z-score is favorable for mean reversion (asset is cheap)
        return -z_score

    def _compute_low_vol_score(
        self, daily_rets: pd.DataFrame, date: pd.Timestamp
    ) -> pd.Series:
        """Compute low-volatility score (inverse of rolling vol)."""
        loc = daily_rets.index.get_loc(date)
        start = loc - self.vol_lookback

        if start < 0:
            return None

        window = daily_rets.iloc[start:loc + 1]
        vol = window.std()

        # Inverse vol: lower vol = higher score
        inv_vol = pd.Series(0.0, index=daily_rets.columns)
        valid = vol > 1e-10
        inv_vol[valid] = 1.0 / vol[valid]
        return inv_vol

    def _rank_normalize(self, series: pd.Series) -> pd.Series:
        """Cross-sectionally rank and normalize to [0, 1]."""
        ranks = series.rank(ascending=True)
        n = len(ranks)
        if n <= 1:
            return pd.Series(0.5, index=series.index)
        return (ranks - 1) / (n - 1)

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
            raise ValueError("MultiFactorStrategy requires 'daily_returns' in features")

        fw = self.factor_weights
        weights_list = []
        base_weight = 1.0 / n_assets

        for date in rebalance_dates:
            mom_score = self._compute_momentum_score(daily_rets, date)
            rev_score = self._compute_reversion_score(daily_rets, date)
            vol_score = self._compute_low_vol_score(daily_rets, date)

            # If any factor unavailable, fall back to equal weight
            if mom_score is None or rev_score is None or vol_score is None:
                w = pd.Series(base_weight, index=assets, name=date)
                weights_list.append(w)
                continue

            # Rank-normalize each factor to [0, 1]
            mom_rank = self._rank_normalize(mom_score)
            rev_rank = self._rank_normalize(rev_score)
            vol_rank = self._rank_normalize(vol_score)

            # Composite score = weighted average of factor ranks
            composite = (
                fw["momentum"] * mom_rank
                + fw["reversion"] * rev_rank
                + fw["low_vol"] * vol_rank
            )

            # Convert scores to weights using softmax-like concentration
            shifted = composite - composite.mean()
            exp_scores = np.exp(self.concentration * shifted)
            w = exp_scores / exp_scores.sum()

            w.name = date
            weights_list.append(w)

        weights = pd.DataFrame(weights_list)
        weights.index.name = "date"
        return weights
