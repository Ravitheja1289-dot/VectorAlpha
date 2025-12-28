"""
Feature Engine

Responsibility:
- Input: daily prices and/or returns
- Output: feature DataFrames

Constraints:
- No weights, signals, or rebalancing
- Features must align with price index and have the same (date x asset) shape

Implemented features:
- daily_returns: simple daily returns aligned to price index
- vol_20d: rolling 20-day volatility of daily returns
- mom_20d: rolling 20-day momentum (price / price_20d_ago - 1)
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

__all__ = [
    "daily_returns",
    "rolling_volatility",
    "rolling_momentum",
    "build_features",
]


def _assert_alignment(prices: pd.DataFrame, df: pd.DataFrame) -> None:
    if not prices.index.equals(df.index):
        raise ValueError("Feature index must equal price index")
    if not prices.columns.equals(df.columns):
        raise ValueError("Feature columns must equal price columns")
    if df.shape != prices.shape:
        raise ValueError("Feature shape must equal price shape")


def daily_returns(prices: pd.DataFrame, returns: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return simple daily returns aligned to `prices.index` and `prices.columns`.

    Returns at time t use prices at t and t-1 (already past data).
    If `returns` is provided, it will be reindexed to the price index
    (first date will be NaN). Otherwise, returns are computed as `prices.pct_change()`.
    """
    if returns is not None:
        rets = returns.reindex(prices.index)
        rets = rets[prices.columns]
    else:
        rets = prices.pct_change()
    _assert_alignment(prices, rets)
    return rets


def rolling_volatility(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    window: int = 20,
) -> pd.DataFrame:
    """Rolling volatility (std) of daily returns over `window` days.

    CRITICAL: Shifted by 1 to avoid look-ahead bias.
    At time t, we see volatility computed using data up to t-1.
    Shape matches `prices`. Initial `window` rows will be NaN.
    """
    rets = daily_returns(prices, returns)
    vol = rets.rolling(window=window).std().shift(1)
    _assert_alignment(prices, vol)
    return vol


def rolling_momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling momentum: price / price.shift(window) - 1, aligned to `prices`.

    CRITICAL: Shifted by 1 to avoid look-ahead bias.
    At time t, we see momentum from t-1 looking back to t-window-1.
    Shape matches `prices`. Initial `window+1` rows will be NaN.
    """
    mom = ((prices / prices.shift(window)) - 1.0).shift(1)
    _assert_alignment(prices, mom)
    return mom


def build_features(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """Build the minimal feature set.

    Returns a dict with keys:
    - "daily_returns"
    - "vol_20d"
    - "mom_20d"
    All DataFrames align with `prices` in index and columns, and share the same shape.
    """
    dr = daily_returns(prices, returns)
    vol = rolling_volatility(prices, dr, window=window)
    mom = rolling_momentum(prices, window=window)
    return {
        "daily_returns": dr,
        "vol_20d": vol,
        "mom_20d": mom,
    }
