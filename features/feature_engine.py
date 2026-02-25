"""
Feature Engine (Enhanced)

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
- rsi_14: 14-day Relative Strength Index
- bollinger_pct: Bollinger Band %B (position within bands)
- mom_12m_skip1m: 12-month momentum skipping most recent month
- mean_reversion_zscore: z-score of returns relative to rolling mean
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

__all__ = [
    "daily_returns",
    "rolling_volatility",
    "rolling_momentum",
    "rsi",
    "bollinger_pct_b",
    "momentum_12m_skip1m",
    "mean_reversion_zscore",
    "build_features",
    "build_enhanced_features",
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


def rsi(
    prices: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Relative Strength Index (RSI).

    RSI = 100 - 100 / (1 + RS)
    RS = average gain / average loss over `window` periods.

    Shifted by 1 to avoid look-ahead bias.
    """
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    rsi_val = rsi_val.shift(1)

    rsi_val = rsi_val.fillna(50.0)
    _assert_alignment(prices, rsi_val)
    return rsi_val


def bollinger_pct_b(
    prices: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Band %B indicator.

    %B = (Price - Lower Band) / (Upper Band - Lower Band)

    %B > 1: above upper band (overbought)
    %B < 0: below lower band (oversold)
    %B = 0.5: at moving average

    Shifted by 1 to avoid look-ahead bias.
    """
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std

    bandwidth = upper - lower
    pct_b = (prices - lower) / bandwidth.replace(0, np.nan)
    pct_b = pct_b.shift(1)
    pct_b = pct_b.fillna(0.5)
    _assert_alignment(prices, pct_b)
    return pct_b


def momentum_12m_skip1m(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """12-month momentum with 1-month skip (Jegadeesh & Titman).

    Cumulative return from 12 months ago to 1 month ago.
    Avoids short-term reversal effect.

    Shifted by 1 for look-ahead bias prevention.
    """
    lookback = 252
    skip = 21
    mom = (prices.shift(skip) / prices.shift(lookback + skip) - 1.0).shift(1)
    _assert_alignment(prices, mom)
    return mom


def mean_reversion_zscore(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    window: int = 60,
) -> pd.DataFrame:
    """Z-score of returns relative to rolling mean.

    z = (r_t - rolling_mean) / rolling_std

    Negative z-score => asset is "cheap" relative to recent trend.
    Shifted by 1 to avoid look-ahead bias.
    """
    rets = daily_returns(prices, returns)
    r_mean = rets.rolling(window=window).mean()
    r_std = rets.rolling(window=window).std()

    z = (rets - r_mean) / r_std.replace(0, np.nan)
    z = z.shift(1).fillna(0)
    _assert_alignment(prices, z)
    return z


def build_features(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
    window: int = 20,
) -> Dict[str, pd.DataFrame]:
    """Build the minimal feature set (backward compatible).

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


def build_enhanced_features(
    prices: pd.DataFrame,
    returns: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """Build the full enhanced feature set.

    Returns a dict with keys:
    - "daily_returns"
    - "vol_20d"
    - "mom_20d"
    - "rsi_14"
    - "bollinger_pctb"
    - "mom_12m_skip1m"
    - "mean_reversion_zscore"
    """
    dr = daily_returns(prices, returns)
    vol = rolling_volatility(prices, dr, window=20)
    mom = rolling_momentum(prices, window=20)
    rsi_14 = rsi(prices, window=14)
    bb_pctb = bollinger_pct_b(prices, window=20, num_std=2.0)
    mom_12_1 = momentum_12m_skip1m(prices)
    mr_zscore = mean_reversion_zscore(prices, dr, window=60)

    return {
        "daily_returns": dr,
        "vol_20d": vol,
        "mom_20d": mom,
        "rsi_14": rsi_14,
        "bollinger_pctb": bb_pctb,
        "mom_12m_skip1m": mom_12_1,
        "mean_reversion_zscore": mr_zscore,
    }
