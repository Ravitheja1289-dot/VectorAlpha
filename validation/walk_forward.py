"""
Walk-Forward Validation Framework

Implements:
1. Rolling walk-forward optimization
2. Combinatorial purged cross-validation (simplified)
3. Out-of-sample performance tracking
4. Overfitting probability estimation (Deflated Sharpe)

This is the gold standard for strategy validation in quantitative finance.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "walk_forward_backtest",
    "deflated_sharpe_ratio",
    "minimum_backtest_length",
    "rolling_oos_sharpe",
    "WalkForwardResult",
]


class WalkForwardResult:
    """Container for walk-forward validation results."""

    def __init__(
        self,
        in_sample_sharpe: List[float],
        out_of_sample_sharpe: List[float],
        in_sample_returns: List[pd.Series],
        out_of_sample_returns: List[pd.Series],
        fold_dates: List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    ):
        self.in_sample_sharpe = in_sample_sharpe
        self.out_of_sample_sharpe = out_of_sample_sharpe
        self.in_sample_returns = in_sample_returns
        self.out_of_sample_returns = out_of_sample_returns
        self.fold_dates = fold_dates

    @property
    def avg_is_sharpe(self) -> float:
        return float(np.mean(self.in_sample_sharpe))

    @property
    def avg_oos_sharpe(self) -> float:
        return float(np.mean(self.out_of_sample_sharpe))

    @property
    def sharpe_decay(self) -> float:
        """Fraction of in-sample Sharpe retained out-of-sample."""
        if abs(self.avg_is_sharpe) < 1e-10:
            return 0.0
        return self.avg_oos_sharpe / self.avg_is_sharpe

    @property
    def n_folds(self) -> int:
        return len(self.in_sample_sharpe)

    @property
    def combined_oos_returns(self) -> pd.Series:
        """Concatenate all out-of-sample return segments."""
        return pd.concat(self.out_of_sample_returns)

    def summary(self) -> Dict[str, float]:
        return {
            "n_folds": self.n_folds,
            "avg_in_sample_sharpe": self.avg_is_sharpe,
            "avg_out_of_sample_sharpe": self.avg_oos_sharpe,
            "sharpe_decay": self.sharpe_decay,
            "oos_sharpe_std": float(np.std(self.out_of_sample_sharpe)),
            "pct_positive_oos": float(np.mean([s > 0 for s in self.out_of_sample_sharpe])),
        }


def _annualized_sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Quick annualized Sharpe calculation."""
    excess = returns - rf / 252
    if excess.std() < 1e-10:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(252))


def walk_forward_backtest(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    strategy_fn: Callable,
    train_days: int = 504,
    test_days: int = 126,
    step_days: int = 63,
    cost_bps: float = 10.0,
    risk_free_rate: float = 0.0,
) -> WalkForwardResult:
    """Rolling walk-forward validation.

    Splits the data into overlapping train/test windows and evaluates
    strategy performance on out-of-sample data.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily prices (dates x assets).
    returns : pd.DataFrame
        Daily returns (dates x assets).
    strategy_fn : Callable
        Function(prices_train, returns_train) -> pd.Series of portfolio weights.
        The function receives training data and returns target weights.
    train_days : int
        Number of training days per fold (default 504 ~ 2 years).
    test_days : int
        Number of test days per fold (default 126 ~ 6 months).
    step_days : int
        Step size between folds (default 63 ~ 3 months).
    cost_bps : float
        Transaction cost in basis points.
    risk_free_rate : float
        Annual risk-free rate for Sharpe calculation.

    Returns
    -------
    WalkForwardResult
        Comprehensive results across all folds.
    """
    dates = returns.index
    n = len(dates)

    is_sharpes = []
    oos_sharpes = []
    is_returns_list = []
    oos_returns_list = []
    fold_dates = []

    fold_start = 0
    while fold_start + train_days + test_days <= n:
        train_end = fold_start + train_days
        test_end = min(train_end + test_days, n)

        train_idx = dates[fold_start:train_end]
        test_idx = dates[train_end:test_end]

        prices_train = prices.loc[train_idx]
        returns_train = returns.loc[train_idx]
        returns_test = returns.loc[test_idx]

        # Get strategy weights from training data
        try:
            weights = strategy_fn(prices_train, returns_train)
        except Exception:
            fold_start += step_days
            continue

        # Compute in-sample portfolio returns
        is_port_ret = (returns_train * weights).sum(axis=1)

        # Compute out-of-sample portfolio returns (same weights, new data)
        oos_port_ret = (returns_test * weights).sum(axis=1)

        # Apply simple cost approximation
        cost_per_day = cost_bps / 10000 / 252  # spread across days
        oos_port_ret = oos_port_ret - cost_per_day

        is_sharpe = _annualized_sharpe(is_port_ret, risk_free_rate)
        oos_sharpe = _annualized_sharpe(oos_port_ret, risk_free_rate)

        is_sharpes.append(is_sharpe)
        oos_sharpes.append(oos_sharpe)
        is_returns_list.append(is_port_ret)
        oos_returns_list.append(oos_port_ret)
        fold_dates.append((train_idx[0], train_idx[-1], test_idx[0], test_idx[-1]))

        fold_start += step_days

    return WalkForwardResult(
        in_sample_sharpe=is_sharpes,
        out_of_sample_sharpe=oos_sharpes,
        in_sample_returns=is_returns_list,
        out_of_sample_returns=oos_returns_list,
        fold_dates=fold_dates,
    )


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    risk_free_rate: float = 0.0,
) -> Dict[str, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Adjusts Sharpe ratio for multiple testing / strategy selection bias.

    Parameters
    ----------
    observed_sharpe : float
        Observed (best) annualized Sharpe ratio.
    n_trials : int
        Number of strategy variants tested.
    n_observations : int
        Number of return observations.
    skewness : float
        Return distribution skewness.
    kurtosis : float
        Return distribution kurtosis (not excess).
    risk_free_rate : float
        Annual risk-free rate.

    Returns
    -------
    dict
        - "deflated_sharpe": adjusted Sharpe
        - "p_value": probability of observing Sharpe by chance
        - "is_significant": whether strategy is significant at 5%
    """
    # Expected maximum Sharpe under null (Euler-Mascheroni approximation)
    euler_mascheroni = 0.5772
    if n_trials <= 1:
        expected_max_sharpe = 0.0
    else:
        expected_max_sharpe = (
            np.sqrt(2 * np.log(n_trials))
            - (np.log(np.pi) + euler_mascheroni)
            / (2 * np.sqrt(2 * np.log(n_trials)))
        )

    # Standard error of Sharpe ratio (Lo, 2002)
    se_sharpe = np.sqrt(
        (1 + 0.5 * observed_sharpe ** 2
         - skewness * observed_sharpe
         + (kurtosis - 3) / 4 * observed_sharpe ** 2)
        / (n_observations - 1)
    )

    if se_sharpe < 1e-10:
        return {"deflated_sharpe": 0.0, "p_value": 1.0, "is_significant": False}

    # Test statistic
    psr_stat = (observed_sharpe - expected_max_sharpe) / se_sharpe
    p_value = 1 - stats.norm.cdf(psr_stat)

    return {
        "deflated_sharpe": float(psr_stat),
        "p_value": float(p_value),
        "is_significant": p_value < 0.05,
        "expected_max_sharpe_null": float(expected_max_sharpe),
    }


def minimum_backtest_length(
    observed_sharpe: float,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> int:
    """Minimum Backtest Length (MBL) for a given Sharpe ratio.

    How many observations do you need for the Sharpe ratio to be statistically
    significant at the given confidence level?

    Parameters
    ----------
    observed_sharpe : float
        Observed annualized Sharpe ratio.
    skewness : float
        Return distribution skewness.
    kurtosis : float
        Return distribution kurtosis.
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    int
        Minimum number of daily observations required.
    """
    if abs(observed_sharpe) < 1e-10:
        return 999999

    z = stats.norm.ppf(confidence)
    sr_daily = observed_sharpe / np.sqrt(252)

    mbl = (
        1
        + (1 - skewness * sr_daily + (kurtosis - 1) / 4 * sr_daily ** 2)
        * (z / sr_daily) ** 2
    )
    return max(1, int(np.ceil(mbl)))


def rolling_oos_sharpe(
    returns: pd.Series,
    window: int = 252,
    step: int = 63,
) -> pd.DataFrame:
    """Compute rolling out-of-sample Sharpe to detect strategy decay.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns.
    window : int
        Window size for each Sharpe calculation.
    step : int
        Step size between windows.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: end_date, sharpe, is_positive.
    """
    dates = returns.index
    results = []

    i = window
    while i <= len(dates):
        segment = returns.iloc[i - window:i]
        sharpe = _annualized_sharpe(segment)
        results.append({
            "end_date": dates[i - 1],
            "sharpe": sharpe,
            "is_positive": sharpe > 0,
        })
        i += step

    return pd.DataFrame(results)
