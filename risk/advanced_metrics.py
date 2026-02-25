"""
Advanced Risk Metrics Module

Extends the base risk metrics with:
- Value-at-Risk (VaR) — Historical, Parametric, Cornish-Fisher
- Conditional VaR / Expected Shortfall (CVaR)
- Sortino Ratio
- Calmar Ratio
- Omega Ratio
- Tail Risk (skewness, kurtosis)
- Information Ratio
- Stress Testing
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "historical_var",
    "parametric_var",
    "cornish_fisher_var",
    "expected_shortfall",
    "sortino_ratio",
    "calmar_ratio",
    "omega_ratio",
    "information_ratio",
    "tail_risk_metrics",
    "stress_test",
    "compute_advanced_metrics",
]


# ============================================================================
# VALUE-AT-RISK
# ============================================================================

def historical_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Historical Value-at-Risk.

    The loss threshold that is only exceeded (1 - confidence)% of the time.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level (default 0.95 = 95%).

    Returns
    -------
    float
        VaR as a positive number (loss magnitude).
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def parametric_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Parametric (Gaussian) Value-at-Risk.

    Assumes returns are normally distributed.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR as a positive number.
    """
    mu = returns.mean()
    sigma = returns.std()
    z = stats.norm.ppf(1 - confidence)
    return -(mu + z * sigma)


def cornish_fisher_var(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Cornish-Fisher VaR (adjusts for skewness and kurtosis).

    More accurate than parametric VaR when returns are non-normal.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        VaR as a positive number.
    """
    mu = returns.mean()
    sigma = returns.std()
    s = returns.skew()
    k = returns.kurtosis()  # excess kurtosis
    z = stats.norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    z_cf = (
        z
        + (z ** 2 - 1) * s / 6
        + (z ** 3 - 3 * z) * k / 24
        - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    )

    return -(mu + z_cf * sigma)


# ============================================================================
# EXPECTED SHORTFALL (CVaR)
# ============================================================================

def expected_shortfall(
    returns: pd.Series, confidence: float = 0.95
) -> float:
    """Expected Shortfall (Conditional VaR).

    Average loss in the worst (1-confidence)% of cases.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    confidence : float
        Confidence level.

    Returns
    -------
    float
        CVaR as a positive number.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return -tail.mean()


# ============================================================================
# RATIO METRICS
# ============================================================================

def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, target: float = 0.0
) -> float:
    """Sortino Ratio — penalizes only downside volatility.

    Sortino = (mean_return - rf) / downside_deviation * sqrt(252)

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    risk_free_rate : float
        Annual risk-free rate.
    target : float
        Minimum acceptable return (daily).

    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    downside = returns[returns < target] - target
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-10
    return (excess.mean() / downside_std) * np.sqrt(252) if downside_std > 1e-10 else 0.0


def calmar_ratio(
    returns: pd.Series, equity_curve: pd.Series
) -> float:
    """Calmar Ratio = CAGR / Max Drawdown.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    equity_curve : pd.Series
        Equity curve.

    Returns
    -------
    float
        Calmar ratio.
    """
    n = len(equity_curve)
    if n < 2:
        return 0.0
    cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / (n - 1)) - 1

    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    max_dd = abs(drawdown.min())

    if max_dd < 1e-10:
        return 0.0
    return cagr / max_dd


def omega_ratio(
    returns: pd.Series, threshold: float = 0.0
) -> float:
    """Omega Ratio = sum of gains above threshold / sum of losses below threshold.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    threshold : float
        Return threshold (daily, default 0).

    Returns
    -------
    float
        Omega ratio.
    """
    excess = returns - threshold
    gains = excess[excess > 0].sum()
    losses = abs(excess[excess <= 0].sum())

    if losses < 1e-12:
        return float("inf") if gains > 0 else 1.0
    return gains / losses


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Information Ratio = active return / tracking error.

    Parameters
    ----------
    returns : pd.Series
        Portfolio daily returns.
    benchmark_returns : pd.Series
        Benchmark daily returns.

    Returns
    -------
    float
        Annualized information ratio.
    """
    active = returns - benchmark_returns
    tracking_error = active.std() * np.sqrt(252)
    if tracking_error < 1e-10:
        return 0.0
    return (active.mean() * 252) / tracking_error


# ============================================================================
# TAIL RISK
# ============================================================================

def tail_risk_metrics(returns: pd.Series) -> Dict[str, float]:
    """Compute tail risk statistics.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.

    Returns
    -------
    dict
        Skewness, kurtosis, Jarque-Bera stat, and tail ratios.
    """
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())  # excess kurtosis

    # Jarque-Bera test for normality
    n = len(returns)
    jb_stat = (n / 6) * (skew ** 2 + (kurt ** 2) / 4)
    jb_pvalue = 1 - stats.chi2.cdf(jb_stat, 2)

    # Tail ratio: 95th percentile / abs(5th percentile)
    p95 = np.percentile(returns.dropna(), 95)
    p5 = np.percentile(returns.dropna(), 5)
    tail_ratio = abs(p95 / p5) if abs(p5) > 1e-10 else float("inf")

    # Gain-to-pain ratio
    total_gain = returns[returns > 0].sum()
    total_loss = abs(returns[returns < 0].sum())
    gain_to_pain = total_gain / total_loss if total_loss > 1e-10 else float("inf")

    return {
        "skewness": skew,
        "excess_kurtosis": kurt,
        "jarque_bera_stat": float(jb_stat),
        "jarque_bera_pvalue": float(jb_pvalue),
        "is_normal": jb_pvalue > 0.05,
        "tail_ratio": tail_ratio,
        "gain_to_pain_ratio": gain_to_pain,
        "positive_days_pct": float((returns > 0).mean()),
        "negative_days_pct": float((returns < 0).mean()),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
    }


# ============================================================================
# STRESS TESTING
# ============================================================================

def stress_test(
    returns: pd.Series,
    equity_curve: pd.Series,
    scenarios: Optional[Dict[str, Tuple[str, str]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Stress test portfolio over predefined crisis periods.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns.
    equity_curve : pd.Series
        Portfolio equity curve.
    scenarios : dict or None
        {name: (start_date, end_date)} crisis scenarios.
        If None, uses default scenarios.

    Returns
    -------
    dict
        {scenario_name: {total_return, max_drawdown, volatility, worst_day}}.
    """
    if scenarios is None:
        scenarios = {
            "COVID Crash (Feb-Mar 2020)": ("2020-02-19", "2020-03-23"),
            "COVID Recovery (Mar-Aug 2020)": ("2020-03-24", "2020-08-31"),
            "2022 Bear Market (Jan-Oct 2022)": ("2022-01-03", "2022-10-12"),
            "2022 Recovery (Oct-Dec 2022)": ("2022-10-13", "2022-12-30"),
            "2023 AI Rally (Jan-Jul 2023)": ("2023-01-03", "2023-07-31"),
            "2024 Full Year": ("2024-01-02", "2024-12-31"),
        }

    results = {}
    for name, (start, end) in scenarios.items():
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        mask = (returns.index >= start_ts) & (returns.index <= end_ts)
        period_rets = returns[mask]

        if len(period_rets) < 5:
            continue

        period_equity = equity_curve[mask]
        total_ret = (1 + period_rets).prod() - 1

        cummax = period_equity.cummax()
        dd = (period_equity - cummax) / cummax
        max_dd = float(dd.min())

        vol = float(period_rets.std() * np.sqrt(252))

        results[name] = {
            "total_return": float(total_ret),
            "max_drawdown": max_dd,
            "annualized_volatility": vol,
            "worst_day": float(period_rets.min()),
            "best_day": float(period_rets.max()),
            "trading_days": len(period_rets),
        }

    return results


# ============================================================================
# AGGREGATED ADVANCED METRICS
# ============================================================================

def compute_advanced_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    risk_free_rate: float = 0.0,
    benchmark_returns: Optional[pd.Series] = None,
) -> Dict[str, object]:
    """Compute all advanced risk metrics at once.

    Parameters
    ----------
    returns : pd.Series
        Daily portfolio returns (net of costs).
    equity_curve : pd.Series
        Equity curve.
    risk_free_rate : float
        Annual risk-free rate.
    benchmark_returns : pd.Series or None
        Benchmark returns for information ratio.

    Returns
    -------
    dict
        Comprehensive risk metrics dictionary.
    """
    metrics = {
        # VaR metrics
        "var_95_historical": historical_var(returns, 0.95),
        "var_99_historical": historical_var(returns, 0.99),
        "var_95_parametric": parametric_var(returns, 0.95),
        "var_95_cornish_fisher": cornish_fisher_var(returns, 0.95),
        "cvar_95": expected_shortfall(returns, 0.95),
        "cvar_99": expected_shortfall(returns, 0.99),

        # Ratio metrics
        "sortino_ratio": sortino_ratio(returns, risk_free_rate),
        "calmar_ratio": calmar_ratio(returns, equity_curve),
        "omega_ratio": omega_ratio(returns),

        # Tail risk
        "tail_risk": tail_risk_metrics(returns),

        # Stress test
        "stress_test": stress_test(returns, equity_curve),
    }

    if benchmark_returns is not None:
        aligned_bench = benchmark_returns.reindex(returns.index).dropna()
        aligned_port = returns.reindex(aligned_bench.index)
        metrics["information_ratio"] = information_ratio(aligned_port, aligned_bench)

    return metrics
