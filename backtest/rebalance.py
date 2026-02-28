"""
Rebalance Calendar

Responsibility:
- From daily dates, select rebalance dates at specified frequency
- Use last trading day of each period

Output: List[pd.Timestamp]

Why last trading day?
- Reflects realistic portfolio ops
- Avoids mid-period churn
"""
from __future__ import annotations

from typing import List

import pandas as pd

__all__ = ["get_weekly_rebalance_dates", "get_rebalance_dates"]


def get_weekly_rebalance_dates(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Return last trading day of each week from a DatetimeIndex.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Daily trading dates (typically from prices.index or returns.index)

    Returns
    -------
    List[pd.Timestamp]
        One rebalance date per week (last trading day of each week)

    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
    >>> rebalance_dates = get_weekly_rebalance_dates(dates)
    >>> len(rebalance_dates)  # ~52 weeks
    52
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a pd.DatetimeIndex")

    if len(dates) == 0:
        return []

    # Create a Series to work with pandas groupby
    df = pd.DataFrame({"date": dates})
    df["week"] = df["date"].dt.to_period("W")

    # Group by week and take the last (latest) date in each week
    weekly = df.groupby("week")["date"].max()

    return weekly.tolist()


def get_rebalance_dates(
    dates: pd.DatetimeIndex,
    frequency: str = "monthly",
) -> List[pd.Timestamp]:
    """Return rebalance dates at the specified frequency.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Daily trading dates
    frequency : str
        One of: "weekly", "monthly", "quarterly", "yearly"

    Returns
    -------
    List[pd.Timestamp]
        Rebalance dates (last trading day of each period)
    """
    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError("dates must be a pd.DatetimeIndex")

    if len(dates) == 0:
        return []

    freq = frequency.lower().strip()

    if freq == "weekly":
        return get_weekly_rebalance_dates(dates)

    period_map = {
        "monthly": "M",
        "quarterly": "Q",
        "yearly": "Y",
    }

    if freq not in period_map:
        raise ValueError(
            f"Unknown frequency '{frequency}'. "
            f"Choose from: weekly, monthly, quarterly, yearly"
        )

    df = pd.DataFrame({"date": dates})
    df["period"] = df["date"].dt.to_period(period_map[freq])
    rebalance = df.groupby("period")["date"].max()

    return rebalance.tolist()
