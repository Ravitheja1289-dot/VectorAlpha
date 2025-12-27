"""
Rebalance Calendar

Responsibility:
- From daily dates, select one rebalance date per week
- Use last trading day of each week (recommended)

Output: List[pd.Timestamp]

Why last trading day?
- Reflects realistic portfolio ops
- Avoids mid-week churn
"""
from __future__ import annotations

from typing import List

import pandas as pd

__all__ = ["get_weekly_rebalance_dates"]


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
