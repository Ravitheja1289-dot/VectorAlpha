"""
Yahoo Finance loader for daily OHLCV data.

Responsibilities:
- Fetch daily OHLCV and include Adjusted Close.
- Return a pandas DataFrame with required columns.
- No saving, renaming, date fixing, or feature creation.

Required columns (must be present):
- Open, High, Low, Close, Adj Close, Volume
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd
import yfinance as yf

from .base_loader import BaseDataLoader

__all__ = ["YahooLoader", "required_columns"]


required_columns: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)


class YahooLoader(BaseDataLoader):
    """Loader that fetches raw daily OHLCV from Yahoo Finance.

    Notes
    -----
    - Returns the DataFrame exactly as provided by `yfinance.download`.
    - Ensures required columns are present, otherwise raises `ValueError`.
    - Does not modify column names, timestamps, or add features.
    """

    def fetch(self, symbol: str, start: date | datetime, end: date | datetime) -> pd.DataFrame:
        # Use yfinance to download daily OHLCV; keep data raw (no auto adjustments)
        df = yf.download(
            tickers=symbol,
            start=_to_datetime(start),
            end=_to_datetime(end),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
        )

        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError(f"YahooLoader: No data returned for {symbol} between {start} and {end}.")

        # Validate presence of required columns; do not alter/rename
        missing: list[str] = [c for c in required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                (
                    "YahooLoader: Missing required columns: "
                    f"{missing}. Expected: {list(required_columns)}."
                )
            )

        return df


def _to_datetime(d: date | datetime) -> datetime:
    """Convert date or datetime to datetime without altering timezone info."""
    if isinstance(d, datetime):
        return d
    return datetime(d.year, d.month, d.day)
