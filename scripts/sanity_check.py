from __future__ import annotations

"""
One-asset sanity check before multi-asset loops.

Flow:
- Load settings from YAML (dates, universe, freq).
- Pick the first symbol (e.g., AAPL).
- Fetch raw daily OHLCV via YahooLoader (must include Adj Close).
- Print head, tail, shape, date range.
- Validate: DatetimeIndex, daily cadence, no duplicate timestamps.

If any check fails, exit with non-zero code. Do not proceed.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is first on sys.path to prefer local packages (avoid 'config' name clashes)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_settings
from data.loaders.yahoo_loader import YahooLoader, required_columns


def _fail(msg: str) -> None:
    print(f"SANITY CHECK FAILED: {msg}")
    sys.exit(1)


def _assert_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        _fail("Index is not a pandas DatetimeIndex")


def _assert_no_duplicates(df: pd.DataFrame) -> None:
    if df.index.has_duplicates:
        _fail("Index has duplicated timestamps")


def _assert_daily_cadence(df: pd.DataFrame) -> None:
    # Use infer_freq as a strong hint; allow business daily ('B') or calendar daily ('D').
    # If infer_freq returns None, fall back to median diff heuristic ~ 1 day.
    idx = df.index.sort_values()
    freq = pd.infer_freq(idx)
    if freq in {"D", "B", "C"}:
        return
    if len(idx) >= 3:
        diffs = idx.to_series().diff().dropna()
        med = diffs.median()
        # Accept median around 1 day (allow a tolerance for DST/holidays)
        if pd.Timedelta("0.5D") <= med <= pd.Timedelta("1.5D"):
            return
    _fail(f"Data does not appear to be daily (infer_freq={freq})")


def main() -> None:
    settings = load_settings()
    symbol = settings.universe_symbols[0]

    print(f"Symbol: {symbol}")
    print(f"Date range (config): {settings.start_date} → {settings.end_date}")
    print(f"Frequency (config): {settings.data_frequency}")

    loader = YahooLoader()
    df = loader.fetch(symbol, settings.start_date, settings.end_date)

    # Print basic info
    print("Head:\n", df.head())
    print("Tail:\n", df.tail())
    print("Shape:", df.shape)
    try:
        rng = (df.index.min(), df.index.max())
        print("Date range (data):", rng)
    except Exception:
        pass

    # Sanity checks
    # Required columns enforced by loader; double-check here as well
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        _fail(f"Missing required columns in DataFrame: {missing}")

    _assert_datetime_index(df)
    _assert_no_duplicates(df)
    _assert_daily_cadence(df)

    print("SANITY CHECK PASSED: Data looks daily with clean timestamps.")


if __name__ == "__main__":
    main()
