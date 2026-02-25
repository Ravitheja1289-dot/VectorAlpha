"""
Real-Time Data Loader

Fetches live / near-live market data via yfinance API.
Supports:
- Current prices and intraday data
- Real-time portfolio valuation
- Market status detection
- Incremental data updates (append to existing Parquet)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "fetch_realtime_prices",
    "fetch_intraday_data",
    "get_market_status",
    "fetch_latest_returns",
    "incremental_update",
    "RealTimeSnapshot",
]


class RealTimeSnapshot:
    """Container for a point-in-time market snapshot."""

    def __init__(
        self,
        prices: pd.Series,
        changes: pd.Series,
        pct_changes: pd.Series,
        volumes: pd.Series,
        timestamp: datetime,
        market_open: bool,
    ):
        self.prices = prices
        self.changes = changes
        self.pct_changes = pct_changes
        self.volumes = volumes
        self.timestamp = timestamp
        self.market_open = market_open

    def to_dict(self) -> Dict:
        return {
            "prices": self.prices.to_dict(),
            "changes": self.changes.to_dict(),
            "pct_changes": self.pct_changes.to_dict(),
            "volumes": self.volumes.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "market_open": self.market_open,
        }


def fetch_realtime_prices(
    symbols: List[str],
    period: str = "1d",
) -> RealTimeSnapshot:
    """Fetch current/latest prices for a list of symbols.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    period : str
        yfinance period string (default "1d" for latest day).

    Returns
    -------
    RealTimeSnapshot
        Current market snapshot.
    """
    import yfinance as yf

    tickers = yf.Tickers(" ".join(symbols))

    prices = {}
    changes = {}
    pct_changes = {}
    volumes = {}

    for symbol in symbols:
        try:
            ticker = tickers.tickers[symbol]
            hist = ticker.history(period=period)
            if len(hist) > 0:
                current = float(hist["Close"].iloc[-1])
                prices[symbol] = current

                if len(hist) > 1:
                    prev = float(hist["Close"].iloc[-2])
                    changes[symbol] = current - prev
                    pct_changes[symbol] = (current - prev) / prev if prev != 0 else 0.0
                else:
                    info = ticker.info
                    prev_close = info.get("previousClose", current)
                    changes[symbol] = current - prev_close
                    pct_changes[symbol] = (
                        (current - prev_close) / prev_close if prev_close != 0 else 0.0
                    )

                volumes[symbol] = float(hist["Volume"].iloc[-1])
        except Exception:
            prices[symbol] = np.nan
            changes[symbol] = 0.0
            pct_changes[symbol] = 0.0
            volumes[symbol] = 0.0

    now = datetime.now()
    market_open = _is_market_open(now)

    return RealTimeSnapshot(
        prices=pd.Series(prices),
        changes=pd.Series(changes),
        pct_changes=pd.Series(pct_changes),
        volumes=pd.Series(volumes),
        timestamp=now,
        market_open=market_open,
    )


def fetch_intraday_data(
    symbols: List[str],
    interval: str = "5m",
    period: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Fetch intraday data for multiple symbols.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    interval : str
        Data interval: "1m", "2m", "5m", "15m", "30m", "60m", "90m".
    period : str
        Period: "1d", "5d", "1mo".

    Returns
    -------
    dict
        {symbol: DataFrame with OHLCV data}.
    """
    import yfinance as yf

    result = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            if not data.empty:
                result[symbol] = data
        except Exception:
            pass

    return result


def fetch_latest_returns(
    symbols: List[str],
    lookback_days: int = 30,
) -> pd.DataFrame:
    """Fetch recent daily returns.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    lookback_days : int
        Number of calendar days of history.

    Returns
    -------
    pd.DataFrame
        Daily returns (dates x assets).
    """
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=lookback_days + 5)  # buffer for weekends

    data = yf.download(
        " ".join(symbols),
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"] if "Adj Close" in data.columns.get_level_values(0) else data["Close"]
    else:
        prices = data[["Adj Close"]].rename(columns={"Adj Close": symbols[0]})

    returns = prices.pct_change().dropna()
    return returns


def get_market_status(
) -> Dict[str, object]:
    """Get current US market status.

    Returns
    -------
    dict
        Market status information.
    """
    now = datetime.now()
    is_open = _is_market_open(now)

    return {
        "timestamp": now.isoformat(),
        "is_open": is_open,
        "status": "OPEN" if is_open else "CLOSED",
        "weekday": now.strftime("%A"),
        "time": now.strftime("%H:%M:%S"),
    }


def incremental_update(
    existing_path: str,
    symbols: List[str],
    price_column: str = "Adj Close",
) -> pd.DataFrame:
    """Append new data to existing Parquet file.

    Fetches data from the last date in the Parquet file to today.

    Parameters
    ----------
    existing_path : str
        Path to existing prices.parquet.
    symbols : list of str
        Ticker symbols.
    price_column : str
        Column to extract.

    Returns
    -------
    pd.DataFrame
        Updated price matrix with new rows appended.
    """
    import yfinance as yf

    existing = pd.read_parquet(existing_path)
    last_date = existing.index[-1]

    start = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    if start >= end:
        print("Data is already up to date.")
        return existing

    data = yf.download(
        " ".join(symbols),
        start=start,
        end=end,
        progress=False,
    )

    if data.empty:
        print("No new data available.")
        return existing

    if isinstance(data.columns, pd.MultiIndex):
        new_prices = data[price_column] if price_column in data.columns.get_level_values(0) else data["Close"]
    else:
        new_prices = data[[price_column]].rename(columns={price_column: symbols[0]})

    # Ensure column alignment
    new_prices = new_prices.reindex(columns=existing.columns)

    # Append
    updated = pd.concat([existing, new_prices])
    updated = updated[~updated.index.duplicated(keep="last")]
    updated = updated.sort_index()

    # Save back
    updated.to_parquet(existing_path)
    print(f"Updated {existing_path}: added {len(new_prices)} new rows.")

    return updated


def _is_market_open(dt: datetime) -> bool:
    """Check if US stock market is currently open (simple heuristic)."""
    # Weekday check (Mon=0 to Fri=4)
    if dt.weekday() > 4:
        return False
    # Market hours: 9:30 AM - 4:00 PM ET (approximate)
    hour = dt.hour
    minute = dt.minute
    market_open_mins = 9 * 60 + 30
    market_close_mins = 16 * 60
    current_mins = hour * 60 + minute
    return market_open_mins <= current_mins <= market_close_mins
