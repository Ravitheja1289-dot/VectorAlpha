"""
Utilities to load backtest settings from YAML.

Avoid hardcoding run parameters in Python; use this loader to read
start/end dates, universe symbols, and data frequency from settings.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import List

import yaml

__all__ = ["Settings", "load_settings"]


@dataclass(frozen=True)
class Settings:
    start_date: date
    end_date: date
    universe_symbols: List[str]
    data_frequency: str


def load_settings(path: str | Path = "config/settings.yaml") -> Settings:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Settings file not found: {p}")
    data = yaml.safe_load(p.read_text()) or {}

    try:
        start = _to_date(data["start_date"])  # ISO string expected
        end = _to_date(data["end_date"])      # ISO string expected
        symbols = list(data["universe_symbols"])  # list[str]
        freq = str(data["data_frequency"])        # e.g., 'daily'
    except KeyError as e:
        raise KeyError(f"Missing required config key in {p}: {e}") from e

    if not symbols:
        raise ValueError("universe_symbols must contain at least one symbol")

    return Settings(start_date=start, end_date=end, universe_symbols=symbols, data_frequency=freq)


def _to_date(s: str | date) -> date:
    if isinstance(s, date):
        return s
    return date.fromisoformat(str(s))
