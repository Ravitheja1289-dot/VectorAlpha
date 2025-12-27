"""
Base interface for raw market data loaders.

Contract:
- Takes a symbol and start/end dates.
- Returns a raw OHLCV DataFrame from the source.

Design rules (enforced by convention):
- No file saving.
- No column renaming.
- No date fixing.
- No feature creation.

This module defines a minimal abstract base class and a callable protocol so
loaders can be implemented either as classes or simple functions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any, Protocol

__all__ = ["BaseDataLoader", "LoaderFn"]


class BaseDataLoader(ABC):
	"""
	Abstract base class for raw OHLCV data fetching.

	Implementations should fetch data directly from the upstream source and
	return it unchanged (raw), suitable for downstream processing.

	Parameters
	----------
	symbol : str
		Asset ticker to fetch.
	start : datetime.date | datetime.datetime
		Inclusive start date for data.
	end : datetime.date | datetime.datetime
		Inclusive end date for data.

	Returns
	-------
	Any
		A pandas DataFrame containing raw OHLCV data as provided by the source.
		Do not modify column names, timestamps, or add features in this layer.

	Notes
	-----
	- Validation should be minimal; avoid massaging data (no resampling, no
	  timezone normalization, etc.).
	- Downstream components are responsible for cleaning, renaming, and feature
	  engineering.
	"""

	@abstractmethod
	def fetch(self, symbol: str, start: date | datetime, end: date | datetime) -> Any:
		"""Fetch raw OHLCV data for `symbol` between `start` and `end` (inclusive)."""
		raise NotImplementedError


class LoaderFn(Protocol):
	"""
	Callable protocol for function-based loaders.

	Use this when implementing a loader as a plain function rather than a class.
	The same design rules apply: return raw OHLCV data without side effects.
	"""

	def __call__(self, symbol: str, start: date | datetime, end: date | datetime) -> Any:  # noqa: D401
		"""Fetch raw OHLCV data for `symbol` between `start` and `end` (inclusive)."""
		...

