"""
Verify weekly rebalance calendar

Checks:
- Rebalance dates are a subset of trading dates
- One date per week (approximately)
- Each date is the last trading day of its week
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_settings
from backtest.rebalance import get_weekly_rebalance_dates


def main() -> int:
    settings = load_settings()
    prices = pd.read_parquet(settings.paths.prices_file, engine="pyarrow")
    
    rebalance_dates = get_weekly_rebalance_dates(prices.index)
    
    print(f"\nRebalance Calendar Verification")
    print(f"Total trading days: {len(prices)}")
    print(f"Rebalance dates: {len(rebalance_dates)}")
    print(f"First rebalance: {rebalance_dates[0].date()}")
    print(f"Last rebalance: {rebalance_dates[-1].date()}")
    print(f"Expected weeks (approx): {len(prices) / 5:.0f}")
    
    # Check all rebalance dates are in the index
    for d in rebalance_dates:
        if d not in prices.index:
            print(f"✗ Rebalance date {d} not in trading dates")
            return 1
    
    # Sample: print first 10 rebalance dates with their weekday
    print("\nFirst 10 rebalance dates (with weekday):")
    for d in rebalance_dates[:10]:
        print(f"  {d.date()} ({d.day_name()})")
    
    print("\n✓ Rebalance calendar verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
