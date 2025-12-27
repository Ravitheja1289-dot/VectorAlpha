"""
Verify equal-weight strategy implementation

Checks:
- Weights shape matches (rebalance_dates × assets)
- All weights equal to 1/N
- Each row sums to 1.0
- No NaNs
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_settings
from features.feature_engine import build_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy


def main() -> int:
    settings = load_settings()
    prices = pd.read_parquet(settings.paths.prices_file, engine="pyarrow")
    returns = pd.read_parquet(settings.paths.returns_file, engine="pyarrow")
    
    # Build features
    features = build_features(prices, returns, window=20)
    
    # Get rebalance calendar
    rebalance_dates = get_weekly_rebalance_dates(prices.index)
    
    # Generate weights
    strategy = EqualWeightStrategy()
    weights = strategy.generate_weights(features, rebalance_dates)
    
    print("\nEqual Weight Strategy Verification")
    print(f"Weights shape: {weights.shape}")
    print(f"Rebalance dates: {len(rebalance_dates)}")
    print(f"Assets: {len(weights.columns)}")
    print(f"Expected weight per asset: {1.0 / len(weights.columns):.6f}")
    
    # Check shape
    if weights.shape[0] != len(rebalance_dates):
        print(f"✗ Row count mismatch: {weights.shape[0]} != {len(rebalance_dates)}")
        return 1
    
    if weights.shape[1] != len(prices.columns):
        print(f"✗ Column count mismatch: {weights.shape[1]} != {len(prices.columns)}")
        return 1
    
    # Check all weights are equal to 1/N
    expected_weight = 1.0 / len(weights.columns)
    if not (weights == expected_weight).all().all():
        print("✗ Not all weights equal to 1/N")
        return 1
    
    # Check each row sums to 1.0
    row_sums = weights.sum(axis=1)
    if not all(abs(row_sums - 1.0) < 1e-10):
        print(f"✗ Weights do not sum to 1.0: min={row_sums.min()}, max={row_sums.max()}")
        return 1
    
    # Check no NaNs
    if weights.isna().any().any():
        print(f"✗ Weights contain NaNs: {weights.isna().sum().sum()}")
        return 1
    
    # Sample first 5 rebalance dates
    print("\nFirst 5 rebalance dates (sample weights):")
    print(weights.head())
    
    print("\n✓ Equal-weight strategy verified")
    print("  - Weights shape correct")
    print("  - All weights = 1/N")
    print("  - Each row sums to 1.0")
    print("  - No NaNs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
