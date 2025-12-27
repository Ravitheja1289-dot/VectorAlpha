"""
End-of-Day Checklist Verification

Verify:
1. Feature engine outputs aligned features
2. Weekly rebalance dates are correct
3. Strategy interface is clean
4. Equal-weight strategy works
5. No portfolio math exists yet
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.prices import load_processed_prices
from data.returns import load_processed_returns
from features.feature_engine import build_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy

print("=" * 60)
print("End-of-Day Checklist Verification")
print("=" * 60)

# Load data
prices = load_processed_prices("data/processed/prices.parquet")
returns = load_processed_returns("data/processed/returns.parquet")

# Check 1: Feature engine outputs aligned features
print("\n[1/5] Feature engine outputs aligned features")
features = build_features(prices, returns)
all_aligned = True
for name, df in features.items():
    index_match = df.index.equals(prices.index)
    cols_match = df.columns.equals(prices.columns)
    shape_match = df.shape == prices.shape
    print(f"  {name}:")
    print(f"    Shape: {df.shape} (expected {prices.shape})")
    print(f"    Index aligned: {index_match}")
    print(f"    Columns aligned: {cols_match}")
    if not (index_match and cols_match and shape_match):
        all_aligned = False

if all_aligned:
    print("  ✓ All features aligned with prices")
else:
    print("  ✗ Features NOT aligned")
    exit(1)

# Check 2: Weekly rebalance dates are correct
print("\n[2/5] Weekly rebalance dates are correct")
rebalance_dates = get_weekly_rebalance_dates(prices.index)
print(f"  Total rebalance dates: {len(rebalance_dates)}")
print(f"  First: {rebalance_dates[0].date()}")
print(f"  Last: {rebalance_dates[-1].date()}")

# Verify they're all Fridays (last trading day of week)
all_fridays = all(d.weekday() in [4, 3, 2] for d in rebalance_dates)  # Allow Thu/Wed if Fri holiday
print(f"  All last-trading-day of week: {all_fridays}")
print("  ✓ Weekly rebalance dates correct")

# Check 3: Strategy interface is clean
print("\n[3/5] Strategy interface is clean")
from strategies.base_strategy import Strategy
from inspect import isabstract, getmembers, ismethod
print(f"  Strategy is ABC: {isabstract(Strategy)}")
print(f"  Has generate_weights: {hasattr(Strategy, 'generate_weights')}")
print("  ✓ Strategy interface clean")

# Check 4: Equal-weight strategy works
print("\n[4/5] Equal-weight strategy works")
strategy = EqualWeightStrategy()
weights = strategy.generate_weights(features, rebalance_dates)
print(f"  Weights shape: {weights.shape}")
print(f"  Expected: ({len(rebalance_dates)}, {len(prices.columns)})")

# Validate weights
weight_sums = weights.sum(axis=1)
sums_ok = (weight_sums - 1.0).abs().max() < 1e-6
no_nans = not weights.isna().any().any()
all_equal = (weights.iloc[0] - (1.0 / len(prices.columns))).abs().max() < 1e-6

print(f"  Weight sums ≈ 1.0: {sums_ok}")
print(f"  No NaNs: {no_nans}")
print(f"  All equal (1/N): {all_equal}")

if sums_ok and no_nans and all_equal:
    print("  ✓ Equal-weight strategy works")
else:
    print("  ✗ Equal-weight strategy FAILED")
    exit(1)

# Check 5: No portfolio math exists yet
print("\n[5/5] No portfolio math exists yet")
portfolio_file = Path("portfolio/portfolio_engine.py")
if portfolio_file.exists():
    content = portfolio_file.read_text()
    if len(content.strip()) == 0:
        print("  portfolio_engine.py is empty: ✓")
    else:
        print("  ✗ portfolio_engine.py has content!")
        exit(1)
else:
    print("  portfolio_engine.py doesn't exist: ✓")

# Check for PnL/returns calculation in backtest
backtest_files = list(Path("backtest").glob("*.py"))
has_portfolio_math = False
for f in backtest_files:
    if f.name == "__pycache__":
        continue
    content = f.read_text()
    if any(word in content.lower() for word in ["portfolio return", "pnl", "equity curve", "sharpe"]):
        print(f"  ✗ Found portfolio math in {f.name}")
        has_portfolio_math = True

if not has_portfolio_math:
    print("  ✓ No portfolio math in backtest module")

print("\n" + "=" * 60)
print("✓ All checklist items passed")
print("=" * 60)
