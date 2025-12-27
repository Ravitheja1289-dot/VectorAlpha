"""
Execution Module Integration Test

Test execution with equal-weight strategy:
- Turnover should be low and stable
- Daily weights should drift slightly, then reset weekly
- Costs should be predictable

If equal-weight behaves strangely, execution is broken.
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
from execution.executor import execute_strategy

print("=" * 60)
print("Execution Module Integration Test (Equal-Weight)")
print("=" * 60)

# Load data
print("\n[1/4] Loading data...")
prices = load_processed_prices("data/processed/prices.parquet")
returns = load_processed_returns("data/processed/returns.parquet")
print(f"  Prices: {prices.shape}")
print(f"  Returns: {returns.shape}")

# Generate features and weights
print("\n[2/4] Generating strategy weights...")
features = build_features(prices, returns)
rebalance_dates = get_weekly_rebalance_dates(prices.index)
strategy = EqualWeightStrategy()
target_weights = strategy.generate_weights(features, rebalance_dates)
print(f"  Target weights: {target_weights.shape}")
print(f"  Rebalance dates: {len(rebalance_dates)}")

# Execute strategy
print("\n[3/4] Executing strategy...")
execution_output = execute_strategy(
    prices=prices,
    returns=returns,
    target_weights=target_weights,
    rebalance_dates=rebalance_dates,
    cost_bps=10.0,
)

daily_weights = execution_output['daily_weights']
turnover = execution_output['turnover']
transaction_costs = execution_output['transaction_costs']

# Validate results
print("\n[4/4] Validating execution output...")

# Check shapes (daily_weights aligned to returns, not original prices)
print(f"  Daily weights shape: {daily_weights.shape}")
print(f"  Expected: {returns.shape} (aligned to returns)")
assert daily_weights.shape == returns.shape, "Daily weights shape mismatch"

print(f"  Turnover length: {len(turnover)}")
print(f"  Expected: {len(rebalance_dates)}")
assert len(turnover) == len(rebalance_dates), "Turnover length mismatch"

print(f"  Transaction costs length: {len(transaction_costs)}")
assert len(transaction_costs) == len(rebalance_dates), "Transaction costs length mismatch"

# Check turnover behavior for equal-weight
print("\n  Equal-weight turnover analysis:")
print(f"    Mean turnover: {turnover[1:].mean():.4f}")  # Skip first (0.0)
print(f"    Std turnover: {turnover[1:].std():.4f}")
print(f"    Min turnover: {turnover[1:].min():.4f}")
print(f"    Max turnover: {turnover[1:].max():.4f}")

# For equal-weight, turnover should be relatively low
# (weights drift slightly, then reset to 1/N)
assert turnover[1:].mean() < 0.1, "Equal-weight turnover too high (mean > 10%)"
print("    ✓ Turnover is low and stable (expected for equal-weight)")

# Check transaction costs
mean_cost = transaction_costs[1:].mean()
print(f"\n  Transaction cost analysis:")
print(f"    Mean cost: {mean_cost:.6f} ({mean_cost * 10000:.2f} bps)")
print(f"    Total costs (all rebalances): {transaction_costs.sum():.6f}")

# Check daily weight drift
print("\n  Daily weight drift analysis:")
# Pick a week and show weight evolution
week_start_idx = 10  # Some arbitrary week
week_end_idx = week_start_idx + 5
week_dates = prices.index[week_start_idx:week_end_idx]
week_weights = daily_weights.loc[week_dates]

print(f"    Sample week ({week_dates[0].date()} to {week_dates[-1].date()}):")
print(f"    First asset weights:")
first_asset = prices.columns[0]
for date, weight in week_weights[first_asset].items():
    print(f"      {date.date()}: {weight:.6f}")

# Check for drift: weights should change slightly day-to-day
weight_changes = week_weights[first_asset].diff().abs()[1:]
assert weight_changes.max() < 0.01, "Weight drift too large between days"
assert weight_changes.sum() > 0, "No weight drift detected (should drift slightly)"
print(f"    ✓ Weights drift slightly between rebalances (max change: {weight_changes.max():.6f})")

# Final validation
print("\n" + "=" * 60)
print("✓ Execution module integration test passed")
print("=" * 60)
print("\nKey results:")
print(f"  - Daily weights: {daily_weights.shape[0]} days")
print(f"  - Mean turnover: {turnover[1:].mean():.4f}")
print(f"  - Mean transaction cost: {mean_cost * 10000:.2f} bps")
print(f"  - Total transaction costs: {transaction_costs.sum():.4f}")
