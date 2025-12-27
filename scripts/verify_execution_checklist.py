"""
End-of-Day Checklist for Execution Module

Verify:
1. Daily weights exist for all 1505 days
2. Weights drift between rebalances
3. Weights reset on rebalance dates
4. Turnover computed correctly
5. Transaction costs applied only on rebalance

If even one fails → fix before moving on.
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
print("Execution Module - End-of-Day Checklist")
print("=" * 60)

# Load data
prices = load_processed_prices("data/processed/prices.parquet")
returns = load_processed_returns("data/processed/returns.parquet")
features = build_features(prices, returns)
rebalance_dates = get_weekly_rebalance_dates(prices.index)
strategy = EqualWeightStrategy()
target_weights = strategy.generate_weights(features, rebalance_dates)

# Execute strategy
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

print("\n" + "=" * 60)
print("CHECKLIST VERIFICATION")
print("=" * 60)

# Check 1: Daily weights exist for all 1505 days
print("\n[1/5] Daily weights exist for all 1505 days")
print(f"  Prices shape: {prices.shape}")
print(f"  Daily weights shape: {daily_weights.shape}")
if daily_weights.shape[0] == prices.shape[0]:
    print("  ✓ PASS: Daily weights cover all trading days")
    check1 = True
else:
    print(f"  ✗ FAIL: Expected {prices.shape[0]} days, got {daily_weights.shape[0]}")
    check1 = False

# Check 2: Weights drift between rebalances
print("\n[2/5] Weights drift between rebalances")
# Pick a period between two rebalances and check if weights change
first_rebal_idx = 0
second_rebal_idx = 1
first_rebal = rebalance_dates[first_rebal_idx]
second_rebal = rebalance_dates[second_rebal_idx]

# Get dates between these two rebalances
mask = (daily_weights.index > first_rebal) & (daily_weights.index < second_rebal)
between_dates = daily_weights.index[mask]

if len(between_dates) > 0:
    # Check if weights change during this period
    first_asset = daily_weights.columns[0]
    weights_between = daily_weights.loc[between_dates, first_asset]
    weight_changes = weights_between.diff().abs()[1:]
    
    if weight_changes.sum() > 0:
        print(f"  Sample period: {between_dates[0].date()} to {between_dates[-1].date()}")
        print(f"  Weight changes detected: {weight_changes.sum():.6f}")
        print("  ✓ PASS: Weights drift between rebalances")
        check2 = True
    else:
        print("  ✗ FAIL: No weight drift detected")
        check2 = False
else:
    print("  ✗ FAIL: No dates between rebalances")
    check2 = False

# Check 3: Weights reset on rebalance dates
print("\n[3/5] Weights reset on rebalance dates")
# Check that weights on rebalance dates match target weights
rebalance_weights = daily_weights.loc[rebalance_dates]
max_deviation = (rebalance_weights - target_weights).abs().max().max()
print(f"  Max deviation from target on rebalance: {max_deviation:.2e}")
if max_deviation < 1e-10:
    print("  ✓ PASS: Weights reset to target on rebalance dates")
    check3 = True
else:
    print("  ✗ FAIL: Weights don't match targets on rebalance")
    check3 = False

# Check 4: Turnover computed correctly
print("\n[4/5] Turnover computed correctly")
# Turnover should be non-negative and reasonable magnitude
turnover_positive = (turnover >= 0).all()
turnover_reasonable = turnover.max() < 2.0  # 200% turnover is extreme
turnover_has_values = len(turnover) == len(rebalance_dates)

print(f"  Turnover entries: {len(turnover)} (expected {len(rebalance_dates)})")
print(f"  All non-negative: {turnover_positive}")
print(f"  Mean turnover: {turnover[1:].mean():.4f}")
print(f"  Max turnover: {turnover.max():.4f}")

if turnover_positive and turnover_reasonable and turnover_has_values:
    print("  ✓ PASS: Turnover computed correctly")
    check4 = True
else:
    print("  ✗ FAIL: Turnover issues detected")
    check4 = False

# Check 5: Transaction costs applied only on rebalance
print("\n[5/5] Transaction costs applied only on rebalance")
# Transaction costs should have entries only for rebalance dates
costs_on_rebalance = len(transaction_costs) == len(rebalance_dates)
costs_positive = (transaction_costs >= 0).all()
costs_index_match = transaction_costs.index.equals(turnover.index)

print(f"  Cost entries: {len(transaction_costs)} (expected {len(rebalance_dates)})")
print(f"  All non-negative: {costs_positive}")
print(f"  Index matches turnover: {costs_index_match}")
print(f"  Mean cost: {transaction_costs[1:].mean():.6f}")

if costs_on_rebalance and costs_positive and costs_index_match:
    print("  ✓ PASS: Transaction costs applied only on rebalance")
    check5 = True
else:
    print("  ✗ FAIL: Transaction cost issues detected")
    check5 = False

# Final result
print("\n" + "=" * 60)
all_pass = check1 and check2 and check3 and check4 and check5
if all_pass:
    print("✓ ALL CHECKS PASSED")
    print("=" * 60)
    print("\nExecution module ready.")
else:
    print("✗ SOME CHECKS FAILED")
    print("=" * 60)
    print("\n⚠ FIX BEFORE MOVING ON")
    exit(1)
