"""
Look-Ahead Bias Check (MANDATORY)

Verify that returns are calculated correctly:
- Return at date t uses prices at t and t-1
- No future prices leak into the calculation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.returns import load_prices_for_returns, calculate_returns


def verify_no_lookahead_bias():
    """
    Manual check to verify return calculation correctness.
    
    Steps:
    1. Pick one asset, one date
    2. Print price at t-1
    3. Print price at t
    4. Hand-calculate return: (P_t / P_{t-1}) - 1
    5. Compare with DataFrame value
    
    If mismatch → STOP
    """
    
    print("=" * 70)
    print("LOOK-AHEAD BIAS CHECK (MANDATORY)")
    print("=" * 70)
    
    # Load data
    print("\nLoading prices and returns...")
    prices = load_prices_for_returns()
    returns = calculate_returns(prices)
    
    print(f"Prices: {len(prices)} days, {len(prices.columns)} assets")
    print(f"Returns: {len(returns)} days, {len(returns.columns)} assets")
    print(f"Price dates: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Return dates: {returns.index.min().date()} to {returns.index.max().date()}")
    
    # Test 1: Pick a date in the middle of the dataset
    print("\n" + "=" * 70)
    print("TEST 1: Middle of dataset")
    print("=" * 70)
    
    asset = "AAPL"
    # Pick a date in the middle (index 500)
    t_idx = 500
    date_t = returns.index[t_idx]
    
    # Find t-1 in prices
    t_idx_in_prices = prices.index.get_loc(date_t)
    date_t_minus_1 = prices.index[t_idx_in_prices - 1]
    
    price_t_minus_1 = prices.loc[date_t_minus_1, asset]
    price_t = prices.loc[date_t, asset]
    return_dataframe = returns.loc[date_t, asset]
    
    # Hand-calculate return
    return_manual = (price_t / price_t_minus_1) - 1
    
    print(f"\nAsset: {asset}")
    print(f"Date t-1: {date_t_minus_1.date()}")
    print(f"Date t:   {date_t.date()}")
    print(f"\nPrice at t-1: ${price_t_minus_1:.6f}")
    print(f"Price at t:   ${price_t:.6f}")
    print(f"\nHand-calculated return: ({price_t:.6f} / {price_t_minus_1:.6f}) - 1 = {return_manual:.10f}")
    print(f"DataFrame return:                                           {return_dataframe:.10f}")
    print(f"\nDifference: {abs(return_manual - return_dataframe):.2e}")
    
    # Check if they match (within floating point tolerance)
    tolerance = 1e-10
    if abs(return_manual - return_dataframe) < tolerance:
        print("✓ PASS: Values match (within tolerance)")
        test1_pass = True
    else:
        print("✗ FAIL: Values DO NOT match!")
        print("STOP - Look-ahead bias or calculation error detected!")
        test1_pass = False
    
    # Test 2: First return date (most critical)
    print("\n" + "=" * 70)
    print("TEST 2: First return date (most critical)")
    print("=" * 70)
    
    asset = "MSFT"
    date_t = returns.index[0]  # First return date
    
    # Find t-1 in prices
    t_idx_in_prices = prices.index.get_loc(date_t)
    date_t_minus_1 = prices.index[t_idx_in_prices - 1]
    
    price_t_minus_1 = prices.loc[date_t_minus_1, asset]
    price_t = prices.loc[date_t, asset]
    return_dataframe = returns.loc[date_t, asset]
    
    # Hand-calculate return
    return_manual = (price_t / price_t_minus_1) - 1
    
    print(f"\nAsset: {asset}")
    print(f"Date t-1: {date_t_minus_1.date()}")
    print(f"Date t:   {date_t.date()}")
    print(f"\nPrice at t-1: ${price_t_minus_1:.6f}")
    print(f"Price at t:   ${price_t:.6f}")
    print(f"\nHand-calculated return: ({price_t:.6f} / {price_t_minus_1:.6f}) - 1 = {return_manual:.10f}")
    print(f"DataFrame return:                                           {return_dataframe:.10f}")
    print(f"\nDifference: {abs(return_manual - return_dataframe):.2e}")
    
    if abs(return_manual - return_dataframe) < tolerance:
        print("✓ PASS: Values match (within tolerance)")
        test2_pass = True
    else:
        print("✗ FAIL: Values DO NOT match!")
        print("STOP - Look-ahead bias or calculation error detected!")
        test2_pass = False
    
    # Test 3: Last return date
    print("\n" + "=" * 70)
    print("TEST 3: Last return date")
    print("=" * 70)
    
    asset = "NVDA"
    date_t = returns.index[-1]  # Last return date
    
    # Find t-1 in prices
    t_idx_in_prices = prices.index.get_loc(date_t)
    date_t_minus_1 = prices.index[t_idx_in_prices - 1]
    
    price_t_minus_1 = prices.loc[date_t_minus_1, asset]
    price_t = prices.loc[date_t, asset]
    return_dataframe = returns.loc[date_t, asset]
    
    # Hand-calculate return
    return_manual = (price_t / price_t_minus_1) - 1
    
    print(f"\nAsset: {asset}")
    print(f"Date t-1: {date_t_minus_1.date()}")
    print(f"Date t:   {date_t.date()}")
    print(f"\nPrice at t-1: ${price_t_minus_1:.6f}")
    print(f"Price at t:   ${price_t:.6f}")
    print(f"\nHand-calculated return: ({price_t:.6f} / {price_t_minus_1:.6f}) - 1 = {return_manual:.10f}")
    print(f"DataFrame return:                                           {return_dataframe:.10f}")
    print(f"\nDifference: {abs(return_manual - return_dataframe):.2e}")
    
    if abs(return_manual - return_dataframe) < tolerance:
        print("✓ PASS: Values match (within tolerance)")
        test3_pass = True
    else:
        print("✗ FAIL: Values DO NOT match!")
        print("STOP - Look-ahead bias or calculation error detected!")
        test3_pass = False
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if test1_pass and test2_pass and test3_pass:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
        print("\nVerified:")
        print("  • Return at date t uses prices at t and t-1")
        print("  • No future prices leak into calculation")
        print("  • Formula: r_t = (P_t / P_{t-1}) - 1 is correct")
        print("\nNo look-ahead bias detected!")
        return True
    else:
        print("\n✗✗✗ TESTS FAILED ✗✗✗")
        print("\nSTOP - Fix the calculation before proceeding!")
        return False


if __name__ == "__main__":
    verify_no_lookahead_bias()
