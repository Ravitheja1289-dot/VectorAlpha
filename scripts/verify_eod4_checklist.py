"""
End-of-Day 4 Checklist Verification

Verify that all return calculation requirements are met.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data.returns import load_prices_for_returns, calculate_returns, load_processed_returns


def verify_eod4_checklist():
    """
    Verify all End-of-Day 4 requirements.
    
    Checklist:
    1. Returns are computed via pct_change
    2. First return date is correct
    3. Hand-calculated value matches
    4. No NaNs or infinities
    5. Saved as Parquet
    """
    
    print("=" * 70)
    print("END-OF-DAY 4 CHECKLIST VERIFICATION")
    print("=" * 70)
    
    all_passed = True
    
    # Load data
    print("\nLoading data...")
    prices = load_prices_for_returns()
    returns = calculate_returns(prices)
    
    # Check 1: Returns are computed via pct_change
    print("\n[1/5] Returns are computed via pct_change")
    print("      Method: Verify by checking formula r_t = (P_t / P_{t-1}) - 1")
    
    # Pick a sample date and asset
    sample_asset = prices.columns[0]
    sample_date = returns.index[10]  # Pick 11th return
    
    # Get prices at t and t-1
    t_idx = prices.index.get_loc(sample_date)
    price_t = prices.iloc[t_idx][sample_asset]
    price_t_minus_1 = prices.iloc[t_idx - 1][sample_asset]
    
    # Calculate expected return
    expected_return = (price_t / price_t_minus_1) - 1
    actual_return = returns.loc[sample_date, sample_asset]
    
    if abs(expected_return - actual_return) < 1e-10:
        print(f"      ✓ PASS - Formula matches pct_change() behavior")
        print(f"        Sample: {sample_asset} on {sample_date.date()}")
        print(f"        Expected: {expected_return:.10f}")
        print(f"        Actual:   {actual_return:.10f}")
        check1_pass = True
    else:
        print(f"      ✗ FAIL - Formula does NOT match")
        all_passed = False
        check1_pass = False
    
    # Check 2: First return date is correct
    print("\n[2/5] First return date is correct")
    first_price_date = prices.index[0]
    second_price_date = prices.index[1]
    first_return_date = returns.index[0]
    
    print(f"      First price date:  {first_price_date.date()}")
    print(f"      Second price date: {second_price_date.date()}")
    print(f"      First return date: {first_return_date.date()}")
    
    if first_return_date == second_price_date:
        print(f"      ✓ PASS - First return date = second price date")
        check2_pass = True
    else:
        print(f"      ✗ FAIL - First return date should be {second_price_date.date()}")
        all_passed = False
        check2_pass = False
    
    # Check 3: Hand-calculated value matches
    print("\n[3/5] Hand-calculated value matches")
    
    # Use first return as critical test
    asset = prices.columns[0]
    price_t_minus_1 = prices.iloc[0][asset]
    price_t = prices.iloc[1][asset]
    
    hand_calc = (price_t / price_t_minus_1) - 1
    dataframe_value = returns.iloc[0][asset]
    
    print(f"      Asset: {asset}")
    print(f"      Date: {returns.index[0].date()}")
    print(f"      Price t-1: ${price_t_minus_1:.6f}")
    print(f"      Price t:   ${price_t:.6f}")
    print(f"      Hand-calculated: {hand_calc:.10f}")
    print(f"      DataFrame value: {dataframe_value:.10f}")
    print(f"      Difference:      {abs(hand_calc - dataframe_value):.2e}")
    
    if abs(hand_calc - dataframe_value) < 1e-10:
        print(f"      ✓ PASS - Values match (within tolerance)")
        check3_pass = True
    else:
        print(f"      ✗ FAIL - Values do NOT match")
        all_passed = False
        check3_pass = False
    
    # Check 4: No NaNs or infinities
    print("\n[4/5] No NaNs or infinities")
    
    # Check NaNs
    nan_count = returns.isna().sum().sum()
    print(f"      NaN count: {nan_count}")
    
    if nan_count == 0:
        print(f"      ✓ PASS - No NaNs")
        nan_pass = True
    else:
        print(f"      ✗ FAIL - Found {nan_count} NaNs")
        nan_cols = returns.isna().sum()[returns.isna().sum() > 0]
        print(f"      Columns with NaNs: {nan_cols.to_dict()}")
        all_passed = False
        nan_pass = False
    
    # Check infinities
    inf_count = np.isinf(returns.values).sum()
    print(f"      Infinity count: {inf_count}")
    
    if inf_count == 0:
        print(f"      ✓ PASS - No infinities")
        inf_pass = True
    else:
        print(f"      ✗ FAIL - Found {inf_count} infinite values")
        all_passed = False
        inf_pass = False
    
    check4_pass = nan_pass and inf_pass
    
    # Check 5: Saved as Parquet
    print("\n[5/5] Saved as Parquet")
    parquet_path = Path("data/processed/returns.parquet")
    
    if parquet_path.exists():
        # Load from Parquet and verify it matches
        loaded_returns = load_processed_returns()
        
        print(f"      ✓ File exists: {parquet_path}")
        print(f"      File size: {parquet_path.stat().st_size / 1024:.2f} KB")
        
        # Verify loaded data matches calculated data
        if loaded_returns.shape == returns.shape:
            print(f"      ✓ Shape matches: {loaded_returns.shape}")
            shape_match = True
        else:
            print(f"      ✗ Shape mismatch: loaded={loaded_returns.shape}, calculated={returns.shape}")
            all_passed = False
            shape_match = False
        
        # Check if values match
        if np.allclose(loaded_returns.values, returns.values, rtol=1e-10):
            print(f"      ✓ Values match between file and calculation")
            values_match = True
        else:
            print(f"      ✗ Values don't match between file and calculation")
            all_passed = False
            values_match = False
        
        check5_pass = shape_match and values_match
    else:
        print(f"      ✗ FAIL - File not found: {parquet_path}")
        all_passed = False
        check5_pass = False
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    print(f"\n[1/5] Returns computed via pct_change:  {'✓ PASS' if check1_pass else '✗ FAIL'}")
    print(f"[2/5] First return date is correct:    {'✓ PASS' if check2_pass else '✗ FAIL'}")
    print(f"[3/5] Hand-calculated value matches:   {'✓ PASS' if check3_pass else '✗ FAIL'}")
    print(f"[4/5] No NaNs or infinities:           {'✓ PASS' if check4_pass else '✗ FAIL'}")
    print(f"[5/5] Saved as Parquet:                {'✓ PASS' if check5_pass else '✗ FAIL'}")
    
    if all_passed:
        print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nYou are DONE and ready to move on!")
    else:
        print("\n✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nFix the issues above before moving on.")
    
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    verify_eod4_checklist()
