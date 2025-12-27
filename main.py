"""
Main entry point for quant backtesting engine.

Orchestration only - no logic leaked here.
"""

from config.loader import load_settings
from data.prices import load_price_matrix, save_price_matrix
from data.returns import calculate_returns, save_returns


def main() -> None:
    """
    Build and persist clean market data layer.
    
    Steps:
    1. Load config
    2. Load universe (from config)
    3. Load raw CSVs
    4. Build aligned price DataFrame
    5. Save prices to Parquet
    6. Build returns
    7. Save returns to Parquet
    8. Exit
    
    No strategies, no portfolio logic yet.
    """
    print("=" * 60)
    print("QUANT BACKTESTING ENGINE - DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load config
    print("\n[1/6] Loading configuration...")
    settings = load_settings()
    print(f"  + Date range: {settings.start_date} to {settings.end_date}")
    print(f"  + Universe: {len(settings.universe_symbols)} assets")
    
    # Steps 2-4: Load universe, load raw CSVs, build aligned price DataFrame
    print("\n[2/6] Loading and aligning price data...")
    prices = load_price_matrix(data_dir="data/raw", price_column="Adj Close")
    print(f"  + Loaded {len(prices.columns)} assets")
    print(f"  + Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  + Total observations: {len(prices)}")
    
    # Step 5: Save prices to Parquet
    print("\n[3/6] Persisting prices to Parquet...")
    save_price_matrix(prices, output_path="data/processed/prices.parquet")
    
    # Step 6: Build returns
    print("\n[4/6] Calculating returns...")
    returns = calculate_returns(prices)
    print(f"  + Calculated {len(returns)} returns")
    print(f"  + Date range: {returns.index.min().date()} to {returns.index.max().date()}")
    print(f"  + Return type: Simple returns (r_t = P_t / P_{{t-1}} - 1)")
    
    # Step 7: Save returns to Parquet
    print("\n[5/6] Persisting returns to Parquet...")
    save_returns(returns, output_path="data/processed/returns.parquet")
    
    # Step 8: Exit
    print("\n[6/6] Complete!")
    print("=" * 60)
    print("\nClean market data layer ready:")
    print("  - data/processed/prices.parquet")
    print("  - data/processed/returns.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()
