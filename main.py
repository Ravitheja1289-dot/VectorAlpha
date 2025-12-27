"""
Main entry point for quant backtesting engine.

Orchestration only - no logic leaked here.
"""

from config.loader import load_settings
from data.prices import load_price_matrix, save_price_matrix


def main() -> None:
    """
    Build and persist the aligned price matrix.
    
    Steps:
    1. Load config
    2. Load universe (from config)
    3. Load raw CSVs
    4. Build aligned price DataFrame
    5. Save Parquet
    6. Exit
    """
    print("=" * 60)
    print("QUANT BACKTESTING ENGINE - DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Load config
    print("\n[1/4] Loading configuration...")
    settings = load_settings()
    print(f"  ✓ Date range: {settings.start_date} to {settings.end_date}")
    print(f"  ✓ Universe: {len(settings.universe_symbols)} assets")
    
    # Steps 2-4: Load universe, load raw CSVs, build aligned price DataFrame
    print("\n[2/4] Loading and aligning price data...")
    prices = load_price_matrix(data_dir="data/raw", price_column="Adj Close")
    print(f"  ✓ Loaded {len(prices.columns)} assets")
    print(f"  ✓ Date range: {prices.index.min().date()} to {prices.index.max().date()}")
    print(f"  ✓ Total observations: {len(prices)}")
    
    # Step 5: Save Parquet
    print("\n[3/4] Persisting to Parquet...")
    save_price_matrix(prices, output_path="data/processed/prices.parquet")
    
    # Step 6: Exit
    print("\n[4/4] Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
