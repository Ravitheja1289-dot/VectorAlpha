"""
Pipeline Idempotency Test

Verify that running main.py multiple times produces identical outputs.

Test:
1. Delete data/processed/
2. Run main.py (first time)
3. Compute file hashes
4. Run main.py (second time)
5. Compute file hashes again
6. Compare - they should match

If outputs differ → hidden state exists (BUG)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import hashlib
import shutil
import subprocess


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def run_main_py():
    """Run main.py and return exit code."""
    result = subprocess.run(
        [sys.executable, "main.py"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    return result.returncode, result.stdout, result.stderr


def verify_idempotency():
    """
    Verify that main.py is idempotent.
    
    Returns True if idempotent, False otherwise.
    """
    
    print("=" * 70)
    print("PIPELINE IDEMPOTENCY TEST")
    print("=" * 70)
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    
    prices_file = processed_dir / "prices.parquet"
    returns_file = processed_dir / "returns.parquet"
    
    # Step 1: Delete data/processed/
    print("\n[Step 1] Deleting data/processed/...")
    if processed_dir.exists():
        for file in processed_dir.glob("*.parquet"):
            file.unlink()
            print(f"  Deleted: {file.name}")
    else:
        processed_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {processed_dir}")
    
    # Verify files are gone
    if prices_file.exists() or returns_file.exists():
        print("  ✗ FAIL - Files still exist after deletion")
        return False
    print("  ✓ Clean slate confirmed")
    
    # Step 2: Run main.py (first time)
    print("\n[Step 2] Running main.py (first time)...")
    exit_code, stdout, stderr = run_main_py()
    
    if exit_code != 0:
        print(f"  ✗ FAIL - main.py failed with exit code {exit_code}")
        print(f"  stderr: {stderr}")
        return False
    
    print("  ✓ First run completed successfully")
    
    # Verify files were created
    if not prices_file.exists() or not returns_file.exists():
        print("  ✗ FAIL - Files not created")
        return False
    
    # Step 3: Compute file hashes (first run)
    print("\n[Step 3] Computing file hashes (first run)...")
    prices_hash_1 = compute_file_hash(prices_file)
    returns_hash_1 = compute_file_hash(returns_file)
    
    prices_size_1 = prices_file.stat().st_size
    returns_size_1 = returns_file.stat().st_size
    
    print(f"  prices.parquet:")
    print(f"    Hash: {prices_hash_1[:16]}...")
    print(f"    Size: {prices_size_1} bytes")
    print(f"  returns.parquet:")
    print(f"    Hash: {returns_hash_1[:16]}...")
    print(f"    Size: {returns_size_1} bytes")
    
    # Step 4: Run main.py (second time)
    print("\n[Step 4] Running main.py (second time)...")
    exit_code, stdout, stderr = run_main_py()
    
    if exit_code != 0:
        print(f"  ✗ FAIL - main.py failed on second run with exit code {exit_code}")
        print(f"  stderr: {stderr}")
        return False
    
    print("  ✓ Second run completed successfully")
    
    # Step 5: Compute file hashes (second run)
    print("\n[Step 5] Computing file hashes (second run)...")
    prices_hash_2 = compute_file_hash(prices_file)
    returns_hash_2 = compute_file_hash(returns_file)
    
    prices_size_2 = prices_file.stat().st_size
    returns_size_2 = returns_file.stat().st_size
    
    print(f"  prices.parquet:")
    print(f"    Hash: {prices_hash_2[:16]}...")
    print(f"    Size: {prices_size_2} bytes")
    print(f"  returns.parquet:")
    print(f"    Hash: {returns_hash_2[:16]}...")
    print(f"    Size: {returns_size_2} bytes")
    
    # Step 6: Compare hashes
    print("\n[Step 6] Comparing outputs...")
    
    prices_match = prices_hash_1 == prices_hash_2
    returns_match = returns_hash_1 == returns_hash_2
    size_match = (prices_size_1 == prices_size_2) and (returns_size_1 == returns_size_2)
    
    print(f"\n  prices.parquet:")
    print(f"    Hashes match:  {'✓ YES' if prices_match else '✗ NO'}")
    print(f"    Sizes match:   {'✓ YES' if prices_size_1 == prices_size_2 else '✗ NO'}")
    
    print(f"\n  returns.parquet:")
    print(f"    Hashes match:  {'✓ YES' if returns_match else '✗ NO'}")
    print(f"    Sizes match:   {'✓ YES' if returns_size_1 == returns_size_2 else '✗ NO'}")
    
    # Final verdict
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    
    if prices_match and returns_match and size_match:
        print("\n✓✓✓ PIPELINE IS IDEMPOTENT ✓✓✓")
        print("\nVerified:")
        print("  • Re-running main.py overwrites files cleanly")
        print("  • No data duplication")
        print("  • No row appending")
        print("  • Outputs are identical (byte-for-byte)")
        print("  • No hidden state detected")
        return True
    else:
        print("\n✗✗✗ PIPELINE IS NOT IDEMPOTENT ✗✗✗")
        print("\nProblem: Hidden state detected!")
        print("Outputs differ between runs - this is a BUG")
        
        if not prices_match:
            print(f"\n  prices.parquet differs:")
            print(f"    Run 1 hash: {prices_hash_1}")
            print(f"    Run 2 hash: {prices_hash_2}")
        
        if not returns_match:
            print(f"\n  returns.parquet differs:")
            print(f"    Run 1 hash: {returns_hash_1}")
            print(f"    Run 2 hash: {returns_hash_2}")
        
        return False


if __name__ == "__main__":
    success = verify_idempotency()
    sys.exit(0 if success else 1)
