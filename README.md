# quant-backtesting-engine

A production-grade quantitative backtesting engine for portfolio strategies.

## Data Contract (IMMUTABLE)

This project maintains two distinct data layers with strict contracts:

### 1. Raw Data Layer
**Location:** `data/raw/*.csv`

**Contract:**
- ✓ **Never modified after download**
- ✓ **Audit layer** - human-readable, traceable
- ✓ Source of truth for all market data
- ✓ CSV format for transparency and inspection

**Purpose:**
- Preserve original data exactly as received
- Enable data quality audits
- Provide rollback capability
- Support reproducibility

### 2. Processed Data Layer
**Location:** 
- `data/processed/prices.parquet`
- `data/processed/returns.parquet`

**Contract:**
- ✓ **Derived from raw data only**
- ✓ **Safe to regenerate** at any time by running `main.py`
- ✓ **Never hand-edited** - always programmatically generated
- ✓ Parquet format for performance and efficiency

**Contents:**
- **prices.parquet**: Aligned, clean adjusted close prices (1505 days × 15 assets)
- **returns.parquet**: Simple returns calculated via pct_change (1504 days × 15 assets)

**Guarantees:**
- No NaNs in final output
- No infinite values
- Forward-filled only (never backward-filled)
- Strictly increasing dates
- Look-ahead bias free
- **Idempotent**: Running `main.py` multiple times produces identical outputs (byte-for-byte)

---

## Why This Matters

**Separation of Concerns:**
- Raw data = audit trail (CSV, human-readable)
- Processed data = performance layer (Parquet, optimized)

**Reproducibility:**
- Delete `data/processed/*` → Run `main.py` → Identical output guaranteed
- Run `main.py` twice → File hashes match → No hidden state

**Safety:**
- You can never accidentally corrupt raw data
- Processed data regeneration is always safe

**Idempotency:**
- Re-running `main.py` overwrites files cleanly
- No data duplication
- No row appending
- Outputs are deterministic
- No hidden state (verified via hash comparison)

**Data Flow:**
```
data/raw/*.csv  →  [data/prices.py]  →  data/processed/prices.parquet
                                      ↓
                   [data/returns.py] →  data/processed/returns.parquet
```

---

## Project Structure

```
data/
  raw/              # Raw CSVs (NEVER MODIFY)
  processed/        # Derived Parquet files (SAFE TO REGENERATE)
  prices.py         # Price alignment and cleaning
  returns.py        # Return calculation

config/
  settings.yaml     # Configuration
  loader.py         # Config loader

main.py             # Orchestration - builds clean market data layer
```

---

## Usage

**Generate clean market data:**
```bash
python main.py
```

This will:
1. Load raw CSVs from `data/raw/`
2. Align and clean prices
3. Calculate returns
4. Save to `data/processed/`

**Regenerate anytime:**
```bash
# Safe - just regenerates from raw data
rm data/processed/*.parquet
python main.py
```

---

## Data Pipeline Guarantees

1. **No Look-Ahead Bias**: Returns at time t use only prices at t and t-1
2. **No NaNs**: All missing data handled via forward-fill policy
3. **No Infinities**: Price data validated before return calculation
4. **Date Alignment**: All assets share identical trading days
5. **Simple Returns**: r_t = (P_t / P_{t-1}) - 1 (industry standard)

Run verification scripts:
```bash
python scripts/verify_checklist.py     # End-of-Day 3 checklist
python scripts/verify_eod4_checklist.py # End-of-Day 4 checklist
python scripts/verify_lookahead.py     # Look-ahead bias check
```