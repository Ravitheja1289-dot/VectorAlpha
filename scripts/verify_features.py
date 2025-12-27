from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.loader import load_settings
from features.feature_engine import build_features


def main() -> int:
    settings = load_settings()
    prices = pd.read_parquet(settings.paths.prices_file, engine="pyarrow")
    returns = pd.read_parquet(settings.paths.returns_file, engine="pyarrow")

    feats = build_features(prices, returns, window=20)
    for name, df in feats.items():
        print(f"{name}: shape={df.shape}, dates={df.index.min().date()}→{df.index.max().date()}")
        # Alignment checks
        assert df.index.equals(prices.index)
        assert df.columns.equals(prices.columns)
        assert df.shape == prices.shape
    print("\nFeatures verified: aligned and same shape as prices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
