# Vector Alpha Research Dashboard

**Read-only interactive research dashboard for institutional portfolio backtesting results.**

## Overview

The Vector Alpha Dashboard is a Streamlit-based visualization layer that enables institutional researchers to explore precomputed backtesting outputs, attribution analysis, and risk metrics. It enforces strict read-only constraints—no strategy recomputation, live data feeds, parameter optimization, or trading signals.

## Architecture

### Data Flow
```
run_experiment.py
    v
Precomputed Outputs:
  - data/processed/prices.parquet
  - data/processed/returns.parquet
  - data/processed/return_attribution.parquet
  - data/processed/risk_attribution.parquet
  - data/processed/risk_metrics.json
  - outputs/*.png (plots)
    v
dashboard/app.py
    v
Interactive Research Views
```

### Features

**Overview**
- Portfolio-level KPIs (CAGR, Volatility, Sharpe, Max Drawdown)
- System performance plots (equity curve, rolling Sharpe)
- Period summary

**Performance**
- Per-asset cumulative return curves
- Return statistics (mean, std, min, max, quantiles)
- Date range and asset filtering

**Attribution**
- Cumulative return attribution by asset
- 2022 drawdown drivers (NVDA, TSLA, META, ORCL analysis)
- Total return contribution ranking

**[WARNING] Risk Analysis**
- Per-asset risk contribution (volatility-based)
- Time-averaged risk metrics
- Risk evolution over time
- Risk attribution table (detailed view)

**System Info**
- Data inventory (row counts, asset counts)
- Dashboard configuration (constraints, data sources)
- Risk metrics summary
- Available assets list

### Filters & Navigation

- **Date Range Slider**: Filter analysis by period (e.g., 2022 drawdown period)
- **Asset Multiselect**: Focus on specific assets or all holdings
- **Page Navigation**: Switch between 5 main views

## Installation & Usage

### Prerequisites
- Python 3.8+
- Virtual environment with project dependencies installed
- Precomputed outputs from `python run_experiment.py`

### Setup

```bash
# Install dashboard dependencies
pip install -r dashboard/requirements.txt

# Run the dashboard
streamlit run dashboard/app.py
```

The dashboard will open at `http://localhost:8501`.

## Constraints & Design Principles

[OK] **What the Dashboard Does**
- Loads and visualizes precomputed equity curves, returns, attribution tables, and risk metrics
- Provides interactive filtering and multi-asset comparison
- Displays institutional-grade plots and tables
- Caches data for performance

[NO] **What the Dashboard Does NOT Do**
- Recompute strategy weights or portfolio allocations
- Simulate execution or calculate PnL
- Generate live market data feeds
- Optimize parameters or generate trading signals
- Perform backtesting or risk calculations

## Data Caching

All data loads are cached with 1-hour TTL (`@st.cache_data(ttl=3600)`). To force a refresh:
- Click **Settings** -> **Clear cache** in Streamlit sidebar, or
- Restart the app with `streamlit run dashboard/app.py`

## Key Insights

### 2022 Drawdown Analysis
The dashboard highlights the 45% maximum drawdown in 2022, driven by:
- **NVDA**: Largest return drag + concentrated risk
- **TSLA**: Significant losses + volatility spike
- **META**: Sector rotation losses
- **ORCL**: Diversification benefit (limiting downside)

### Attribution Validation
- Return attribution contributions sum to portfolio gross returns (within numerical precision)
- Risk attribution contributions sum to portfolio volatility
- Segmented analysis available for pre-2022, 2022, post-2022 periods

## Troubleshooting

**"Precomputed data not found"**
-> Run `python run_experiment.py` to generate `data/processed/` and `outputs/` directories.

**Data not updating**
-> Clear Streamlit cache: Settings -> Clear cache, or restart the app.

**Missing plots**
-> Ensure `outputs/equity_drawdown.png`, `outputs/rolling_sharpe.png`, etc. exist from `run_experiment.py`.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | >=1.28.0 | Dashboard framework |
| pandas | >=1.5.0 | Data manipulation |
| numpy | >=1.24.0 | Numerical computation |
| matplotlib | >=3.7.0 | Plotting |
| pyarrow | >=13.0.0 | Parquet file I/O |

## Future Enhancements

- Regime-based analysis (trending vs. ranging markets)
- Correlation heatmaps (asset cross-correlations)
- Leverage/concentration risk alerts
- Factor exposure breakdowns
- Custom date range comparison tool
- Export attribution tables to CSV

## Contributing

This is a read-only research layer. Changes should:
1. Not introduce computation (filtering/visualization only)
2. Not modify precomputed data
3. Add UI clarity or interactivity
4. Document any new data assumptions

## License

Institutional use only. Contact authors for commercial licensing.
