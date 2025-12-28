# Vector Alpha

**Institutional-Grade Multi-Asset Portfolio Backtesting and Risk Attribution Framework**

---

## Overview

Vector Alpha is a research and evaluation framework for quantitative portfolio strategies. It provides end-to-end infrastructure from raw price data to attribution-level analytics, with emphasis on execution realism, accounting correctness, and deterministic reproducibility.

**This is not a trading system.** Vector Alpha is designed for:
- Strategy research and prototyping
- Portfolio construction evaluation
- Risk decomposition and attribution analysis
- Methodology validation and peer review

The framework implements weekly portfolio rebalancing with daily weight drift, explicit transaction cost modeling, lagged-weight accounting (no look-ahead bias), and covariance-based risk attribution. All outputs are deterministic and auditable.

**Key Links:**
- [Research Paper (PDF)](research_paper/VectorAlpha.pdf)
- [Interactive Dashboard](https://vectoralpha.streamlit.app) *(read-only visualization)*
- [GitHub Repository](https://github.com/Ravitheja1289-dot/VectorAlpha)

---

## System Architecture

The framework is structured as a modular pipeline with strict separation of concerns:

### 1. Data Layer
- **Input**: Raw CSV files containing OHLCV data (audit layer)
- **Processing**: Deterministic transformation to immutable Parquet files
- **Contracts**: `prices.parquet`, `returns.parquet` with aligned DatetimeIndex
- **Guarantees**: Idempotent regeneration; no data leakage; explicit NA handling

### 2. Feature Engineering
- Time-series feature construction on processed prices and returns
- Supports momentum indicators, volatility signals, mean reversion metrics
- Aligned to portfolio rebalance schedule; no forward-looking features

### 3. Strategy Interface
- Pluggable strategy API: input (date, prices, features)  output (target weights)
- **Baseline implementation**: Equal-weight allocation across 15 large-cap tech stocks
- Extensible to momentum, mean-reversion, factor-based, or ML-driven strategies
- Weekly rebalancing frequency (313 periods in sample)

### 4. Execution Engine
- **Drift modeling**: Daily weight evolution via asset return dynamics
- **Transaction costs**: Linear model (10 basis points per unit turnover)
- **Cost application**: Applied on rebalance days only
- **Renormalization**: Weights sum to 1.0 post-drift and post-cost
- **Constraints**: Long-only; no leverage; fully invested

### 5. Portfolio Accounting
- **PnL calculation**: Lagged weight convention: `return_t = Σ(w_{t-1}  r_t)`
- **Return decomposition**: Gross returns - transaction costs = net returns
- **Equity curve**: Iterative NAV calculation with compounding
- **No look-ahead bias**: All portfolio decisions use t-1 information

### 6. Risk & Attribution
- **Performance metrics**: CAGR, volatility, Sharpe ratio, maximum drawdown, rolling metrics
- **Return attribution**: Per-asset contribution decomposition (sums exactly to portfolio return)
- **Risk attribution**: Covariance-based volatility decomposition (sums to portfolio volatility)
- **Validation**: Float-precision sum checks; deterministic outputs
- **Persistence**: Parquet files for downstream analysis and visualization

### 7. Visualization Layer
- Streamlit dashboard for interactive exploration
- Read-only interface; no parameter tuning or live optimization
- Plotly-based charts: cumulative returns, drawdown analysis, attribution breakdowns, rolling metrics

---

## Methodology & Assumptions

### Portfolio Specification
- **Universe**: 15 large-cap technology and growth stocks (AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, etc.)
- **Rebalancing frequency**: Weekly (every Friday or last business day)
- **Sample period**: 313 rebalance periods (approximately 6 years of daily data)
- **Data frequency**: Daily close prices; simple returns (not log returns)

### Execution Model
- Weekly target weights drift daily via asset-specific returns
- Linear transaction cost model: 10 basis points  turnover on rebalance days
- Turnover = Σ|w_target - w_drift|
- Costs deducted from portfolio value before weight renormalization
- No intraday execution modeling; assumes close-to-close execution
- No market impact; no slippage beyond linear cost model

### Accounting Conventions
- **Lagged weight PnL**: Uses t-1 weights with t returns (standard practice)
- **Gross vs. net returns**: Gross return + cost = net return (accounting identity)
- **No leverage**: Weights constrained to [0, 1]; sum to 1.0
- **Full investment**: No cash drag modeled separately

### Attribution Framework
- **Return attribution**: `Contribution_i = w_{i,t-1}  r_{i,t}` (sums to portfolio return)
- **Risk attribution**: Covariance-based decomposition: `Risk_i = w_i  (Σw)_i / σ_p`
- **Validation**: Daily sum checks with float-precision tolerance (max error: 0.00e+00)

---

## Validation & Sanity Checks

The framework enforces multiple layers of validation to ensure correctness:

### Data Integrity
- Index alignment across all dataframes (prices, returns, weights, attribution)
- No NaN or Inf values in critical calculation paths
- Type enforcement (float64 for prices/returns; DatetimeIndex for time series)
- Deterministic output: same input data  same results (no randomness)

### Accounting Correctness
- **Lagged weight validation**: Portfolio return matches Σ(w_{t-1}  r_t) within float tolerance
- **Cost reconciliation**: Net return = gross return - transaction costs
- **Weight normalization**: Daily weights sum to 1.0  1e-10 after drift and costs
- **Non-negativity**: Turnover and costs are strictly non-negative

### Attribution Accuracy
- **Return attribution**: Asset contributions sum exactly to portfolio return (verified daily)
- **Risk attribution**: Asset risk contributions sum to portfolio volatility (covariance-based)
- **Float-precision checks**: Maximum summation error tracked and reported (typically 0.00e+00)

### Reproducibility
- Processed data files (`prices.parquet`, `returns.parquet`) are immutable
- Attribution outputs regenerate identically from raw data
- No wall-clock time dependencies; no external API calls during backtest
- Idempotent pipeline: running twice produces identical results

---

## Key Findings

The baseline equal-weight strategy was evaluated over the full sample period with the following results:

### Aggregate Performance
- **CAGR**: 27.4%
- **Annualized Volatility**: 29.7%
- **Sharpe Ratio**: 0.97 (assuming risk-free rate  2%)
- **Maximum Drawdown**: -45.0%
- **Drawdown Duration**: Prolonged decline through 2022

### 2022 Drawdown Analysis
The framework captured a significant stress period in 2022, driven by:
- **Correlation concentration**: Diversification failed as tech sector correlations spiked to ~0.9
- **Concentrated exposure**: 15-stock equal-weight portfolio lacks sector diversification
- **Volatility amplification**: High-beta names (NVDA, TSLA) dominated both return drag and risk contribution
- **Top risk contributors**: NVDA, TSLA, META (high covariance + elevated standalone volatility)
- **Modest diversifiers**: ORCL provided limited downside protection during same period

### Attribution Insights
- Return attribution precisely isolates per-asset P&L contribution (verified to sum exactly to portfolio return)
- Risk attribution shows volatility contribution  risk contribution under equal-weight
- 2022 case study demonstrates limitations of naive diversification under correlation breakdown

### Strategy Limitations
- Equal-weight baseline has no adaptive capacity (no correlation hedging, no volatility targeting)
- Static universe ignores sector rotation and changing market structure
- No factor exposure management (momentum, value, quality, low-vol)

---

## What This Framework Does NOT Do

To set appropriate expectations:

- **No live trading**: This is a research tool, not an execution system
- **No optimization**: No parameter tuning, walk-forward optimization, or hyperparameter search
- **No factor attribution**: Does not decompose returns into Fama-French or custom factor exposures
- **No dynamic universe**: Static 15-stock set; no additions/removals during sample period
- **No intraday modeling**: Daily close-to-close execution only
- **No market impact**: Linear cost model; no price impact, liquidity constraints, or adverse selection
- **No stress testing**: No scenario analysis, Monte Carlo, or tail risk simulation
- **No regime detection**: No explicit bull/bear/crisis regime classification
- **No forecasting**: Strategies use historical data only; no predictive modeling in baseline

These are deliberate design choices to maintain simplicity, auditability, and reproducibility.

---

## Repository Structure

```
quant-backtesting-engine/

 backtest/
    backtester.py          # Core backtesting engine
    rebalance.py            # Weekly rebalancing logic with drift and costs

 config/
    loader.py               # Configuration loader
    settings.yaml           # Universe, parameters, paths

 data/
    loaders/
       base_loader.py      # Abstract data loader interface
       yahoo_loader.py     # Yahoo Finance data fetcher
    raw/                    # Raw CSV files (audit layer)
    processed/              # Immutable Parquet files
    prices.py               # Price data processing
    returns.py              # Return calculation

 dashboard/
    app.py                  # Streamlit main app
    components_*.py         # Page components (overview, performance, attribution, risk)
    data_loader.py          # Dashboard data loader
    utils_*.py              # Helper functions and plotting utilities
    requirements.txt        # Dashboard-specific dependencies

 execution/
    executor.py             # Execution engine (drift + costs)
    costs.py                # Transaction cost modeling
    slippage.py             # Slippage estimation (unused in baseline)

 features/
    feature_engine.py       # Feature engineering pipeline

 portfolio/
    portfolio_engine.py     # Portfolio accounting and PnL calculation

 research_paper/
    VectorAlpha.pdf         # LaTeX research paper with full methodology

 risk/
    metrics.py              # Risk metrics (vol, Sharpe, drawdown, rolling stats)
    attribution.py          # Return and risk attribution logic

 scripts/
    save_raw.py             # Download raw data from Yahoo Finance
    compute_attribution.py  # Compute and persist attribution
    sanity_check.py         # Validation script

 strategies/
    base_strategy.py        # Abstract strategy interface
    equal_weight.py         # Equal-weight baseline
    momentum.py             # Momentum strategy (placeholder)
    mean_reversion.py       # Mean reversion strategy (placeholder)

 universe/
    universe.py             # Universe definition and filtering

 validation/
    preflight.py            # Pre-execution validation checks
    walk_forward.py         # Walk-forward validation (future)

 visualization/
    plots.py                # Matplotlib/Plotly plotting utilities
    attribution_plots.py    # Attribution-specific charts

 main.py                     # CLI entry point for backtests
 run_experiment.py           # Full pipeline execution
 README.md                   # This file
```

---

## Reproducibility

Vector Alpha is designed for deterministic, reproducible research:

### Deterministic Pipeline
- No random number generation (no `random.seed()` or `np.random`)
- No wall-clock time dependencies (no `datetime.now()` in calculations)
- No external API calls during backtest execution (data fetch is separate step)
- Fixed universe; no dynamic filtering based on run-time conditions

### Immutable Data Contracts
- Raw CSVs in `data/raw/` serve as audit layer (never modified)
- Processed Parquet files in `data/processed/` are regenerated from raw only
- Attribution outputs (`return_attribution.parquet`, `risk_attribution.parquet`) are deterministic

### Regeneration
To fully regenerate all outputs from raw data:
```bash
python scripts/save_raw.py          # Fetch raw data (if needed)
python run_experiment.py            # Regenerate processed data, run backtest, compute attribution
python scripts/sanity_check.py      # Validate accounting and attribution
streamlit run dashboard/app.py      # Launch dashboard
```

Running the pipeline twice will produce byte-identical Parquet files (modulo Parquet metadata timestamps).

### Version Control
- All configuration in `config/settings.yaml` (universe, costs, frequency)
- Git history provides audit trail for code and configuration changes
- Research paper (LaTeX) versioned alongside code

---

## Dashboard

The framework includes a Streamlit-based dashboard for visualization and exploration:

**Access**: [https://vectoralpha.streamlit.app](https://vectoralpha.streamlit.app)

### Features
- **Overview**: System architecture, methodology summary, key performance indicators
- **Performance**: Cumulative returns, return distribution, statistical summary
- **Risk & Drawdowns**: Underwater plot, rolling volatility, rolling Sharpe ratio
- **Attribution**: Asset-level return and risk contribution, time-series and aggregate views

### Important Notes
- **Read-only**: No parameter tuning, strategy switching, or optimization in the UI
- **Visualization only**: Dashboard displays pre-computed results from Parquet files
- **No live data**: All charts reflect historical backtest results (not real-time)
- **No investment advice**: For research and educational purposes only

The dashboard is a communication tool for presenting results to stakeholders (PMs, risk teams, researchers). It is not a trading interface.

---

## Limitations & Future Work

### Current Limitations

**Execution Model:**
- Linear cost model (no market impact or liquidity constraints)
- No intraday execution modeling (close-to-close assumption)
- No slippage beyond fixed transaction costs
- No partial fills or execution risk

**Portfolio Construction:**
- Static universe (15 stocks; no additions/removals)
- Equal-weight baseline only (no adaptive strategies in production)
- No sector constraints, correlation caps, or exposure limits
- No dynamic leverage or cash management

**Attribution & Risk:**
- No factor-based attribution (Fama-French, custom factors)
- Costs not allocated to individual assets (portfolio-level only)
- No regime-specific risk metrics
- No stress testing or scenario analysis

**Data & Universe:**
- Single asset class (US equities only)
- No fixed income, commodities, currencies, or alternatives
- No survivorship bias correction (requires broader universe)
- No dividend or corporate action modeling

### Planned Extensions

**Phase 1: Enhanced Execution**
- Nonlinear cost models (square-root market impact)
- Intraday execution simulation (VWAP, TWAP benchmarks)
- Slippage estimation from bid-ask spreads
- Latency and delay modeling

**Phase 2: Factor Attribution**
- Fama-French 3-factor or 5-factor decomposition
- Custom factor exposures (momentum, quality, low-vol)
- Time-varying factor loadings
- Factor timing analysis

**Phase 3: Advanced Strategies**
- Momentum and mean-reversion implementations
- Covariance-based optimization (minimum variance, risk parity)
- Volatility targeting and dynamic leverage
- Correlation hedging and sector rotation

**Phase 4: Risk & Validation**
- Stress testing framework (historical scenarios, hypothetical shocks)
- Monte Carlo simulation for confidence intervals
- Walk-forward optimization with out-of-sample validation
- Regime detection and regime-conditional metrics

**Phase 5: Multi-Asset Extension**
- Fixed income (bonds, rates)
- Commodities and currencies
- Alternative risk premia strategies
- Cross-asset correlation modeling

---

## Usage

### Prerequisites
- Python 3.10 or higher
- Virtual environment recommended

### Installation
```bash
git clone https://github.com/Ravitheja1289-dot/VectorAlpha.git
cd VectorAlpha
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r dashboard/requirements.txt
```

### Run Pipeline
```bash
# Fetch raw data (if not already present)
python scripts/save_raw.py

# Run full backtest and attribution
python run_experiment.py

# Validate accounting and attribution
python scripts/sanity_check.py

# Launch dashboard
streamlit run dashboard/app.py
```

### Configuration
Edit `config/settings.yaml` to modify:
- Asset universe
- Rebalancing frequency
- Transaction cost assumptions
- Date ranges

---

## Research Paper

The complete methodology, mathematical derivations, and empirical results are documented in the accompanying research paper:

**[VectorAlpha.pdf](research_paper/VectorAlpha.pdf)**

The paper includes:
- Formal definitions of return and risk attribution
- Proof of summation properties (return attribution sums to portfolio return)
- Detailed analysis of 2022 drawdown
- Comparison with alternative execution models
- Discussion of limitations and modeling assumptions
- Recommendations for portfolio managers

---

## Citation

If you use this framework in academic or professional research, please cite:

```
@misc{vectoralpha2025,
  title={Vector Alpha: Institutional-Grade Multi-Asset Portfolio Backtesting Framework},
  author={[Ravi Teja Reddy Shyamala]},
  year={2025},
  url={https://github.com/Ravitheja1289-dot/VectorAlpha}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

**This software is provided for research and educational purposes only.**

- Vector Alpha is not a trading system and provides no investment recommendations.
- Past performance (simulated or actual) does not guarantee future results.
- No warranty or guarantee of correctness, profitability, or fitness for any purpose.
- Users assume all responsibility for validating methodology and results.
- Not suitable for production trading without extensive additional validation.

Use at your own risk. Consult qualified financial professionals before making investment decisions.

---

## Contact

For questions, contributions, or collaboration:
- **GitHub Issues**: [https://github.com/Ravitheja1289-dot/VectorAlpha/issues](https://github.com/Ravitheja1289-dot/VectorAlpha/issues)
- **Email**: ravithejareddy1289@gmail.com
- **Dashboard**: [https://vectoralpha.streamlit.app](https://vectoralpha.streamlit.app)

Contributions, bug reports, and methodology discussions are welcome.

---

**Target Audience:** Quantitative researchers, portfolio managers, risk analysts, and quant developers seeking a transparent, auditable framework for strategy research and attribution analysis.
