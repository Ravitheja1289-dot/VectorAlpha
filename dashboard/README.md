# Vector Alpha Dashboard

Interactive Portfolio Lab for Finance Students.

## Architecture

```
app.py                    # Main Streamlit orchestrator
config.py                 # Constants, colors, asset metadata, educational text
engine.py                 # Compute bridge: UI inputs -> backend modules -> results
insights_engine.py        # Plain-English insight generator + diversification score
plotting.py               # Clean Plotly chart library

section_build.py          # Tab 1: Build Your Portfolio (inputs)
section_results.py        # Tab 2: See What Happened (KPIs, charts, highlights)
section_insights.py       # Tab 3: What This Means (educational insights)
tab_risk_explorer.py      # Tab 4: Risk Explorer (correlation, diversification)
tab_return_attribution.py # Tab 5: Return Breakdown (asset contributions)
tab_advanced.py           # Tab 6: Advanced Mode (hidden by default)
```

## Data Flow

```
User Input (assets, weights, dates, frequency)
  -> engine.run_portfolio_lab()
    -> backtest/rebalance.py (get rebalance dates)
    -> execution/executor.py (weight drift + costs)
    -> portfolio/portfolio_engine.py (PnL + equity curve)
    -> risk/metrics.py (CAGR, Sharpe, drawdown, etc.)
    -> risk/attribution.py (return + risk decomposition)
  -> results dict
    -> insights_engine.py (plain-English insights)
    -> plotting.py (Plotly figures)
    -> Streamlit render
```

## Running

```bash
streamlit run dashboard/app.py
```
