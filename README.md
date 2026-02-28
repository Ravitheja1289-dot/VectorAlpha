# Vector Alpha

**Interactive Portfolio Lab for Finance Students**

---

## What is Vector Alpha?

Vector Alpha is an interactive portfolio learning tool. Build a portfolio, simulate performance, and understand where returns and risks come from — visually and intuitively.

**It teaches through experimentation, not lectures.**

When you use this tool, the goal is for you to say:

> "I finally understand how portfolio mechanics actually work."

**Key Links:**
- [Interactive Dashboard](https://vectoralpha.streamlit.app)
- [GitHub Repository](https://github.com/Ravitheja1289-dot/VectorAlpha)
- [Research Paper (PDF)](research_paper/VectorAlpha.pdf) *(technical methodology)*

---

## Who is this for?

Finance students who:
- Are learning portfolio theory (CAPM, Modern Portfolio Theory)
- Want to visualize how diversification actually works
- Learn better through interaction than textbooks
- Want to understand risk attribution intuitively

---

## What can you do?

### 1. Build Your Portfolio
- Select from 15 real stocks (Apple, Microsoft, NVIDIA, Tesla, etc.)
- Set custom weights (or use equal weight)
- Choose a time period (2020–2025)
- Pick a rebalancing frequency: None, Monthly, Quarterly, or Yearly
- Toggle transaction costs on/off

### 2. See What Happened
- View your portfolio's equity curve (growth of $1)
- See the drawdown chart (worst declines)
- Get key metrics: Annualized Return, Volatility, Sharpe Ratio, Max Drawdown
- Identify the best performer, worst performer, and largest risk contributor

### 3. What This Means (Learning Insights)
The system generates plain-English explanations like:
- *"70% of your portfolio risk comes from just two assets."*
- *"Monthly rebalancing reduced weight drift by X%."*
- *"Your portfolio suffered most during high correlation regimes."*

Every insight answers three questions:
1. **What happened?**
2. **Why did it happen?**
3. **What concept does this demonstrate?**

### 4. Risk Explorer
- Correlation heatmap between your assets
- Risk contribution bar chart (which assets drive risk)
- Diversification score (0–10) with breakdown
- Teaches the difference between *real diversification* and *illusion of diversification*

### 5. Return Breakdown
- Contribution by asset (who helped, who hurt)
- Positive vs negative contributors
- Year-by-year attribution with narrative explanations

### 6. Advanced Mode (Hidden by Default)
Toggle to see:
- Weight drift over time
- Turnover at each rebalance
- Transaction cost impact
- Rolling volatility and Sharpe ratio
- Detailed attribution tables

---

## Concepts You'll Learn

| Concept | What the tool shows |
|---------|-------------------|
| Rebalancing effect | How resetting weights controls drift and costs |
| Concentration risk | When 2 assets dominate 70% of your risk |
| Diversification illusion | Holding 5 correlated stocks isn't diversified |
| Correlation spikes | Assets that move apart in calm markets crash together |
| Volatility clustering | Risk is not constant — it comes in waves |
| Risk vs return tradeoff | Higher returns usually require higher volatility |

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation
```bash
git clone https://github.com/Ravitheja1289-dot/VectorAlpha.git
cd VectorAlpha
pip install -r dashboard/requirements.txt
```

### Generate Data (first time only)
```bash
python scripts/save_raw.py
python run_experiment.py
```

### Launch the Dashboard
```bash
streamlit run dashboard/app.py
```

Open your browser to `http://localhost:8501` and start experimenting.

---

## Project Structure

```
VectorAlpha/
  dashboard/              # Interactive Streamlit app
    app.py                # Main entry point
    config.py             # Colors, asset info, educational text
    engine.py             # Compute bridge (UI -> backend)
    insights_engine.py    # Plain-English insight generator
    plotting.py           # Clean Plotly charts
    section_build.py      # Build Your Portfolio (inputs)
    section_results.py    # See What Happened (visualizations)
    section_insights.py   # What This Means (learning insights)
    tab_risk_explorer.py  # Risk Explorer tab
    tab_return_attribution.py  # Return Breakdown tab
    tab_advanced.py       # Advanced Mode (hidden by default)

  backtest/               # Rebalance calendar
  execution/              # Weight drift + transaction costs
  portfolio/              # PnL calculation + equity curve
  risk/                   # Metrics, attribution, factor models
  strategies/             # Strategy implementations
  data/raw/               # Historical price data (15 stocks)
  data/processed/         # Pre-computed Parquet files
```

---

## How It Works (Under the Hood)

The dashboard is interactive — when you click "Simulate Portfolio", it:

1. Loads historical prices for your selected assets
2. Generates rebalance dates at your chosen frequency
3. Simulates daily weight drift between rebalances
4. Applies transaction costs on rebalance days
5. Calculates portfolio returns using lagged weights (no look-ahead bias)
6. Computes risk metrics (CAGR, Sharpe, drawdown, etc.)
7. Decomposes return and risk attribution by asset
8. Generates plain-English insights from the results

All computations use real market data (2020–2025) and institutional-grade accounting.

---

## Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
pyarrow>=13.0.0
plotly>=5.0.0
scipy>=1.11.0
scikit-learn>=1.3.0
yfinance>=0.2.18
```

---

## Research Paper

The full mathematical methodology is documented in [VectorAlpha.pdf](research_paper/VectorAlpha.pdf), including:
- Formal definitions of return and risk attribution
- Proof of summation properties
- Execution model specifications
- Empirical analysis

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Disclaimer

**For educational purposes only.** Not investment advice. Past simulated performance does not guarantee future results.

---

## Contact

- **GitHub Issues**: [Report a bug or request a feature](https://github.com/Ravitheja1289-dot/VectorAlpha/issues)
- **Email**: ravithejareddy1289@gmail.com
- **Dashboard**: [vectoralpha.streamlit.app](https://vectoralpha.streamlit.app)
