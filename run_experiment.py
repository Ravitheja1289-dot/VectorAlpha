"""Multi-strategy experiment runner.

- Loads processed data
- Builds features (basic + enhanced)
- Runs EIGHT strategies: Equal-Weight, Momentum, Mean-Reversion, Multi-Factor,
  Low-Vol, Risk Parity, HRP, Min Variance
- Computes risk metrics + advanced metrics for all strategies
- Computes attribution
- Persists all results for the dashboard
- Saves comparison plots

Usage:
    python run_experiment.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import json
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from data.prices import load_processed_prices
from data.returns import load_processed_returns
from features.feature_engine import build_features, build_enhanced_features
from backtest.rebalance import get_weekly_rebalance_dates
from strategies.equal_weight import EqualWeightStrategy
from strategies.momentum import MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.multi_factor import MultiFactorStrategy
from strategies.low_volatility import LowVolatilityStrategy
from portfolio.optimizer import OptimizedStrategy
from execution.executor import execute_strategy
from portfolio.portfolio_engine import run_backtest
from risk.metrics import compute_risk_metrics, save_risk_metrics_summary
from risk.advanced_metrics import compute_advanced_metrics
from risk.attribution import (
    compute_return_attribution,
    compute_risk_attribution,
    save_return_attribution,
    save_risk_attribution,
)
from risk.factor_model import build_factor_report

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data/processed")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _run_single_strategy(
    name: str,
    strategy,
    features: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rebalance_dates: list,
    cost_bps: float = 10.0,
) -> dict:
    """Run a single strategy through the full pipeline and return results."""
    print(f"\n{'='*60}")
    print(f"  STRATEGY: {name}")
    print(f"{'='*60}")

    target_weights = strategy.generate_weights(features, rebalance_dates)

    exec_out = execute_strategy(
        prices, returns, target_weights, rebalance_dates, cost_bps=cost_bps
    )

    backtest = run_backtest(
        exec_out["daily_weights"], returns, exec_out["transaction_costs"],
        initial_capital=1.0,
    )

    metrics = compute_risk_metrics(backtest["net_returns"], backtest["equity"])

    # Advanced metrics
    adv_metrics = compute_advanced_metrics(
        backtest["net_returns"], backtest["equity"]
    )

    # Return attribution
    ret_attr = compute_return_attribution(
        daily_weights=exec_out["daily_weights"],
        returns=returns,
        portfolio_returns=backtest["gross_returns"],
    )

    # Risk attribution
    risk_attr = compute_risk_attribution(
        returns=returns, daily_weights=exec_out["daily_weights"]
    )

    return {
        "name": name,
        "exec_out": exec_out,
        "backtest": backtest,
        "metrics": metrics,
        "advanced_metrics": adv_metrics,
        "return_attribution": ret_attr,
        "risk_attribution": risk_attr,
        "target_weights": target_weights,
    }


def _print_comparison_table(results: dict) -> None:
    """Print a comparison table of all strategies."""
    print(f"\n{'='*90}")
    print(f"  STRATEGY COMPARISON")
    print(f"{'='*90}")
    header = f"{'Strategy':<25} {'CAGR':>8} {'Vol':>8} {'Sharpe':>8} {'MaxDD':>8} {'Sortino':>8} {'Calmar':>8}"
    print(header)
    print("-" * 90)

    for name, res in results.items():
        s = res["metrics"]["static"]
        adv = res["advanced_metrics"]
        print(
            f"{name:<25} "
            f"{s['annualized_return_cagr']*100:>7.1f}% "
            f"{s['annualized_volatility']*100:>7.1f}% "
            f"{s['sharpe_ratio']:>8.2f} "
            f"{s['max_drawdown']*100:>7.1f}% "
            f"{adv['sortino_ratio']:>8.2f} "
            f"{adv['calmar_ratio']:>8.2f}"
        )
    print("=" * 90)


def _save_strategy_results(results: dict) -> None:
    """Persist all strategy results for the dashboard."""

    # 1. Combined equity curves
    equity_curves = {}
    net_returns_all = {}
    for name, res in results.items():
        equity_curves[name] = res["backtest"]["equity"]
        net_returns_all[name] = res["backtest"]["net_returns"]

    equity_df = pd.DataFrame(equity_curves)
    equity_df.to_parquet(DATA_DIR / "strategy_equity_curves.parquet")

    returns_df = pd.DataFrame(net_returns_all)
    returns_df.to_parquet(DATA_DIR / "strategy_returns.parquet")

    # 2. Comparison metrics table
    comparison = []
    for name, res in results.items():
        s = res["metrics"]["static"]
        adv = res["advanced_metrics"]
        tail = adv.get("tail_risk", {})
        row = {
            "strategy": name,
            "cagr": s["annualized_return_cagr"],
            "volatility": s["annualized_volatility"],
            "sharpe": s["sharpe_ratio"],
            "max_drawdown": s["max_drawdown"],
            "dd_duration_days": s["drawdown_duration_days"],
            "sortino": adv["sortino_ratio"],
            "calmar": adv["calmar_ratio"],
            "omega": adv["omega_ratio"],
            "var_95": adv["var_95_historical"],
            "cvar_95": adv["cvar_95"],
            "skewness": tail.get("skewness", 0),
            "kurtosis": tail.get("excess_kurtosis", 0),
            "best_day": tail.get("best_day", 0),
            "worst_day": tail.get("worst_day", 0),
            "positive_days_pct": tail.get("positive_days_pct", 0),
        }
        comparison.append(row)

    comp_df = pd.DataFrame(comparison).set_index("strategy")
    comp_df.to_parquet(DATA_DIR / "strategy_comparison.parquet")

    # 3. Stress test results
    stress_all = {}
    for name, res in results.items():
        stress = res["advanced_metrics"].get("stress_test", {})
        for scenario, metrics in stress.items():
            for metric_name, val in metrics.items():
                stress_all.setdefault(scenario, {})[f"{name}_{metric_name}"] = val

    with open(DATA_DIR / "stress_test_results.json", "w") as f:
        json.dump(stress_all, f, indent=2, default=str)

    # 4. Average weight snapshots
    weight_summary = {}
    for name, res in results.items():
        dw = res["exec_out"]["daily_weights"]
        valid = dw.dropna(how="all")
        weight_summary[name] = valid.mean()

    weight_df = pd.DataFrame(weight_summary)
    weight_df.to_parquet(DATA_DIR / "strategy_avg_weights.parquet")

    # 5. Primary strategy attribution (backward compat)
    primary = "Multi-Factor"
    if primary not in results:
        primary = list(results.keys())[0]

    res = results[primary]
    save_return_attribution(res["return_attribution"], str(DATA_DIR / "return_attribution.parquet"))
    save_risk_attribution(res["risk_attribution"], str(DATA_DIR / "risk_attribution.parquet"))
    save_risk_metrics_summary(res["metrics"], str(DATA_DIR / "risk_metrics.json"))

    print(f"\nAll results saved to {DATA_DIR}/")


def _generate_plots(results: dict, returns: pd.DataFrame) -> None:
    """Generate comparison plots using matplotlib."""
    import matplotlib.pyplot as plt

    colors = {
        "Equal Weight": "#1f77b4",
        "Momentum": "#ff7f0e",
        "Mean Reversion": "#2ca02c",
        "Multi-Factor": "#d62728",
        "Low Volatility": "#9467bd",
        "Risk Parity": "#8c564b",
        "HRP": "#e377c2",
        "Min Variance": "#17becf",
    }

    # ---- Plot 1: Equity Curves + Drawdown + Rolling Sharpe ----
    fig, axes = plt.subplots(3, 1, figsize=(14, 16),
                             gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

    ax1 = axes[0]
    for name, res in results.items():
        eq = res["backtest"]["equity"]
        ax1.plot(eq.index, eq.values, label=name,
                 color=colors.get(name, "gray"), linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_title("Strategy Comparison: Equity Curves (log scale)", fontsize=14)
    ax1.set_ylabel("Cumulative Return")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    for name in ["Multi-Factor", "Equal Weight", "Momentum"]:
        if name not in results:
            continue
        eq = results[name]["backtest"]["equity"]
        dd = (eq - eq.cummax()) / eq.cummax()
        ax2.fill_between(dd.index, dd.values, alpha=0.3,
                         label=name, color=colors.get(name))
        ax2.plot(dd.index, dd.values, linewidth=0.8, color=colors.get(name))
    ax2.set_title("Drawdown Comparison", fontsize=14)
    ax2.set_ylabel("Drawdown")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    for name in list(results.keys())[:4]:
        rets = results[name]["backtest"]["net_returns"]
        rs = (rets.rolling(63).mean() / rets.rolling(63).std()) * np.sqrt(252)
        ax3.plot(rs.index, rs.values, label=name,
                 color=colors.get(name), linewidth=1)
    ax3.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_title("Rolling Sharpe (63-day)", fontsize=14)
    ax3.set_ylabel("Sharpe Ratio")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_drawdown.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {OUTPUT_DIR / 'equity_drawdown.png'}")

    # ---- Plot 2: Per-Asset Performance ----
    fig2, ax = plt.subplots(figsize=(14, 8))
    asset_colors = plt.cm.tab20(np.linspace(0, 1, len(returns.columns)))
    for i, col in enumerate(returns.columns):
        cum_ret = (1 + returns[col]).cumprod()
        ax.plot(cum_ret.index, cum_ret.values, label=col,
                color=asset_colors[i], linewidth=1.2)
    ax.set_yscale("log")
    ax.set_title("Individual Asset Performance (Normalized)", fontsize=14)
    ax.set_ylabel("Cumulative Return (log)")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(OUTPUT_DIR / "asset_performance.png", dpi=150)
    plt.close(fig2)
    print(f"Saved: {OUTPUT_DIR / 'asset_performance.png'}")

    # ---- Plot 3: Weight Heatmaps ----
    fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
    weight_strats = ["Multi-Factor", "Momentum", "Risk Parity", "Low Volatility"]
    for idx, name in enumerate(weight_strats):
        if name not in results:
            continue
        ax = axes3[idx // 2][idx % 2]
        tw = results[name]["target_weights"]
        sampled = tw.iloc[::10]
        im = ax.imshow(sampled.T.values, aspect="auto", cmap="RdYlGn", vmin=0)
        ax.set_title(f"{name} Weights Over Time", fontsize=11)
        ax.set_yticks(range(len(sampled.columns)))
        ax.set_yticklabels(sampled.columns, fontsize=7)
        n_ticks = min(6, len(sampled))
        tick_pos = np.linspace(0, len(sampled) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(
            [sampled.index[i].strftime("%Y-%m") for i in tick_pos],
            fontsize=7, rotation=45,
        )
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig3.savefig(OUTPUT_DIR / "weight_heatmaps.png", dpi=150)
    plt.close(fig3)
    print(f"Saved: {OUTPUT_DIR / 'weight_heatmaps.png'}")

    # ---- Plot 4: Risk Contribution Comparison ----
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    bar_width = 0.15
    strats_to_plot = [s for s in ["Equal Weight", "Momentum", "Multi-Factor", "Risk Parity"]
                      if s in results]
    x = np.arange(len(returns.columns))
    for i, name in enumerate(strats_to_plot):
        risk_sum = results[name]["risk_attribution"]["summary"]
        pct_vol = risk_sum["pct_of_portfolio_vol"].reindex(returns.columns).fillna(0)
        ax4.bar(x + i * bar_width, pct_vol.values * 100, bar_width,
                label=name, color=colors.get(name), alpha=0.85)
    ax4.set_xlabel("Asset")
    ax4.set_ylabel("% of Portfolio Volatility")
    ax4.set_title("Risk Contribution by Asset & Strategy", fontsize=14)
    ax4.set_xticks(x + bar_width * len(strats_to_plot) / 2)
    ax4.set_xticklabels(returns.columns, fontsize=8, rotation=45)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig4.savefig(OUTPUT_DIR / "risk_contribution.png", dpi=150)
    plt.close(fig4)
    print(f"Saved: {OUTPUT_DIR / 'risk_contribution.png'}")

    # ---- Plot 5: Rolling Sharpe (all strategies) ----
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    for name, res in results.items():
        rets = res["backtest"]["net_returns"]
        rs = (rets.rolling(63).mean() / rets.rolling(63).std()) * np.sqrt(252)
        ax5.plot(rs.index, rs.values, label=name,
                 color=colors.get(name, "gray"), linewidth=1)
    ax5.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax5.axhline(y=1, color="green", linestyle=":", linewidth=0.6, alpha=0.5)
    ax5.axhline(y=-1, color="red", linestyle=":", linewidth=0.6, alpha=0.5)
    ax5.set_title("Rolling Sharpe Ratio (63-day) - All Strategies", fontsize=14)
    ax5.set_ylabel("Sharpe Ratio")
    ax5.legend(loc="upper left", fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    fig5.savefig(OUTPUT_DIR / "rolling_sharpe.png", dpi=150)
    plt.close(fig5)
    print(f"Saved: {OUTPUT_DIR / 'rolling_sharpe.png'}")


def main() -> None:
    print("Running multi-strategy experiment...")

    prices = load_processed_prices("data/processed/prices.parquet")
    returns = load_processed_returns("data/processed/returns.parquet")

    features = build_features(prices, returns)
    rebalance_dates = get_weekly_rebalance_dates(prices.index)

    strategies = {
        "Equal Weight": EqualWeightStrategy(),
        "Momentum": MomentumStrategy(lookback=252, skip=21, top_pct=0.4),
        "Mean Reversion": MeanReversionStrategy(lookback=60, z_threshold=1.0),
        "Multi-Factor": MultiFactorStrategy(
            momentum_lookback=252, reversion_lookback=20,
            vol_lookback=60, concentration=2.0,
        ),
        "Low Volatility": LowVolatilityStrategy(vol_lookback=60),
        "Risk Parity": OptimizedStrategy(method="risk_parity", lookback=252),
        "HRP": OptimizedStrategy(method="hrp", lookback=252),
        "Min Variance": OptimizedStrategy(method="min_var", lookback=252),
    }

    results = {}
    for name, strategy in strategies.items():
        try:
            res = _run_single_strategy(
                name=name, strategy=strategy, features=features,
                prices=prices, returns=returns,
                rebalance_dates=rebalance_dates, cost_bps=10.0,
            )
            results[name] = res
        except Exception as e:
            print(f"\n  [ERROR] Strategy '{name}' failed: {e}")

    _print_comparison_table(results)

    # Factor analysis
    primary = "Multi-Factor" if "Multi-Factor" in results else list(results.keys())[0]
    try:
        print("\nRunning factor analysis...")
        factor_report = build_factor_report(
            returns=returns,
            daily_weights=results[primary]["exec_out"]["daily_weights"],
            n_factors=5,
        )
        model = factor_report["model"]
        model["factor_loadings"].to_parquet(DATA_DIR / "factor_loadings.parquet")
        pd.DataFrame({
            "explained_variance": model["explained_variance_ratio"],
            "cumulative_variance": model["cumulative_variance"],
        }).to_parquet(DATA_DIR / "factor_variance.parquet")
        print("  Factor analysis complete.")
    except Exception as e:
        print(f"  Factor analysis failed: {e}")

    _save_strategy_results(results)
    _generate_plots(results, returns)

    print("\nExperiment complete. All outputs are reproducible via this script.")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
