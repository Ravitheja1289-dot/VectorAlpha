"""
Vector Alpha Dashboard - Strategy Comparison Component
=====================================================

Display multi-strategy comparison: equity curves, metrics table,
drawdowns, rolling Sharpe, and weight allocations.
"""

import streamlit as st
import pandas as pd

from utils_plotting import (
    plot_strategy_equity_curves,
    plot_strategy_drawdowns,
    plot_strategy_rolling_sharpe,
    plot_strategy_comparison_bars,
    plot_weight_heatmap,
)
from config import STRATEGY_COLORS, ADVANCED_METRICS


def show_strategies(data: dict) -> None:
    """Render the Strategy Comparison section."""

    st.subheader("Multi-Strategy Equity Curves")

    equity_df = data.get("strategy_equity_curves")
    returns_df = data.get("strategy_returns")
    comparison_df = data.get("strategy_comparison")
    weights_df = data.get("strategy_avg_weights")

    if equity_df is None or equity_df.empty:
        st.warning("Strategy comparison data not available. Run `python run_experiment.py` first.")
        return

    # Strategy selector
    available = equity_df.columns.tolist()
    selected = st.multiselect(
        "Select Strategies to Compare",
        available,
        default=available[:5],
        key="strategy_select"
    )

    if not selected:
        st.info("Select at least one strategy.")
        return

    # 1. Equity curves
    fig_eq = plot_strategy_equity_curves(
        equity_df[selected],
        title="Strategy Equity Curves (log scale)"
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("---")

    # 2. Comparison metrics table
    if comparison_df is not None and not comparison_df.empty:
        st.subheader("Performance Metrics Comparison")

        display_df = comparison_df.loc[
            comparison_df.index.isin(selected)
        ].copy()

        # Format for display
        fmt_df = display_df.copy()
        for col in ["cagr", "volatility", "max_drawdown", "var_95", "cvar_95"]:
            if col in fmt_df.columns:
                fmt_df[col] = (fmt_df[col] * 100).round(2).astype(str) + "%"
        for col in ["sharpe", "sortino", "calmar", "omega"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].round(3)
        for col in ["skewness", "kurtosis"]:
            if col in fmt_df.columns:
                fmt_df[col] = fmt_df[col].round(3)
        for col in ["best_day", "worst_day"]:
            if col in fmt_df.columns:
                fmt_df[col] = (fmt_df[col] * 100).round(2).astype(str) + "%"
        if "positive_days_pct" in fmt_df.columns:
            fmt_df["positive_days_pct"] = (fmt_df["positive_days_pct"] * 100).round(1).astype(str) + "%"

        st.dataframe(fmt_df, use_container_width=True)

        st.markdown("---")

        # 3. Bar chart comparison
        st.subheader("Metric Comparison (Bar Charts)")
        col1, col2 = st.columns(2)

        with col1:
            metric = st.selectbox(
                "Select Metric",
                ["sharpe", "cagr", "sortino", "calmar", "omega", "max_drawdown", "volatility"],
                key="metric_select"
            )

        filtered_comp = comparison_df.loc[comparison_df.index.isin(selected)]
        fig_bars = plot_strategy_comparison_bars(filtered_comp, metric=metric)
        st.plotly_chart(fig_bars, use_container_width=True)

    st.markdown("---")

    # 4. Drawdown comparison
    st.subheader("Drawdown Comparison")
    fig_dd = plot_strategy_drawdowns(equity_df, strategies=selected)
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # 5. Rolling Sharpe
    if returns_df is not None and not returns_df.empty:
        st.subheader("Rolling Sharpe Comparison")
        window = st.selectbox("Rolling Window (days)", [63, 126, 252], index=0, key="rs_window")
        fig_rs = plot_strategy_rolling_sharpe(
            returns_df[selected], window=window, strategies=selected
        )
        st.plotly_chart(fig_rs, use_container_width=True)

    st.markdown("---")

    # 6. Weight allocation heatmap
    if weights_df is not None and not weights_df.empty:
        st.subheader("Average Weight Allocation by Strategy")
        weight_display = weights_df[[c for c in selected if c in weights_df.columns]]
        if not weight_display.empty:
            fig_w = plot_weight_heatmap(
                weight_display,
                title="Average Portfolio Weights (Asset x Strategy)"
            )
            st.plotly_chart(fig_w, use_container_width=True)
