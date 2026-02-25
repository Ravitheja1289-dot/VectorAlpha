"""
Vector Alpha Dashboard - Stress Test & Advanced Risk Component
=============================================================

Display stress test results, VaR/CVaR metrics, and tail risk analysis.
"""

import streamlit as st
import pandas as pd
import json

from utils_plotting import plot_stress_test_bars, plot_strategy_comparison_bars
from config import PARQUET_FILES


def show_stress_test(data: dict) -> None:
    """Render the Stress Test & Advanced Risk section."""

    st.subheader("Stress Testing & Advanced Risk Metrics")

    comparison_df = data.get("strategy_comparison")
    stress_data = data.get("stress_test_results")

    # 1. Advanced metrics comparison
    if comparison_df is not None and not comparison_df.empty:
        st.markdown("#### Advanced Risk Metrics")

        col1, col2, col3, col4 = st.columns(4)

        adv_metrics = [
            ("var_95", "VaR 95%", "Daily loss threshold exceeded 5% of the time"),
            ("cvar_95", "CVaR 95%", "Average loss in worst 5% of days"),
            ("sortino", "Sortino Ratio", "Return per unit of downside risk"),
            ("calmar", "Calmar Ratio", "CAGR / Max Drawdown"),
        ]

        for i, (metric, label, desc) in enumerate(adv_metrics):
            if metric in comparison_df.columns:
                col = [col1, col2, col3, col4][i]
                best_strat = comparison_df[metric].idxmax() if metric != "var_95" else comparison_df[metric].idxmin()
                best_val = comparison_df.loc[best_strat, metric]
                with col:
                    if metric in ["var_95", "cvar_95"]:
                        st.metric(f"Best {label}", f"{best_val*100:.2f}%", delta=best_strat)
                    else:
                        st.metric(f"Best {label}", f"{best_val:.3f}", delta=best_strat)

        st.markdown("---")

        # VaR / CVaR comparison bars
        st.markdown("#### Value-at-Risk Comparison")
        col1, col2 = st.columns(2)
        with col1:
            if "var_95" in comparison_df.columns:
                fig_var = plot_strategy_comparison_bars(
                    comparison_df, metric="var_95",
                    title="95% VaR by Strategy (lower = better)"
                )
                st.plotly_chart(fig_var, use_container_width=True)
        with col2:
            if "cvar_95" in comparison_df.columns:
                fig_cvar = plot_strategy_comparison_bars(
                    comparison_df, metric="cvar_95",
                    title="95% CVaR (Expected Shortfall) by Strategy"
                )
                st.plotly_chart(fig_cvar, use_container_width=True)

        st.markdown("---")

        # Tail risk metrics
        st.markdown("#### Tail Risk Analysis")
        col1, col2 = st.columns(2)
        with col1:
            if "skewness" in comparison_df.columns:
                fig_skew = plot_strategy_comparison_bars(
                    comparison_df, metric="skewness",
                    title="Return Skewness (negative = left tail risk)"
                )
                st.plotly_chart(fig_skew, use_container_width=True)
        with col2:
            if "kurtosis" in comparison_df.columns:
                fig_kurt = plot_strategy_comparison_bars(
                    comparison_df, metric="kurtosis",
                    title="Excess Kurtosis (>0 = fat tails)"
                )
                st.plotly_chart(fig_kurt, use_container_width=True)

    st.markdown("---")

    # 2. Stress test results
    if stress_data and isinstance(stress_data, dict) and len(stress_data) > 0:
        st.subheader("Crisis Period Stress Tests")

        st.caption(
            "How each strategy performed during major market events. "
            "Compares total return and max drawdown across crisis periods."
        )

        fig_stress_ret = plot_stress_test_bars(
            stress_data, metric="total_return",
            title="Stress Test: Total Return by Crisis Period"
        )
        st.plotly_chart(fig_stress_ret, use_container_width=True)

        fig_stress_dd = plot_stress_test_bars(
            stress_data, metric="max_drawdown",
            title="Stress Test: Max Drawdown by Crisis Period"
        )
        st.plotly_chart(fig_stress_dd, use_container_width=True)

        # Table view
        st.markdown("#### Detailed Stress Test Results")
        rows = []
        for scenario, metrics in stress_data.items():
            row = {"Scenario": scenario}
            for key, val in metrics.items():
                if "total_return" in key:
                    strat = key.replace("_total_return", "")
                    row[f"{strat} Return"] = f"{val*100:.1f}%"
                elif "max_drawdown" in key:
                    strat = key.replace("_max_drawdown", "")
                    row[f"{strat} MaxDD"] = f"{val*100:.1f}%"
            rows.append(row)
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Scenario"), use_container_width=True)
    else:
        st.info("Stress test data not available. Run `python run_experiment.py` to generate.")
