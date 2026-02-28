"""
Vector Alpha — Advanced Mode Tab
==================================

Hidden by default. Shows detailed analytics for deeper exploration:
weight drift, turnover, transaction costs, rolling metrics.
"""

import streamlit as st
import pandas as pd

from config import CONCEPT_DEFINITIONS, SECTION_DESCRIPTIONS
from plotting import (
    plot_weight_drift,
    plot_turnover,
    plot_rolling_volatility,
    plot_rolling_sharpe,
)


def show_advanced_mode(results: dict):
    """Render the Advanced Mode tab."""

    st.markdown(
        f"<p style='color: #94A3B8; margin-top: -10px;'>"
        f"{SECTION_DESCRIPTIONS['advanced']}</p>",
        unsafe_allow_html=True,
    )

    config = results["config"]

    # ---- Weight Drift ----
    st.markdown("#### Weight Drift Over Time")
    st.caption(
        "This chart shows how your portfolio weights changed each day. "
        "Even with rebalancing, weights shift between rebalance dates "
        "because asset prices change at different rates."
    )

    daily_weights = results.get("daily_weights")
    if daily_weights is not None and not daily_weights.empty:
        fig_drift = plot_weight_drift(daily_weights)
        st.plotly_chart(fig_drift, use_container_width=True)

        # Show drift statistics
        if len(daily_weights) > 1:
            initial = daily_weights.iloc[0]
            final = daily_weights.iloc[-1]
            drift = (final - initial).abs()
            max_drift_asset = drift.idxmax()
            max_drift_val = drift[max_drift_asset]

            st.markdown(
                f"Largest drift: **{max_drift_asset}** moved "
                f"**{max_drift_val*100:.1f} percentage points** "
                f"from its starting weight."
            )

    with st.expander("Learn more: Weight Drift"):
        st.markdown(CONCEPT_DEFINITIONS["weight_drift"])

    st.markdown("---")

    # ---- Turnover ----
    if config["rebalance_freq"] != "none":
        turnover = results.get("turnover")
        if turnover is not None and len(turnover) > 0:
            st.markdown("#### Turnover at Each Rebalance")
            st.caption(
                "Turnover measures how much of the portfolio is traded "
                "at each rebalance. Higher turnover means more trading "
                "and higher transaction costs."
            )

            fig_turnover = plot_turnover(turnover)
            st.plotly_chart(fig_turnover, use_container_width=True)

            # Summary stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Turnover", f"{turnover.mean()*100:.1f}%")
            with col2:
                st.metric("Max Turnover", f"{turnover.max()*100:.1f}%")
            with col3:
                st.metric("Total Rebalances", len(turnover))

            st.markdown("---")

    # ---- Transaction Cost Impact ----
    total_costs = results.get("total_costs", 0)
    if total_costs > 0:
        st.markdown("#### Transaction Cost Impact")

        cagr = results["cagr"]
        st.markdown(
            f"Total transaction costs over the period: "
            f"**{total_costs*100:.3f}%** of portfolio value."
        )
        if cagr != 0:
            cost_drag = total_costs / abs(cagr) * 100
            st.markdown(
                f"This represents **{cost_drag:.1f}%** of your total return — "
                f"the price paid for maintaining your target allocation."
            )

        st.markdown("---")

    # ---- Rolling Metrics ----
    st.markdown("#### Rolling Volatility")
    st.caption(
        "Volatility is not constant. This chart shows how your portfolio's "
        "risk level changed over time."
    )

    roll_vol = results.get("rolling_volatility")
    if roll_vol is not None and not roll_vol.dropna().empty:
        fig_rvol = plot_rolling_volatility(roll_vol.dropna())
        st.plotly_chart(fig_rvol, use_container_width=True)

    st.markdown("#### Rolling Sharpe Ratio")
    st.caption(
        "The Sharpe Ratio also changes over time. Periods where it drops "
        "below zero mean the portfolio was losing money."
    )

    roll_sharpe = results.get("rolling_sharpe")
    if roll_sharpe is not None and not roll_sharpe.dropna().empty:
        fig_rsr = plot_rolling_sharpe(roll_sharpe.dropna())
        st.plotly_chart(fig_rsr, use_container_width=True)

    st.markdown("---")

    # ---- Detailed Attribution Table ----
    st.markdown("#### Detailed Attribution Data")

    ret_attr = results.get("return_attribution")
    risk_attr = results.get("risk_attribution")

    if ret_attr is not None:
        with st.expander("Return Attribution Table"):
            summary = ret_attr["summary"].copy()
            summary["cumulative_contribution"] = summary["cumulative_contribution"] * 100
            summary["pct_contribution"] = summary["pct_contribution"] * 100
            summary.columns = ["Cumulative Contribution (%)", "% of Total Return"]
            st.dataframe(summary.style.format("{:.2f}"), use_container_width=True)

    if risk_attr is not None:
        with st.expander("Risk Attribution Table"):
            summary = risk_attr["summary"].copy()
            display = summary[["avg_weight", "risk_contribution", "pct_of_portfolio_vol"]].copy()
            display.columns = ["Average Weight", "Risk Contribution", "% of Portfolio Risk"]
            display["Average Weight"] = display["Average Weight"] * 100
            display["% of Portfolio Risk"] = display["% of Portfolio Risk"] * 100
            st.dataframe(display.style.format("{:.2f}"), use_container_width=True)

            st.metric(
                "Portfolio Volatility (Daily)",
                f"{risk_attr['portfolio_volatility']*100:.2f}%",
            )
