"""
Vector Alpha — Section 2: See What Happened
=============================================

Visualize portfolio performance clearly.
KPI cards + equity curve + drawdown + highlights.
"""

import streamlit as st
import pandas as pd

from config import (
    CONCEPT_DEFINITIONS,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_ACCENT,
    SECTION_DESCRIPTIONS,
)
from plotting import (
    plot_equity_curve,
    plot_drawdown,
    plot_asset_cumulative_returns,
)


def show_results_section(results: dict):
    """Render the performance results page."""

    st.markdown(
        f"<p style='color: #94A3B8; margin-top: -10px;'>"
        f"{SECTION_DESCRIPTIONS['results']}</p>",
        unsafe_allow_html=True,
    )

    # ---- KPI Cards ----
    st.markdown("#### Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    cagr = results["cagr"]
    vol = results["volatility"]
    sharpe = results["sharpe"]
    max_dd = results["max_drawdown"]

    with col1:
        st.metric(
            label="Annualized Return",
            value=f"{cagr*100:.1f}%",
            help=CONCEPT_DEFINITIONS["cagr"],
        )
    with col2:
        st.metric(
            label="Volatility",
            value=f"{vol*100:.1f}%",
            help=CONCEPT_DEFINITIONS["volatility"],
        )
    with col3:
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            help=CONCEPT_DEFINITIONS["sharpe_ratio"],
        )
    with col4:
        st.metric(
            label="Max Drawdown",
            value=f"{max_dd*100:.1f}%",
            help=CONCEPT_DEFINITIONS["max_drawdown"],
        )

    # ---- Plain-English metric descriptions ----
    with st.expander("What do these numbers mean?", expanded=False):
        st.markdown(
            f"**Annualized Return ({cagr*100:.1f}%)** — "
            f"{CONCEPT_DEFINITIONS['cagr']}\n\n"
            f"**Volatility ({vol*100:.1f}%)** — "
            f"{CONCEPT_DEFINITIONS['volatility']}\n\n"
            f"**Sharpe Ratio ({sharpe:.2f})** — "
            f"{CONCEPT_DEFINITIONS['sharpe_ratio']}\n\n"
            f"**Max Drawdown ({max_dd*100:.1f}%)** — "
            f"{CONCEPT_DEFINITIONS['max_drawdown']}"
        )

    st.markdown("---")

    # ---- Equity Curve ----
    st.markdown("#### Portfolio Growth")
    st.caption("How $1 invested at the start would have grown over time.")

    fig_equity = plot_equity_curve(results["equity"])
    st.plotly_chart(fig_equity, use_container_width=True)

    # ---- Drawdown ----
    st.markdown("#### Drawdowns")
    st.caption(
        "The chart below shows how far the portfolio dropped from its peak at any point. "
        "Deeper dips mean larger losses before recovery."
    )

    fig_dd = plot_drawdown(results["drawdown_series"])
    st.plotly_chart(fig_dd, use_container_width=True)

    st.markdown("---")

    # ---- Highlights Box ----
    st.markdown("#### Highlights")

    ret_attr = results.get("return_attribution")
    risk_attr = results.get("risk_attribution")

    col_a, col_b, col_c = st.columns(3)

    if ret_attr is not None:
        summary = ret_attr["summary"]
        best = summary["cumulative_contribution"].idxmax()
        best_val = summary.loc[best, "cumulative_contribution"]
        worst = summary["cumulative_contribution"].idxmin()
        worst_val = summary.loc[worst, "cumulative_contribution"]

        with col_a:
            st.markdown(
                f"<div style='background: #14532D; padding: 16px; border-radius: 8px; "
                f"border-left: 4px solid {COLOR_POSITIVE}; color: #F1F5F9;'>"
                f"<strong>Best Performer</strong><br>"
                f"<span style='font-size: 1.3em; color: #4ADE80;'>{best}</span><br>"
                f"Contributed {best_val*100:+.1f}% to returns"
                f"</div>",
                unsafe_allow_html=True,
            )

        with col_b:
            st.markdown(
                f"<div style='background: #7F1D1D; padding: 16px; border-radius: 8px; "
                f"border-left: 4px solid {COLOR_NEGATIVE}; color: #F1F5F9;'>"
                f"<strong>Worst Performer</strong><br>"
                f"<span style='font-size: 1.3em; color: #F87171;'>{worst}</span><br>"
                f"Contributed {worst_val*100:+.1f}% to returns"
                f"</div>",
                unsafe_allow_html=True,
            )

    if risk_attr is not None:
        risk_summary = risk_attr["summary"]
        top_risk = risk_summary["pct_of_portfolio_vol"].idxmax()
        top_risk_pct = risk_summary.loc[top_risk, "pct_of_portfolio_vol"]

        with col_c:
            st.markdown(
                f"<div style='background: #78350F; padding: 16px; border-radius: 8px; "
                f"border-left: 4px solid {COLOR_ACCENT}; color: #F1F5F9;'>"
                f"<strong>Largest Risk Contributor</strong><br>"
                f"<span style='font-size: 1.3em; color: #FCD34D;'>{top_risk}</span><br>"
                f"Accounts for {top_risk_pct*100:.0f}% of portfolio risk"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ---- Individual Asset Performance ----
    st.markdown("#### Individual Asset Performance")
    st.caption("How each asset in your portfolio performed on its own.")

    fig_assets = plot_asset_cumulative_returns(results["returns"])
    st.plotly_chart(fig_assets, use_container_width=True)
