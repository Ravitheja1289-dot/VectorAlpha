"""
Vector Alpha Dashboard - Performance Component (Enhanced)
========================================================

Display portfolio and per-asset return metrics with distinct colors per asset.
"""

import streamlit as st
import pandas as pd
from utils_plotting import (
    plot_returns_histogram,
    plot_cumulative_attribution,
    plot_asset_cumulative_returns,
)


def show_performance(returns: pd.DataFrame, return_attribution: pd.DataFrame) -> None:
    """Render the Performance section."""

    # Daily returns histogram (portfolio)
    st.subheader("Daily Returns Distribution")

    fig_hist = plot_returns_histogram(
        returns["TOTAL"],
        title="Portfolio Daily Returns Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        "**Interpretation**: Shows how daily returns are distributed. "
        "Skew and tail behavior indicate risk characteristics."
    )

    st.markdown("---")

    # Individual asset cumulative returns (distinct colors + line styles)
    st.subheader("Individual Asset Performance")

    assets = sorted([col for col in returns.columns if col != "TOTAL"])
    selected_assets = st.multiselect(
        "Select assets to compare",
        assets,
        default=assets[:8],
        key="perf_asset_select"
    )

    if selected_assets:
        fig_asset = plot_asset_cumulative_returns(
            returns,
            assets=selected_assets,
            title="Cumulative Returns by Asset (Distinct Colors)"
        )
        st.plotly_chart(fig_asset, use_container_width=True)

        st.caption(
            "**Note**: Each asset has a unique color and line style (solid, dash, dot) "
            "for easy visual differentiation."
        )
    else:
        st.info("Select at least one asset to compare")

    st.markdown("---")

    # Attribution stacked area
    st.subheader("Return Attribution (Stacked)")

    attr_assets = sorted([col for col in return_attribution.columns if col != "TOTAL"])
    selected_attr = st.multiselect(
        "Select assets for attribution",
        attr_assets,
        default=attr_assets[:6],
        key="perf_attr_select"
    )

    if selected_attr:
        fig_cumret = plot_cumulative_attribution(
            return_attribution,
            assets=selected_attr,
            title="Cumulative Return Contribution (Selected Assets)"
        )
        st.plotly_chart(fig_cumret, use_container_width=True)

    st.markdown("---")

    # Quick stats
    st.subheader("Return Statistics")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Mean Daily Return", f"{returns['TOTAL'].mean() * 100:.3f}%")
    with col2:
        st.metric("Std Dev (Daily)", f"{returns['TOTAL'].std() * 100:.3f}%")
    with col3:
        st.metric("Min Daily Return", f"{returns['TOTAL'].min() * 100:.2f}%")
    with col4:
        st.metric("Max Daily Return", f"{returns['TOTAL'].max() * 100:.2f}%")
    with col5:
        st.metric("Skewness", f"{returns['TOTAL'].skew():.3f}")

    st.markdown("---")

    # Per-asset stats table
    st.subheader("Per-Asset Return Statistics")

    asset_stats = []
    for asset in assets:
        r = returns[asset]
        cum_ret = (1 + r).prod() - 1
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * (252 ** 0.5)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        asset_stats.append({
            "Asset": asset,
            "Cumulative Return": f"{cum_ret*100:.1f}%",
            "Ann. Return": f"{ann_ret*100:.1f}%",
            "Ann. Volatility": f"{ann_vol*100:.1f}%",
            "Sharpe": f"{sharpe:.2f}",
            "Max Daily Gain": f"{r.max()*100:.1f}%",
            "Max Daily Loss": f"{r.min()*100:.1f}%",
        })

    stats_df = pd.DataFrame(asset_stats).set_index("Asset")
    st.dataframe(stats_df, use_container_width=True)
