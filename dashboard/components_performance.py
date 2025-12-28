"""
Vector Alpha Dashboard - Performance Component
=============================================

Display portfolio and per-asset return metrics.

Design:
- Returns histogram (not cumulative - shows daily volatility)
- Per-asset cumulative returns (optional detail)
- No calculations; use precomputed data only
"""

import streamlit as st
import pandas as pd
from utils_plotting import plot_returns_histogram, plot_cumulative_attribution


def show_performance(returns: pd.DataFrame, return_attribution: pd.DataFrame) -> None:
    """
    Render the Performance section.
    
    Args:
        returns: DataFrame of daily returns (all assets + TOTAL)
        return_attribution: DataFrame of daily return attribution
        
    Returns:
        None (renders Streamlit components)
        
    Purpose:
        - Understand return distribution (daily volatility, tail risk)
        - Identify which assets contributed most to portfolio returns
        - See period-specific attribution
        
    Why histogram over time series?
        - Histogram shows distribution: skew, kurtosis, outliers
        - Time series would just be noisy daily returns
        - For tracking returns over time, use cumulative in Attribution tab
    """
    
    # Daily returns histogram (portfolio)
    st.subheader("Daily Returns Distribution")
    
    fig_hist = plot_returns_histogram(
        returns["TOTAL"],
        title="Portfolio Daily Returns Distribution"
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Shows how daily returns are distributed. "
        "Skew and tail behavior indicate risk characteristics. "
        "Outliers reveal stress periods."
    )
    
    st.markdown("---")
    
    # Asset-level cumulative returns (for comparison)
    st.subheader("Asset Cumulative Returns")
    
    # Multi-select to pick assets
    assets = sorted([col for col in returns.columns if col != "TOTAL"])
    selected_assets = st.multiselect(
        "Select assets to compare",
        assets,
        default=["NVDA", "TSLA", "META", "ORCL"],
        key="perf_asset_select"
    )
    
    if selected_assets:
        # Plot cumulative returns for selected assets
        fig_cumret = plot_cumulative_attribution(
            return_attribution,
            assets=selected_assets,
            title="Cumulative Return Contribution (Selected Assets)"
        )
        st.plotly_chart(fig_cumret, use_container_width=True)
        
        st.caption(
            "**Interpretation**: Stacked cumulative returns show each asset's contribution "
            "to total portfolio returns over time. Useful for tracking diversification."
        )
    else:
        st.info("Select at least one asset to compare")
    
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
