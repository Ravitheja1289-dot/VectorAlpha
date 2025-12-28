"""
Vector Alpha Dashboard - Drawdown & Risk Component
=================================================

Display drawdown characteristics and rolling risk metrics.

Design:
- Drawdown time series with fill
- Rolling volatility (selectable window)
- Rolling Sharpe ratio (selectable window)
- Controls: Rolling window selector (63/126/252 days)
"""

import streamlit as st
import pandas as pd
from config import ROLLING_WINDOWS, RISK_FREE_RATE
from utils_plotting import (
    plot_drawdown,
    plot_rolling_volatility,
    plot_rolling_sharpe
)
from utils_helpers import compute_rolling_volatility, compute_rolling_sharpe


def show_drawdown_risk(returns: pd.DataFrame, risk_attribution: pd.DataFrame) -> None:
    """
    Render the Drawdown & Risk section.
    
    Args:
        returns: DataFrame of daily returns (all assets + TOTAL)
        risk_attribution: DataFrame of daily risk attribution (volatility by asset)
        
    Returns:
        None (renders Streamlit components)
        
    Purpose:
        - Understand max loss from peak (drawdown)
        - Track volatility over time (rolling)
        - Assess risk-adjusted returns (rolling Sharpe)
        - Identify periods of elevated risk
        
    Why rolling windows?
        - Market conditions change (bull, bear, transition periods)
        - Fixed volatility masks structural changes
        - Rolling windows reveal 63-day (Q), 126-day (2Q), 252-day (Y) regimes
    """
    
    portfolio_returns = returns["TOTAL"]
    
    # Sidebar control: rolling window selection
    rolling_window = st.sidebar.radio(
        "Rolling Window (days)",
        options=ROLLING_WINDOWS,
        index=2,  # Default to 252 (annual)
        key="risk_window_select"
    )
    
    st.subheader("Drawdown Analysis")
    
    fig_dd = plot_drawdown(
        portfolio_returns,
        title="Portfolio Drawdown (Maximum Loss from Peak)"
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Shows maximum loss from the peak value at any point in time. "
        "Red area = underwater; peak-to-trough losses. "
        "Key metric for risk management and redemption risk."
    )
    
    st.markdown("---")
    
    st.subheader(f"Rolling Volatility ({rolling_window}-day window)")
    
    fig_vol = plot_rolling_volatility(
        portfolio_returns,
        window=rolling_window,
        title=f"Annualized Rolling Volatility ({rolling_window}-day)"
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Volatility changes over time. "
        f"Window = {rolling_window} days; allows detection of risk regime changes. "
        "Rising volatility = increasing market stress; falling = market calming."
    )
    
    st.markdown("---")
    
    st.subheader(f"Rolling Sharpe Ratio ({rolling_window}-day window)")
    
    fig_sharpe = plot_rolling_sharpe(
        portfolio_returns,
        window=rolling_window,
        rf_rate=RISK_FREE_RATE,
        title=f"Rolling Sharpe Ratio ({rolling_window}-day, RF={RISK_FREE_RATE*100:.1f}%)"
    )
    st.plotly_chart(fig_sharpe, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Risk-adjusted returns over time. "
        "Sharpe > 1.0 = strong risk-adjusted returns; Sharpe < 0.5 = weak. "
        "Dips below zero = periods where losses exceeded risk-free rate."
    )
    
    st.markdown("---")
    
    st.subheader("Risk Metrics Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Compute drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown_series = (cumulative_returns - running_max) / running_max
    max_dd = drawdown_series.min()
    
    # Compute rolling metrics
    rolling_vol = compute_rolling_volatility(portfolio_returns, rolling_window)
    rolling_sharpe = compute_rolling_sharpe(portfolio_returns, rolling_window, RISK_FREE_RATE)
    
    with col1:
        st.metric("Max Drawdown", f"{max_dd * 100:.2f}%")
    with col2:
        st.metric(f"Avg Rolling Vol ({rolling_window}d)", f"{rolling_vol.mean() * 100:.2f}%")
    with col3:
        st.metric(f"Max Rolling Vol ({rolling_window}d)", f"{rolling_vol.max() * 100:.2f}%")
    with col4:
        st.metric(f"Avg Rolling Sharpe ({rolling_window}d)", f"{rolling_sharpe.mean():.2f}")
