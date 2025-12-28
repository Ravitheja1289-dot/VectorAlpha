"""
Vector Alpha Dashboard - Overview Component
===========================================

Display project description, metadata, and portfolio-level KPIs.

Design:
- Pure Streamlit rendering (no data transformation)
- Accept data as parameters
- Single responsibility: Display overview only
"""

import streamlit as st
from config import OVERVIEW_DESCRIPTION, DISPLAY_METRICS


def show_overview(risk_metrics: dict) -> None:
    """
    Render the Overview section.
    
    Args:
        risk_metrics: Dictionary of portfolio KPIs from risk_metrics.json
        
    Returns:
        None (renders Streamlit components)
        
    Purpose:
        - Project description and context
        - Portfolio metadata (rebalancing, costs, universe)
        - Key performance metrics (CAGR, Sharpe, DD, etc.)
    """
    
    # Project description
    st.markdown(OVERVIEW_DESCRIPTION)
    
    st.markdown("---")
    
    # KPI Cards
    st.subheader("Portfolio Metrics")
    
    # Arrange KPIs in columns (up to 3 per row)
    cols = st.columns(3)
    col_idx = 0
    
    for metric_key, metric_label in DISPLAY_METRICS.items():
        if metric_key in risk_metrics:
            value = risk_metrics[metric_key]
            
            # Format values appropriately
            if "return" in metric_key or "volatility" in metric_key or "drawdown" in metric_key:
                # Percentage metrics
                display_value = f"{value * 100:.2f}%"
            elif "sharpe" in metric_key:
                # Ratio metrics
                display_value = f"{value:.2f}"
            else:
                # Days or counts
                display_value = f"{int(value)}"
            
            with cols[col_idx % 3]:
                st.metric(metric_label, display_value)
            
            col_idx += 1
    
    st.markdown("---")
    
    # Additional insights
    st.subheader("Key Takeaways")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(
            "**Returns**: This strategy generated strong absolute returns "
            "with a healthy risk-adjusted Sharpe ratio."
        )
    
    with col2:
        st.warning(
            "**Drawdown**: 2022 tech crash caused a significant drawdown. "
            "Concentration in growth stocks amplified the decline."
        )
