"""
Vector Alpha Dashboard - Attribution Component
============================================

Display which assets drove portfolio returns and risk.

Design:
- Cumulative return attribution (stacked area)
- Total return contribution by asset (bar chart)
- Average risk contribution by asset (bar chart)
- Controls: Date range slider, top-N selector (3/5/10)
"""

import streamlit as st
import pandas as pd

from utils_plotting import (
    plot_cumulative_attribution,
    plot_attribution_bars,
    plot_risk_contribution_bars
)
from utils_helpers import (
    filter_by_date_range,
    get_top_contributors,
    get_date_range_stats
)


def show_attribution(return_attribution: pd.DataFrame, risk_attribution: pd.DataFrame) -> None:
    """
    Render the Attribution section.
    
    Args:
        return_attribution: DataFrame of daily return attribution by asset
        risk_attribution: DataFrame of daily risk (volatility) attribution by asset
        
    Returns:
        None (renders Streamlit components)
        
    Purpose:
        - Answer: which assets drove returns?
        - Answer: which assets took on the risk?
        - Answer: did we get paid for the risk we took?
        - Attribution to individual positions
        
    Why three charts?
        1. Cumulative (stacked): See how each asset's contribution grew over time
        2. Total bars: Compare final contribution across assets (simple ranking)
        3. Risk bars: See risk contribution vs return contribution (reward/risk)
    """
    
    # Sidebar controls
    st.sidebar.subheader("Attribution Filters")
    
    # Date range selector
    min_date = return_attribution.index.min().date()
    max_date = return_attribution.index.max().date()
    
    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    with col_end:
        end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Top-N selector
    top_n = st.sidebar.selectbox("Show Top N Assets", [3, 5, 10], index=1)
    
    # Filter data by date range
    filtered_ret_attr = filter_by_date_range(return_attribution, start_date, end_date)
    # risk_attribution is already aggregated (one row per asset), no date filtering needed
    filtered_risk_attr = risk_attribution
    
    # Get date range stats for display
    date_stats = get_date_range_stats(filtered_ret_attr)
    
    st.subheader("Return Attribution Analysis")
    st.caption(f"Period: {date_stats['start_date'].strftime('%Y-%m-%d')} to {date_stats['end_date'].strftime('%Y-%m-%d')} ({date_stats['num_days']} days)")
    
    # 1. Cumulative attribution (stacked area)
    st.markdown("#### Cumulative Return Contribution (Stacked)")
    
    # Get top contributors by total return contribution
    ret_sum = filtered_ret_attr.sum()
    top_assets_series = get_top_contributors(ret_sum, n=top_n)
    top_assets = top_assets_series.index.tolist()
    
    fig_cumret = plot_cumulative_attribution(
        filtered_ret_attr,
        assets=top_assets,
        title=f"Cumulative Return Contribution (Top {top_n} Assets)"
    )
    st.plotly_chart(fig_cumret, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Stacked area shows each asset's cumulative contribution to total returns. "
        "Wider area = larger contribution. Heights show relative importance over time."
    )
    
    st.markdown("---")
    
    # 2. Total return attribution (bars)
    st.markdown("#### Total Return Contribution (By Asset)")
    
    ret_sum = filtered_ret_attr.sum().sort_values(ascending=False)
    top_ret_assets = ret_sum.head(top_n).index.tolist()
    
    fig_bars = plot_attribution_bars(
        ret_sum,
        title=f"Total Return Contribution (Top {top_n} Assets)"
    )
    st.plotly_chart(fig_bars, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Bar chart shows total return contribution by asset over the period. "
        "Positive bars = long/winning positions; negative bars = short/losing positions."
    )
    
    st.markdown("---")
    
    # 3. Risk contribution (bars)
    st.markdown("#### Average Risk Contribution (By Asset)")
    
    # Use risk_contribution column from the aggregated risk_attribution data
    if 'risk_contribution' in filtered_risk_attr.columns:
        risk_contrib = filtered_risk_attr['risk_contribution'].sort_values(ascending=False)
    else:
        # Fallback to marginal_contribution if risk_contribution doesn't exist
        risk_contrib = filtered_risk_attr.get('marginal_contribution', filtered_risk_attr.iloc[:, 0]).sort_values(ascending=False)
    
    fig_risk = plot_risk_contribution_bars(
        risk_contrib.head(top_n),
        title=f"Average Daily Risk Contribution (Top {top_n} Assets)"
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    st.caption(
        "**Interpretation**: Bar chart shows average daily risk (volatility) contribution by asset. "
        "Compares with return contribution to assess if we were compensated for the risk taken."
    )
    
    st.markdown("---")
    
    st.subheader("Attribution Summary Table")
    
    # Create summary table
    summary_data = []
    for asset in top_ret_assets:
        ret_contrib = filtered_ret_attr[asset].sum()
        
        # Get risk contribution from the aggregated risk_attribution data
        if asset in filtered_risk_attr.index:
            if 'risk_contribution' in filtered_risk_attr.columns:
                risk_contrib = filtered_risk_attr.loc[asset, 'risk_contribution']
            else:
                risk_contrib = filtered_risk_attr.loc[asset, 'marginal_contribution']
        else:
            risk_contrib = 0
        
        ratio = ret_contrib / risk_contrib if risk_contrib != 0 else 0
        
        summary_data.append({
            "Asset": asset,
            "Return Contribution": f"{ret_contrib * 100:.2f}%",
            "Avg Risk (Daily Vol)": f"{risk_contrib * 100:.2f}%",
            "Return/Risk Ratio": f"{ratio:.2f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    st.caption(
        "**Return/Risk Ratio**: How much return per unit of risk. "
        "Higher is better. Compare across assets to identify best alpha sources."
    )
