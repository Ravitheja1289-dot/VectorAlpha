"""
Vector Alpha Dashboard - Factor Analysis Component
=================================================

Display PCA factor model results, factor loadings,
variance explained, and correlation analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils_plotting import (
    plot_factor_variance_explained,
    plot_factor_loadings_heatmap,
    plot_correlation_heatmap,
)


def show_factors(data: dict) -> None:
    """Render the Factor Analysis section."""

    st.subheader("Factor Risk Model (PCA)")

    factor_loadings = data.get("factor_loadings")
    factor_variance = data.get("factor_variance")
    returns = data.get("returns")

    if factor_loadings is None or factor_variance is None:
        st.warning(
            "Factor analysis data not available. "
            "Run `python run_experiment.py` to generate factor model outputs."
        )

        # Still show correlation heatmap from returns
        if returns is not None and not returns.empty:
            st.markdown("---")
            st.subheader("Asset Correlation Matrix")
            asset_cols = [c for c in returns.columns if c != "TOTAL"]
            fig_corr = plot_correlation_heatmap(
                returns[asset_cols],
                title="Asset Return Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        return

    # 1. Variance explained
    st.markdown("#### Variance Explained by Principal Components")
    fig_var = plot_factor_variance_explained(
        factor_variance,
        title="PCA Factor Model: Variance Explained"
    )
    st.plotly_chart(fig_var, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "PC1 Explains",
            f"{factor_variance['explained_variance'].iloc[0]*100:.1f}%"
        )
    with col2:
        st.metric(
            "Top 3 Factors",
            f"{factor_variance['cumulative_variance'].iloc[min(2, len(factor_variance)-1)]*100:.1f}%"
        )
    with col3:
        st.metric(
            "All Factors",
            f"{factor_variance['cumulative_variance'].iloc[-1]*100:.1f}%"
        )

    st.caption(
        "**Interpretation**: PC1 typically represents the market factor. "
        "Higher cumulative variance means the factors explain more of the return variation."
    )

    st.markdown("---")

    # 2. Factor loadings heatmap
    st.markdown("#### Factor Loadings (Asset Exposure to Each Factor)")
    fig_load = plot_factor_loadings_heatmap(
        factor_loadings,
        title="Factor Loadings: How Each Asset Loads on Each Factor"
    )
    st.plotly_chart(fig_load, use_container_width=True)

    st.caption(
        "**Interpretation**: Red = positive loading (moves with factor), "
        "Blue = negative loading (moves against factor). "
        "Assets with similar loading patterns are correlated."
    )

    st.markdown("---")

    # 3. Correlation heatmap
    if returns is not None and not returns.empty:
        st.subheader("Asset Correlation Matrix")
        asset_cols = [c for c in returns.columns if c != "TOTAL"]
        fig_corr = plot_correlation_heatmap(
            returns[asset_cols],
            title="Asset Return Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.caption(
            "**Interpretation**: High positive correlation (red) means assets move together. "
            "Low or negative correlation (blue) provides diversification benefit."
        )

    st.markdown("---")

    # 4. Factor loadings table
    st.subheader("Factor Loadings Table")
    st.dataframe(
        factor_loadings.round(4),
        use_container_width=True
    )
