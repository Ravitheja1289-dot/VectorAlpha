"""
Vector Alpha — Risk Explorer Tab
==================================

Correlation heatmap, risk contribution, concentration, diversification score.
Teaches real diversification vs illusion of diversification.
"""

import numpy as np
import streamlit as st

from config import CONCEPT_DEFINITIONS, SECTION_DESCRIPTIONS
from plotting import (
    plot_correlation_heatmap,
    plot_risk_contribution,
    plot_diversification_gauge,
)
from insights_engine import compute_diversification_score


def show_risk_explorer(results: dict):
    """Render the Risk Explorer tab."""

    st.markdown(
        f"<p style='color: #94A3B8; margin-top: -10px;'>"
        f"{SECTION_DESCRIPTIONS['risk_explorer']}</p>",
        unsafe_allow_html=True,
    )

    # ---- Diversification Score ----
    st.markdown("#### Diversification Score")

    div_score = compute_diversification_score(results)

    col_gauge, col_breakdown = st.columns([1, 1])

    with col_gauge:
        fig_gauge = plot_diversification_gauge(div_score["score"])
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_breakdown:
        score = div_score["score"]

        if score <= 3:
            verdict = "Your portfolio is **poorly diversified**."
            advice = (
                "Consider adding assets from different sectors or "
                "reducing concentration in your top holdings."
            )
        elif score <= 6:
            verdict = "Your portfolio has **moderate diversification**."
            advice = (
                "There's room for improvement. Assets with lower correlation "
                "to your existing holdings would improve the score."
            )
        else:
            verdict = "Your portfolio is **well diversified**."
            advice = (
                "You have a good spread of risk across assets with "
                "relatively low correlation."
            )

        st.markdown(verdict)
        st.markdown(advice)

        st.markdown("**Score Breakdown:**")
        st.markdown(
            f"- Weight balance: **{div_score['hhi_score']}/10** "
            f"(how evenly spread your weights are)"
        )
        st.markdown(
            f"- Correlation: **{div_score['corr_score']}/10** "
            f"(how independently assets move)"
        )
        st.markdown(
            f"- Risk balance: **{div_score['risk_score']}/10** "
            f"(how evenly risk is distributed)"
        )

        if div_score["avg_correlation"] is not None:
            st.caption(
                f"Average pairwise correlation: {div_score['avg_correlation']:.2f} | "
                f"Weight concentration (HHI): {div_score['hhi']:.3f}"
            )

    st.markdown("---")

    # ---- Correlation Heatmap ----
    st.markdown("#### Asset Correlations")
    st.caption(
        "This heatmap shows how closely each pair of assets moves together. "
        "Red = move together (high correlation). Blue = move apart (low/negative correlation). "
        "For good diversification, you want lower correlations."
    )

    corr = results.get("correlation")
    if corr is not None and not corr.empty:
        fig_corr = plot_correlation_heatmap(corr)
        st.plotly_chart(fig_corr, use_container_width=True)

        # Highlight key finding
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        corr_vals = corr.where(mask).stack()
        if len(corr_vals) > 0:
            max_pair = corr_vals.idxmax()
            max_corr = corr_vals.max()
            min_pair = corr_vals.idxmin()
            min_corr = corr_vals.min()

            col1, col2 = st.columns(2)
            with col1:
                st.info(
                    f"**Most correlated:** {max_pair[0]} & {max_pair[1]} "
                    f"({max_corr:.2f}) — these move very similarly."
                )
            with col2:
                st.info(
                    f"**Least correlated:** {min_pair[0]} & {min_pair[1]} "
                    f"({min_corr:.2f}) — these provide the best diversification."
                )

    st.markdown("---")

    # ---- Risk Contribution ----
    st.markdown("#### Risk Contribution by Asset")
    st.caption(
        "This chart shows what percentage of your portfolio's total risk "
        "comes from each asset. An asset's risk contribution depends on its "
        "volatility AND how correlated it is with other assets in your portfolio."
    )

    risk_attr = results.get("risk_attribution")
    if risk_attr is not None:
        fig_risk = plot_risk_contribution(risk_attr["summary"])
        st.plotly_chart(fig_risk, use_container_width=True)

        # Concentration ratio
        summary = risk_attr["summary"]
        pct = summary["pct_of_portfolio_vol"].sort_values(ascending=False)
        n_assets = len(pct)
        top_2_share = pct.head(2).sum() * 100

        st.markdown(
            f"<div style='background: #7F1D1D; padding: 12px 16px; "
            f"border-radius: 8px; border-left: 4px solid #DC2626; color: #F1F5F9;'>"
            f"Even though you hold <strong>{n_assets} assets</strong>, "
            f"<strong>{top_2_share:.0f}%</strong> of the risk comes from just 2. "
            f"This is the <em>diversification illusion</em> — owning more assets "
            f"doesn't guarantee lower risk if they behave similarly."
            f"</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Learn more: Risk Contribution", expanded=False):
        st.markdown(CONCEPT_DEFINITIONS["risk_contribution"])
