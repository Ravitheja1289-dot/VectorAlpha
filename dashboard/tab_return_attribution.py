"""
Vector Alpha — Return Breakdown Tab
=====================================

Shows which assets helped, which hurt, and how contributions changed over time.
Turns attribution into a story.
"""

import streamlit as st
import pandas as pd

from config import SECTION_DESCRIPTIONS, COLOR_POSITIVE, COLOR_NEGATIVE
from plotting import (
    plot_return_contribution,
    plot_yearly_attribution,
)


def show_return_attribution(results: dict):
    """Render the Return Breakdown tab."""

    st.markdown(
        f"<p style='color: #94A3B8; margin-top: -10px;'>"
        f"{SECTION_DESCRIPTIONS['return_breakdown']}</p>",
        unsafe_allow_html=True,
    )

    ret_attr = results.get("return_attribution")
    if ret_attr is None:
        st.info("No attribution data available.")
        return

    summary = ret_attr["summary"]
    daily_contrib = ret_attr["daily_contributions"]

    # ---- Overall contribution ----
    st.markdown("#### Contribution by Asset")
    st.caption(
        "This chart shows how much each asset added (or subtracted) "
        "from your total portfolio return. Green bars helped, red bars hurt."
    )

    fig_contrib = plot_return_contribution(summary)
    st.plotly_chart(fig_contrib, use_container_width=True)

    st.markdown("---")

    # ---- Positive vs Negative split ----
    st.markdown("#### Winners vs Losers")

    positive = summary[summary["cumulative_contribution"] >= 0].sort_values(
        "cumulative_contribution", ascending=False
    )
    negative = summary[summary["cumulative_contribution"] < 0].sort_values(
        "cumulative_contribution", ascending=True
    )

    col_pos, col_neg = st.columns(2)

    with col_pos:
        st.markdown(
            f"<div style='background: #14532D; padding: 12px 16px; "
            f"border-radius: 8px; border-left: 4px solid {COLOR_POSITIVE}; color: #F1F5F9;'>"
            f"<strong>Positive Contributors</strong></div>",
            unsafe_allow_html=True,
        )
        if len(positive) > 0:
            for asset in positive.index:
                val = positive.loc[asset, "cumulative_contribution"] * 100
                pct = positive.loc[asset, "pct_contribution"] * 100
                st.markdown(
                    f"**{asset}**: +{val:.1f}% "
                    f"({pct:.0f}% of total return)"
                )
        else:
            st.markdown("No assets with positive contribution.")

    with col_neg:
        st.markdown(
            f"<div style='background: #7F1D1D; padding: 12px 16px; "
            f"border-radius: 8px; border-left: 4px solid {COLOR_NEGATIVE}; color: #F1F5F9;'>"
            f"<strong>Negative Contributors</strong></div>",
            unsafe_allow_html=True,
        )
        if len(negative) > 0:
            for asset in negative.index:
                val = negative.loc[asset, "cumulative_contribution"] * 100
                st.markdown(f"**{asset}**: {val:+.1f}%")
        else:
            st.markdown("All assets contributed positively.")

    st.markdown("---")

    # ---- Time-segment attribution ----
    st.markdown("#### Return Contribution by Year")
    st.caption(
        "See how each asset's contribution changed over time. "
        "An asset that helped one year may have hurt the next."
    )

    if len(daily_contrib) > 0:
        fig_yearly = plot_yearly_attribution(daily_contrib)
        st.plotly_chart(fig_yearly, use_container_width=True)

        # Generate narrative
        yearly = daily_contrib.groupby(daily_contrib.index.year).sum()
        narratives = []

        for year in yearly.index:
            year_data = yearly.loc[year]
            best_asset = year_data.idxmax()
            worst_asset = year_data.idxmin()
            best_val = year_data[best_asset] * 100
            worst_val = year_data[worst_asset] * 100

            if best_val > 0 and worst_val < 0:
                narratives.append(
                    f"**{year}**: {best_asset} was the top contributor "
                    f"(+{best_val:.1f}%), while {worst_asset} dragged "
                    f"returns ({worst_val:+.1f}%)."
                )
            elif best_val > 0:
                narratives.append(
                    f"**{year}**: {best_asset} led the way (+{best_val:.1f}%). "
                    f"All assets contributed positively."
                )
            else:
                narratives.append(
                    f"**{year}**: A difficult year. {worst_asset} was hit "
                    f"hardest ({worst_val:+.1f}%)."
                )

        if narratives:
            st.markdown("**Year-by-Year Story:**")
            for n in narratives:
                st.markdown(f"- {n}")
