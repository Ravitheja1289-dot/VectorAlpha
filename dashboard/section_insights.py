"""
Vector Alpha — Section 3: What This Means
===========================================

Plain-English educational insights.
Each answers: What happened? Why? What concept does this demonstrate?
"""

import streamlit as st

from config import SECTION_DESCRIPTIONS
from insights_engine import generate_all_insights


# Category icons (text-based, no emoji)
CATEGORY_STYLES = {
    "Performance": {"bg": "#1E3A8A", "border": "#60A5FA", "icon": "P"},
    "Risk":        {"bg": "#7F1D1D", "border": "#F87171", "icon": "R"},
    "Rebalancing": {"bg": "#78350F", "border": "#FBBF24", "icon": "B"},
    "Diversification": {"bg": "#14532D", "border": "#4ADE80", "icon": "D"},
    "Attribution": {"bg": "#4C1D95", "border": "#A78BFA", "icon": "A"},
}


def show_insights_section(results: dict):
    """Render the learning insights page."""

    st.markdown(
        f"<p style='color: #94A3B8; margin-top: -10px;'>"
        f"{SECTION_DESCRIPTIONS['insights']}</p>",
        unsafe_allow_html=True,
    )

    insights = generate_all_insights(results)

    if not insights:
        st.info("Run a simulation to see insights about your portfolio.")
        return

    # Group by category
    categories = {}
    for insight in insights:
        cat = insight["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(insight)

    for category, items in categories.items():
        style = CATEGORY_STYLES.get(category, {"bg": "#F9FAFB", "border": "#6B7280", "icon": "?"})

        st.markdown(f"#### {category}")

        for item in items:
            st.markdown(
                f"""<div style="
                    background: {style['bg']};
                    border-left: 4px solid {style['border']};
                    padding: 16px 20px;
                    border-radius: 8px;
                    margin-bottom: 12px;
                ">
                    <strong style="font-size: 1.05em; color: #F1F5F9;">{item['headline']}</strong><br>
                    <span style="color: #E2E8F0;">{item['explanation']}</span><br>
                    <span style="color: #94A3B8; font-size: 0.9em; font-style: italic;">
                        Concept: {item['concept']}
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("")  # Spacing
