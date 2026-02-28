"""
Vector Alpha — Interactive Portfolio Lab for Finance Students
==============================================================

Main Streamlit application. Clean, educational, minimal.

Experiment with portfolios and see how risk, rebalancing,
and diversification actually behave in real markets.
"""

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    APP_TAGLINE,
    LANDING_DESCRIPTION,
    SECTION_HEADERS,
)
from engine import run_portfolio_lab
from section_build import show_build_section
from section_results import show_results_section
from section_insights import show_insights_section
from tab_risk_explorer import show_risk_explorer
from tab_return_attribution import show_return_attribution
from tab_advanced import show_advanced_mode


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title=f"{APP_TITLE} | {APP_SUBTITLE}",
    page_icon="chart-line",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — Clean, educational look
# ============================================================================

st.markdown("""
<style>
    /* Reduce top padding */
    .block-container { padding-top: 2rem; }

    /* Dark theme for tab content */
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 8px;
        color: #F1F5F9;
    }
    
    /* Force light text in tab panels */
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] span,
    .stTabs [data-baseweb="tab-panel"] div,
    .stTabs [data-baseweb="tab-panel"] label {
        color: #F1F5F9 !important;
    }
    
    /* Dark theme headers in tabs */
    .stTabs [data-baseweb="tab-panel"] h1,
    .stTabs [data-baseweb="tab-panel"] h2,
    .stTabs [data-baseweb="tab-panel"] h3,
    .stTabs [data-baseweb="tab-panel"] h4 {
        color: #F8FAFC !important;
    }

    /* Dark metric cards */
    .stTabs [data-testid="stMetric"] {
        background: #334155;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 12px 16px;
    }
    
    .stTabs [data-testid="stMetric"] label {
        color: #94A3B8 !important;
    }
    
    .stTabs [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F1F5F9 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 6px;
    }
    
    /* Dark expanders in tabs */
    .stTabs [data-baseweb="tab-panel"] .streamlit-expanderHeader {
        background-color: #334155;
        color: #F1F5F9;
        border-radius: 6px;
    }
    
    .stTabs [data-baseweb="tab-panel"] .streamlit-expanderContent {
        background-color: #1E293B;
        border-left: 2px solid #475569;
    }
    
    /* Dark dataframes in tabs */
    .stTabs [data-baseweb="tab-panel"] .stDataFrame {
        background-color: #334155;
    }
    
    /* Input fields in tabs remain readable */
    .stTabs [data-baseweb="tab-panel"] input,
    .stTabs [data-baseweb="tab-panel"] textarea,
    .stTabs [data-baseweb="tab-panel"] select {
        background-color: #334155 !important;
        color: #F1F5F9 !important;
        border: 1px solid #475569 !important;
    }

    /* Button styling */
    .stButton > button[kind="primary"] {
        background: #2563EB;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1.05em;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================

st.markdown(
    f"<h1 style='margin-bottom: 0;'>{APP_TITLE}</h1>"
    f"<p style='color: #2563EB; font-size: 1.2em; margin-top: 0;'>"
    f"{APP_SUBTITLE}</p>",
    unsafe_allow_html=True,
)
st.markdown(f"*{APP_TAGLINE}*")


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(f"### {APP_TITLE}")
    st.caption(LANDING_DESCRIPTION)

    st.markdown("---")

    # Advanced mode toggle
    advanced_mode = st.toggle(
        "Advanced Mode",
        value=False,
        key="advanced_toggle",
        help="Show detailed analytics like weight drift, turnover, and rolling metrics.",
    )

    st.markdown("---")

    # Info
    st.markdown("**How to use:**")
    st.markdown(
        "1. Select assets and set weights\n"
        "2. Choose your rebalancing strategy\n"
        "3. Click **Simulate Portfolio**\n"
        "4. Explore the results across tabs"
    )

    st.markdown("---")
    st.caption(
        "Vector Alpha | Portfolio Learning Lab\n\n"
        "For educational purposes only. Not investment advice."
    )


# ============================================================================
# MAIN CONTENT — TABS
# ============================================================================

# Define tabs based on whether results exist
has_results = "results" in st.session_state and st.session_state["results"] is not None

if has_results and advanced_mode:
    tab_names = [
        SECTION_HEADERS["build"],
        SECTION_HEADERS["results"],
        SECTION_HEADERS["insights"],
        SECTION_HEADERS["risk_explorer"],
        SECTION_HEADERS["return_breakdown"],
        SECTION_HEADERS["advanced"],
    ]
elif has_results:
    tab_names = [
        SECTION_HEADERS["build"],
        SECTION_HEADERS["results"],
        SECTION_HEADERS["insights"],
        SECTION_HEADERS["risk_explorer"],
        SECTION_HEADERS["return_breakdown"],
    ]
else:
    tab_names = [SECTION_HEADERS["build"]]

tabs = st.tabs(tab_names)


# ---- Tab 1: Build Your Portfolio ----
with tabs[0]:
    st.markdown(f"### {SECTION_HEADERS['build']}")
    portfolio_config = show_build_section()

    if portfolio_config is not None:
        # Run simulation
        with st.spinner("Simulating your portfolio..."):
            try:
                results = run_portfolio_lab(
                    assets=portfolio_config["assets"],
                    weights=portfolio_config["weights"],
                    start_date=portfolio_config["start_date"],
                    end_date=portfolio_config["end_date"],
                    rebalance_freq=portfolio_config["rebalance_freq"],
                    cost_bps=portfolio_config["cost_bps"],
                    show_costs=portfolio_config["show_costs"],
                )
                st.session_state["results"] = results
                st.session_state["results_show_drift"] = portfolio_config["show_drift"]
                st.success("Simulation complete! Navigate to the other tabs to explore results.")
                st.rerun()
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                st.session_state["results"] = None


# ---- Remaining tabs (only if results exist) ----
if has_results:
    results = st.session_state["results"]

    with tabs[1]:
        st.markdown(f"### {SECTION_HEADERS['results']}")
        show_results_section(results)

    with tabs[2]:
        st.markdown(f"### {SECTION_HEADERS['insights']}")
        show_insights_section(results)

    with tabs[3]:
        st.markdown(f"### {SECTION_HEADERS['risk_explorer']}")
        show_risk_explorer(results)

    with tabs[4]:
        st.markdown(f"### {SECTION_HEADERS['return_breakdown']}")
        show_return_attribution(results)

    if advanced_mode and len(tabs) > 5:
        with tabs[5]:
            st.markdown(f"### {SECTION_HEADERS['advanced']}")
            show_advanced_mode(results)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #9CA3AF; font-size: 12px; "
    "margin-top: 1rem;'>"
    "Vector Alpha — Interactive Portfolio Lab for Finance Students<br>"
    "Built for learning. Not investment advice."
    "</div>",
    unsafe_allow_html=True,
)
