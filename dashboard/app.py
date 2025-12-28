"""
Vector Alpha Dashboard - Main App
================================

Streamlit orchestrator for the Vector Alpha research dashboard.

Architecture:
- Page config (title, layout, logo)
- Data loading (centralized, cached)
- Sidebar navigation
- Component dispatcher

Design Principles:
- Zero business logic (all precomputed data)
- Pure orchestration (route to components)
- Centralized error handling
- Loud failures (user-visible errors)
"""

import streamlit as st
from config import ASSETS, PROJECT_SUBTITLE
from data_loader import load_all_data
from components_overview import show_overview
from components_performance import show_performance
from components_drawdown_risk import show_drawdown_risk
from components_attribution import show_attribution

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Vector Alpha | Research Dashboard",
    page_icon="chart-line",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Vector Alpha Research Dashboard")
st.markdown(PROJECT_SUBTITLE)

# ============================================================================
# DATA LOADING (Centralized, Cached)
# ============================================================================

@st.cache_data
def load_data():
    """Load all data once per session."""
    return load_all_data()

# Try to load data; fail loudly if there's an issue
try:
    data = load_data()
    st.session_state["data_loaded"] = True
except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select View",
    ["Overview", "Performance", "Drawdown & Risk", "Attribution"],
    index=0,
    key="main_nav"
)

st.sidebar.markdown("---")

# Sidebar info
st.sidebar.subheader("Portfolio")
st.sidebar.metric("Assets Tracked", len(ASSETS))
st.sidebar.metric("Data Points", len(data["returns"]))
st.sidebar.metric("Period (Years)", f"{(data['returns'].index[-1] - data['returns'].index[0]).days / 365:.1f}")

# ============================================================================
# PAGE DISPATCHER
# ============================================================================

if page == "Overview":
    show_overview(data["risk_metrics"])

elif page == "Performance":
    show_performance(data["returns"], data["return_attribution"])

elif page == "Drawdown & Risk":
    show_drawdown_risk(data["returns"], data["risk_attribution"])

elif page == "Attribution":
    show_attribution(data["return_attribution"], data["risk_attribution"])

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px; margin-top: 2rem;">
        Vector Alpha Research Dashboard<br>
        <strong>Disclaimer:</strong> For research and backtesting purposes only. Not investment advice.
    </div>
    """,
    unsafe_allow_html=True
)
