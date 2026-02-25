"""
Vector Alpha Dashboard - Main App (Enhanced)
============================================

Streamlit orchestrator for the Vector Alpha research dashboard.

Pages:
1. Overview - KPIs and system summary
2. Performance - Returns, distributions, asset-level analysis
3. Drawdown & Risk - Drawdown, rolling vol, rolling Sharpe
4. Attribution - Return and risk attribution by asset
5. Strategy Comparison - Multi-strategy equity curves, metrics, weights
6. Factor Analysis - PCA factors, loadings, correlation
7. Stress Test & VaR - Advanced risk, stress testing, tail risk
8. Real-Time Monitor - Live prices, market status, intraday data
"""

import sys
from pathlib import Path

# Ensure project root is on path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from config import ASSETS, PROJECT_SUBTITLE
from data_loader import load_all_data
from components_overview import show_overview
from components_performance import show_performance
from components_drawdown_risk import show_drawdown_risk
from components_attribution import show_attribution
from components_strategies import show_strategies
from components_factors import show_factors
from components_stress import show_stress_test
from components_realtime import show_realtime

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
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    return load_all_data()

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
    [
        "Overview",
        "Performance",
        "Drawdown & Risk",
        "Attribution",
        "Strategy Comparison",
        "Factor Analysis",
        "Stress Test & VaR",
        "Real-Time Monitor",
    ],
    index=0,
    key="main_nav"
)

st.sidebar.markdown("---")

# Sidebar info
st.sidebar.subheader("Portfolio")
st.sidebar.metric("Assets Tracked", len(ASSETS))
st.sidebar.metric("Data Points", len(data["returns"]))
st.sidebar.metric("Period (Years)", f"{(data['returns'].index[-1] - data['returns'].index[0]).days / 365:.1f}")

# Strategy count
strategy_eq = data.get("strategy_equity_curves")
if strategy_eq is not None and not strategy_eq.empty:
    st.sidebar.metric("Strategies", len(strategy_eq.columns))

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

elif page == "Strategy Comparison":
    show_strategies(data)

elif page == "Factor Analysis":
    show_factors(data)

elif page == "Stress Test & VaR":
    show_stress_test(data)

elif page == "Real-Time Monitor":
    show_realtime(data)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px; margin-top: 2rem;">
        Vector Alpha Research Dashboard | 8 Strategies | 15 Assets | Real-Time Monitoring<br>
        <strong>Disclaimer:</strong> For research and backtesting purposes only. Not investment advice.
    </div>
    """,
    unsafe_allow_html=True
)
