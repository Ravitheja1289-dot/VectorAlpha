"""
Vector Alpha Dashboard - Real-Time Monitoring Component
=====================================================

Display live/latest market data, portfolio valuation,
and market status using yfinance API.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from config import ASSETS, ASSET_COLORS
from utils_plotting import plot_asset_cumulative_returns


def show_realtime(data: dict) -> None:
    """Render the Real-Time Monitoring section."""

    st.subheader("Real-Time Market Monitor")

    st.caption(
        "Live market data powered by Yahoo Finance API. "
        "Click 'Refresh Data' to fetch latest prices."
    )

    # Refresh button
    if st.button("Refresh Data", key="refresh_realtime"):
        st.cache_data.clear()
        st.rerun()

    # Fetch real-time data
    try:
        snapshot = _fetch_snapshot(ASSETS)
    except Exception as e:
        st.error(f"Failed to fetch real-time data: {e}")
        st.info("Make sure you have `yfinance` installed: `pip install yfinance`")
        return

    if snapshot is None:
        st.warning("Could not retrieve market data.")
        return

    # 1. Market Status
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "OPEN" if snapshot["market_open"] else "CLOSED"
        st.metric("Market Status", status)
    with col2:
        st.metric("Last Updated", snapshot["timestamp"].strftime("%H:%M:%S"))
    with col3:
        st.metric("Assets Tracked", len(snapshot["prices"]))

    st.markdown("---")

    # 2. Price ticker board
    st.subheader("Live Price Board")

    # Create rows of 5 columns
    assets_with_data = [a for a in ASSETS if a in snapshot["prices"] and not np.isnan(snapshot["prices"].get(a, np.nan))]

    for i in range(0, len(assets_with_data), 5):
        batch = assets_with_data[i:i+5]
        cols = st.columns(len(batch))
        for j, asset in enumerate(batch):
            price = snapshot["prices"].get(asset, 0)
            change = snapshot["pct_changes"].get(asset, 0)
            with cols[j]:
                st.metric(
                    asset,
                    f"${price:.2f}",
                    delta=f"{change*100:+.2f}%",
                    delta_color="normal" if change >= 0 else "inverse"
                )

    st.markdown("---")

    # 3. Daily movers
    st.subheader("Today's Movers")
    col1, col2 = st.columns(2)

    pct_changes = snapshot["pct_changes"]
    sorted_changes = pct_changes.sort_values(ascending=False)

    with col1:
        st.markdown("**Top Gainers**")
        for asset in sorted_changes.head(5).index:
            val = sorted_changes[asset]
            color = ASSET_COLORS.get(asset, "#2ca02c")
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{asset}</span>: "
                f"<span style='color:green;'>{val*100:+.2f}%</span>",
                unsafe_allow_html=True
            )

    with col2:
        st.markdown("**Top Losers**")
        for asset in sorted_changes.tail(5).index:
            val = sorted_changes[asset]
            color = ASSET_COLORS.get(asset, "#d62728")
            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>{asset}</span>: "
                f"<span style='color:red;'>{val*100:+.2f}%</span>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # 4. Volume analysis
    st.subheader("Volume Analysis")
    volumes = snapshot["volumes"]
    if len(volumes) > 0:
        vol_df = pd.DataFrame({
            "Asset": volumes.index,
            "Volume": volumes.values,
            "Volume (M)": (volumes.values / 1e6).round(1),
        }).sort_values("Volume", ascending=False)
        st.dataframe(vol_df.set_index("Asset"), use_container_width=True)

    st.markdown("---")

    # 5. Recent returns chart (from precomputed data)
    returns = data.get("returns")
    if returns is not None and not returns.empty:
        st.subheader("Recent Performance (Last 30 Trading Days)")
        recent = returns.tail(30)
        asset_cols = [c for c in recent.columns if c != "TOTAL"]
        fig = plot_asset_cumulative_returns(
            recent, assets=asset_cols,
            title="Last 30 Days: Cumulative Asset Returns"
        )
        st.plotly_chart(fig, use_container_width=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def _fetch_snapshot(symbols: list) -> dict:
    """Fetch latest market snapshot. Cached for 5 minutes."""
    try:
        from data.loaders.realtime_loader import fetch_realtime_prices
        snap = fetch_realtime_prices(symbols)
        return {
            "prices": snap.prices,
            "changes": snap.changes,
            "pct_changes": snap.pct_changes,
            "volumes": snap.volumes,
            "timestamp": snap.timestamp,
            "market_open": snap.market_open,
        }
    except ImportError:
        # If running from dashboard dir, try relative import
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.loaders.realtime_loader import fetch_realtime_prices
        snap = fetch_realtime_prices(symbols)
        return {
            "prices": snap.prices,
            "changes": snap.changes,
            "pct_changes": snap.pct_changes,
            "volumes": snap.volumes,
            "timestamp": snap.timestamp,
            "market_open": snap.market_open,
        }
