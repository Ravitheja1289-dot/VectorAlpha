"""
Vector Alpha — Section 1: Build Your Portfolio
================================================

Clean, minimal inputs for portfolio construction.
Students select assets, set weights, choose settings, and simulate.
"""

import streamlit as st
import pandas as pd

from config import (
    ASSET_INFO,
    ASSETS,
    DEFAULT_ASSETS,
    DEFAULT_REBALANCE,
    DEFAULT_COST_BPS,
    REBALANCE_OPTIONS,
    REBALANCE_MAP,
    SECTION_DESCRIPTIONS,
)


def show_build_section():
    """Render the portfolio builder UI. Returns config dict or None."""

    st.markdown(
        f"<p style='color: #94A3B8; margin-top: -10px;'>"
        f"{SECTION_DESCRIPTIONS['build']}</p>",
        unsafe_allow_html=True,
    )

    # ----- Asset selection -----
    st.markdown("#### Select Assets")
    st.caption("Choose the stocks you want in your portfolio.")

    # Format labels with company name and sector
    asset_labels = {
        t: f"{t} — {ASSET_INFO[t]['name']} ({ASSET_INFO[t]['sector']})"
        for t in ASSETS
    }

    selected = st.multiselect(
        "Assets",
        options=ASSETS,
        default=DEFAULT_ASSETS,
        format_func=lambda t: asset_labels[t],
        key="asset_selector",
        label_visibility="collapsed",
    )

    if len(selected) < 2:
        st.warning("Please select at least 2 assets to build a portfolio.")
        return None

    # ----- Weight allocation -----
    st.markdown("#### Set Weights")

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.caption("Allocate a percentage to each asset. Weights must add up to 100%.")
    with col_btn:
        equal_weight = st.button(
            "Equal Weight",
            key="equal_weight_btn",
            use_container_width=True,
        )

    weights = {}
    equal_val = round(100.0 / len(selected), 1)

    cols = st.columns(min(len(selected), 3))
    for i, asset in enumerate(selected):
        col_idx = i % min(len(selected), 3)
        with cols[col_idx]:
            default_w = equal_val if equal_weight or f"weight_{asset}" not in st.session_state else None
            w = st.number_input(
                f"{asset}",
                min_value=0.0,
                max_value=100.0,
                value=default_w if default_w is not None else equal_val,
                step=1.0,
                key=f"weight_{asset}" if not equal_weight else f"weight_{asset}_eq",
                format="%.1f",
            )
            weights[asset] = w

    total_weight = sum(weights.values())
    remaining = 100.0 - total_weight

    if abs(remaining) < 0.1:
        st.success(f"Weights total: {total_weight:.1f}%")
    elif remaining > 0:
        st.warning(f"Weights total: {total_weight:.1f}% — {remaining:.1f}% unallocated")
    else:
        st.error(f"Weights total: {total_weight:.1f}% — exceeds 100% by {abs(remaining):.1f}%")

    st.markdown("---")

    # ----- Settings -----
    st.markdown("#### Settings")

    col1, col2 = st.columns(2)

    with col1:
        rebalance = st.radio(
            "Rebalancing Frequency",
            options=REBALANCE_OPTIONS,
            index=REBALANCE_OPTIONS.index(DEFAULT_REBALANCE),
            key="rebalance_freq",
            help="How often to reset weights back to your target allocation.",
        )

        date_col1, date_col2 = st.columns(2)
        with date_col1:
            start_date = st.date_input(
                "Start Date",
                value=pd.Timestamp("2020-01-02"),
                min_value=pd.Timestamp("2020-01-01"),
                max_value=pd.Timestamp("2025-12-31"),
                key="start_date",
            )
        with date_col2:
            end_date = st.date_input(
                "End Date",
                value=pd.Timestamp("2025-06-30"),
                min_value=pd.Timestamp("2020-01-01"),
                max_value=pd.Timestamp("2025-12-31"),
                key="end_date",
            )

    with col2:
        show_costs = st.checkbox(
            "Include transaction costs",
            value=True,
            key="input_show_costs",
            help=f"Apply {DEFAULT_COST_BPS} basis points per trade when rebalancing.",
        )
        show_drift = st.checkbox(
            "Show weight drift effect",
            value=False,
            key="input_show_drift",
            help="Visualize how asset weights change between rebalances.",
        )

    st.markdown("---")

    # ----- Simulate button -----
    can_run = abs(remaining) < 0.5 and len(selected) >= 2 and start_date < end_date

    simulate = st.button(
        "Simulate Portfolio",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
        key="simulate_btn",
    )

    if simulate and can_run:
        # Normalize weights to sum to 1.0
        w_normalized = {k: v / total_weight for k, v in weights.items()}

        return {
            "assets": selected,
            "weights": w_normalized,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "rebalance_freq": REBALANCE_MAP[rebalance],
            "cost_bps": DEFAULT_COST_BPS if show_costs else 0.0,
            "show_costs": show_costs,
            "show_drift": show_drift,
        }

    return None
