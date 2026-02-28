"""
Vector Alpha — Educational Chart Library
==========================================

Clean, minimal Plotly charts designed for learning.
No clutter. Large labels. Educational annotations.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from config import (
    ASSET_COLORS,
    COLOR_PRIMARY,
    COLOR_POSITIVE,
    COLOR_NEGATIVE,
    COLOR_NEUTRAL,
    COLOR_ACCENT,
    PLOT_HEIGHT,
    CHART_TEMPLATE,
    CHART_MARGIN,
    TITLE_FONT_SIZE,
    PLOT_FONT_SIZE,
)


def _base_layout(title: str, height: int = PLOT_HEIGHT) -> dict:
    """Standard layout kwargs for all charts."""
    return dict(
        template=CHART_TEMPLATE,
        height=height,
        margin=CHART_MARGIN,
        title=dict(text=title, font=dict(size=TITLE_FONT_SIZE)),
        font=dict(size=PLOT_FONT_SIZE),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )


# ============================================================================
# EQUITY CURVE
# ============================================================================

def plot_equity_curve(equity: pd.Series, title: str = "Portfolio Growth") -> go.Figure:
    """Clean equity curve with drawdown shading."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity.index,
        y=equity.values,
        mode="lines",
        name="Portfolio Value",
        line=dict(color=COLOR_PRIMARY, width=2.5),
        hovertemplate="$%{y:.4f}<extra></extra>",
    ))

    # Add $1.00 reference line
    fig.add_hline(
        y=1.0, line_dash="dot", line_color=COLOR_NEUTRAL,
        annotation_text="Starting value ($1.00)",
        annotation_position="bottom right",
        annotation_font_color=COLOR_NEUTRAL,
    )

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Portfolio Value ($1 invested)",
        xaxis_title="",
    )

    return fig


# ============================================================================
# DRAWDOWN CHART
# ============================================================================

def plot_drawdown(dd_series: pd.Series, title: str = "Drawdown (Decline from Peak)") -> go.Figure:
    """Underwater chart showing peak-to-trough declines."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dd_series.index,
        y=dd_series.values * 100,
        mode="lines",
        fill="tozeroy",
        name="Drawdown",
        line=dict(color=COLOR_NEGATIVE, width=1.5),
        fillcolor="rgba(220, 38, 38, 0.15)",
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))

    # Annotate worst drawdown
    worst_idx = dd_series.idxmin()
    worst_val = dd_series.min() * 100
    fig.add_annotation(
        x=worst_idx, y=worst_val,
        text=f"Worst: {worst_val:.1f}%",
        showarrow=True, arrowhead=2,
        font=dict(color=COLOR_NEGATIVE, size=12),
    )

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Drawdown (%)",
        xaxis_title="",
    )

    return fig


# ============================================================================
# CORRELATION HEATMAP
# ============================================================================

def plot_correlation_heatmap(
    corr: pd.DataFrame,
    title: str = "Asset Correlations",
) -> go.Figure:
    """Annotated correlation heatmap."""
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=11),
        colorbar=dict(title="Correlation"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title, height=max(400, 50 * len(corr))),
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )

    return fig


# ============================================================================
# RISK CONTRIBUTION BAR CHART
# ============================================================================

def plot_risk_contribution(
    risk_summary: pd.DataFrame,
    title: str = "Risk Contribution by Asset",
) -> go.Figure:
    """Horizontal bar chart of risk contributions."""
    pct = risk_summary["pct_of_portfolio_vol"].sort_values(ascending=True)

    colors = [ASSET_COLORS.get(asset, COLOR_PRIMARY) for asset in pct.index]

    fig = go.Figure(data=go.Bar(
        x=pct.values * 100,
        y=pct.index.tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in pct.values * 100],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:.1f}% of portfolio risk<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title),
        xaxis_title="Share of Portfolio Risk (%)",
        yaxis_title="",
        showlegend=False,
    )

    return fig


# ============================================================================
# RETURN CONTRIBUTION BAR CHART
# ============================================================================

def plot_return_contribution(
    return_summary: pd.DataFrame,
    title: str = "Return Contribution by Asset",
) -> go.Figure:
    """Bar chart of return contributions, colored green/red."""
    contrib = return_summary["cumulative_contribution"].sort_values(ascending=True)

    colors = [COLOR_POSITIVE if v >= 0 else COLOR_NEGATIVE for v in contrib.values]

    fig = go.Figure(data=go.Bar(
        x=contrib.values * 100,
        y=contrib.index.tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in contrib.values * 100],
        textposition="outside",
        hovertemplate="<b>%{y}</b>: %{x:+.1f}% contribution<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title),
        xaxis_title="Cumulative Return Contribution (%)",
        yaxis_title="",
        showlegend=False,
    )

    return fig


# ============================================================================
# WEIGHT DRIFT AREA CHART
# ============================================================================

def plot_weight_drift(
    daily_weights: pd.DataFrame,
    title: str = "Portfolio Weight Changes Over Time",
) -> go.Figure:
    """Stacked area chart showing how weights drift."""
    fig = go.Figure()

    for col in daily_weights.columns:
        fig.add_trace(go.Scatter(
            x=daily_weights.index,
            y=daily_weights[col].values * 100,
            mode="lines",
            name=col,
            stackgroup="one",
            line=dict(width=0.5, color=ASSET_COLORS.get(col, None)),
            hovertemplate=f"<b>{col}</b>: " + "%{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Weight (%)",
        yaxis=dict(range=[0, 100]),
        xaxis_title="",
    )

    return fig


# ============================================================================
# ROLLING VOLATILITY
# ============================================================================

def plot_rolling_volatility(
    rolling_vol: pd.Series,
    title: str = "Rolling Volatility",
) -> go.Figure:
    """Line chart of rolling annualized volatility."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values * 100,
        mode="lines",
        name="Volatility",
        line=dict(color=COLOR_ACCENT, width=2),
        hovertemplate="%{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Annualized Volatility (%)",
        xaxis_title="",
        showlegend=False,
    )

    return fig


# ============================================================================
# ROLLING SHARPE
# ============================================================================

def plot_rolling_sharpe(
    rolling_sr: pd.Series,
    title: str = "Rolling Sharpe Ratio",
) -> go.Figure:
    """Rolling Sharpe with reference line at 1.0."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_sr.index,
        y=rolling_sr.values,
        mode="lines",
        name="Sharpe",
        line=dict(color=COLOR_PRIMARY, width=2),
        hovertemplate="%{y:.2f}<extra></extra>",
    ))

    fig.add_hline(
        y=1.0, line_dash="dot", line_color=COLOR_POSITIVE,
        annotation_text="Good (1.0)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=0, line_dash="dot", line_color=COLOR_NEUTRAL,
    )

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Sharpe Ratio",
        xaxis_title="",
        showlegend=False,
    )

    return fig


# ============================================================================
# TURNOVER BAR CHART
# ============================================================================

def plot_turnover(
    turnover: pd.Series,
    title: str = "Portfolio Turnover at Each Rebalance",
) -> go.Figure:
    """Bar chart of turnover on rebalance dates."""
    fig = go.Figure(data=go.Bar(
        x=turnover.index,
        y=turnover.values * 100,
        marker_color=COLOR_ACCENT,
        hovertemplate="%{x|%Y-%m-%d}: %{y:.1f}% turnover<extra></extra>",
    ))

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Turnover (%)",
        xaxis_title="",
        showlegend=False,
    )

    return fig


# ============================================================================
# DIVERSIFICATION GAUGE
# ============================================================================

def plot_diversification_gauge(
    score: float,
    title: str = "Diversification Score",
) -> go.Figure:
    """Gauge chart showing 0-10 diversification score."""
    if score <= 3:
        bar_color = COLOR_NEGATIVE
    elif score <= 6:
        bar_color = COLOR_ACCENT
    else:
        bar_color = COLOR_POSITIVE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(text=title, font=dict(size=TITLE_FONT_SIZE)),
        number=dict(font=dict(size=40), suffix="/10"),
        gauge=dict(
            axis=dict(range=[0, 10], tickwidth=1),
            bar=dict(color=bar_color, thickness=0.75),
            bgcolor="white",
            steps=[
                dict(range=[0, 3], color="rgba(220,38,38,0.1)"),
                dict(range=[3, 6], color="rgba(245,158,11,0.1)"),
                dict(range=[6, 10], color="rgba(22,163,74,0.1)"),
            ],
            threshold=dict(
                line=dict(color="black", width=2),
                thickness=0.75,
                value=score,
            ),
        ),
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        font=dict(size=PLOT_FONT_SIZE),
    )

    return fig


# ============================================================================
# TIME-SEGMENT ATTRIBUTION
# ============================================================================

def plot_yearly_attribution(
    daily_contributions: pd.DataFrame,
    title: str = "Return Contribution by Year",
) -> go.Figure:
    """Grouped bar chart showing each asset's contribution per year."""
    # Group by year
    yearly = daily_contributions.groupby(daily_contributions.index.year).sum()

    fig = go.Figure()

    for col in yearly.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=yearly.index.astype(str),
            y=yearly[col].values * 100,
            marker_color=ASSET_COLORS.get(col, None),
            hovertemplate=f"<b>{col}</b> " + "%{x}: %{y:+.1f}%<extra></extra>",
        ))

    fig.update_layout(
        **_base_layout(title),
        barmode="group",
        xaxis_title="Year",
        yaxis_title="Return Contribution (%)",
    )

    return fig


# ============================================================================
# INDIVIDUAL ASSET RETURNS
# ============================================================================

def plot_asset_cumulative_returns(
    returns: pd.DataFrame,
    title: str = "Individual Asset Performance",
) -> go.Figure:
    """Line chart of cumulative returns for each asset."""
    cum_returns = (1 + returns).cumprod()

    fig = go.Figure()

    for col in cum_returns.columns:
        fig.add_trace(go.Scatter(
            x=cum_returns.index,
            y=cum_returns[col].values,
            mode="lines",
            name=col,
            line=dict(
                color=ASSET_COLORS.get(col, None),
                width=1.5,
            ),
            hovertemplate=f"<b>{col}</b>: " + "$%{y:.2f}<extra></extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dot", line_color=COLOR_NEUTRAL)

    fig.update_layout(
        **_base_layout(title),
        yaxis_title="Growth of $1",
        xaxis_title="",
    )

    return fig
