"""
Vector Alpha Dashboard - Plotting Utilities (Enhanced)
=====================================================

All Plotly chart creation functions.
NO Streamlit calls (pure plotting library).
Each function returns a Plotly figure object.

Key improvement: Every asset and strategy gets a DISTINCT color and line style
so charts are visually differentiable.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import (
    COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_ACCENT,
    PLOT_HEIGHT, TITLE_FONT_SIZE,
    STRATEGY_COLORS, ASSET_COLORS, ASSET_DASH,
)


def _get_asset_color(asset: str) -> str:
    return ASSET_COLORS.get(asset, "#636efa")


def _get_asset_dash(asset: str) -> str:
    return ASSET_DASH.get(asset, "solid")


def _get_strategy_color(strategy: str) -> str:
    return STRATEGY_COLORS.get(strategy, "#636efa")


# ============================================================================
# EQUITY CURVE
# ============================================================================

def plot_equity_curve(returns: pd.Series, title: str = "Equity Curve") -> go.Figure:
    cumulative = (1 + returns).cumprod()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cumulative.index, y=cumulative.values,
        mode='lines', name='Equity Curve',
        line=dict(color=COLOR_POSITIVE, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Cumulative: %{y:.2f}x<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Date",
        yaxis_title="Cumulative Return (log scale)",
        height=PLOT_HEIGHT, hovermode='x unified', yaxis_type="log",
        template="plotly_white", font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# STRATEGY COMPARISON EQUITY CURVES
# ============================================================================

def plot_strategy_equity_curves(
    equity_df: pd.DataFrame,
    title: str = "Strategy Comparison: Equity Curves",
) -> go.Figure:
    """Plot equity curves for multiple strategies with distinct colors."""
    fig = go.Figure()
    for col in equity_df.columns:
        color = _get_strategy_color(col)
        fig.add_trace(go.Scatter(
            x=equity_df.index, y=equity_df[col].values,
            mode='lines', name=col,
            line=dict(color=color, width=2),
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{col}: %{{y:.3f}}<extra></extra>'
        ))
    fig.update_layout(
        title=title, xaxis_title="Date",
        yaxis_title="Cumulative Return (log scale)",
        height=PLOT_HEIGHT, hovermode='x unified', yaxis_type="log",
        template="plotly_white", font=dict(size=TITLE_FONT_SIZE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# INDIVIDUAL ASSET PERFORMANCE (DISTINCT COLORS + LINE STYLES)
# ============================================================================

def plot_asset_cumulative_returns(
    returns: pd.DataFrame,
    assets: list = None,
    title: str = "Individual Asset Performance",
) -> go.Figure:
    """Plot cumulative returns per asset with unique color and dash style."""
    if assets is None:
        assets = [c for c in returns.columns if c != "TOTAL"]

    fig = go.Figure()
    for asset in assets:
        if asset not in returns.columns:
            continue
        cum = (1 + returns[asset]).cumprod()
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            mode='lines', name=asset,
            line=dict(color=_get_asset_color(asset), width=2,
                      dash=_get_asset_dash(asset)),
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{asset}: %{{y:.2f}}x<extra></extra>'
        ))
    fig.update_layout(
        title=title, xaxis_title="Date",
        yaxis_title="Cumulative Return (log scale)",
        height=PLOT_HEIGHT, hovermode='x unified', yaxis_type="log",
        template="plotly_white", font=dict(size=TITLE_FONT_SIZE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# RETURNS DISTRIBUTION
# ============================================================================

def plot_returns_histogram(returns: pd.Series, title: str = "Daily Returns Distribution") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values * 100, name='Daily Returns', nbinsx=50,
        marker=dict(color=COLOR_POSITIVE, opacity=0.7),
        hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        height=PLOT_HEIGHT, showlegend=False, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# DRAWDOWN
# ============================================================================

def plot_drawdown(returns: pd.Series, title: str = "Drawdown Over Time") -> go.Figure:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        mode='lines', name='Drawdown', fill='tozeroy',
        line=dict(color=COLOR_NEGATIVE, width=1),
        fillcolor='rgba(214, 39, 40, 0.3)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=PLOT_HEIGHT, hovermode='x unified', template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


def plot_strategy_drawdowns(
    equity_df: pd.DataFrame,
    strategies: list = None,
    title: str = "Drawdown Comparison",
) -> go.Figure:
    if strategies is None:
        strategies = equity_df.columns.tolist()

    fig = go.Figure()
    for strat in strategies:
        if strat not in equity_df.columns:
            continue
        eq = equity_df[strat]
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        color = _get_strategy_color(strat)
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            mode='lines', name=strat,
            line=dict(color=color, width=1.5),
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{strat}: %{{y:.2f}}%<extra></extra>'
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Drawdown (%)",
        height=PLOT_HEIGHT, hovermode='x unified', template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# ROLLING METRICS
# ============================================================================

def plot_rolling_volatility(returns: pd.Series, window: int = 63, title: str = None) -> go.Figure:
    if title is None:
        title = f"Rolling Volatility ({window}-day)"
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol.values,
        mode='lines', name='Rolling Volatility',
        line=dict(color=COLOR_NEUTRAL, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volatility: %{y:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Annualized Volatility (%)",
        height=PLOT_HEIGHT, hovermode='x unified', template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 63, rf_rate: float = 0.0, title: str = None) -> go.Figure:
    if title is None:
        title = f"Rolling Sharpe Ratio ({window}-day)"
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - rf_rate) / rolling_std

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe.values,
        mode='lines', name='Rolling Sharpe',
        line=dict(color=COLOR_ACCENT, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Sharpe: %{y:.2f}<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Sharpe Ratio",
        height=PLOT_HEIGHT, hovermode='x unified', template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


def plot_strategy_rolling_sharpe(
    returns_df: pd.DataFrame, window: int = 63,
    strategies: list = None, title: str = None,
) -> go.Figure:
    if title is None:
        title = f"Rolling Sharpe Ratio ({window}-day) - All Strategies"
    if strategies is None:
        strategies = returns_df.columns.tolist()

    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_hline(y=1, line_dash="dot", line_color="green", opacity=0.4)
    fig.add_hline(y=-1, line_dash="dot", line_color="red", opacity=0.4)

    for strat in strategies:
        if strat not in returns_df.columns:
            continue
        rets = returns_df[strat]
        rs = (rets.rolling(window).mean() / rets.rolling(window).std()) * np.sqrt(252)
        color = _get_strategy_color(strat)
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            mode='lines', name=strat,
            line=dict(color=color, width=1.5),
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{strat}: %{{y:.2f}}<extra></extra>'
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title="Sharpe Ratio",
        height=PLOT_HEIGHT, hovermode='x unified', template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ============================================================================
# ATTRIBUTION
# ============================================================================

def plot_cumulative_attribution(attribution: pd.DataFrame, assets: list = None, title: str = "Cumulative Return Attribution") -> go.Figure:
    if assets is None:
        assets = [col for col in attribution.columns if col != "TOTAL"]
    cumulative_attr = attribution[assets].cumsum()
    fig = go.Figure()
    for asset in assets:
        fig.add_trace(go.Scatter(
            x=cumulative_attr.index, y=cumulative_attr[asset].values,
            mode='lines', name=asset, stackgroup='one',
            line=dict(color=_get_asset_color(asset)),
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{asset}: %{{y:.4f}}<extra></extra>'
        ))
    fig.update_layout(
        title=title, xaxis_title="Date",
        yaxis_title="Cumulative Return Contribution",
        height=PLOT_HEIGHT, hovermode='x unified', template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


def plot_attribution_bars(attribution_sum: pd.Series, title: str = "Total Return Contribution") -> go.Figure:
    colors = [_get_asset_color(a) if a in ASSET_COLORS else
              (COLOR_POSITIVE if x > 0 else COLOR_NEGATIVE)
              for a, x in zip(attribution_sum.index, attribution_sum.values)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=attribution_sum.index, y=attribution_sum.values,
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in attribution_sum.values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Contribution: %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Asset", yaxis_title="Return Contribution",
        height=PLOT_HEIGHT, showlegend=False, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


def plot_risk_contribution_bars(risk_attr_mean: pd.Series, title: str = "Average Risk Contribution") -> go.Figure:
    colors = [_get_asset_color(a) if a in ASSET_COLORS else COLOR_NEUTRAL
              for a in risk_attr_mean.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=risk_attr_mean.index, y=risk_attr_mean.values,
        marker=dict(color=colors),
        hovertemplate='<b>%{x}</b><br>Risk Contrib: %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Asset", yaxis_title="Risk Contribution",
        height=PLOT_HEIGHT, showlegend=False, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# STRATEGY COMPARISON BAR CHART
# ============================================================================

def plot_strategy_comparison_bars(
    comparison_df: pd.DataFrame, metric: str = "sharpe", title: str = None,
) -> go.Figure:
    if title is None:
        title = f"Strategy Comparison: {metric.replace('_', ' ').title()}"
    values = comparison_df[metric]
    colors = [_get_strategy_color(s) for s in values.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values.index, y=values.values,
        marker=dict(color=colors),
        text=[f"{v:.3f}" for v in values.values],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>%{y:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=title, xaxis_title="Strategy",
        yaxis_title=metric.replace("_", " ").title(),
        height=PLOT_HEIGHT, showlegend=False, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# WEIGHT HEATMAP
# ============================================================================

def plot_weight_heatmap(weights_df: pd.DataFrame, title: str = "Portfolio Weights") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=weights_df.values, x=weights_df.columns.tolist(),
        y=weights_df.index.tolist(), colorscale="RdYlGn",
        hovertemplate='<b>%{y}</b> / %{x}<br>Weight: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=title, height=PLOT_HEIGHT, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# FACTOR ANALYSIS PLOTS
# ============================================================================

def plot_factor_variance_explained(variance_df: pd.DataFrame, title: str = "Factor Variance Explained") -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    factors = [f"PC{i+1}" for i in range(len(variance_df))]
    fig.add_trace(go.Bar(
        x=factors, y=variance_df["explained_variance"].values * 100,
        name="Individual", marker_color=COLOR_POSITIVE, opacity=0.7,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=factors, y=variance_df["cumulative_variance"].values * 100,
        name="Cumulative", mode="lines+markers",
        line=dict(color=COLOR_ACCENT, width=2),
    ), secondary_y=True)
    fig.update_layout(
        title=title, height=PLOT_HEIGHT, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    fig.update_yaxes(title_text="Individual (%)", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative (%)", secondary_y=True)
    return fig


def plot_factor_loadings_heatmap(loadings_df: pd.DataFrame, title: str = "Factor Loadings") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(
        z=loadings_df.values, x=loadings_df.columns.tolist(),
        y=loadings_df.index.tolist(), colorscale="RdBu", zmid=0,
        text=np.round(loadings_df.values, 2), texttemplate="%{text}",
        hovertemplate='<b>%{y}</b> / %{x}<br>Loading: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=title, height=PLOT_HEIGHT, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# STRESS TEST CHART
# ============================================================================

def plot_stress_test_bars(
    stress_data: dict, metric: str = "total_return",
    strategies: list = None, title: str = "Stress Test Results",
) -> go.Figure:
    fig = go.Figure()
    scenarios = list(stress_data.keys())
    if strategies is None:
        sample_keys = list(stress_data.get(scenarios[0], {}).keys())
        strategies = sorted(set(
            k.replace(f"_{metric}", "") for k in sample_keys if k.endswith(f"_{metric}")
        ))

    for strat in strategies:
        values = []
        for scenario in scenarios:
            key = f"{strat}_{metric}"
            values.append(stress_data.get(scenario, {}).get(key, 0))
        color = _get_strategy_color(strat)
        fig.add_trace(go.Bar(
            name=strat, x=scenarios, y=[v * 100 for v in values],
            marker_color=color,
        ))
    fig.update_layout(
        title=title, barmode='group',
        xaxis_title="Scenario", yaxis_title=f"{metric.replace('_', ' ').title()} (%)",
        height=PLOT_HEIGHT, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig


# ============================================================================
# CORRELATION HEATMAP
# ============================================================================

def plot_correlation_heatmap(returns: pd.DataFrame, title: str = "Asset Correlation Matrix") -> go.Figure:
    corr = returns.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>',
    ))
    fig.update_layout(
        title=title, height=700, template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    return fig
