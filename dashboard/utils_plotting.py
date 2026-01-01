"""
Vector Alpha Dashboard - Plotting Utilities
===========================================

All Plotly chart creation functions.
NO Streamlit calls (pure plotting library).
Each function returns a Plotly figure object.

Design:
- Single responsibility: Each function creates one type of chart
- No data manipulation: Accept pre-computed data
- Reusable: Called from multiple components
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from config import COLOR_POSITIVE, COLOR_NEGATIVE, COLOR_NEUTRAL, COLOR_ACCENT, PLOT_HEIGHT, TITLE_FONT_SIZE


# ============================================================================
# EQUITY CURVE
# ============================================================================

def plot_equity_curve(returns: pd.Series, title: str = "Equity Curve") -> go.Figure:
    """
    Plot cumulative equity curve from daily returns.
    
    Args:
        returns: Series of daily returns (TOTAL column)
        title: Chart title
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Visual overview of portfolio performance
        - Identifies drawdown periods visually
        - Log scale shows consistent percentage growth
    """
    cumulative = (1 + returns).cumprod()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative.values,
        mode='lines',
        name='Equity Curve',
        line=dict(color=COLOR_POSITIVE, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Cumulative Return: %{y:.2f}x<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return (log scale)",
        height=PLOT_HEIGHT,
        hovermode='x unified',
        yaxis_type="log",
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


# ============================================================================
# RETURNS DISTRIBUTION
# ============================================================================

def plot_returns_histogram(returns: pd.Series, title: str = "Daily Returns Distribution") -> go.Figure:
    """
    Plot histogram of daily returns.
    
    Args:
        returns: Series of daily returns
        title: Chart title
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Understand return distribution (skew, tail risk)
        - Identify outliers
        - No cumulative calculations (pure daily returns)
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=returns.values * 100,  # Convert to percentage
        name='Daily Returns',
        nbinsx=50,
        marker=dict(color=COLOR_POSITIVE, opacity=0.7),
        hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency",
        height=PLOT_HEIGHT,
        showlegend=False,
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


# ============================================================================
# DRAWDOWN
# ============================================================================

def plot_drawdown(returns: pd.Series, title: str = "Drawdown Over Time") -> go.Figure:
    """
    Plot cumulative drawdown from daily returns.
    
    Args:
        returns: Series of daily returns
        title: Chart title
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Show peak-to-trough declines
        - Identify prolonged underwater periods
        - Understand recovery duration
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,  # Convert to percentage
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        line=dict(color=COLOR_NEGATIVE, width=1),
        fillcolor=f'rgba(214, 39, 40, 0.3)',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=PLOT_HEIGHT,
        hovermode='x unified',
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


# ============================================================================
# ROLLING METRICS
# ============================================================================

def plot_rolling_volatility(returns: pd.Series, window: int = 63, title: str = None) -> go.Figure:
    """
    Plot rolling volatility (annualized).
    
    Args:
        returns: Series of daily returns
        window: Rolling window size (days)
        title: Chart title (auto-generated if None)
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Track changing market volatility
        - Identify periods of elevated risk
        - Compare current vs. historical volatility
    """
    if title is None:
        title = f"Rolling Volatility ({window}-day)"
    
    rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100  # Annualized, percentage
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=rolling_vol.index,
        y=rolling_vol.values,
        mode='lines',
        name='Rolling Volatility',
        line=dict(color=COLOR_NEUTRAL, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volatility: %{y:.2f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Annualized Volatility (%)",
        height=PLOT_HEIGHT,
        hovermode='x unified',
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


def plot_rolling_sharpe(returns: pd.Series, window: int = 63, rf_rate: float = 0.0, title: str = None) -> go.Figure:
    """
    Plot rolling Sharpe ratio.
    
    Args:
        returns: Series of daily returns
        window: Rolling window size (days)
        rf_rate: Risk-free rate (annualized, default 0%)
        title: Chart title (auto-generated if None)
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Track changing risk-adjusted returns
        - Identify periods of outperformance/underperformance
        - Assess strategy consistency
    """
    if title is None:
        title = f"Rolling Sharpe Ratio ({window}-day)"
    
    rolling_mean = returns.rolling(window).mean() * 252
    rolling_std = returns.rolling(window).std() * np.sqrt(252)
    rolling_sharpe = (rolling_mean - rf_rate) / rolling_std
    
    fig = go.Figure()
    
    # Background shading for positive/negative Sharpe
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="0")
    
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        mode='lines',
        name='Rolling Sharpe',
        line=dict(color=COLOR_ACCENT, width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Sharpe: %{y:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Sharpe Ratio",
        height=PLOT_HEIGHT,
        hovermode='x unified',
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


# ============================================================================
# ATTRIBUTION
# ============================================================================

def plot_cumulative_attribution(attribution: pd.DataFrame, assets: list = None, title: str = "Cumulative Return Attribution") -> go.Figure:
    """
    Plot stacked cumulative return attribution by asset.
    
    Args:
        attribution: DataFrame of daily return attribution
        assets: List of asset names to plot (default: all except TOTAL)
        title: Chart title
        
    Returns:
        Plotly figure object
        
    Purpose:
        - See which assets drive cumulative returns
        - Identify periods of positive/negative contribution
        - Track diversification benefit
    """
    if assets is None:
        assets = [col for col in attribution.columns if col != "TOTAL"]
    
    cumulative_attr = attribution[assets].cumsum()
    
    fig = go.Figure()
    
    for asset in assets:
        fig.add_trace(go.Scatter(
            x=cumulative_attr.index,
            y=cumulative_attr[asset].values,
            mode='lines',
            name=asset,
            stackgroup='one',
            hovertemplate=f'<b>%{{x|%Y-%m-%d}}</b><br>{asset}: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Cumulative Return Contribution",
        height=PLOT_HEIGHT,
        hovermode='x unified',
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


def plot_attribution_bars(attribution_sum: pd.Series, title: str = "Total Return Contribution") -> go.Figure:
    """
    Plot bar chart of total return contribution by asset.
    
    Args:
        attribution_sum: Series of summed return contributions by asset
        title: Chart title
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Quick ranking of top performers / detractors
        - Identify key drivers of overall returns
    """
    colors = [COLOR_POSITIVE if x > 0 else COLOR_NEGATIVE for x in attribution_sum.values]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=attribution_sum.index,
        y=attribution_sum.values,
        marker=dict(color=colors),
        text=attribution_sum.values,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Contribution: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Return Contribution",
        height=PLOT_HEIGHT,
        showlegend=False,
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig


def plot_risk_contribution_bars(risk_attr_mean: pd.Series, title: str = "Average Risk Contribution") -> go.Figure:
    """
    Plot bar chart of average risk contribution by asset.
    
    Args:
        risk_attr_mean: Series of mean risk contributions by asset
        title: Chart title
        
    Returns:
        Plotly figure object
        
    Purpose:
        - Identify which assets create portfolio volatility
        - Understand concentration of risk
        - Compare to return contribution
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=risk_attr_mean.index,
        y=risk_attr_mean.values,
        marker=dict(color=COLOR_NEUTRAL),
        hovertemplate='<b>%{x}</b><br>Avg Risk Contrib: %{y:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Asset",
        yaxis_title="Risk Contribution",
        height=PLOT_HEIGHT,
        showlegend=False,
        template="plotly_white",
        font=dict(size=TITLE_FONT_SIZE),
    )
    
    return fig
