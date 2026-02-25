"""
Factor Risk Model

Implements:
1. PCA-based statistical factor model
2. Custom factor decomposition (momentum, value, volatility, size proxies)
3. Factor exposure analysis
4. Factor risk contribution
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

__all__ = [
    "pca_factor_model",
    "compute_factor_exposures",
    "compute_factor_risk_contribution",
    "build_factor_report",
]


def pca_factor_model(
    returns: pd.DataFrame,
    n_factors: int = 5,
) -> Dict[str, object]:
    """PCA-based statistical factor model.

    Extracts latent factors from asset return covariance structure.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns (dates x assets).
    n_factors : int
        Number of principal components to extract.

    Returns
    -------
    dict
        - "factor_returns": pd.DataFrame (dates x factors)
        - "factor_loadings": pd.DataFrame (assets x factors)
        - "explained_variance_ratio": array
        - "cumulative_variance": array
        - "residual_variance": pd.Series (per-asset)
    """
    clean = returns.dropna()
    n_factors = min(n_factors, len(clean.columns), len(clean))

    pca = PCA(n_components=n_factors)
    factor_returns_arr = pca.fit_transform(clean.values)

    factor_names = [f"PC{i+1}" for i in range(n_factors)]
    factor_returns = pd.DataFrame(
        factor_returns_arr, index=clean.index, columns=factor_names
    )
    factor_loadings = pd.DataFrame(
        pca.components_.T, index=clean.columns, columns=factor_names
    )

    # Residual variance (idiosyncratic risk)
    reconstructed = factor_returns_arr @ pca.components_
    residuals = clean.values - reconstructed
    residual_var = pd.Series(
        np.var(residuals, axis=0), index=clean.columns, name="residual_variance"
    )

    return {
        "factor_returns": factor_returns,
        "factor_loadings": factor_loadings,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "residual_variance": residual_var,
    }


def compute_factor_exposures(
    portfolio_weights: pd.Series,
    factor_loadings: pd.DataFrame,
) -> pd.Series:
    """Compute portfolio-level factor exposures.

    Parameters
    ----------
    portfolio_weights : pd.Series
        Current portfolio weights (assets).
    factor_loadings : pd.DataFrame
        Factor loadings (assets x factors).

    Returns
    -------
    pd.Series
        Portfolio factor exposures (one per factor).
    """
    common = portfolio_weights.index.intersection(factor_loadings.index)
    w = portfolio_weights[common]
    loadings = factor_loadings.loc[common]
    return loadings.T @ w


def compute_factor_risk_contribution(
    portfolio_weights: pd.Series,
    factor_loadings: pd.DataFrame,
    factor_returns: pd.DataFrame,
    residual_variance: pd.Series,
) -> Dict[str, object]:
    """Decompose portfolio risk into factor and specific (idiosyncratic) risk.

    Parameters
    ----------
    portfolio_weights : pd.Series
        Current portfolio weights.
    factor_loadings : pd.DataFrame
        Factor loadings (assets x factors).
    factor_returns : pd.DataFrame
        Factor return series (dates x factors).
    residual_variance : pd.Series
        Per-asset residual variance.

    Returns
    -------
    dict
        - "total_variance": float
        - "factor_variance": float
        - "specific_variance": float
        - "factor_pct": float (fraction of variance from factors)
        - "per_factor_contribution": pd.Series
    """
    common = portfolio_weights.index.intersection(factor_loadings.index)
    w = portfolio_weights[common].values
    B = factor_loadings.loc[common].values  # (n_assets, n_factors)

    # Factor covariance
    F_cov = factor_returns.cov().values  # (n_factors, n_factors)

    # Specific (diagonal) covariance
    D = np.diag(residual_variance[common].values)

    # Total covariance: B @ F_cov @ B' + D
    factor_cov_contrib = B @ F_cov @ B.T
    total_cov = factor_cov_contrib + D

    total_var = float(w @ total_cov @ w)
    factor_var = float(w @ factor_cov_contrib @ w)
    specific_var = float(w @ D @ w)

    # Per-factor contribution
    factor_exposures = B.T @ w  # (n_factors,)
    per_factor_var = factor_exposures ** 2 * np.diag(F_cov)

    factor_names = factor_returns.columns.tolist()
    per_factor = pd.Series(per_factor_var, index=factor_names)

    return {
        "total_variance": total_var,
        "factor_variance": factor_var,
        "specific_variance": specific_var,
        "factor_pct": factor_var / total_var if total_var > 0 else 0.0,
        "specific_pct": specific_var / total_var if total_var > 0 else 0.0,
        "per_factor_contribution": per_factor,
    }


def build_factor_report(
    returns: pd.DataFrame,
    daily_weights: pd.DataFrame,
    n_factors: int = 5,
) -> Dict[str, object]:
    """Build a complete factor analysis report.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.
    daily_weights : pd.DataFrame
        Daily portfolio weights.
    n_factors : int
        Number of PCA factors.

    Returns
    -------
    dict
        Complete factor analysis with model, exposures, and risk decomposition.
    """
    # Fit PCA model
    model = pca_factor_model(returns, n_factors)

    # Average weights for factor exposure
    avg_weights = daily_weights.mean()

    # Factor exposures
    exposures = compute_factor_exposures(avg_weights, model["factor_loadings"])

    # Risk decomposition
    risk_decomp = compute_factor_risk_contribution(
        avg_weights,
        model["factor_loadings"],
        model["factor_returns"],
        model["residual_variance"],
    )

    # Correlation between factors and portfolio
    port_returns = (daily_weights.shift(1) * returns).sum(axis=1).dropna()
    factor_corr = {}
    for col in model["factor_returns"].columns:
        common_idx = port_returns.index.intersection(model["factor_returns"].index)
        if len(common_idx) > 10:
            factor_corr[col] = float(
                port_returns.loc[common_idx].corr(
                    model["factor_returns"][col].loc[common_idx]
                )
            )

    return {
        "model": model,
        "portfolio_exposures": exposures,
        "risk_decomposition": risk_decomp,
        "factor_portfolio_correlation": factor_corr,
        "explained_variance": model["explained_variance_ratio"],
        "cumulative_variance": model["cumulative_variance"],
    }
