"""
Portfolio Optimization Module

Implements multiple optimization approaches:
1. Mean-Variance Optimization (Markowitz)
2. Risk Parity (Equal Risk Contribution)
3. Hierarchical Risk Parity (HRP) - Lopez de Prado
4. Minimum Variance
5. Maximum Sharpe

No strategy logic, no execution. Pure weight optimization.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

__all__ = [
    "mean_variance_optimize",
    "minimum_variance",
    "maximum_sharpe",
    "risk_parity",
    "hierarchical_risk_parity",
    "inverse_volatility",
]


# ============================================================================
# MEAN-VARIANCE OPTIMIZATION (Markowitz)
# ============================================================================

def mean_variance_optimize(
    returns: pd.DataFrame,
    target_return: Optional[float] = None,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> pd.Series:
    """Classic Markowitz mean-variance optimization.

    If target_return is None, maximizes Sharpe ratio.
    Otherwise, minimizes variance subject to target return.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns (dates x assets).
    target_return : float or None
        Target portfolio return (annualized). If None, maximize Sharpe.
    risk_free_rate : float
        Annual risk-free rate.
    max_weight : float
        Maximum weight per asset.
    min_weight : float
        Minimum weight per asset.

    Returns
    -------
    pd.Series
        Optimal weights indexed by asset name.
    """
    n = len(returns.columns)
    mu = returns.mean().values * 252  # annualized expected returns
    cov = returns.cov().values * 252  # annualized covariance

    if target_return is None:
        # Maximize Sharpe ratio => minimize negative Sharpe
        def neg_sharpe(w):
            port_ret = w @ mu - risk_free_rate
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol < 1e-12:
                return 1e6
            return -port_ret / port_vol

        objective = neg_sharpe
    else:
        # Minimize variance subject to target return
        def port_var(w):
            return w @ cov @ w

        objective = port_var

    # Constraints
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if target_return is not None:
        constraints.append(
            {"type": "eq", "fun": lambda w: w @ mu - target_return}
        )

    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if not result.success:
        # Fall back to equal weight on failure
        weights = np.ones(n) / n
    else:
        weights = result.x

    # Renormalize (numerical precision)
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


# ============================================================================
# MINIMUM VARIANCE
# ============================================================================

def minimum_variance(
    returns: pd.DataFrame,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> pd.Series:
    """Minimum variance portfolio.

    Minimizes portfolio variance without any return target.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.
    max_weight : float
        Maximum weight per asset.
    min_weight : float
        Minimum weight per asset.

    Returns
    -------
    pd.Series
        Optimal weights.
    """
    n = len(returns.columns)
    cov = returns.cov().values * 252

    def port_var(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(
        port_var, x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 1000, "ftol": 1e-12},
    )

    weights = result.x if result.success else np.ones(n) / n
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


# ============================================================================
# MAXIMUM SHARPE
# ============================================================================

def maximum_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.25,
    min_weight: float = 0.0,
) -> pd.Series:
    """Maximum Sharpe ratio portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.
    risk_free_rate : float
        Annual risk-free rate.
    max_weight : float
        Maximum weight per asset.
    min_weight : float
        Minimum weight per asset.

    Returns
    -------
    pd.Series
        Optimal weights.
    """
    return mean_variance_optimize(
        returns,
        target_return=None,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        min_weight=min_weight,
    )


# ============================================================================
# RISK PARITY (Equal Risk Contribution)
# ============================================================================

def risk_parity(
    returns: pd.DataFrame,
    max_weight: float = 0.25,
    min_weight: float = 0.01,
) -> pd.Series:
    """Risk parity: equalize risk contribution across assets.

    Solves: minimize sum_i (RC_i - sigma_p/N)^2
    where RC_i = w_i * (Sigma @ w)_i / sigma_p

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.
    max_weight : float
        Maximum weight per asset.
    min_weight : float
        Minimum weight per asset.

    Returns
    -------
    pd.Series
        Risk parity weights.
    """
    n = len(returns.columns)
    cov = returns.cov().values * 252

    def risk_parity_obj(w):
        port_var = w @ cov @ w
        if port_var < 1e-16:
            return 1e6
        port_vol = np.sqrt(port_var)
        marginal = cov @ w
        risk_contrib = w * marginal / port_vol
        target_rc = port_vol / n
        return np.sum((risk_contrib - target_rc) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n
    x0 = np.ones(n) / n

    result = minimize(
        risk_parity_obj, x0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"maxiter": 2000, "ftol": 1e-14},
    )

    weights = result.x if result.success else np.ones(n) / n
    weights = weights / weights.sum()
    return pd.Series(weights, index=returns.columns)


# ============================================================================
# HIERARCHICAL RISK PARITY (HRP) - Lopez de Prado
# ============================================================================

def hierarchical_risk_parity(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (HRP) by Marcos Lopez de Prado.

    Steps:
    1. Compute correlation distance matrix
    2. Hierarchical clustering (single linkage)
    3. Quasi-diagonalization (reorder by cluster)
    4. Recursive bisection to allocate weights

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.

    Returns
    -------
    pd.Series
        HRP weights.
    """
    corr = returns.corr()
    cov = returns.cov() * 252
    assets = returns.columns.tolist()
    n = len(assets)

    # Step 1: Distance matrix from correlation
    dist = np.sqrt(0.5 * (1 - corr.values))
    np.fill_diagonal(dist, 0)

    # Step 2: Hierarchical clustering
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")

    # Step 3: Quasi-diagonalization (reorder assets by cluster)
    sorted_idx = leaves_list(link).tolist()
    sorted_assets = [assets[i] for i in sorted_idx]

    # Step 4: Recursive bisection
    def _get_cluster_var(cov_matrix, cluster_items):
        """Compute inverse-variance portfolio variance for a cluster."""
        sub_cov = cov_matrix.loc[cluster_items, cluster_items]
        ivp = 1.0 / np.diag(sub_cov.values)
        ivp = ivp / ivp.sum()
        return float(ivp @ sub_cov.values @ ivp)

    def _recursive_bisection(cov_matrix, sorted_items):
        """Recursively bisect and allocate weights."""
        w = pd.Series(1.0, index=sorted_items)

        cluster_items = [sorted_items]
        while len(cluster_items) > 0:
            next_clusters = []
            for cluster in cluster_items:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = _get_cluster_var(cov_matrix, left)
                var_right = _get_cluster_var(cov_matrix, right)

                # Allocate based on inverse variance
                alpha = 1.0 - var_left / (var_left + var_right)

                w[left] *= alpha
                w[right] *= (1.0 - alpha)

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)

            cluster_items = next_clusters

        return w

    weights = _recursive_bisection(cov, sorted_assets)
    weights = weights / weights.sum()

    # Reindex to original order
    return weights.reindex(assets)


# ============================================================================
# INVERSE VOLATILITY
# ============================================================================

def inverse_volatility(returns: pd.DataFrame) -> pd.Series:
    """Simple inverse-volatility weighting.

    w_i = (1/vol_i) / sum(1/vol_j)

    Parameters
    ----------
    returns : pd.DataFrame
        Historical asset returns.

    Returns
    -------
    pd.Series
        Inverse-vol weights.
    """
    vol = returns.std() * np.sqrt(252)
    inv_vol = 1.0 / vol.clip(lower=1e-10)
    return inv_vol / inv_vol.sum()


# ============================================================================
# STRATEGY WRAPPER: Optimization-based Strategy
# ============================================================================

class OptimizedStrategy:
    """Wrapper that uses optimization methods within the Strategy interface.

    Parameters
    ----------
    method : str
        Optimization method: "mvo", "min_var", "max_sharpe",
        "risk_parity", "hrp", "inv_vol".
    lookback : int
        Number of days of historical returns to use for optimization.
    max_weight : float
        Maximum weight per asset.
    """

    def __init__(
        self,
        method: str = "risk_parity",
        lookback: int = 252,
        max_weight: float = 0.25,
    ):
        self.method = method
        self.lookback = lookback
        self.max_weight = max_weight

    def generate_weights(
        self,
        features: Dict[str, pd.DataFrame],
        rebalance_dates: list,
    ) -> pd.DataFrame:
        daily_rets = features.get("daily_returns")
        if daily_rets is None:
            raise ValueError("OptimizedStrategy requires 'daily_returns' in features")

        assets = daily_rets.columns.tolist()
        n_assets = len(assets)
        base_weight = 1.0 / n_assets

        opt_func = {
            "mvo": lambda r: mean_variance_optimize(r, max_weight=self.max_weight),
            "min_var": lambda r: minimum_variance(r, max_weight=self.max_weight),
            "max_sharpe": lambda r: maximum_sharpe(r, max_weight=self.max_weight),
            "risk_parity": lambda r: risk_parity(r, max_weight=self.max_weight),
            "hrp": hierarchical_risk_parity,
            "inv_vol": inverse_volatility,
        }

        if self.method not in opt_func:
            raise ValueError(f"Unknown method: {self.method}. Choose from {list(opt_func.keys())}")

        weights_list = []
        for date in rebalance_dates:
            loc = daily_rets.index.get_loc(date)
            start = loc - self.lookback

            if start < 0:
                w = pd.Series(base_weight, index=assets, name=date)
            else:
                hist = daily_rets.iloc[start:loc + 1]
                try:
                    w = opt_func[self.method](hist)
                    w.name = date
                except Exception:
                    w = pd.Series(base_weight, index=assets, name=date)

            weights_list.append(w)

        return pd.DataFrame(weights_list)
