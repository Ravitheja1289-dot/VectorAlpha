"""
Vector Alpha — Insight Generator
==================================

Generates plain-English, educational insights from portfolio results.
No jargon. Every insight answers: What happened? Why? What concept?
"""

import numpy as np
import pandas as pd


def generate_all_insights(results: dict) -> list:
    """Generate all insights from portfolio results.

    Returns a list of dicts: {"category", "headline", "explanation", "concept"}
    """
    insights = []
    insights.extend(generate_performance_insights(results))
    insights.extend(generate_risk_insights(results))
    insights.extend(generate_rebalance_insights(results))
    insights.extend(generate_diversification_insights(results))
    insights.extend(generate_attribution_insights(results))
    return insights


# ============================================================================
# PERFORMANCE INSIGHTS
# ============================================================================

def generate_performance_insights(results: dict) -> list:
    insights = []

    cagr = results["cagr"]
    vol = results["volatility"]
    sharpe = results["sharpe"]
    max_dd = results["max_drawdown"]

    # Overall return assessment
    if cagr > 0.20:
        tone = "strong"
    elif cagr > 0.10:
        tone = "moderate"
    elif cagr > 0:
        tone = "modest"
    else:
        tone = "negative"

    insights.append({
        "category": "Performance",
        "headline": f"Your portfolio returned {cagr*100:.1f}% per year",
        "explanation": (
            f"This is a {tone} annualized return. "
            f"For context, the long-term average for the S&P 500 is about 10% per year. "
            f"Your result reflects the specific assets and time period you chose."
        ),
        "concept": "CAGR (Compound Annual Growth Rate) smooths out year-to-year fluctuations to show the average annual return.",
    })

    # Risk-return tradeoff
    if vol > 0.30:
        vol_desc = "high"
    elif vol > 0.15:
        vol_desc = "moderate"
    else:
        vol_desc = "low"

    insights.append({
        "category": "Performance",
        "headline": f"Volatility of {vol*100:.1f}% means {vol_desc} risk",
        "explanation": (
            f"Your portfolio fluctuates by roughly {vol*100:.1f}% per year. "
            f"In practical terms, you could see swings of "
            f"{vol*100*2:.0f}% in a bad year. "
            f"Higher return usually requires accepting higher volatility."
        ),
        "concept": "Volatility measures uncertainty. It's the standard deviation of returns, scaled to a yearly number.",
    })

    # Sharpe assessment
    if sharpe > 1.0:
        sharpe_desc = "good"
    elif sharpe > 0.5:
        sharpe_desc = "acceptable"
    else:
        sharpe_desc = "poor"

    insights.append({
        "category": "Performance",
        "headline": f"Risk-adjusted return is {sharpe_desc} (Sharpe: {sharpe:.2f})",
        "explanation": (
            f"The Sharpe Ratio of {sharpe:.2f} means you earned "
            f"{sharpe:.2f} units of return for every unit of risk. "
            f"A ratio above 1.0 is generally considered good."
        ),
        "concept": "The Sharpe Ratio helps compare strategies that have different risk levels. More return per unit of risk is better.",
    })

    # Drawdown
    insights.append({
        "category": "Performance",
        "headline": f"Worst decline was {abs(max_dd)*100:.1f}%",
        "explanation": (
            f"At the worst point, your portfolio dropped {abs(max_dd)*100:.1f}% "
            f"from its peak. Recovery took {results['drawdown_duration']} trading days. "
            f"This is the pain you would have felt as an investor."
        ),
        "concept": "Maximum drawdown shows your worst-case scenario. It's the largest peak-to-trough drop in portfolio value.",
    })

    return insights


# ============================================================================
# RISK & CONCENTRATION INSIGHTS
# ============================================================================

def generate_risk_insights(results: dict) -> list:
    insights = []

    risk_attr = results.get("risk_attribution")
    if risk_attr is None:
        return insights

    summary = risk_attr["summary"]
    port_vol = risk_attr["portfolio_volatility"]

    # Top risk contributors
    pct_risk = summary["pct_of_portfolio_vol"].sort_values(ascending=False)
    top_2 = pct_risk.head(2)
    top_2_pct = top_2.sum() * 100

    if top_2_pct > 50:
        insights.append({
            "category": "Risk",
            "headline": f"{top_2_pct:.0f}% of your portfolio risk comes from just 2 assets",
            "explanation": (
                f"{top_2.index[0]} and {top_2.index[1]} together account for "
                f"{top_2_pct:.0f}% of total portfolio risk. "
                f"Even if their portfolio weights are smaller, their volatility "
                f"and correlation with other assets amplify their risk impact."
            ),
            "concept": "Risk contribution is not the same as weight. A volatile, correlated asset can dominate portfolio risk even at a modest allocation.",
        })

    # Weight vs risk mismatch
    if "avg_weight" in summary.columns:
        for asset in summary.index:
            w = summary.loc[asset, "avg_weight"]
            r = summary.loc[asset, "pct_of_portfolio_vol"]
            if r > w * 1.5 and r > 0.15:
                insights.append({
                    "category": "Risk",
                    "headline": f"{asset} contributes more risk than its weight suggests",
                    "explanation": (
                        f"{asset} has a {w*100:.1f}% weight but accounts for "
                        f"{r*100:.1f}% of portfolio risk. This mismatch happens "
                        f"because {asset} is more volatile or more correlated "
                        f"with your other holdings."
                    ),
                    "concept": "Risk contribution depends on volatility AND correlation, not just weight. This is why equal-weight portfolios aren't equal-risk.",
                })
                break  # One example is enough

    return insights


# ============================================================================
# REBALANCING INSIGHTS
# ============================================================================

def generate_rebalance_insights(results: dict) -> list:
    insights = []

    config = results["config"]
    freq = config["rebalance_freq"]
    daily_weights = results["daily_weights"]
    initial_weights = config["weights"]

    if freq == "none":
        # Measure drift for buy-and-hold
        if len(daily_weights) > 0:
            final_weights = daily_weights.iloc[-1]
            max_drift = 0
            drifted_asset = ""
            for asset in initial_weights:
                if asset in final_weights.index:
                    drift = abs(final_weights[asset] - initial_weights[asset])
                    if drift > max_drift:
                        max_drift = drift
                        drifted_asset = asset

            if max_drift > 0.05:
                insights.append({
                    "category": "Rebalancing",
                    "headline": f"Without rebalancing, {drifted_asset} drifted by {max_drift*100:.1f} percentage points",
                    "explanation": (
                        f"You started with a {initial_weights.get(drifted_asset, 0)*100:.1f}% "
                        f"allocation to {drifted_asset}, but it ended at "
                        f"{final_weights[drifted_asset]*100:.1f}%. "
                        f"This happened because {drifted_asset}'s price changed more "
                        f"than other assets, causing its share to grow or shrink."
                    ),
                    "concept": "Weight drift is the natural consequence of not rebalancing. Winners grow larger, losers shrink — changing your risk profile over time.",
                })
    else:
        # Rebalanced portfolio insights
        turnover = results.get("turnover")
        total_costs = results.get("total_costs", 0)
        rebal_dates = results.get("rebalance_dates", [])

        if turnover is not None and len(turnover) > 0:
            avg_turnover = turnover.mean()
            insights.append({
                "category": "Rebalancing",
                "headline": f"{freq.capitalize()} rebalancing traded {avg_turnover*100:.1f}% of the portfolio on average",
                "explanation": (
                    f"Across {len(rebal_dates)} rebalancing events, the average turnover "
                    f"was {avg_turnover*100:.1f}%. This means roughly {avg_turnover*100:.1f}% "
                    f"of the portfolio had to be bought or sold each time to return to "
                    f"target weights."
                ),
                "concept": "Rebalancing controls drift but creates trading costs. More frequent rebalancing means less drift but more costs.",
            })

        if total_costs > 0.001:
            insights.append({
                "category": "Rebalancing",
                "headline": f"Transaction costs consumed {total_costs*100:.2f}% of portfolio value",
                "explanation": (
                    f"Over the full period, rebalancing costs added up to "
                    f"{total_costs*100:.2f}% of cumulative return. "
                    f"This is the price of maintaining your target allocation."
                ),
                "concept": "Transaction costs are the hidden cost of rebalancing. They reduce net returns but the discipline of rebalancing often compensates.",
            })

    return insights


# ============================================================================
# DIVERSIFICATION INSIGHTS
# ============================================================================

def generate_diversification_insights(results: dict) -> list:
    insights = []

    corr = results.get("correlation")
    config = results["config"]
    n_assets = len(config["assets"])

    if corr is None or corr.empty:
        return insights

    # Average pairwise correlation
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    avg_corr = corr.values[mask].mean()

    if avg_corr > 0.6:
        insights.append({
            "category": "Diversification",
            "headline": f"Your assets are highly correlated (average: {avg_corr:.2f})",
            "explanation": (
                f"The average correlation between your {n_assets} assets is {avg_corr:.2f}. "
                f"When assets move together, diversification provides less protection. "
                f"In a downturn, they tend to fall together."
            ),
            "concept": "True diversification requires low correlation. Holding many similar assets gives the illusion of diversification without the benefit.",
        })
    elif avg_corr > 0.3:
        insights.append({
            "category": "Diversification",
            "headline": f"Moderate correlation between assets ({avg_corr:.2f} average)",
            "explanation": (
                f"Your {n_assets} assets have moderate correlation. "
                f"This provides some diversification benefit, but during market stress, "
                f"correlations tend to increase — reducing the protection you expect."
            ),
            "concept": "Correlations are not constant. They tend to spike during market crashes, exactly when diversification is needed most.",
        })

    # Find the most correlated pair
    corr_values = corr.where(mask).stack()
    if len(corr_values) > 0:
        max_pair = corr_values.idxmax()
        max_corr = corr_values.max()
        if max_corr > 0.7:
            insights.append({
                "category": "Diversification",
                "headline": f"{max_pair[0]} and {max_pair[1]} are very similar (correlation: {max_corr:.2f})",
                "explanation": (
                    f"These two assets move almost in lockstep. "
                    f"Holding both provides little additional diversification. "
                    f"You might achieve better diversification by replacing one with "
                    f"an asset from a different sector."
                ),
                "concept": "Highly correlated assets are near-substitutes in a portfolio. Adding the second one doesn't reduce risk much.",
            })

    return insights


# ============================================================================
# ATTRIBUTION INSIGHTS
# ============================================================================

def generate_attribution_insights(results: dict) -> list:
    insights = []

    ret_attr = results.get("return_attribution")
    if ret_attr is None:
        return insights

    summary = ret_attr["summary"]

    # Best and worst contributors
    best = summary["cumulative_contribution"].idxmax()
    worst = summary["cumulative_contribution"].idxmin()
    best_val = summary.loc[best, "cumulative_contribution"]
    worst_val = summary.loc[worst, "cumulative_contribution"]

    insights.append({
        "category": "Attribution",
        "headline": f"{best} was your best performer, {worst} was your worst",
        "explanation": (
            f"{best} contributed {best_val*100:+.1f}% to total returns, "
            f"while {worst} contributed {worst_val*100:+.1f}%. "
            f"The difference between your best and worst picks was "
            f"{(best_val - worst_val)*100:.1f} percentage points."
        ),
        "concept": "Return attribution breaks down where your portfolio's returns actually came from. It helps you understand which decisions helped and which hurt.",
    })

    # Concentration of returns
    sorted_contrib = summary["pct_contribution"].sort_values(ascending=False)
    if len(sorted_contrib) > 2:
        top_pct = sorted_contrib.head(2).sum() * 100
        if top_pct > 60:
            top_names = sorted_contrib.head(2).index.tolist()
            insights.append({
                "category": "Attribution",
                "headline": f"{top_pct:.0f}% of returns came from just {top_names[0]} and {top_names[1]}",
                "explanation": (
                    f"Your portfolio returns were heavily concentrated. "
                    f"If either of these assets had performed differently, "
                    f"your overall result would look very different."
                ),
                "concept": "Return concentration is a form of hidden risk. If most returns come from one or two assets, your portfolio is less diversified than it appears.",
            })

    return insights


# ============================================================================
# DIVERSIFICATION SCORE
# ============================================================================

def compute_diversification_score(results: dict) -> dict:
    """Compute a 0-10 diversification score with breakdown.

    Components:
    - Weight concentration (HHI-based, lower is better)
    - Correlation (lower average is better)
    - Risk concentration (lower top-2 share is better)
    """
    config = results["config"]
    weights = config["weights"]
    corr = results.get("correlation")
    risk_attr = results.get("risk_attribution")

    n_assets = len(weights)

    # 1. Weight concentration (HHI)
    w_vals = np.array(list(weights.values()))
    hhi = np.sum(w_vals ** 2)
    # Perfect equal weight of N assets has HHI = 1/N
    # Single asset has HHI = 1.0
    # Score: 10 when HHI = 1/N, 0 when HHI = 1
    if n_assets > 1:
        hhi_min = 1.0 / n_assets
        hhi_score = max(0, 10 * (1 - (hhi - hhi_min) / (1 - hhi_min)))
    else:
        hhi_score = 0

    # 2. Correlation score
    if corr is not None and not corr.empty:
        mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        avg_corr = corr.values[mask].mean()
        # Score: 10 when avg_corr = 0, 0 when avg_corr = 1
        corr_score = max(0, 10 * (1 - avg_corr))
    else:
        corr_score = 5  # neutral if unknown

    # 3. Risk concentration
    if risk_attr is not None:
        risk_pct = risk_attr["summary"]["pct_of_portfolio_vol"].sort_values(ascending=False)
        top2_share = risk_pct.head(2).sum()
        # Score: 10 when top2 share = 2/N (equal), 0 when top2 = 1.0
        if n_assets > 1:
            ideal_top2 = 2.0 / n_assets
            risk_score = max(0, 10 * (1 - (top2_share - ideal_top2) / (1 - ideal_top2)))
        else:
            risk_score = 0
    else:
        risk_score = 5

    # Composite score
    composite = (
        0.30 * hhi_score +
        0.35 * corr_score +
        0.35 * risk_score
    )

    return {
        "score": round(composite, 1),
        "hhi_score": round(hhi_score, 1),
        "corr_score": round(corr_score, 1),
        "risk_score": round(risk_score, 1),
        "avg_correlation": round(avg_corr, 3) if corr is not None else None,
        "hhi": round(hhi, 4),
        "n_assets": n_assets,
    }
