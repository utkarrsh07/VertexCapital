import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def estimate_covariance(
    historical_returns: pd.DataFrame,
) -> pd.DataFrame:
    clean = historical_returns.dropna(axis=1, how="any")

    if clean.shape[0] < 30 or clean.shape[1] == 0:
        raise ValueError("Not enough returns for covariance estimation.")

    estimator = LedoitWolf().fit(clean.values)

    annual_covariance = estimator.covariance_ * 252

    return pd.DataFrame(
        annual_covariance,
        index=clean.columns,
        columns=clean.columns,
    )


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    previous_weights=None,
    max_weight=0.20,
    cash_reserve=0.05,
    risk_aversion=8.0,
    turnover_penalty=0.01,
    target_volatility=0.20,
) -> pd.Series:
    assets = expected_returns.index.intersection(covariance.index)

    mu = expected_returns.loc[assets].clip(-0.50, 0.50)
    sigma = covariance.loc[assets, assets]

    number_assets = len(assets)
    weights = cp.Variable(number_assets)

    if previous_weights is None:
        previous = np.zeros(number_assets)
    else:
        previous = (
            previous_weights.reindex(assets)
            .fillna(0)
            .to_numpy()
        )

    portfolio_variance = cp.quad_form(
        weights,
        cp.psd_wrap(sigma.to_numpy()),
    )

    turnover = cp.norm1(weights - previous)

    objective = cp.Maximize(
        mu.to_numpy() @ weights
        - risk_aversion * portfolio_variance
        - turnover_penalty * turnover
    )

    constraints = [
        weights >= 0,
        weights <= max_weight,
        cp.sum(weights) == 1 - cash_reserve,
        portfolio_variance <= target_volatility**2,
    ]

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.CLARABEL)
    except cp.error.SolverError:
        problem.solve()

    if weights.value is None:
        fallback = np.repeat(
            (1 - cash_reserve) / number_assets,
            number_assets,
        )
        return pd.Series(fallback, index=assets)

    optimized = np.maximum(weights.value, 0)

    return pd.Series(optimized, index=assets)