import numpy as np
import pandas as pd


def calculate_drawdown(
    portfolio_returns: pd.Series,
) -> pd.Series:
    wealth = (1 + portfolio_returns.fillna(0)).cumprod()
    running_peak = wealth.cummax()
    return wealth / running_peak - 1


def portfolio_risk_report(
    portfolio_returns: pd.Series,
    benchmark_returns=None,
    confidence=0.95,
) -> dict:
    returns = portfolio_returns.dropna()

    if returns.empty:
        raise ValueError("Portfolio returns are empty.")

    annual_return = (1 + returns).prod() ** (
        252 / len(returns)
    ) - 1

    annual_volatility = returns.std() * np.sqrt(252)

    sharpe = (
        annual_return / annual_volatility
        if annual_volatility > 0
        else np.nan
    )

    downside = returns[returns < 0].std() * np.sqrt(252)

    sortino = (
        annual_return / downside
        if downside > 0
        else np.nan
    )

    drawdown = calculate_drawdown(returns)

    historical_var = returns.quantile(1 - confidence)
    expected_shortfall = returns[
        returns <= historical_var
    ].mean()

    report = {
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "maximum_drawdown": drawdown.min(),
        "daily_var_95": historical_var,
        "expected_shortfall_95": expected_shortfall,
        "positive_day_rate": (returns > 0).mean(),
        "trading_days": len(returns),
    }

    if benchmark_returns is not None:
        aligned = pd.concat(
            [returns, benchmark_returns],
            axis=1,
        ).dropna()

        aligned.columns = ["portfolio", "benchmark"]

        benchmark_variance = aligned["benchmark"].var()

        if benchmark_variance > 0:
            beta = (
                aligned["portfolio"].cov(aligned["benchmark"])
                / benchmark_variance
            )
        else:
            beta = np.nan

        report["market_beta"] = beta
        report["correlation_to_benchmark"] = aligned.corr().iloc[0, 1]

    return report


def risk_kill_switch(
    portfolio_returns: pd.Series,
    maximum_drawdown_limit=-0.15,
    volatility_limit=0.30,
) -> tuple[bool, list[str]]:
    reasons = []

    drawdown = calculate_drawdown(portfolio_returns).min()
    volatility = portfolio_returns.std() * np.sqrt(252)

    if drawdown <= maximum_drawdown_limit:
        reasons.append(
            f"Drawdown limit breached: {drawdown:.2%}"
        )

    if volatility >= volatility_limit:
        reasons.append(
            f"Volatility limit breached: {volatility:.2%}"
        )

    return len(reasons) > 0, reasons