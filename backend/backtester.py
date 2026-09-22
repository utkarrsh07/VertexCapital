import numpy as np
import pandas as pd

from optimizer import estimate_covariance, optimize_portfolio
from risk_engine import portfolio_risk_report


def build_portfolio_weights(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    lookback=252,
) -> pd.DataFrame:
    daily_returns = prices.pct_change(fill_method=None)
    prediction_dates = sorted(predictions["date"].unique())

    previous_weights = None
    weight_records = []

    for date in prediction_dates:
        if date not in prices.index:
            continue

        date_position = prices.index.get_loc(date)

        if not isinstance(date_position, (int, np.integer)):
            continue

        history_start = max(0, date_position - lookback)
        historical_returns = daily_returns.iloc[
            history_start:date_position
        ]

        scores = (
            predictions[predictions["date"] == date]
            .set_index("ticker")["prediction"]
            .dropna()
        )

        eligible = [
            ticker
            for ticker in scores.index
            if ticker in historical_returns.columns
            and historical_returns[ticker].count() >= 126
        ]

        if len(eligible) < 3:
            continue

        historical_returns = (
            historical_returns[eligible]
            .dropna(axis=1, how="any")
        )

        eligible = list(historical_returns.columns)

        if len(eligible) < 3:
            continue

        covariance = estimate_covariance(historical_returns)

        # Model predicts approximately 20-day excess returns.
        annualized_alpha = (
            scores.reindex(eligible) * (252 / 20)
        )

        weights = optimize_portfolio(
            expected_returns=annualized_alpha,
            covariance=covariance,
            previous_weights=previous_weights,
        )

        record = weights.rename("weight").reset_index()
        record.columns = ["ticker", "weight"]
        record["date"] = date

        weight_records.append(record)
        previous_weights = weights

    if not weight_records:
        raise RuntimeError("The optimizer produced no portfolios.")

    return pd.concat(weight_records, ignore_index=True)


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    transaction_cost_bps=10,
    benchmark="SPY",
    secondary_benchmark="XEQT.TO",
) -> tuple[pd.DataFrame, dict]:
    returns = prices.pct_change(fill_method=None).fillna(0)

    if benchmark not in returns.columns:
        raise ValueError(
            f"Primary benchmark {benchmark} is missing from prices."
        )

    if secondary_benchmark not in returns.columns:
        raise ValueError(
            f"Secondary benchmark {secondary_benchmark} "
            "is missing from prices."
        )

    weight_matrix = weights.pivot(
        index="date",
        columns="ticker",
        values="weight",
    )

    weight_matrix = (
        weight_matrix.reindex(returns.index)
        .ffill()
        .fillna(0)
    )

    weight_matrix = weight_matrix.reindex(
        columns=returns.columns,
        fill_value=0,
    )

    # A signal created today becomes tomorrow's position.
    portfolio_returns = (
        weight_matrix.shift(1).fillna(0) * returns
    ).sum(axis=1)

    turnover = weight_matrix.diff().abs().sum(axis=1)

    trading_costs = (
        turnover * (transaction_cost_bps / 10_000)
    )

    net_returns = portfolio_returns - trading_costs

    benchmark_returns = returns[benchmark]
    secondary_benchmark_returns = returns[
        secondary_benchmark
    ]

    results = pd.DataFrame(
        {
            "gross_return": portfolio_returns,
            "trading_cost": trading_costs,
            "net_return": net_returns,
            "benchmark_return": benchmark_returns,
            "xeqt_return": secondary_benchmark_returns,
        }
    )

    first_trade = weights["date"].min()
    results = results.loc[first_trade:].copy()

    results["portfolio_value"] = (
        1 + results["net_return"]
    ).cumprod()

    results["benchmark_value"] = (
        1 + results["benchmark_return"]
    ).cumprod()

    results["xeqt_value"] = (
        1 + results["xeqt_return"]
    ).cumprod()

    # Main portfolio report remains relative to SPY.
    report = portfolio_risk_report(
        results["net_return"],
        results["benchmark_return"],
    )

    # Separate comparison against XEQT.
    xeqt_comparison = portfolio_risk_report(
        results["net_return"],
        results["xeqt_return"],
    )

    spy_report = portfolio_risk_report(
        results["benchmark_return"]
    )

    xeqt_report = portfolio_risk_report(
        results["xeqt_return"]
    )

    report["total_turnover"] = turnover.loc[
        results.index
    ].sum()

    report["spy_annual_return"] = spy_report[
        "annual_return"
    ]

    report["xeqt_annual_return"] = xeqt_report[
        "annual_return"
    ]

    report["excess_return_vs_spy"] = (
        report["annual_return"]
        - report["spy_annual_return"]
    )

    report["excess_return_vs_xeqt"] = (
        report["annual_return"]
        - report["xeqt_annual_return"]
    )

    report["beta_to_xeqt"] = xeqt_comparison[
        "market_beta"
    ]

    report["correlation_to_xeqt"] = xeqt_comparison[
        "correlation_to_benchmark"
    ]

    return results, report