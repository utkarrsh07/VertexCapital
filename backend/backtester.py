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
    """Daily buy-and-hold accounting with target rebalances on the next bar.

    Signals dated t are executed at the beginning of the t+1 return
    interval (a prior-close execution proxy). Real execution prices,
    overnight gaps, and market impact are NOT modeled.
    """
    if prices.empty or weights.empty:
        raise ValueError("Prices and weights must be nonempty.")
    if transaction_cost_bps < 0:
        raise ValueError("Transaction costs cannot be negative.")
    if not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError("Price index must be unique and sorted.")
    if prices.columns.duplicated().any():
        raise ValueError("Duplicate price columns.")
    for name in (benchmark, secondary_benchmark):
        if name not in prices.columns:
            raise ValueError(f"Benchmark {name} is missing from prices.")
    required = {"date", "ticker", "weight"}
    if not required.issubset(weights.columns):
        raise ValueError("Weights require date, ticker, weight columns.")
    w = weights.copy()
    w["date"] = pd.to_datetime(w["date"])
    if w.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker weight records.")
    if not np.isfinite(w["weight"].to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite portfolio weights.")
    if (w["weight"] < -1e-10).any():
        raise ValueError("Negative portfolio weights are unsupported.")
    if (w.groupby("date")["weight"].sum() > 1.000001).any():
        raise ValueError("Target allocations exceed 100%.")
    if not set(w["ticker"]).issubset(prices.columns):
        raise ValueError("Weights reference tickers missing from prices.")
    if not w["date"].isin(prices.index).all():
        raise ValueError("Signal dates must exist in the price index.")
    first_trade = w["date"].min()
    if first_trade >= prices.index[-1]:
        raise ValueError("No trading interval follows the first signal.")
    returns = prices.pct_change(fill_method=None)
    # Signals at t become active in the t+1 price-to-price interval.
    targets = w.pivot(index="date", columns="ticker", values="weight")
    targets = targets.reindex(columns=prices.columns, fill_value=0.0).fillna(0.0)
    signal_map = {date: row.to_numpy(dtype=float) for date, row in targets.iterrows()}
    columns = list(prices.columns)
    current = np.zeros(len(columns), dtype=float)
    pending = None
    records = []
    total_turnover = 0.0
    cost_rate = transaction_cost_bps / 10000.0
    for date in prices.index:
        daily = returns.loc[date].to_numpy(dtype=float)
        trade_cost = 0.0
        turnover = 0.0
        if pending is not None:
            # Rebalance from drifted weights; cash is the residual.
            turnover = float(np.abs(pending - current).sum())
            trade_cost = turnover * cost_rate
            current = pending.copy()
            total_turnover += turnover
        active = current > 1e-12
        if active.any() and not np.isfinite(daily[active]).all():
            raise ValueError(f"Missing return for held asset on {date}.")
        if date >= first_trade and (not np.isfinite(returns.loc[date, benchmark])
                                    or not np.isfinite(returns.loc[date, secondary_benchmark])):
            # First signal date may be the first price row; there is no return yet.
            if date != prices.index[0]:
                raise ValueError(f"Missing benchmark return on {date}.")
        gross = float(np.dot(current[active], daily[active])) if active.any() else 0.0
        net = gross - trade_cost
        if 1 + gross <= 0:
            raise ValueError("Portfolio lost 100% or more in a single period.")
        # Drift invested weights after market move; cost is deducted from cash/equity.
        end_equity = 1.0 + net
        if end_equity <= 0:
            raise ValueError("Portfolio equity exhausted by returns/costs.")
        current = current * np.where(np.isfinite(daily), 1.0 + daily, 1.0) / end_equity
        if current.sum() > 1.000001:
            # Trading costs are funded proportionally by the entire portfolio,
            # not from an assumed unlimited cash balance.
            current = current / current.sum()
        if date >= first_trade:
            records.append((date, gross, trade_cost, net,
                            float(returns.loc[date, benchmark]) if pd.notna(returns.loc[date, benchmark]) else 0.0,
                            float(returns.loc[date, secondary_benchmark]) if pd.notna(returns.loc[date, secondary_benchmark]) else 0.0))
        pending = signal_map.get(date)
    results = pd.DataFrame(records, columns=["date", "gross_return", "trading_cost", "net_return", "benchmark_return", "xeqt_return"]).set_index("date")
    results["portfolio_value"] = (1 + results["net_return"]).cumprod()
    results["benchmark_value"] = (1 + results["benchmark_return"]).cumprod()
    results["xeqt_value"] = (1 + results["xeqt_return"]).cumprod()
    report = portfolio_risk_report(results["net_return"], results["benchmark_return"])
    xeqt_comparison = portfolio_risk_report(results["net_return"], results["xeqt_return"])
    spy_report = portfolio_risk_report(results["benchmark_return"])
    xeqt_report = portfolio_risk_report(results["xeqt_return"])
    report["total_turnover"] = total_turnover
    report["spy_annual_return"] = spy_report["annual_return"]
    report["xeqt_annual_return"] = xeqt_report["annual_return"]
    report["excess_return_vs_spy"] = report["annual_return"] - report["spy_annual_return"]
    report["excess_return_vs_xeqt"] = report["annual_return"] - report["xeqt_annual_return"]
    report["beta_to_xeqt"] = xeqt_comparison["market_beta"]
    report["correlation_to_xeqt"] = xeqt_comparison["correlation_to_benchmark"]
    return results, report
