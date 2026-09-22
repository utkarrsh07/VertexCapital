import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "backend"),
)

import backtester  # noqa: E402
from backtester import (  # noqa: E402
    build_portfolio_weights,
    run_backtest,
)


def make_prices(days=400):
    dates = pd.date_range("2024-01-01", periods=days, freq="B")

    return pd.DataFrame(
        {
            "AAPL": 100 * (1.001 ** np.arange(days)),
            "MSFT": 120 * (1.0008 ** np.arange(days)),
            "GOOGL": 90 * (1.0006 ** np.arange(days)),
            "SPY": 200 * (1.0007 ** np.arange(days)),
            "XEQT.TO": 150 * (1.0005 ** np.arange(days)),
        },
        index=dates,
    )


def test_build_weights_creates_portfolio(monkeypatch):
    prices = make_prices()
    prediction_date = prices.index[300]

    predictions = pd.DataFrame(
        {
            "date": [prediction_date] * 3,
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "prediction": [0.03, 0.02, 0.01],
        }
    )

    def fake_covariance(returns):
        assets = returns.columns

        return pd.DataFrame(
            np.eye(len(assets)) * 0.04,
            index=assets,
            columns=assets,
        )

    def fake_optimizer(expected_returns, covariance, previous_weights=None):
        return pd.Series(
            0.95 / len(expected_returns),
            index=expected_returns.index,
        )

    monkeypatch.setattr(
        backtester,
        "estimate_covariance",
        fake_covariance,
    )
    monkeypatch.setattr(
        backtester,
        "optimize_portfolio",
        fake_optimizer,
    )

    weights = build_portfolio_weights(predictions, prices)

    assert len(weights) == 3
    assert set(weights["ticker"]) == {"AAPL", "MSFT", "GOOGL"}
    assert weights["weight"].sum() == pytest.approx(0.95)
    assert (weights["date"] == prediction_date).all()


def test_predictions_are_annualized(monkeypatch):
    prices = make_prices()
    prediction_date = prices.index[300]

    predictions = pd.DataFrame(
        {
            "date": [prediction_date] * 3,
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "prediction": [0.02, 0.01, -0.01],
        }
    )

    captured = {}

    def fake_covariance(returns):
        assets = returns.columns

        return pd.DataFrame(
            np.eye(len(assets)),
            index=assets,
            columns=assets,
        )

    def fake_optimizer(expected_returns, covariance, previous_weights=None):
        captured["expected_returns"] = expected_returns.copy()

        return pd.Series(
            0.95 / len(expected_returns),
            index=expected_returns.index,
        )

    monkeypatch.setattr(
        backtester,
        "estimate_covariance",
        fake_covariance,
    )
    monkeypatch.setattr(
        backtester,
        "optimize_portfolio",
        fake_optimizer,
    )

    build_portfolio_weights(predictions, prices)

    assert captured["expected_returns"]["AAPL"] == pytest.approx(
        0.02 * (252 / 20)
    )
    assert captured["expected_returns"]["GOOGL"] == pytest.approx(
        -0.01 * (252 / 20)
    )


def test_build_weights_rejects_insufficient_history():
    prices = make_prices(days=100)
    prediction_date = prices.index[-1]

    predictions = pd.DataFrame(
        {
            "date": [prediction_date] * 3,
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "prediction": [0.03, 0.02, 0.01],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="optimizer produced no portfolios",
    ):
        build_portfolio_weights(predictions, prices)


def test_build_weights_skips_dates_missing_from_prices():
    prices = make_prices()

    predictions = pd.DataFrame(
        {
            "date": [pd.Timestamp("2035-01-01")] * 3,
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "prediction": [0.03, 0.02, 0.01],
        }
    )

    with pytest.raises(RuntimeError):
        build_portfolio_weights(predictions, prices)


def test_backtest_applies_weights_on_following_day():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 110.0, 121.0, 121.0],
            "SPY": [100.0, 100.0, 100.0, 100.0],
            "XEQT.TO": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    weights = pd.DataFrame(
        {
            "date": [dates[1]],
            "ticker": ["AAPL"],
            "weight": [1.0],
        }
    )

    results, _ = run_backtest(
        prices,
        weights,
        transaction_cost_bps=0,
    )

    assert results.loc[dates[1], "gross_return"] == pytest.approx(0.0)
    assert results.loc[dates[2], "gross_return"] == pytest.approx(0.10)


def test_transaction_cost_reduces_net_return():
    dates = pd.date_range("2026-01-01", periods=3, freq="D")

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 100.0, 100.0],
            "SPY": [100.0, 100.0, 100.0],
            "XEQT.TO": [100.0, 100.0, 100.0],
        },
        index=dates,
    )

    weights = pd.DataFrame(
        {
            "date": [dates[1]],
            "ticker": ["AAPL"],
            "weight": [1.0],
        }
    )

    results, _ = run_backtest(
        prices,
        weights,
        transaction_cost_bps=10,
    )

    assert results.loc[dates[1], "trading_cost"] == pytest.approx(
        0.001
    )
    assert results.loc[dates[1], "net_return"] == pytest.approx(
        -0.001
    )


def test_zero_transaction_cost_makes_net_equal_gross():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 101.0, 102.0, 103.0],
            "SPY": [100.0, 100.0, 100.0, 100.0],
            "XEQT.TO": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    weights = pd.DataFrame(
        {
            "date": [dates[1]],
            "ticker": ["AAPL"],
            "weight": [0.95],
        }
    )

    results, _ = run_backtest(
        prices,
        weights,
        transaction_cost_bps=0,
    )

    pd.testing.assert_series_equal(
        results["gross_return"],
        results["net_return"],
        check_names=False,
    )


def test_portfolio_value_matches_compounded_returns():
    dates = pd.date_range("2026-01-01", periods=4, freq="D")

    prices = pd.DataFrame(
        {
            "AAPL": [100.0, 100.0, 110.0, 121.0],
            "SPY": [100.0, 100.0, 100.0, 100.0],
            "XEQT.TO": [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )

    weights = pd.DataFrame(
        {
            "date": [dates[1]],
            "ticker": ["AAPL"],
            "weight": [1.0],
        }
    )

    results, _ = run_backtest(
        prices,
        weights,
        transaction_cost_bps=0,
    )

    assert results.iloc[-1]["portfolio_value"] == pytest.approx(
        1.21
    )


def test_backtest_report_contains_total_turnover():
    prices = make_prices(days=20)
    trade_date = prices.index[5]

    weights = pd.DataFrame(
        {
            "date": [trade_date],
            "ticker": ["AAPL"],
            "weight": [0.95],
        }
    )

    _, report = run_backtest(prices, weights)

    assert "total_turnover" in report
    assert report["total_turnover"] == pytest.approx(0.95)


def test_backtest_returns_expected_columns():
    prices = make_prices(days=20)
    trade_date = prices.index[5]

    weights = pd.DataFrame(
        {
            "date": [trade_date],
            "ticker": ["AAPL"],
            "weight": [0.95],
        }
    )

    results, _ = run_backtest(prices, weights)

    expected_columns = {
        "gross_return",
        "trading_cost",
        "net_return",
        "benchmark_return",
        "portfolio_value",
        "benchmark_value",
    }

    assert expected_columns.issubset(results.columns)