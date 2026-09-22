from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from factors import RAW_FACTOR_COLUMNS, build_factor_panel


@pytest.fixture
def synthetic_market_data():
    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        start="2020-01-01",
        periods=500,
    )

    tickers = [
        "AAPL",
        "MSFT",
        "JPM",
        "SPY",
    ]

    daily_returns = rng.normal(
        loc=0.0004,
        scale=0.015,
        size=(len(dates), len(tickers)),
    )

    close = pd.DataFrame(
        100 * np.exp(np.cumsum(daily_returns, axis=0)),
        index=dates,
        columns=tickers,
    )

    volume = pd.DataFrame(
        rng.integers(
            low=1_000_000,
            high=100_000_000,
            size=(len(dates), len(tickers)),
        ),
        index=dates,
        columns=tickers,
    ).astype(float)

    return close, volume


@pytest.fixture
def factor_panel(synthetic_market_data):
    close, volume = synthetic_market_data

    return build_factor_panel(
        close=close,
        volume=volume,
        benchmark="SPY",
        horizon=20,
    )


def test_factor_panel_has_expected_columns(factor_panel):
    expected_columns = set(
        RAW_FACTOR_COLUMNS + ["target_20d_excess"]
    )

    assert expected_columns.issubset(
        set(factor_panel.columns)
    )


def test_factor_panel_uses_date_ticker_index(factor_panel):
    assert isinstance(
        factor_panel.index,
        pd.MultiIndex,
    )

    assert factor_panel.index.names == [
        "date",
        "ticker",
    ]


def test_factor_values_are_cross_sectional_ranks(factor_panel):
    factor_values = factor_panel[RAW_FACTOR_COLUMNS]

    finite_values = factor_values.stack().dropna()

    assert not finite_values.empty
    assert finite_values.min() >= -0.5
    assert finite_values.max() <= 0.5


def test_volume_momentum_is_not_completely_empty(factor_panel):
    assert factor_panel["volume_momentum"].notna().any()


def test_target_matches_future_excess_return(
    synthetic_market_data,
    factor_panel,
):
    close, _ = synthetic_market_data

    test_date = close.index[350]
    ticker = "AAPL"
    horizon = 20

    stock_future_return = (
        close.loc[
            close.index[350 + horizon],
            ticker,
        ]
        / close.loc[test_date, ticker]
        - 1
    )

    benchmark_future_return = (
        close.loc[
            close.index[350 + horizon],
            "SPY",
        ]
        / close.loc[test_date, "SPY"]
        - 1
    )

    expected_target = (
        stock_future_return
        - benchmark_future_return
    )

    actual_target = factor_panel.loc[
        (test_date, ticker),
        "target_20d_excess",
    ]

    assert actual_target == pytest.approx(
        expected_target,
        abs=1e-12,
    )


def test_final_horizon_has_no_future_target(
    synthetic_market_data,
    factor_panel,
):
    close, _ = synthetic_market_data

    final_dates = close.index[-20:]

    final_targets = factor_panel.loc[
        (
            final_dates,
            slice(None),
        ),
        "target_20d_excess",
    ]

    assert final_targets.isna().all()


def test_panel_contains_no_infinite_values(factor_panel):
    numeric_values = factor_panel.select_dtypes(
        include=[np.number]
    )

    assert not np.isinf(
        numeric_values.to_numpy()
    ).any()