from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from optimizer import estimate_covariance, optimize_portfolio


@pytest.fixture
def historical_returns():
    rng = np.random.default_rng(42)

    assets = [
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "JPM",
        "SPY",
    ]

    returns = rng.normal(
        loc=0.0004,
        scale=0.01,
        size=(300, len(assets)),
    )

    return pd.DataFrame(
        returns,
        columns=assets,
    )


@pytest.fixture
def covariance(historical_returns):
    return estimate_covariance(historical_returns)


@pytest.fixture
def expected_returns():
    return pd.Series(
        {
            "AAPL": 0.15,
            "MSFT": 0.12,
            "GOOGL": 0.10,
            "AMZN": 0.08,
            "JPM": 0.06,
            "SPY": 0.05,
        }
    )


def test_covariance_has_correct_assets(
    historical_returns,
    covariance,
):
    assert list(covariance.index) == list(
        historical_returns.columns
    )

    assert list(covariance.columns) == list(
        historical_returns.columns
    )


def test_covariance_is_symmetric(covariance):
    assert np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
        atol=1e-12,
    )


def test_covariance_is_positive_semidefinite(covariance):
    eigenvalues = np.linalg.eigvalsh(
        covariance.to_numpy()
    )

    assert eigenvalues.min() >= -1e-10


def test_covariance_rejects_insufficient_history():
    short_returns = pd.DataFrame(
        {
            "AAPL": [0.01] * 20,
            "MSFT": [0.02] * 20,
        }
    )

    with pytest.raises(
        ValueError,
        match="Not enough returns",
    ):
        estimate_covariance(short_returns)


def test_optimizer_is_long_only(
    expected_returns,
    covariance,
):
    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
    )

    assert (weights >= -1e-8).all()


def test_optimizer_respects_maximum_weight(
    expected_returns,
    covariance,
):
    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        max_weight=0.20,
    )

    assert weights.max() <= 0.20001


def test_optimizer_maintains_cash_reserve(
    expected_returns,
    covariance,
):
    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        cash_reserve=0.05,
    )

    assert weights.sum() == pytest.approx(
        0.95,
        abs=1e-5,
    )


def test_optimizer_respects_volatility_limit(
    expected_returns,
    covariance,
):
    target_volatility = 0.20

    weights = optimize_portfolio(
        expected_returns=expected_returns,
        covariance=covariance,
        target_volatility=target_volatility,
    )

    aligned_covariance = covariance.loc[
        weights.index,
        weights.index,
    ]

    portfolio_volatility = np.sqrt(
        weights.to_numpy()
        @ aligned_covariance.to_numpy()
        @ weights.to_numpy()
    )

    assert portfolio_volatility <= (
        target_volatility + 1e-5
    )


def test_highest_expected_return_cannot_break_cap(
    expected_returns,
    covariance,
):
    aggressive_returns = expected_returns.copy()
    aggressive_returns["AAPL"] = 10.0

    weights = optimize_portfolio(
        expected_returns=aggressive_returns,
        covariance=covariance,
        max_weight=0.20,
    )

    assert weights["AAPL"] <= 0.20001


def test_optimizer_only_uses_common_assets(
    expected_returns,
    covariance,
):
    returns_with_unknown_asset = expected_returns.copy()
    returns_with_unknown_asset["UNKNOWN"] = 0.50

    weights = optimize_portfolio(
        expected_returns=returns_with_unknown_asset,
        covariance=covariance,
    )

    assert "UNKNOWN" not in weights.index
    assert set(weights.index) == set(covariance.index)