import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from recommendations import make_trade_plans


def example_prices(days=320):
    # Alternating daily returns create a realistic, nonzero 20-day distribution.
    # Small periodic changes avoid an artificial 40% one-month target.
    dates = pd.bdate_range("2025-01-01", periods=days)
    daily = 0.0004 + 0.006 * np.sin(np.arange(days) * 0.37)
    prices = 100 * np.cumprod(1 + daily)
    return pd.DataFrame({"AAA": prices}, index=dates)


def example_predictions(prices, alpha=0.02):
    return pd.DataFrame({
        "date": [prices.index[-1]],
        "ticker": ["AAA"],
        "predicted_20d_excess_return": [alpha],
    })


def test_plans_have_ordered_levels_and_deadline():
    prices = example_prices()
    result = make_trade_plans(example_predictions(prices), prices)
    assert len(result) == 1
    row = result.iloc[0]
    assert row.illustrative_stop < row.suggested_limit_entry < row.illustrative_target
    assert row.reward_risk_ratio > 0
    assert row.historical_samples >= 252
    assert pd.Timestamp(row.time_exit_no_later_than) > prices.index[-1]
    assert row.status == "RESEARCH_ONLY_NOT_VALIDATED"


def test_negative_excess_alpha_not_a_buy():
    prices = example_prices()
    assert make_trade_plans(example_predictions(prices, -0.02), prices).empty


def test_insufficient_history_returns_no_trade():
    prices = example_prices(days=100)
    assert make_trade_plans(example_predictions(prices), prices).empty


def test_unrealistic_upside_returns_no_trade():
    dates = pd.bdate_range("2025-01-01", periods=320)
    # 1% daily growth implies ~22% over 20 sessions; rejected by 15% cap.
    prices = pd.DataFrame({"AAA": 100 * (1.01 ** np.arange(320))}, index=dates)
    assert make_trade_plans(example_predictions(prices), prices).empty
