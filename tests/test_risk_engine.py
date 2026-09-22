import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "backend"),
)

from risk_engine import (  # noqa: E402
    calculate_drawdown,
    portfolio_risk_report,
    risk_kill_switch,
)


def test_drawdown_starts_at_zero():
    returns = pd.Series([0.10, 0.05, 0.02])

    drawdown = calculate_drawdown(returns)

    assert drawdown.iloc[0] == pytest.approx(0.0)


def test_drawdown_calculation_is_correct():
    returns = pd.Series([0.10, -0.10, 0.05])

    drawdown = calculate_drawdown(returns)

    assert drawdown.iloc[0] == pytest.approx(0.0)
    assert drawdown.iloc[1] == pytest.approx(-0.10)
    assert drawdown.iloc[2] == pytest.approx(-0.055)


def test_drawdown_handles_missing_values():
    returns = pd.Series([0.10, np.nan, -0.05])

    drawdown = calculate_drawdown(returns)

    assert len(drawdown) == 3
    assert drawdown.isna().sum() == 0


def test_risk_report_contains_expected_metrics():
    returns = pd.Series(
        [0.01, -0.005, 0.008, -0.002, 0.006]
    )

    report = portfolio_risk_report(returns)

    expected_keys = {
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "maximum_drawdown",
        "daily_var_95",
        "expected_shortfall_95",
        "positive_day_rate",
        "trading_days",
    }

    assert expected_keys.issubset(report.keys())


def test_risk_report_counts_non_missing_days():
    returns = pd.Series([0.01, np.nan, -0.01, 0.02])

    report = portfolio_risk_report(returns)

    assert report["trading_days"] == 3


def test_positive_day_rate_is_correct():
    returns = pd.Series([0.01, -0.01, 0.02, 0.00])

    report = portfolio_risk_report(returns)

    assert report["positive_day_rate"] == pytest.approx(0.50)


def test_expected_shortfall_is_not_above_var():
    returns = pd.Series(
        [-0.05, -0.03, -0.01, 0.00, 0.01, 0.02]
    )

    report = portfolio_risk_report(returns)

    assert (
        report["expected_shortfall_95"]
        <= report["daily_var_95"]
    )


def test_identical_benchmark_has_beta_and_correlation_one():
    returns = pd.Series(
        [0.01, -0.02, 0.015, 0.005, -0.01]
    )

    report = portfolio_risk_report(
        returns,
        benchmark_returns=returns.copy(),
    )

    assert report["market_beta"] == pytest.approx(1.0)
    assert report["correlation_to_benchmark"] == pytest.approx(1.0)


def test_empty_returns_are_rejected():
    returns = pd.Series([], dtype=float)

    with pytest.raises(
        ValueError,
        match="Portfolio returns are empty",
    ):
        portfolio_risk_report(returns)


def test_kill_switch_remains_off_for_safe_returns():
    returns = pd.Series(
        [0.001, -0.001] * 50
    )

    triggered, reasons = risk_kill_switch(returns)

    assert triggered is False
    assert reasons == []


def test_kill_switch_detects_drawdown_breach():
    returns = pd.Series([0.01, -0.20, 0.01])

    triggered, reasons = risk_kill_switch(returns)

    assert triggered is True
    assert any(
        "Drawdown limit breached" in reason
        for reason in reasons
    )


def test_kill_switch_detects_volatility_breach():
    returns = pd.Series([0.03, -0.03] * 50)

    triggered, reasons = risk_kill_switch(returns)

    assert triggered is True
    assert any(
        "Volatility limit breached" in reason
        for reason in reasons
    )