import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from portfolio_simulation_v2 import simulate_portfolio

def sample():
    dates = pd.bdate_range("2025-01-01", periods=5)
    prices = pd.DataFrame({"AAA": [100, 110, 120, 120, 120],
                           "BBB": [100]*5, "SPY": [100, 101, 102, 103, 104]}, index=dates)
    trades = pd.DataFrame([{"ticker": "AAA", "entry_date": dates[0],
                            "exit_date": dates[2], "entry_price": 100,
                            "exit_price": 120, "outcome": "TARGET"}])
    return dates, prices, trades

def test_cash_and_returns():
    dates, prices, trades = sample()
    equity, fills, rejected, report = simulate_portfolio(trades, prices, allocation_fraction=0.5)
    assert report["executed_trades"] == 1
    assert report["final_value"] == pytest.approx(11000)
    assert report["total_return"] == pytest.approx(.1)
    assert report["benchmark_return"] == pytest.approx(.02)
    assert len(rejected) == 0

def test_no_duplicate_position():
    dates, prices, trades = sample()
    trades = pd.concat([trades, trades], ignore_index=True)
    _, fills, rejected, _ = simulate_portfolio(trades, prices)
    assert len(fills) == 1
    assert "already_held" in set(rejected.reason)

def test_missing_prices_rejected():
    dates, prices, trades = sample()
    prices.loc[dates[1], "AAA"] = float("nan")
    with pytest.raises(ValueError, match="No eligible"):
        simulate_portfolio(trades, prices)

def test_missing_benchmark():
    _, prices, trades = sample()
    with pytest.raises(ValueError, match="Benchmark"):
        simulate_portfolio(trades, prices.drop(columns="SPY"))
