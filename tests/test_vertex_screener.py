"""Offline tests for the V19 screener (synthetic prices, no network)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import vertex_screener as vs  # noqa: E402


def fake_prices(seed: int, n: int = 1260, drift: float = 0.0004) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))
    idx = pd.bdate_range(end="2026-11-04", periods=n)
    high = close * (1 + rng.uniform(0.002, 0.02, n))
    low = close * (1 - rng.uniform(0.002, 0.02, n))
    return pd.DataFrame({"Open": close * (1 + rng.normal(0, 0.004, n)), "High": high, "Low": low,
                         "Close": close, "Volume": rng.integers(1e6, 5e6, n)}, index=idx)


def test_trade_levels_are_ordered():
    lv = vs.trade_levels(fake_prices(1))
    assert lv["stop"] < lv["entry"] <= lv["price"] + 1e-9
    assert lv["target"] > lv["entry"]
    assert lv["rr"] >= 1.5


def test_score_is_sector_relative_and_bounded():
    rng = np.random.default_rng(0)
    n = 40
    df = pd.DataFrame({
        "symbol": [f"S{i}" for i in range(n)],
        "sector": ["Tech"] * 20 + ["Banks"] * 20,
        "trailing_pe": rng.uniform(5, 40, n), "forward_pe": rng.uniform(5, 35, n),
        "peg": rng.uniform(0.5, 3, n), "revenue_growth": rng.normal(0.08, 0.1, n),
        "profit_margin": rng.uniform(0.02, 0.4, n), "roe": rng.uniform(0.02, 0.4, n),
        "debt_to_equity": rng.uniform(0, 3, n), "beta": rng.uniform(0.5, 1.8, n),
        "momentum_6m": rng.normal(0.05, 0.2, n),
    })
    df.loc[0, "trailing_pe"] = -12  # loss-making: must rank worst, not best
    df.loc[1, ["peg", "forward_pe", "roe", "beta"]] = np.nan  # 5/9 factors -> unscored
    out = vs.score_universe(df)
    assert out.loc[0, "trailing_pe_pct"] == out[out.sector == "Tech"]["trailing_pe_pct"].min()
    assert np.isnan(out.loc[1, "score"])
    s = out["score"].dropna()
    assert s.between(0, 100).all() and len(s) == n - 1


def test_signals_are_decisive_tiers():
    n = 100
    df = pd.DataFrame({"symbol": [f"S{i}" for i in range(n)], "score": np.linspace(10, 90, n),
                       "ml_pct": np.linspace(0, 1, n), "price": 10.0, "sma50": 9.0})
    out = vs.composite_and_signals(df, {"state": "RISK-ON"})
    counts = out["signal"].value_counts()
    assert counts["STRONG BUY"] == 15 and counts["SELL"] == 15 and counts["HOLD"] == 30
    assert out["composite"].between(0, 100).all()


def test_portfolio_respects_caps():
    rows = []
    for i in range(30):
        rows.append({"symbol": f"S{i}", "name": f"S{i}", "sector": f"Sec{i % 5}", "score": 90 - i,
                     "signal": "LONG", "atr_pct": 0.01 + 0.002 * i, "price": 50.0})
    port = vs.build_portfolio(pd.DataFrame(rows))
    assert len(port) <= vs.MAX_NAMES
    assert port["weight"].max() <= vs.MAX_WEIGHT + 1e-9
    assert port.groupby("sector")["weight"].sum().max() <= vs.MAX_SECTOR_WEIGHT + 1e-9
    assert port["weight"].sum() <= vs.INVESTED + 1e-6


def test_full_pipeline_offline(tmp_path, monkeypatch):
    uni = vs.load_universe(tmp_path).head(60)
    monkeypatch.setattr(vs, "load_universe", lambda root, limit=None: uni)
    syms = uni["symbol"].tolist() + ["SPY"]
    monkeypatch.setattr(vs, "download_prices",
                        lambda s: {x: fake_prices(i, drift=0.0002 + 0.00002 * (i % 20)) for i, x in enumerate(syms)})
    rng = np.random.default_rng(3)
    monkeypatch.setattr(vs, "fetch_fundamentals", lambda s, p, w: {x: {
        "trailing_pe": rng.uniform(8, 40), "forward_pe": rng.uniform(8, 30), "peg": rng.uniform(0.6, 3),
        "revenue_growth": rng.normal(0.07, 0.08), "profit_margin": rng.uniform(0.03, 0.35),
        "roe": rng.uniform(0.03, 0.35), "debt_to_equity": rng.uniform(0, 2.5), "beta": rng.uniform(0.6, 1.6),
        "market_cap": 5e10, "currency": "USD"} for x in s})
    assert vs.run(tmp_path, None, 2) == 0
    rep = json.loads((tmp_path / "data" / "vertex" / "screener.json").read_text())
    assert rep["universe"]["scored"] == 60
    assert rep["portfolio"]["holdings"]
    assert (tmp_path / "data" / "vertex" / "candles.json").exists()
