"""Offline tests for the options chain script (mock mode, no network)."""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import vertex_options as vo  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "backend" / "vertex_options.py"


def test_pick_expiry_prefers_25_to_60_days():
    today = date(2026, 11, 16)
    exps = ["2026-11-21", "2026-11-28", "2026-12-18", "2027-01-15", "2027-06-18"]
    assert vo.pick_expiry(exps, today) == "2026-12-18"


def test_mock_chain_prices_are_sane():
    c = vo.mock_chain("MU", 120.0, None)
    assert 25 <= c["dte"] <= 60 and c["expiry"] in c["expirations"]
    calls = {r["strike"]: r for r in c["calls"]}
    puts = {r["strike"]: r for r in c["puts"]}
    ks = sorted(calls)
    assert all(calls[a]["ask"] >= calls[b]["ask"] for a, b in zip(ks, ks[1:]))   # calls cheaper as strike rises
    assert all(puts[a]["ask"] <= puts[b]["ask"] for a, b in zip(ks, ks[1:]))     # puts dearer as strike rises
    assert all(r["bid"] <= r["ask"] for r in c["calls"] + c["puts"])
    k = min(ks, key=lambda x: abs(x - 120))
    parity = (calls[k]["last"] - puts[k]["last"]) - (120 - k * 2.718281828 ** (-c["rate"] * c["dte"] / 365))
    assert abs(parity) < 0.05                                                    # put-call parity holds
