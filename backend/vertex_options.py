"""
VertexCapital options chain fetcher (V21)
=========================================
Prints one JSON document with the option chain for a US-listed symbol, used by the Options Lab.
yfinance handles Yahoo's cookie/crumb handshake, which the options endpoint requires.

Usage:
  python backend/vertex_options.py MU                 # picks an expiry ~25-60 days out
  python backend/vertex_options.py MU --expiry 2026-12-18
  python backend/vertex_options.py MU --mock --spot 120   # synthetic chain for tests/offline demos
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timezone

STRIKE_RANGE = (0.6, 1.5)        # keep strikes within 60%-150% of spot
DEFAULT_RATE = 0.04


def _f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _ncdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(spot, k, t, r, iv, call):
    if t <= 0 or iv <= 0:
        return max(0.0, (spot - k) if call else (k - spot))
    d1 = (math.log(spot / k) + (r + iv * iv / 2) * t) / (iv * math.sqrt(t))
    d2 = d1 - iv * math.sqrt(t)
    if call:
        return spot * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
    return k * math.exp(-r * t) * _ncdf(-d2) - spot * _ncdf(-d1)


def pick_expiry(expiries: list[str], today: date) -> str | None:
    dte = [(e, (date.fromisoformat(e) - today).days) for e in expiries]
    for lo, hi in ((25, 60), (14, 90), (7, 400)):
        ok = [e for e, d in dte if lo <= d <= hi]
        if ok:
            return ok[0]
    return expiries[0] if expiries else None


def mock_chain(symbol: str, spot: float, expiry: str | None) -> dict:
    """Black-Scholes priced chain with a volatility smile and realistic bid/ask spreads."""
    today = date.today()
    exps = []
    for days in (12, 19, 33, 47, 75, 110):
        d = date.fromordinal(today.toordinal() + days)
        while d.weekday() != 4:
            d = date.fromordinal(d.toordinal() + 1)
        exps.append(d.isoformat())
    exp = expiry if expiry in exps else pick_expiry(exps, today)
    t = max((date.fromisoformat(exp) - today).days, 1) / 365
    step = 1 if spot < 60 else 2.5 if spot < 150 else 5 if spot < 400 else 10
    strikes = [round(k * step, 2) for k in range(int(spot * STRIKE_RANGE[0] / step), int(spot * STRIKE_RANGE[1] / step) + 1)]
    calls, puts = [], []
    for k in strikes:
        m = math.log(k / spot)
        iv = 0.38 + 0.25 * m * m - 0.08 * m
        for call, out in ((True, calls), (False, puts)):
            mid = _bs(spot, k, t, DEFAULT_RATE, iv, call)
            spr = max(0.02, mid * 0.04)
            oi = int(4000 * math.exp(-8 * m * m))
            out.append({"strike": k, "bid": round(max(0.0, mid - spr / 2), 2), "ask": round(mid + spr / 2, 2),
                        "last": round(mid, 2), "iv": round(iv, 4), "volume": oi // 6, "oi": oi})
    return {"symbol": symbol, "spot": spot, "rate": DEFAULT_RATE, "expiry": exp,
            "dte": (date.fromisoformat(exp) - today).days, "expirations": exps,
            "calls": calls, "puts": puts, "source": "mock", "fetched": datetime.now(timezone.utc).isoformat()}
