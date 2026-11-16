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


def live_chain(symbol: str, expiry: str | None) -> dict:
    import yfinance as yf
    t = yf.Ticker(symbol)
    exps = list(t.options or [])
    if not exps:
        return {"symbol": symbol, "error": f"No listed options for {symbol} on Yahoo Finance."}
    today = date.today()
    exp = expiry if expiry in exps else pick_expiry(exps, today)
    oc = t.option_chain(exp)
    spot = None
    try:
        spot = _f(t.fast_info["last_price"])
    except Exception:
        pass
    if not spot:
        h = t.history(period="5d")
        spot = _f(h["Close"].iloc[-1]) if len(h) else None
    if not spot:
        return {"symbol": symbol, "error": "Could not get the current share price."}
    rate = DEFAULT_RATE
    try:
        irx = yf.Ticker("^IRX").history(period="5d")["Close"]
        if len(irx):
            rate = float(irx.iloc[-1]) / 100
    except Exception:
        pass

    def rows(df):
        out = []
        for _, r in df.iterrows():
            k = _f(r.get("strike"))
            if k is None or not (spot * STRIKE_RANGE[0] <= k <= spot * STRIKE_RANGE[1]):
                continue
            out.append({"strike": k, "bid": _f(r.get("bid")), "ask": _f(r.get("ask")), "last": _f(r.get("lastPrice")),
                        "iv": _f(r.get("impliedVolatility")), "volume": int(_f(r.get("volume")) or 0),
                        "oi": int(_f(r.get("openInterest")) or 0)})
        return out

    return {"symbol": symbol, "spot": spot, "rate": rate, "expiry": exp, "dte": (date.fromisoformat(exp) - today).days,
            "expirations": exps, "calls": rows(oc.calls), "puts": rows(oc.puts), "source": "yahoo",
            "fetched": datetime.now(timezone.utc).isoformat()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--expiry")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--spot", type=float, default=100.0)
    a = ap.parse_args()
    sym = a.symbol.upper()
    if sym.endswith(".TO"):
        out = {"symbol": sym, "error": "Options Lab covers US-listed stocks only."}
    else:
        try:
            out = mock_chain(sym, a.spot, a.expiry) if a.mock else live_chain(sym, a.expiry)
        except Exception as exc:
            out = {"symbol": sym, "error": f"Option chain unavailable: {type(exc).__name__}: {exc}"[:300]}
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
