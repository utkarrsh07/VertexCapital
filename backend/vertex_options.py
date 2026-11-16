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
