"""
VertexCapital ML engine (V20)
=============================
Cross-sectional return model in the style used by systematic equity desks:

  * Features: 16 price/volume signals from the academic & practitioner literature
    (12-1 momentum, short-term reversal, low volatility, idiosyncratic volatility,
    beta, 52-week-high proximity, trend, lottery/max effect, illiquidity, volume shocks,
    sector-relative momentum), cross-sectionally rank-normalised each month.
  * Target: next-21-session return in excess of the universe median, rank-normalised.
  * Models: Ridge (linear, explainable) + Histogram Gradient Boosting (non-linear),
    ensembled by averaging cross-sectional ranks.
  * Validation: expanding-window walk-forward with a purge gap (no overlapping labels),
    quarterly re-training. Every reported metric is OUT-OF-SAMPLE.
  * Reporting: rank IC and t-stat, quintile spreads, top-quintile portfolio net of
    transaction costs vs equal-weight universe and SPY, calibration table, rolling IC,
    feature importance, market regime.

Known limits (also written into the report): the universe is today's index members
(survivorship bias inflates backtests), Yahoo prices are not point-in-time, and the
fundamental factors cannot be backtested with free data, so only the price model is validated.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

HORIZON = 21              # sessions per rebalance / label horizon
MIN_TRAIN_DATES = 24      # months of history before the first out-of-sample month
RETRAIN_EVERY = 3         # re-fit every N months
COST_BPS = 10.0           # one-way transaction cost
MIN_NAMES = 50            # skip dates with fewer usable stocks

FEATURES = [
    ("mom_12_1", "12-1 month momentum"),
    ("mom_6_1", "6-1 month momentum"),
    ("mom_3", "3 month momentum"),
    ("rev_1m", "1 month reversal"),
    ("vol_63", "3M volatility"),
    ("dvol_63", "3M downside volatility"),
    ("beta_252", "1Y beta"),
    ("ivol_63", "Idiosyncratic volatility"),
    ("hi52", "Distance to 52W high"),
    ("ma50", "Price vs 50-day avg"),
    ("ma200", "Price vs 200-day avg"),
    ("max21", "Max daily return (lottery)"),
    ("illiq", "Amihud illiquidity"),
    ("volshock", "Volume shock 20d/120d"),
    ("rsi14", "RSI(14)"),
    ("smom", "Sector-relative momentum"),
]
FEATURE_LABEL = dict(FEATURES)


# ---------------------------------------------------------------- panel + features
def build_panel(prices: dict[str, pd.DataFrame], bench: str = "SPY"):
    idx = prices[bench].index
    close = pd.DataFrame({s: d["Close"] for s, d in prices.items()}).reindex(idx)
    volume = pd.DataFrame({s: d["Volume"] for s, d in prices.items()}).reindex(idx)
    close = close.ffill(limit=3)
    return close, volume


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
