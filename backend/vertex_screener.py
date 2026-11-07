"""
VertexCapital Screener (V20)
============================
Nine-factor, sector-relative equity screener with volatility-based trade levels
and a risk-weighted model portfolio.

What it does, end to end:
  1. Loads the universe (S&P 500 + TSX 60) from backend/vertex_universe.csv
  2. Downloads ~1 year of daily OHLCV from Yahoo Finance (batched)
  3. Pulls fundamentals per ticker (cached for 24h)
  4. Ranks 9 factors as percentiles WITHIN each sector, combines them with fixed weights
  5. Computes ATR(14)-based entry / stop / target levels and Jensen's alpha vs SPY
  6. Builds a hypothetical $100k model portfolio (score / volatility weighting, caps)
  7. Writes data/vertex/screener.json, candles.json and scan_log.txt

Research tool only. Scores are a transparent ranking, not a validated forecast.
Trade levels are volatility heuristics, not predictions. No orders are placed.

Usage:  python -u backend/vertex_screener.py --root .   [--limit 60] [--workers 6]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "20.0"
BENCHMARK = "SPY"

# key, label, weight, higher_is_better, group
FACTORS = [
    ("trailing_pe",    "Trailing P/E",      0.15, False, "value"),
    ("forward_pe",     "Forward P/E",       0.10, False, "value"),
    ("peg",            "PEG Ratio",         0.10, False, "value"),
    ("revenue_growth", "Revenue Growth",    0.15, True,  "growth"),
    ("profit_margin",  "Profit Margin",     0.10, True,  "quality"),
    ("roe",            "Return on Equity",  0.10, True,  "quality"),
    ("debt_to_equity", "Debt / Equity",     0.10, False, "quality"),
    ("beta",           "Beta",              0.10, False, "quality"),
    ("momentum_6m",    "6M Momentum",       0.10, True,  "growth"),
]
PRICE_YEARS = 13        # 13 years: 1y feature warm-up + 2y initial training + ~10y of walk-forward out-of-sample testing
# Benchmarks are ETFs (dividends reinvested, like the stock data) so the comparison is like-for-like.
BENCH_ETFS = {"SPY": "S&P 500 (SPY)", "QQQ": "Nasdaq 100 (QQQ)", "DIA": "Dow Jones (DIA)",
              "IWM": "Russell 2000 (IWM)", "XIU.TO": "TSX 60 (XIU)", "ACWI": "MSCI World (ACWI)"}
FUND_CACHE_VERSION = 2
MIN_FACTORS = 6          # a stock needs at least 6 of 9 factors to be scored
MIN_SECTOR_SIZE = 8      # smaller sectors are ranked against the whole universe
CANDLE_DAYS = 160
PORTFOLIO_CAPITAL = 100_000.0
MAX_NAMES = 15
MAX_PER_SECTOR = 4
MAX_WEIGHT = 0.10
MAX_SECTOR_WEIGHT = 0.30
INVESTED = 0.95

_T0 = time.time()
_LOG: list[str] = []


def log(tag: str, msg: str) -> None:
    line = f"[{time.time() - _T0:7.2f}s] {tag:<8} {msg}"
    _LOG.append(line)
    print(line, flush=True)


# ---------------------------------------------------------------- data loading
def load_universe(root: Path, limit: int | None = None) -> pd.DataFrame:
    path = Path(__file__).with_name("vertex_universe.csv")
    if not path.exists():
        path = root / "backend" / "vertex_universe.csv"
    uni = pd.read_csv(path)
    if limit:
        # keep the mix of both indexes when limiting (useful for quick demos)
        uni = (uni.groupby("index", group_keys=False)
                  .apply(lambda g: g.head(max(1, int(limit * len(g) / len(uni))))))
    return uni.reset_index(drop=True)


def download_prices(symbols: list[str], chunk: int = 80) -> dict[str, pd.DataFrame]:
    import yfinance as yf
    out: dict[str, pd.DataFrame] = {}
    batches = [symbols[i:i + chunk] for i in range(0, len(symbols), chunk)]
    for bi, batch in enumerate(batches, 1):
        t = time.time()
        try:
            start = (pd.Timestamp.today() - pd.DateOffset(years=PRICE_YEARS)).strftime("%Y-%m-%d")
            raw = yf.download(batch, start=start, interval="1d", auto_adjust=True,
                              group_by="ticker", threads=True, progress=False)
        except Exception as exc:  # network / Yahoo hiccup
            log("PRICES", f"batch {bi}/{len(batches)} failed: {exc}")
            continue
        got = 0
        for s in batch:
            try:
                df = raw[s] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
                if len(df) >= 130:
                    out[s] = df
                    got += 1
            except Exception:
                pass
        log("PRICES", f"batch {bi}/{len(batches)}  {got}/{len(batch)} usable histories  "
                      f"({time.time() - t:.1f}s)")
    return out


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _div_yield(info: dict):
    y = _num(info.get("dividendYield"))
    if y is None:
        y = _num(info.get("trailingAnnualDividendYield"))
    if y is not None and y > 0.25:      # some Yahoo versions report percent, not fraction
        y = y / 100.0
    return y


def fetch_one_fundamental(symbol: str) -> dict:
    import yfinance as yf
    last_exc = None
    for attempt in range(3):
        try:
            info = yf.Ticker(symbol).info or {}
            if not info:
                raise ValueError("empty info")
            de = _num(info.get("debtToEquity"))
            peg = _num(info.get("pegRatio")) or _num(info.get("trailingPegRatio"))
            return {
                "trailing_pe": _num(info.get("trailingPE")),
                "forward_pe": _num(info.get("forwardPE")),
                "peg": peg,
                "revenue_growth": _num(info.get("revenueGrowth")),
                "profit_margin": _num(info.get("profitMargins")),
                "roe": _num(info.get("returnOnEquity")),
                "debt_to_equity": de / 100.0 if de is not None else None,  # Yahoo reports %
                "beta": _num(info.get("beta")),
                "market_cap": _num(info.get("marketCap")),
                "currency": info.get("currency"),
                "exchange": info.get("exchange"),
                "eps_trailing": _num(info.get("trailingEps")),
                "dividend_yield": _div_yield(info),
                "v": FUND_CACHE_VERSION,
                "fetched": time.time(),
            }
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    return {"error": str(last_exc)[:120], "fetched": time.time()}


def fetch_fundamentals(symbols: list[str], cache_path: Path, workers: int) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    fresh = {s: cache[s] for s in symbols
             if s in cache and "error" not in cache[s] and cache[s].get("v") == FUND_CACHE_VERSION
             and time.time() - cache[s].get("fetched", 0) < 86400}
    todo = [s for s in symbols if s not in fresh]
    log("FUNDS", f"{len(fresh)} cached (<24h), fetching {len(todo)} with {workers} workers")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_one_fundamental, s): s for s in todo}
        for f in as_completed(futs):
            s = futs[f]
            d = f.result()
            fresh[s] = d
            done += 1
            if "error" in d:
                log("FUNDS", f"{s:<9} unavailable ({d['error']})")
            else:
                def p(v, pct=False, x=False):
                    if v is None:
                        return "  n/a"
                    return f"{v * 100:+.1f}%" if pct else (f"{v:.2f}x" if x else f"{v:.1f}")
                log("FUNDS", f"{s:<9} pe={p(d['trailing_pe'])} fpe={p(d['forward_pe'])} "
                             f"peg={p(d['peg'])} rev={p(d['revenue_growth'], pct=True)} "
                             f"mgn={p(d['profit_margin'], pct=True)} roe={p(d['roe'], pct=True)} "
                             f"d/e={p(d['debt_to_equity'], x=True)} beta={p(d['beta'])}  "
                             f"[{done}/{len(todo)}]")
    cache.update(fresh)
    cache_path.write_text(json.dumps(cache))
    return fresh


# ---------------------------------------------------------------- calculations
def atr(df: pd.DataFrame, n: int = 14) -> float:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def jensen_alpha(stock: pd.Series, bench: pd.Series) -> tuple[float | None, float | None]:
    """Annualised Jensen's alpha and realised beta from 1y of daily returns."""
    r = pd.concat([stock.pct_change(), bench.pct_change()], axis=1, join="inner").dropna()
    if len(r) < 120:
        return None, None
    x, y = r.iloc[:, 1].values, r.iloc[:, 0].values
    b = float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))
    a = float(np.mean(y) - b * np.mean(x)) * 252
    return a, b


def trade_levels(df: pd.DataFrame) -> dict:
    """Volatility-based research levels: pullback entry, 2xATR stop, 2R / resistance target."""
    close = float(df["Close"].iloc[-1])
    a = atr(df)
    sma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    sma50 = float(df["Close"].rolling(50).mean().iloc[-1])
    high60 = float(df["High"].tail(60).max())
    low20 = float(df["Low"].tail(20).min())
    entry = min(close, max(sma20, close - 1.5 * a))          # pull back toward 20-day mean, max 1.5 ATR
    stop = max(entry - 2.0 * a, low20 - 0.25 * a) if low20 < entry else entry - 2.0 * a
    stop = min(stop, entry - 1.0 * a)                        # never tighter than 1 ATR
    risk = entry - stop
    target = entry + 2.0 * risk
    if entry + 1.5 * risk < high60 < entry + 4.0 * risk:     # use prior resistance if it's a sensible target
        target = high60
    return {
        "price": close, "atr": a, "atr_pct": a / close, "sma20": sma20, "sma50": sma50,
        "entry": entry, "stop": stop, "target": target, "rr": (target - entry) / risk if risk > 0 else None,
        "high_60d": high60, "low_20d": low20,
    }


def score_universe(df: pd.DataFrame) -> pd.DataFrame:
    """Sector-relative percentile per factor -> weighted 0-100 composite."""
    df = df.copy()
    # Negative earnings / PEG / equity make a multiple meaningless: rank them as worst, don't drop them.
    for k in ("trailing_pe", "forward_pe", "peg"):
        df.loc[df[k] <= 0, k] = np.inf
    df.loc[df["debt_to_equity"] < 0, "debt_to_equity"] = np.inf
    sector_sizes = df["sector"].map(df["sector"].value_counts())
    group = np.where(sector_sizes >= MIN_SECTOR_SIZE, df["sector"], "__universe__")
    for key, _, _, hib, _ in FACTORS:
        col = df[key].astype(float)
        pct = col.groupby(group).rank(pct=True, ascending=hib, method="average")
        # rank(pct) on "ascending=hib": highest pct = best
        df[f"{key}_pct"] = pct
        df.loc[col.isna(), f"{key}_pct"] = np.nan
    pct_cols = [f"{k}_pct" for k, *_ in FACTORS]
    w = np.array([wt for _, _, wt, _, _ in FACTORS])
    P = df[pct_cols].values
    mask = ~np.isnan(P)
    df["factor_count"] = mask.sum(axis=1)
    num = np.nansum(np.where(mask, P, 0) * w, axis=1)
    den = (mask * w).sum(axis=1)
    df["score"] = np.where(df["factor_count"] >= MIN_FACTORS, 100 * num / np.where(den > 0, den, 1), np.nan)
    for g in ("value", "growth", "quality"):
        keys = [k for k, _, _, _, gg in FACTORS if gg == g]
        df[f"{g}_score"] = 100 * df[[f"{k}_pct" for k in keys]].mean(axis=1)
    # restore inf -> NaN-safe display values
    for k in ("trailing_pe", "forward_pe", "peg", "debt_to_equity"):
        df.loc[np.isinf(df[k]), k] = np.nan
    return df


SIGNAL_TIERS = [(0.85, "STRONG BUY"), (0.65, "BUY"), (0.35, "HOLD"), (0.15, "REDUCE"), (0.0, "SELL")]
W_ML, W_FUND = 0.55, 0.45
MAX_NAMES = 20
MAX_WEIGHT = 0.08
MAX_SECTOR_WEIGHT = 0.25
VOL_TARGET = 0.16
RISK_AVERSION = 6.0


def composite_and_signals(df: pd.DataFrame, reg: dict | None) -> pd.DataFrame:
    """Blend validated ML rank with fundamental rank; tiers are relative, so signals are decisive."""
    df = df.copy()
    df["fund_pct"] = df["score"].rank(pct=True)
    has_ml = df["ml_pct"].notna() if "ml_pct" in df else pd.Series(False, index=df.index)
    comp = np.where(has_ml & df["fund_pct"].notna(), W_ML * df.get("ml_pct", 0) + W_FUND * df["fund_pct"],
                    np.where(has_ml, df.get("ml_pct", np.nan), df["fund_pct"]))
    df["composite"] = pd.Series(comp, index=df.index).rank(pct=True) * 100
    pr = df["composite"].rank(pct=True)

    def tier(p):
        if pd.isna(p):
            return "N/A"
        for cut, name in SIGNAL_TIERS:
            if p > cut or cut == 0.0:
                return name
        return "SELL"
    df["signal"] = pr.map(tier)
    if reg and reg.get("state") == "RISK-OFF":
        # trend filter in risk-off markets: no STRONG BUY below the 50-day average
        weak = (df["signal"] == "STRONG BUY") & (df["price"] < df["sma50"])
        df.loc[weak, "signal"] = "BUY"
    return df


def _project(w: np.ndarray, sectors: np.ndarray, cap: float, scap: float) -> np.ndarray:
    w = np.clip(w, 0, cap)
    for _ in range(60):
        if w.sum() <= 0:
            w = np.ones_like(w)
        w = w / w.sum()
        w = np.clip(w, 0, cap)
        for sc in np.unique(sectors):
            m = sectors == sc
            t = w[m].sum()
            if t > scap:
                w[m] *= scap / t
        if abs(w.sum() - 1) < 1e-6 and w.max() <= cap + 1e-9:
            break
    return w / w.sum()


def build_portfolio(df: pd.DataFrame, prices: dict[str, pd.DataFrame], reg: dict | None) -> tuple[pd.DataFrame, dict]:
    """Long-only mean-variance optimisation with a Ledoit-Wolf covariance, name/sector caps and a volatility target."""
    from sklearn.covariance import LedoitWolf
    cands = df[df["signal"].isin(["STRONG BUY", "BUY"])].sort_values("composite", ascending=False).head(45)
    if len(cands) < 8:
        cands = df[df["composite"].notna()].sort_values("composite", ascending=False).head(45)
    syms = [s for s in cands["symbol"] if s in prices]
    cands = cands.set_index("symbol").loc[syms].reset_index()
    rets = pd.DataFrame({s: prices[s]["Close"].pct_change() for s in syms}).tail(252).dropna(axis=1, thresh=200).fillna(0)
    cands = cands[cands["symbol"].isin(rets.columns)].reset_index(drop=True)
    rets = rets[cands["symbol"]]
    cov = LedoitWolf().fit(rets.values).covariance_ * 252
    if "exp_excess" in cands and cands["exp_excess"].notna().any():
        mu = cands["exp_excess"].fillna(0).values * 12 + 0.04 * (cands["composite"].values / 100 - 0.5)
    else:
        mu = 0.08 * (cands["composite"].values / 100 - 0.5)
    sec = cands["sector"].values
    w = _project(np.ones(len(cands)) / len(cands), sec, MAX_WEIGHT, MAX_SECTOR_WEIGHT)
    step = 0.5 / (RISK_AVERSION * np.linalg.eigvalsh(cov).max())
    for _ in range(400):                                       # projected gradient ascent on mu'w - (lambda/2) w'Sw
        w = _project(w + step * (mu - RISK_AVERSION * cov @ w), sec, MAX_WEIGHT, MAX_SECTOR_WEIGHT)
    keep = np.argsort(-w)[:MAX_NAMES]
    keep = keep[w[keep] > 0.005]
    p = cands.iloc[keep].copy()
    wk = _project(w[keep], sec[keep], MAX_WEIGHT * 1.25, MAX_SECTOR_WEIGHT)
    sub = cov[np.ix_(keep, keep)]
    vol = float(np.sqrt(wk @ sub @ wk))
    exposure = min(INVESTED, (reg or {}).get("exposure", 1.0) * INVESTED, VOL_TARGET / vol if vol > 0 else INVESTED)
    p["weight"] = wk * exposure
    p["value"] = p["weight"] * PORTFOLIO_CAPITAL
    p["shares"] = np.floor(p["value"] / p["price"])
    beta = float(np.nansum(p["weight"] * p["beta_realized"].fillna(1.0)))
    stats = {"ex_ante_vol": vol * exposure, "unlevered_vol": vol, "beta": beta, "invested": float(p["weight"].sum()),
             "effective_n": float(1 / np.sum(wk ** 2)), "expected_excess_1m": float(np.sum(wk * cands.iloc[keep]["exp_excess"].fillna(0).values)) if "exp_excess" in cands else None,
             "method": f"Mean-variance (risk aversion {RISK_AVERSION:g}), Ledoit-Wolf covariance, max {MAX_WEIGHT:.0%}/name, "
                       f"{MAX_SECTOR_WEIGHT:.0%}/sector, {VOL_TARGET:.0%} volatility target, regime-scaled exposure"}
    return p.sort_values("weight", ascending=False), stats


def trailing_returns(port: pd.DataFrame, prices: dict[str, pd.DataFrame]) -> list[dict]:
    out = []
    for label, n in (("1W", 5), ("1M", 21), ("3M", 63), ("6M", 126)):
        pr = 0.0
        for _, r in port.iterrows():
            c = prices[r["symbol"]]["Close"]
            pr += r["weight"] * (float(c.iloc[-1]) / float(c.iloc[-n - 1]) - 1)
        b = prices.get(BENCHMARK)
        br = float(b["Close"].iloc[-1]) / float(b["Close"].iloc[-n - 1]) - 1 if b is not None else None
        out.append({"period": label, "portfolio": pr, "benchmark": br})
    return out


# ---------------------------------------------------------------- pipeline
def clean(v, nd=4):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return round(f, nd) if math.isfinite(f) else None


def run(root: Path, limit: int | None, workers: int) -> int:
    outdir = root / "data" / "vertex"
    outdir.mkdir(parents=True, exist_ok=True)
    log("INIT", f"VertexCapital Screener v{VERSION}  root={root}")
    uni = load_universe(root, limit)
    counts = uni["index"].value_counts().to_dict()
    log("UNIVERSE", f"{len(uni)} symbols  " + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    log("UNIVERSE", f"{uni['sector'].nunique()} GICS sectors; factor weights: "
                    + ", ".join(f"{lbl} {w:.0%}" for _, lbl, w, _, _ in FACTORS))

    symbols = uni["symbol"].tolist()
    prices = download_prices(symbols + [b for b in BENCH_ETFS if b not in symbols])
    if BENCHMARK not in prices:
        log("ERROR", "benchmark SPY unavailable - check internet connection")
        return 2
    log("PRICES", f"{len(prices) - 1}/{len(symbols)} symbols with >=130 sessions; "
                  f"latest bar {prices[BENCHMARK].index[-1].date()}")

    usable = [s for s in symbols if s in prices]
    funds = fetch_fundamentals(usable, outdir / "fundamentals_cache.json", workers)

    rows = []
    bench = prices[BENCHMARK]["Close"]
    for _, u in uni.iterrows():
        s = u["symbol"]
        if s not in prices:
            continue
        df = prices[s]
        f = funds.get(s, {})
        lv = trade_levels(df)
        c = df["Close"]
        a, hb = jensen_alpha(c.tail(253), bench)
        rows.append({
            "symbol": s, "name": u["name"], "sector": u["sector"], "index": u["index"],
            **{k: f.get(k) for k, *_ in FACTORS if k != "momentum_6m"},
            "momentum_6m": float(c.iloc[-1] / c.iloc[-127] - 1) if len(c) > 127 else None,
            "alpha": a, "beta_realized": hb,
            "change_1d": float(c.iloc[-1] / c.iloc[-2] - 1),
            "vol_1y": float(c.pct_change().tail(252).std() * np.sqrt(252)),
            "dividend_yield": f.get("dividend_yield"),
            "market_cap": f.get("market_cap"), "currency": f.get("currency") or ("CAD" if s.endswith(".TO") else "USD"),
            **lv,
        })
    df = pd.DataFrame(rows)
    for k, *_ in FACTORS:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    coverage = {k: int(df[k].notna().sum()) for k, *_ in FACTORS}
    log("FACTORS", "coverage  " + "  ".join(f"{k}={v}" for k, v in coverage.items()))

    df = score_universe(df)
    for sec, g in df.groupby("sector"):
        mode = "sector-relative" if len(g) >= MIN_SECTOR_SIZE else "universe-relative"
        log("RANK", f"{sec:<24} n={len(g):<3} {mode}  median score {g['score'].median():.1f}")
    scored = df[df["score"].notna()]
    log("SCORE", f"{len(scored)} fundamentally scored, {len(df) - len(scored)} with <{MIN_FACTORS} of 9 factors")
    if len(scored) < 0.5 * len(df):
        log("WARN", "low fundamentals coverage (Yahoo rate limit?) - run the scan again; successful lookups are cached 24h")

    # ---------- machine-learning layer (walk-forward validated)
    ml_report, reg = None, None
    try:
        import vertex_ml
        res = vertex_ml.run_ml({s_: prices[s_] for s_ in usable + [BENCHMARK]},
                               dict(zip(df["symbol"], df["sector"])), log=lambda m: log("ML", m))
        if res:
            sc = res["scores"]
            df["ml_pct"] = df["symbol"].map(sc["ml_pct"])
            df["exp_excess"] = df["symbol"].map(sc["exp_excess"])
            df["hit_prob"] = df["symbol"].map(sc["hit_prob"])
            df["ml_quintile"] = df["symbol"].map(sc["quintile"])
            df["drivers"] = df["symbol"].map(sc["drivers"])
            ml_report, reg = res["report"], res["report"]["regime"]
    except Exception as exc:  # the screener still works on fundamentals alone
        log("ML", f"model skipped: {type(exc).__name__}: {exc}")
    if "ml_pct" not in df:
        df["ml_pct"] = np.nan
        df["exp_excess"] = np.nan
        df["hit_prob"] = np.nan
        df["ml_quintile"] = np.nan
        df["drivers"] = None

    df = composite_and_signals(df, reg)
    top = df.sort_values("composite", ascending=False).head(10)
    for _, r in top.iterrows():
        log("LEVELS", f"{r['symbol']:<9} comp={r['composite']:5.1f} ml={100 * (r['ml_pct'] if pd.notna(r['ml_pct']) else 0):5.1f} "
                      f"fund={r['score'] if pd.notna(r['score']) else 0:5.1f}  px={r['price']:.2f}  "
                      f"entry={r['entry']:.2f} stop={r['stop']:.2f} target={r['target']:.2f}  {r['signal']}")
    sig = df["signal"].value_counts().to_dict()
    log("SIGNAL", "  ".join(f"{k}={v}" for k, v in sig.items()))

    port, pstats = build_portfolio(df, prices, reg)
    trail = trailing_returns(port, prices) if len(port) else []
    if len(port):
        log("PORTF", f"{len(port)} names, invested {port['weight'].sum():.1%}, cash {1 - port['weight'].sum():.1%}, "
                     f"max name {port['weight'].max():.1%}, sectors {port['sector'].nunique()}, "
                     f"ex-ante vol {pstats['ex_ante_vol']:.1%}, beta {pstats['beta']:.2f}, effective N {pstats['effective_n']:.1f}")
        for _, r in port.iterrows():
            log("PORTF", f"{r['symbol']:<9} w={r['weight']:6.2%}  ${r['value']:>9,.0f}  {r['sector']}")

    # ---------- outputs
    stocks = []
    for _, r in df.iterrows():
        c = prices[r["symbol"]]["Close"]
        stocks.append({
            "symbol": r["symbol"], "name": r["name"], "sector": r["sector"], "index": r["index"],
            "currency": r["currency"], "market_cap": clean(r["market_cap"], 0),
            "price": clean(r["price"]), "change_1d": clean(r["change_1d"]),
            "score": clean(r["composite"], 2), "fund_score": clean(r["score"], 2),
            "ml_score": clean(100 * r["ml_pct"], 1) if pd.notna(r["ml_pct"]) else None,
            "exp_excess_1m": clean(r["exp_excess"]), "hit_prob": clean(r["hit_prob"], 3),
            "ml_quintile": int(r["ml_quintile"]) if pd.notna(r["ml_quintile"]) else None,
            "drivers": [[n, round(v, 4)] for n, v in r["drivers"]] if isinstance(r["drivers"], list) else [],
            "vol_1y": clean(r["vol_1y"]), "dividend_yield": clean(r["dividend_yield"]),
            "signal": r["signal"], "factor_count": int(r["factor_count"]),
            "value_score": clean(r["value_score"], 1), "growth_score": clean(r["growth_score"], 1),
            "quality_score": clean(r["quality_score"], 1),
            "factors": {k: {"value": clean(r[k]), "pct": clean(r[f"{k}_pct"], 3)} for k, *_ in FACTORS},
            "alpha": clean(r["alpha"]), "beta_realized": clean(r["beta_realized"], 3),
            "atr": clean(r["atr"]), "atr_pct": clean(r["atr_pct"]), "sma20": clean(r["sma20"]),
            "sma50": clean(r["sma50"]), "entry": clean(r["entry"]), "stop": clean(r["stop"]),
            "target": clean(r["target"]), "rr": clean(r["rr"], 2),
            "spark": [round(float(v), 2) for v in c.tail(60)],
        })
    report = {
        "version": VERSION,
        "status": "RESEARCH_ONLY",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "price_date": str(prices[BENCHMARK].index[-1].date()),
        "source": "Yahoo Finance (delayed, not point-in-time)",
        "universe": {"requested": len(uni), "priced": len(usable), "scored": int(len(scored)), "by_index": counts},
        "factors": [{"key": k, "label": l, "weight": w, "higher_is_better": h, "group": g} for k, l, w, h, g in FACTORS],
        "coverage": coverage,
        "signals": sig,
        "methodology": {
            "scoring": "Each factor is ranked as a percentile within its GICS sector (universe-wide if the sector has "
                       f"<{MIN_SECTOR_SIZE} names). Weighted average of available percentiles x 100; needs {MIN_FACTORS}/9 factors.",
            "levels": "Entry: pullback toward 20-day average (max 1.5 ATR below price). Stop: 2 ATR below entry or just under "
                      "20-day low. Target: 2R, or the 60-day high when it sits 1.5R-4R above entry.",
            "composite": f"Score = {W_ML:.0%} machine-learning rank (price/volume model, walk-forward validated) + "
                         f"{W_FUND:.0%} fundamental rank (9 sector-relative factors), re-ranked 0-100 across the universe.",
            "signal": "Relative tiers by score: top 15% STRONG BUY, next 20% BUY, middle 30% HOLD, next 20% REDUCE, bottom 15% SELL. "
                      "In a risk-off market regime, STRONG BUY also requires price above the 50-day average.",
            "portfolio": pstats.get("method", "") if len(port) else "",
            "alpha": "Jensen's alpha vs SPY from 1 year of daily returns, annualised. Descriptive, not predictive.",
            "disclaimer": "Research and demonstration tool. Not investment advice. No trades are placed.",
        },
        "stocks": stocks,
        "ml": ml_report,
        "regime": reg,
        "portfolio": {
            "capital": PORTFOLIO_CAPITAL,
            "hypothetical": True,
            "cash_weight": clean(1 - float(port["weight"].sum()) if len(port) else 1.0),
            "stats": {k: clean(v) if not isinstance(v, str) else v for k, v in pstats.items()} if len(port) else {},
            "holdings": [{"symbol": r["symbol"], "name": r["name"], "sector": r["sector"],
                          "weight": clean(r["weight"]), "value": clean(r["value"], 2), "shares": int(r["shares"]),
                          "price": clean(r["price"]), "score": clean(r["score"], 2)} for _, r in port.iterrows()],
            "trailing": [{k: clean(v) if k != "period" else v for k, v in t.items()} for t in trail],
            "trailing_note": "Trailing returns of TODAY's selection held over past windows (hindsight; not a backtest).",
        },
    }
    candles = {}
    for s in usable + [BENCHMARK]:
        t = prices[s].tail(CANDLE_DAYS)
        candles[s] = [[d.strftime("%Y-%m-%d"), round(float(o), 2), round(float(h), 2), round(float(l), 2),
                       round(float(c), 2), int(v) if not pd.isna(v) else 0]
                      for d, o, h, l, c, v in zip(t.index, t["Open"], t["High"], t["Low"], t["Close"], t["Volume"])]
    (outdir / "screener.json").write_text(json.dumps(report))
    (outdir / "candles.json").write_text(json.dumps(candles))
    log("DONE", f"wrote {outdir / 'screener.json'}  ({len(stocks)} stocks, {len(port)} holdings)")
    (outdir / "scan_log.txt").write_text("\n".join(_LOG[-1500:]))
    print("VERTEX SCREENER COMPLETE", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--limit", type=int, default=None, help="scan a subset for a quick demo")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    try:
        return run(Path(a.root).resolve(), a.limit, a.workers)
    except KeyboardInterrupt:
        log("STOP", "interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
