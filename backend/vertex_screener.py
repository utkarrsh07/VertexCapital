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
PRICE_PERIOD = "5y"     # 5 years: 1y for levels + 4y of walk-forward history for the ML model
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
            raw = yf.download(batch, period=PRICE_PERIOD, interval="1d", auto_adjust=True,
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


def classify(row) -> str:
    if pd.isna(row["score"]):
        return "N/A"
    if row["score"] >= 65 and row["price"] > row["sma50"]:
        return "LONG"
    if row["score"] >= 50:
        return "WATCH"
    return "AVOID"


def build_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    cands = df[df["signal"] == "LONG"].sort_values("score", ascending=False)
    if len(cands) < 8:  # fall back to top scores if the LONG list is thin
        cands = df[df["score"].notna()].sort_values("score", ascending=False)
    picks, per_sector = [], {}
    for _, r in cands.iterrows():
        if per_sector.get(r["sector"], 0) >= MAX_PER_SECTOR:
            continue
        picks.append(r)
        per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        if len(picks) >= MAX_NAMES:
            break
    if not picks:
        return pd.DataFrame()
    p = pd.DataFrame(picks).reset_index(drop=True)
    raw = (p["score"] / 100.0) / p["atr_pct"].clip(lower=0.005)   # conviction / volatility
    wts = raw / raw.sum() * INVESTED
    for _ in range(50):  # iterative capping (name + sector), redistribute excess
        wts = wts.clip(upper=MAX_WEIGHT)
        sec = wts.groupby(p["sector"]).transform("sum")
        over = sec > MAX_SECTOR_WEIGHT
        wts[over] = wts[over] * MAX_SECTOR_WEIGHT / sec[over]
        gap = INVESTED - wts.sum()
        if gap < 1e-6:
            break
        free = (wts < MAX_WEIGHT - 1e-9) & (~over)
        if not free.any():
            break
        wts[free] += gap * raw[free] / raw[free].sum()
    p["weight"] = wts
    p["value"] = p["weight"] * PORTFOLIO_CAPITAL
    p["shares"] = np.floor(p["value"] / p["price"])
    return p.sort_values("weight", ascending=False)


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
    prices = download_prices(symbols + [BENCHMARK])
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
    df["signal"] = df.apply(classify, axis=1)
    scored = df[df["score"].notna()]
    log("SCORE", f"{len(scored)} scored, {len(df) - len(scored)} skipped (<{MIN_FACTORS} of 9 factors)")
    if len(scored) < 0.5 * len(df):
        log("WARN", "low fundamentals coverage (Yahoo rate limit?) - run the scan again; successful lookups are cached 24h")
    top = scored.sort_values("score", ascending=False).head(10)
    for _, r in top.iterrows():
        log("LEVELS", f"{r['symbol']:<9} score={r['score']:5.1f}  px={r['price']:.2f}  atr={r['atr']:.2f}  "
                      f"entry={r['entry']:.2f} stop={r['stop']:.2f} target={r['target']:.2f}  R:R={r['rr']:.2f}")
    sig = df["signal"].value_counts().to_dict()
    log("SIGNAL", "  ".join(f"{k}={v}" for k, v in sig.items()))

    port = build_portfolio(df)
    trail = trailing_returns(port, prices) if len(port) else []
    if len(port):
        log("PORTF", f"{len(port)} names, invested {port['weight'].sum():.1%}, cash {1 - port['weight'].sum():.1%}, "
                     f"max name {port['weight'].max():.1%}, sectors {port['sector'].nunique()}")
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
            "score": clean(r["score"], 2), "signal": r["signal"], "factor_count": int(r["factor_count"]),
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
            "signal": "LONG = score >= 65 and price above 50-day average; WATCH = score >= 50; AVOID otherwise.",
            "portfolio": f"Top {MAX_NAMES} LONG names, max {MAX_PER_SECTOR}/sector, weight ~ score / ATR%, "
                         f"capped {MAX_WEIGHT:.0%} per name and {MAX_SECTOR_WEIGHT:.0%} per sector, {INVESTED:.0%} invested.",
            "alpha": "Jensen's alpha vs SPY from 1 year of daily returns, annualised. Descriptive, not predictive.",
            "disclaimer": "Research and demonstration tool. Not investment advice. No trades are placed.",
        },
        "stocks": stocks,
        "portfolio": {
            "capital": PORTFOLIO_CAPITAL,
            "hypothetical": True,
            "cash_weight": clean(1 - float(port["weight"].sum()) if len(port) else 1.0),
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
