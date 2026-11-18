"""
Regenerates README.md from the latest scan (data/vertex/screener.json), so every number in the
README is the real out-of-sample result of the most recent run.

    python scripts/update_readme.py            # writes README.md in the project root
    python scripts/update_readme.py --repo owner/name   # sets the CI badge link
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURE_REFS = {
    "mom_12_1": "Jegadeesh & Titman (1993)", "mom_6_1": "Jegadeesh & Titman (1993)", "mom_3": "Jegadeesh & Titman (1993)",
    "rev_1m": "Jegadeesh (1990)", "vol_63": "Ang, Hodrick, Xing & Zhang (2006)", "dvol_63": "Ang, Chen & Xing (2006)",
    "beta_252": "Frazzini & Pedersen (2014)", "ivol_63": "Ang, Hodrick, Xing & Zhang (2006)", "hi52": "George & Hwang (2004)",
    "ma50": "Brock, Lakonishok & LeBaron (1992)", "ma200": "Brock, Lakonishok & LeBaron (1992)",
    "max21": "Bali, Cakici & Whitelaw (2011)", "illiq": "Amihud (2002)", "volshock": "Gervais, Kaniel & Mingelgrin (2001)",
    "rsi14": "Wilder (1978)", "smom": "Moskowitz & Grinblatt (1999)", "rmom": "Blitz, Huij & Martens (2011)",
    "imom": "Moskowitz & Grinblatt (1999)",
}


def pct(x, d=1, sign=True):
    if x is None:
        return "n/a"
    s = f"{x * 100:+.{d}f}%" if sign else f"{x * 100:.{d}f}%"
    return s


def repo_slug() -> str | None:
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None
    if "github.com" not in url:
        return None
    slug = url.split("github.com")[-1].lstrip(":/").removesuffix(".git")
    return slug or None


def results_section(r: dict) -> str:
    ml = r.get("ml")
    if not ml:
        return "_Run a scan (`python backend/vertex_screener.py --root .`) and re-run `python scripts/update_readme.py` to fill in results._\n"
    v = ml["validation"]
    tn, un, spy = v.get("top_net") or {}, v.get("universe") or {}, v.get("spy") or {}
    ic = v["ic"]["ens"]
    years = v["oos_months"] * ml["horizon_days"] / 252
    rows = [("VertexCapital strategy (top quintile, net of costs)", tn), ("Equal-weight universe", un)]
    for b in v.get("benchmarks", []):
        rows.append((b["name"], b["stats"]))
    if not any(n.startswith("S&P 500") for n, _ in rows):
        rows.append(("S&P 500 (SPY)", spy))
    rows.sort(key=lambda x: -(x[1].get("cagr") or -9))
    table = "\n".join(f"| {n} | {pct(s.get('cagr'))} | {pct(s.get('vol'), sign=False)} | "
                      f"{s.get('sharpe', 0) or 0:.2f} | {pct(s.get('max_dd'))} |" for n, s in rows if s)
    edge = (tn.get("cagr") or 0) - (un.get("cagr") or 0)
    return f"""Walk-forward, out-of-sample: **{v['first_oos']} to {v['last_oos']}** ({v['oos_months']} monthly rebalances, ~{years:.1f} years). The model never saw a month before predicting it. Generated from the scan of {r.get('price_date')}.

| Metric | Value |
|---|---|
| Rank information coefficient (ensemble) | {ic['mean']:+.3f} (t-stat {v['ic_t']:.2f}, positive in {ic['hit'] * 100:.0f}% of months) |
| Strategy vs equal-weight universe | {pct(edge)} per year |
| Average monthly turnover | {pct(v.get('avg_turnover'), 0, sign=False)} |

| Line | CAGR | Volatility | Sharpe | Max drawdown |
|---|---|---|---|---|
{table}

**How to read this.** The fair comparison is the strategy against the equal-weight universe, because both share the same stock list (and the same survivorship bias, below). A rank IC around 0.03-0.05 with a t-stat above 2 is generally considered a useful signal; the current t-stat of {v['ic_t']:.2f} {"meets" if v['ic_t'] >= 2 else "does not yet meet"} that bar, which is why the score gives the machine-learning model a weight set by its own evidence.
"""


def build(repo: str | None) -> str:
    rp = ROOT / "data" / "vertex" / "screener.json"
    r = json.loads(rp.read_text()) if rp.exists() else {}
    ml = r.get("ml") or {}
    feats = ml.get("features") or []
    uni = r.get("universe") or {}
    n_stocks = uni.get("requested", 567)
    badge = (f"[![CI](https://github.com/{repo}/actions/workflows/ci.yml/badge.svg)](https://github.com/{repo}/actions/workflows/ci.yml) "
             if repo else "")
    shots = sorted((ROOT / "docs" / "screenshots").glob("*.png")) if (ROOT / "docs" / "screenshots").exists() else []
    shot_md = ("\n## Screenshots\n\n" + "\n\n".join(f"![{p.stem.replace('-', ' ').replace('_', ' ')}](docs/screenshots/{p.name})" for p in shots) + "\n") if shots else ""
    ftable = "\n".join(f"| {f['label']} | {FEATURE_REFS.get(f['key'], '')} |" for f in feats) if feats else \
        "| (run a scan to list features) | |"
    return f"""# VertexCapital

{badge}![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Next.js](https://img.shields.io/badge/Next.js-15-black) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

A quantitative equity research terminal. VertexCapital ranks {n_stocks} S&P 500 and TSX 60 stocks every day by combining a machine-learning model of price and volume behaviour with sector-relative fundamentals, tests the model walk-forward over roughly ten years of data it never trained on, and presents the results in a live web terminal with trade levels, a market heatmap, news, and defined-risk option ideas.

> Research and education project. Not investment advice; no orders are ever placed.
{shot_md}
## What it does

- **Ranks the whole universe daily** on a 0-100 score and assigns relative signals (Strong buy, Buy, Hold, Reduce, Sell).
- **Machine-learning model** (ridge regression + histogram gradient boosting) trained on {len(feats) or 18} price/volume features from the academic literature, validated with a purged, expanding-window walk-forward test.
- **Fundamental model**: nine factors (valuation, growth, profitability, balance sheet, risk, momentum) ranked within each GICS sector.
- **Evidence-weighted blend**: the machine-learning share of each score is set every scan by its own out-of-sample t-statistic (25% with no edge, up to 65%).
- **Trade levels** from volatility (ATR): entry zone, stop and target for every stock.
- **Portfolio construction**: top-ranked names sized by score and volatility, sector caps, Ledoit-Wolf covariance, a volatility target and a market-regime filter.
- **Web terminal** (Next.js): Top Picks with plain-language reasons and live news, a sector heatmap with market breadth, a strategy terminal, an Options Lab, and a Model page that compares the strategy with the S&P 500, Nasdaq, Dow, Russell 2000, TSX and MSCI World.

## Architecture

```mermaid
flowchart LR
  Y[(Yahoo Finance)] --> S[vertex_screener.py<br/>prices, fundamentals, ATR levels]
  S --> F[vertex_ml.py<br/>features, walk-forward training,<br/>evaluation, regime]
  F --> C[Composite score<br/>evidence-weighted blend]
  S --> C
  C --> P[Signals and portfolio<br/>Ledoit-Wolf, vol target]
  P --> J[(data/vertex/screener.json)]
  J --> A[Next.js API routes]
  Y -. live quotes, news, option chains .-> A
  O[vertex_options.py] --> A
  A --> U[Terminal UI<br/>Top Picks, Market, Strategy Terminal,<br/>Options Lab, Model]
```

## How the model works

**Universe and data.** {n_stocks} stocks (S&P 500 and TSX 60), 13 years of daily adjusted prices and current fundamentals from Yahoo Finance, refreshed by each scan.

**Features.** Every month, each feature is ranked across all stocks (rank-normalised to [-0.5, 0.5]) so the model learns relative, not absolute, patterns and is robust to outliers.

| Feature | Research basis |
|---|---|
{ftable}

**Target.** Each stock's next-21-trading-day return minus the universe median, rank-normalised. Predicting relative performance removes the market's direction, which no stock-level model can forecast reliably.

**Models.** Ridge regression (linear, interpretable) and histogram gradient boosting (non-linear interactions), combined by averaging their cross-sectional ranks.

**Validation.** Expanding-window walk-forward: the first 24 months train the model, which then predicts the next month, retraining every 3 months. A one-month purge gap stops training labels overlapping the test month (Lopez de Prado, 2018). Turnover is charged 10 basis points each way, and holdings use a buffer rule (keep names while they stay in the top 40%) to avoid trading on small rank changes.

**Calibration.** The out-of-sample average excess return and hit rate of each model quintile become each stock's "expected 1-month excess" and "beat the universe" figures, so expectations come from measured history rather than assumptions.

## Results
{results_section(r)}
## Limitations

- **Survivorship bias.** The universe is today's index members, so companies that were removed or went bankrupt are missing. This flatters every backtest on this page, often by 1-3% a year over a decade.
- **Not point-in-time data.** Yahoo adjusted prices can be revised; a professional system would use a point-in-time database.
- **Fundamentals are not backtested.** Free data only provides current values, so the nine fundamental factors cannot be validated historically.
- **Options ideas are not backtested.** Free data has no option price history.
- Past out-of-sample performance does not guarantee future results.

## Tech stack

Python (pandas, NumPy, scikit-learn, yfinance) for data, modelling and backtesting; Next.js 15 / React 19 / TypeScript for the terminal; hand-built SVG charts; GitHub Actions for tests.

## Running locally

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install scikit-learn yfinance pandas numpy scipy pytest
python -u backend/vertex_screener.py --root .     # full scan: ~25 minutes, 13 years of prices
cd terminal; npm install; npm run dev              # open http://localhost:3000/vertex
```

Tests: `python -m pytest tests/test_vertex_screener.py tests/test_vertex_options.py` (offline, synthetic data, including a check that the walk-forward model finds a planted signal and finds nothing in pure noise).

## Key files

| Path | Purpose |
|---|---|
| `backend/vertex_screener.py` | Scan pipeline: data, fundamentals, trade levels, scoring, signals, portfolio, report |
| `backend/vertex_ml.py` | Features, walk-forward training, evaluation, calibration, regime, benchmarks |
| `backend/vertex_options.py` | Option chains and Black-Scholes helpers for the Options Lab |
| `terminal/app/vertex/page.tsx` | Terminal UI |
| `terminal/app/api/vertex/` | API routes: report, candles, live quotes, market snapshot, news, options, scan |
| `tests/` | Offline tests for the pipeline, the model and the options code |

## References

Jegadeesh & Titman (1993); Jegadeesh (1990); Moskowitz & Grinblatt (1999); Blitz, Huij & Martens (2011); George & Hwang (2004); Ang, Hodrick, Xing & Zhang (2006); Frazzini & Pedersen (2014); Bali, Cakici & Whitelaw (2011); Amihud (2002); Novy-Marx (2013); Asness, Frazzini & Pedersen (2019); Gu, Kelly & Xiu (2020), "Empirical Asset Pricing via Machine Learning"; Ledoit & Wolf (2004); Lopez de Prado (2018), *Advances in Financial Machine Learning*.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    a = ap.parse_args()
    out = build(a.repo or repo_slug())
    (ROOT / "README.md").write_text(out, encoding="utf-8")
    print(f"README.md written ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
