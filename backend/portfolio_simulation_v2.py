"""VertexCapital Upgrade 8: constrained, exploratory daily portfolio simulation.

Uses previously simulated fills/exits from Strategy v2. Not an independent backtest.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

AUDIT = Path("data/recommendations/strategy_v2/trade_audit.csv")
DEST = Path("data/recommendations/strategy_v2/portfolio")
OUTCOMES = {"STOP", "STOP_GAP", "TARGET", "TARGET_GAP", "TIME_EXIT"}


def simulate_portfolio(trades, prices, initial_cash=10000.0,
                       max_positions=5, allocation_fraction=0.15,
                       benchmark="SPY"):
    """One share-count position per ticker, no leverage, no fractional shares.

    Entry/exit fills are inherited from the strategy audit. Daily valuation uses
    adjusted close, so results remain approximations and must not be treated as
    executable performance.
    """
    if initial_cash <= 0 or not (0 < allocation_fraction <= 1) or max_positions < 1:
        raise ValueError("Invalid portfolio configuration")
    required = {"ticker", "entry_date", "exit_date", "entry_price",
                "exit_price", "outcome"}
    if not required.issubset(trades.columns):
        raise ValueError("Missing trade audit fields: " + str(sorted(required - set(trades.columns))))
    if benchmark not in prices:
        raise ValueError("Benchmark missing from price history")
    px = prices.copy().sort_index()
    px.index = pd.to_datetime(px.index).tz_localize(None) if px.index.tz is not None else pd.to_datetime(px.index)
    t = trades.copy()
    t = t[t.outcome.isin(OUTCOMES)].copy()
    for field in ("entry_date", "exit_date"):
        t[field] = pd.to_datetime(t[field], errors="coerce")
    for field in ("entry_price", "exit_price"):
        t[field] = pd.to_numeric(t[field], errors="coerce")
    t = t.dropna(subset=["entry_date", "exit_date", "entry_price", "exit_price"])
    t = t[(t.entry_date <= t.exit_date) & (t.entry_price > 0) & (t.exit_price > 0)]
    t = t[t.ticker.isin(px.columns)].sort_values(["entry_date", "ticker", "exit_date"], kind="stable")
    if t.empty:
        raise ValueError("No eligible completed trades")
    dates = px.index[(px.index >= t.entry_date.min()) & (px.index <= t.exit_date.max())]
    if len(dates) < 2:
        raise ValueError("Insufficient overlapping market dates")
    # Skip trades with missing daily valuations; do not invent missing prices.
    valid = []
    rejected = []
    for i, r in t.iterrows():
        window = px.loc[(px.index >= r.entry_date) & (px.index <= r.exit_date), r.ticker]
        if r.entry_date not in px.index or r.exit_date not in px.index or window.empty or window.isna().any() or (window <= 0).any():
            rejected.append({"ticker": r.ticker, "entry_date": str(r.entry_date.date()), "reason": "missing_market_prices"})
        else:
            valid.append(r.to_dict())
    if not valid:
        raise ValueError("No eligible completed trades with full market prices")
    by_entry = {}
    for r in valid:
        by_entry.setdefault(r["entry_date"], []).append(r)
    cash = float(initial_cash)
    active = {}
    equity_records, fills = [], []
    for date in dates:
        # Exit first: proceeds become available for entries that session.
        for ticker, p in list(active.items()):
            if p["exit_date"] == date:
                cash += p["shares"] * p["exit_price"]
                fills.append({"ticker": ticker, "entry_date": str(p["entry_date"].date()),
                              "exit_date": str(date.date()), "shares": p["shares"],
                              "entry_price": p["entry_price"], "exit_price": p["exit_price"],
                              "pnl": p["shares"] * (p["exit_price"] - p["entry_price"])})
                del active[ticker]
        for r in by_entry.get(date, []):
            ticker = r["ticker"]
            if ticker in active:
                rejected.append({"ticker": ticker, "entry_date": str(date.date()), "reason": "already_held"})
                continue
            if len(active) >= max_positions:
                rejected.append({"ticker": ticker, "entry_date": str(date.date()), "reason": "position_limit"})
                continue
            equity_before = cash + sum(p["shares"] * float(px.at[date, name]) for name,p in active.items())
            budget = min(cash, equity_before * allocation_fraction)
            shares = int(np.floor(budget / r["entry_price"]))
            if shares < 1:
                rejected.append({"ticker": ticker, "entry_date": str(date.date()), "reason": "insufficient_cash"})
                continue
            cash -= shares * r["entry_price"]
            active[ticker] = {**r, "shares": shares}
        marked = cash + sum(p["shares"] * float(px.at[date, ticker]) for ticker,p in active.items())
        equity_records.append({"date": date, "portfolio_value": marked,
                               "cash": cash, "open_positions": len(active)})
    equity = pd.DataFrame(equity_records).set_index("date")
    benchmark_px = px.loc[equity.index, benchmark]
    if benchmark_px.isna().any() or (benchmark_px <= 0).any():
        raise ValueError("Benchmark has missing or invalid prices")
    equity["benchmark_value"] = initial_cash * benchmark_px / benchmark_px.iloc[0]
    equity["portfolio_return"] = equity.portfolio_value.pct_change(fill_method=None).fillna(0)
    equity["benchmark_return"] = equity.benchmark_value.pct_change(fill_method=None).fillna(0)
    equity["drawdown"] = equity.portfolio_value / equity.portfolio_value.cummax() - 1
    report = {
        "initial_cash": initial_cash,
        "final_value": float(equity.portfolio_value.iloc[-1]),
        "total_return": float(equity.portfolio_value.iloc[-1] / initial_cash - 1),
        "benchmark_return": float(equity.benchmark_value.iloc[-1] / initial_cash - 1),
        "maximum_drawdown": float(equity.drawdown.min()),
        "executed_trades": len(fills),
        "rejected_trades": len(rejected),
        "max_concurrent_positions": int(equity.open_positions.max()),
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "warning": ("Exploratory reconstruction using precomputed historical fills, "
                    "adjusted closes for daily marks, no additional fees beyond audit fill prices, "
                    "no FX conversion, no cash interest, and no independently verified signal provenance. "
                    "Same-day entry/exit ordering is approximate. Not live-trading performance.")
    }
    return equity, pd.DataFrame(fills), pd.DataFrame(rejected, columns=["ticker","entry_date","reason"]), report


def main():
    if not AUDIT.exists():
        raise SystemExit("Missing trade audit. Run python backend/strategy_research_v2.py first.")
    from market_data import download_market_data
    trades = pd.read_csv(AUDIT)
    tickers = sorted(set(trades.get("ticker", pd.Series(dtype=str)).dropna().astype(str)) | {"SPY"})
    print("Downloading prices for portfolio reconstruction...")
    prices = download_market_data(tickers=tickers, period="10y")["close"]
    equity, fills, rejected, report = simulate_portfolio(trades, prices)
    DEST.mkdir(parents=True, exist_ok=True)
    equity.to_csv(DEST / "equity_curve.csv")
    fills.to_csv(DEST / "executed_trades.csv", index=False)
    rejected.to_csv(DEST / "rejected_trades.csv", index=False)
    (DEST / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("VERTEXCAPITAL UPGRADE 8 - EXPLORATORY PORTFOLIO")
    for k, v in report.items():
        print(f"{k}: {v}")
    print("Saved:", DEST)


if __name__ == "__main__":
    main()
