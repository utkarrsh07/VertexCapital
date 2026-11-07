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


def compute_features(close: pd.DataFrame, volume: pd.DataFrame, sectors: dict[str, str], bench: str = "SPY"):
    r = close.pct_change(fill_method=None)
    m = r[bench]
    f: dict[str, pd.DataFrame] = {}
    f["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    f["mom_6_1"] = close.shift(21) / close.shift(126) - 1
    f["mom_3"] = close / close.shift(63) - 1
    f["rev_1m"] = -(close / close.shift(21) - 1)                 # sign flipped: losers score high
    f["vol_63"] = -r.rolling(63, min_periods=50).std() * math.sqrt(252)   # low vol scores high
    f["dvol_63"] = -r.clip(upper=0).rolling(63, min_periods=50).std() * math.sqrt(252)
    cov = r.rolling(252, min_periods=180).cov(m)
    beta = cov.div(m.rolling(252, min_periods=180).var(), axis=0)
    f["beta_252"] = -beta                                         # betting-against-beta
    resid = r - beta.mul(m, axis=0)
    f["ivol_63"] = -resid.rolling(63, min_periods=50).std() * math.sqrt(252)
    f["hi52"] = close / close.rolling(252, min_periods=200).max() - 1
    f["ma50"] = close / close.rolling(50).mean() - 1
    f["ma200"] = close / close.rolling(200, min_periods=180).mean() - 1
    f["max21"] = -r.rolling(21).max()
    dollar = (close * volume).replace(0, np.nan)
    f["illiq"] = -(r.abs() / dollar).rolling(63, min_periods=40).mean() * 1e9
    f["volshock"] = volume.rolling(20).mean() / volume.rolling(120, min_periods=80).mean() - 1
    f["rsi14"] = _rsi(close)
    # sector-relative 12-1 momentum: stock minus its sector median each day
    sec = pd.Series(sectors)
    mom = f["mom_12_1"]
    smed = pd.DataFrame(index=mom.index, columns=mom.columns, dtype=float)
    for s_name, members in sec.groupby(sec).groups.items():
        cols = [c for c in members if c in mom.columns]
        if cols:
            med = mom[cols].median(axis=1)
            for c in cols:
                smed[c] = med
    f["smom"] = mom - smed
    return f, beta


def _xs_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank to [-0.5, 0.5] per date (robust to outliers and scale)."""
    return df.rank(axis=1, pct=True) - 0.5


def build_dataset(close, volume, sectors, bench="SPY"):
    feats, beta = compute_features(close, volume, sectors, bench)
    stocks = [c for c in close.columns if c != bench]
    fwd = close[stocks].shift(-HORIZON) / close[stocks] - 1
    n = len(close)
    # sample dates every HORIZON sessions, anchored on the LAST date so the newest month is predictable
    pos = list(range(n - 1, 251, -HORIZON))[::-1]
    dates = close.index[pos]
    rows = []
    for d in dates:
        X = pd.DataFrame({k: feats[k].loc[d, stocks] for k, _ in FEATURES})
        ok = X.notna().sum(axis=1) >= len(FEATURES) - 3
        X = X[ok]
        if len(X) < MIN_NAMES:
            continue
        Xr = X.rank(pct=True) - 0.5
        Xr = Xr.fillna(0.0)                          # missing -> neutral
        y = fwd.loc[d, X.index]
        ex = y - y.median()
        out = Xr.copy()
        out["date"] = d
        out["symbol"] = X.index
        out["fwd"] = y.values
        out["fwd_ex"] = ex.values
        out["target"] = ex.rank(pct=True).values - 0.5
        rows.append(out)
    data = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return data, feats, beta


# ---------------------------------------------------------------- models
def _models():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    return {
        "ridge": Ridge(alpha=5.0),
        "gbm": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
                                             min_samples_leaf=150, l2_regularization=1.0, random_state=7),
    }


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = [k for k, _ in FEATURES]
    out = {}
    for name, mdl in _models().items():
        mdl.fit(train[cols].values, train["target"].values)
        out[name] = mdl.predict(test[cols].values)
    return out


def _ens(test: pd.DataFrame, preds: dict[str, np.ndarray]) -> pd.Series:
    s = sum(pd.Series(p, index=test.index).groupby(test["date"]).rank(pct=True) for p in preds.values())
    return s / len(preds)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


# ---------------------------------------------------------------- walk-forward
def walk_forward(data: pd.DataFrame, log=print) -> pd.DataFrame:
    dates = sorted(data["date"].unique())
    labelled = [d for d in dates if data.loc[data["date"] == d, "fwd"].notna().mean() > 0.8]
    oos = []
    fit_at = None
    preds_cache = None
    for i, d in enumerate(labelled):
        if i < MIN_TRAIN_DATES:
            continue
        # purge: the previous sample's 21-day label overlaps this date, so train only on dates <= i-2
        train_dates = labelled[: i - 1]
        if fit_at is None or (i - fit_at) >= RETRAIN_EVERY:
            fit_at = i
            train = data[data["date"].isin(train_dates) & data["fwd"].notna()]
            block = data[data["date"].isin(labelled[i: i + RETRAIN_EVERY])]
            t0 = time.time()
            preds_cache = (block, _fit_predict(train, block))
            log(f"fold {pd.Timestamp(d).date()}  train={len(train):,} rows ({len(train_dates)} months)  "
                f"test={block['date'].nunique()} months  fit {time.time() - t0:.1f}s")
        block, preds = preds_cache
        mask = (block["date"] == d).values
        t = block[mask].copy()
        for name, p in preds.items():
            t[f"p_{name}"] = p[mask]
        oos.append(t)
    if not oos:
        return pd.DataFrame()
    res = pd.concat(oos, ignore_index=True)
    res["p_ens"] = _ens(res, {k: res[f"p_{k}"].values for k in _models()})
    return res


def evaluate(oos: pd.DataFrame, bench_fwd: pd.Series, others: dict[str, pd.Series] | None = None) -> dict:
    by = oos.groupby("date")
    ic = {m: by.apply(lambda g, m=m: _spearman(g[f"p_{m}"], g["fwd_ex"]), include_groups=False) for m in ("ridge", "gbm", "ens")}
    ens_ic = ic["ens"]
    oos = oos.copy()
    oos["q"] = by["p_ens"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1)
    qret = oos.groupby(["date", "q"])["fwd"].mean().unstack()
    ew = by["fwd"].mean()
    # top-quintile portfolio with turnover costs
    members = oos[oos["q"] == 5].groupby("date")["symbol"].apply(set)
    prev, cost = None, []
    for d, mem in members.items():
        turn = 1.0 if prev is None else 1 - len(mem & prev) / max(len(mem), 1)
        cost.append(2 * turn * COST_BPS / 1e4)    # sell + buy
        prev = mem
    top_net = qret[5] - pd.Series(cost, index=members.index)
    spy = bench_fwd.reindex(qret.index)

    def stats(r: pd.Series) -> dict:
        r = r.dropna()
        if len(r) < 3:
            return {}
        eq = (1 + r).cumprod()
        yrs = len(r) * HORIZON / 252
        ann = eq.iloc[-1] ** (1 / yrs) - 1
        vol = r.std() * math.sqrt(252 / HORIZON)
        dd = float((eq / eq.cummax() - 1).min())
        return {"cagr": float(ann), "vol": float(vol), "sharpe": float(ann / vol) if vol > 0 else None,
                "max_dd": dd, "hit": float((r > 0).mean()), "months": int(len(r))}

    n = len(ens_ic.dropna())
    t_stat = float(ens_ic.mean() / ens_ic.std() * math.sqrt(n)) if n > 2 and ens_ic.std() > 0 else 0.0
    excess = (top_net - ew).dropna()
    ex_t = float(excess.mean() / excess.std() * math.sqrt(len(excess))) if len(excess) > 2 and excess.std() > 0 else 0.0
    calib = []
    for q in range(1, 6):
        sub = oos[oos["q"] == q]
        calib.append({"quintile": q, "mean_excess": float(sub["fwd_ex"].mean()),
                      "hit_rate": float((sub["fwd_ex"] > 0).mean()), "n": int(len(sub))})
    strength = ("strong" if t_stat >= 3 and ex_t >= 2 else
                "moderate" if t_stat >= 2 else
                "weak" if t_stat >= 1 else "none")
    curve = pd.DataFrame({"top_net": (1 + top_net).cumprod(), "universe": (1 + ew).cumprod(),
                          "spy": (1 + spy.fillna(0)).cumprod()}).dropna(how="all")
    roll = ens_ic.rolling(12, min_periods=6).mean()
    benches = []
    for name, fwd_b in (others or {}).items():
        rb = fwd_b.reindex(qret.index)
        if rb.notna().mean() < 0.9:
            continue                       # not enough history to compare fairly
        rb = rb.fillna(0)
        benches.append({"name": name, "stats": stats(rb),
                        "curve": [[str(pd.Timestamp(d).date()), round(float(v), 4)] for d, v in (1 + rb).cumprod().items()]})
    return {
        "ic": {m: {"mean": float(v.mean()), "std": float(v.std()), "hit": float((v > 0).mean()),
                   "t": float(v.mean() / v.std() * math.sqrt(len(v))) if v.std() > 0 else 0.0} for m, v in ic.items()},
        "ic_t": t_stat, "evidence": strength,
        "quintiles": [{"q": int(q), "ann": float((1 + qret[q].mean()) ** (252 / HORIZON) - 1)} for q in qret.columns],
        "long_short_ann": float((1 + (qret[5] - qret[1]).mean()) ** (252 / HORIZON) - 1),
        "top_net": stats(top_net), "universe": stats(ew), "spy": stats(spy),
        "excess_t": ex_t, "avg_turnover": float(np.mean(cost) / (2 * COST_BPS / 1e4)) if cost else None,
        "calibration": calib,
        "curve": [[str(pd.Timestamp(d).date()), round(float(a), 4), round(float(b), 4), round(float(c), 4)]
                  for d, a, b, c in zip(curve.index, curve["top_net"], curve["universe"], curve["spy"])],
        "benchmarks": benches,
        "rolling_ic": [[str(pd.Timestamp(d).date()), round(float(v), 4)] for d, v in roll.dropna().items()],
        "ic_series": [[str(pd.Timestamp(d).date()), round(float(v), 4)] for d, v in ens_ic.dropna().items()],
        "first_oos": str(pd.Timestamp(ens_ic.index.min()).date()) if n else None,
        "last_oos": str(pd.Timestamp(ens_ic.index.max()).date()) if n else None,
        "oos_months": n,
    }


def regime(spy: pd.Series) -> dict:
    px = float(spy.iloc[-1])
    ma200 = float(spy.rolling(200).mean().iloc[-1])
    vol20 = float(spy.pct_change().tail(20).std() * math.sqrt(252))
    dd = float(px / spy.tail(252).max() - 1)
    if px > ma200 and vol20 < 0.22 and dd > -0.10:
        state, exposure = "RISK-ON", 1.0
    elif px < ma200 and (vol20 > 0.25 or dd < -0.15):
        state, exposure = "RISK-OFF", 0.6
    else:
        state, exposure = "NEUTRAL", 0.85
    return {"state": state, "exposure": exposure, "spy_vs_200d": px / ma200 - 1, "vol_20d": vol20, "drawdown_52w": dd}


def run_ml(prices: dict[str, pd.DataFrame], sectors: dict[str, str], log=print, bench: str = "SPY",
           bench_closes: dict[str, pd.Series] | None = None) -> dict | None:
    """Returns {'scores': DataFrame(symbol -> ml_pct, exp_excess, hit_prob, drivers), 'report': {...}}"""
    close, volume = build_panel(prices, bench)
    if len(close) < 252 + HORIZON * (MIN_TRAIN_DATES + 4):
        log(f"not enough history for walk-forward ({len(close)} sessions)")
        return None
    t0 = time.time()
    data, feats, beta = build_dataset(close, volume, sectors, bench)
    log(f"panel {close.shape[0]} sessions x {close.shape[1] - 1} stocks -> {len(data):,} samples, "
        f"{data['date'].nunique()} monthly cross-sections, {len(FEATURES)} features ({time.time() - t0:.1f}s)")
    oos = walk_forward(data, log)
    if oos.empty:
        log("walk-forward produced no out-of-sample predictions")
        return None
    spy_c = close[bench]
    bench_fwd = (spy_c.shift(-HORIZON) / spy_c - 1)
    others = {}
    for name, ser in (bench_closes or {}).items():
        sc = ser.reindex(close.index).ffill(limit=5)
        others[name] = sc.shift(-HORIZON) / sc - 1
    ev = evaluate(oos, bench_fwd, others)
    for b in ev["benchmarks"]:
        log(f"benchmark {b['name']:<24} CAGR {b['stats'].get('cagr', 0):+.1%}  Sharpe {b['stats'].get('sharpe') or 0:.2f}  "
            f"maxDD {b['stats'].get('max_dd', 0):.1%}")
    e = ev["ic"]["ens"]
    log(f"OOS {ev['oos_months']} months: rank IC {e['mean']:+.4f} (t={ev['ic_t']:.2f}, hit {e['hit']:.0%})  "
        f"ridge {ev['ic']['ridge']['mean']:+.4f}  gbm {ev['ic']['gbm']['mean']:+.4f}")
    tn, un = ev["top_net"], ev["universe"]
    if tn and un:
        log(f"top quintile net of {COST_BPS:.0f}bps: CAGR {tn['cagr']:+.1%} Sharpe {tn['sharpe']:.2f} maxDD {tn['max_dd']:.1%}  "
            f"| equal-weight universe CAGR {un['cagr']:+.1%}  | evidence: {ev['evidence'].upper()}")

    # ---- final model on all labelled dates, predict newest cross-section
    dates = sorted(data["date"].unique())
    latest = dates[-1]
    train = data[(data["date"] < dates[-2]) & data["fwd"].notna()]
    test = data[data["date"] == latest].copy()
    cols = [k for k, _ in FEATURES]
    mdls = _models()
    for name, mdl in mdls.items():
        mdl.fit(train[cols].values, train["target"].values)
        test[f"p_{name}"] = mdl.predict(test[cols].values)
    test["ml_pct"] = (test["p_ridge"].rank(pct=True) + test["p_gbm"].rank(pct=True)) / 2
    test["ml_pct"] = test["ml_pct"].rank(pct=True)
    q = np.clip(np.ceil(test["ml_pct"] * 5), 1, 5).astype(int)
    cal = {c["quintile"]: c for c in ev["calibration"]}
    test["exp_excess"] = q.map(lambda k: cal[k]["mean_excess"])
    test["hit_prob"] = q.map(lambda k: cal[k]["hit_rate"])
    test["quintile"] = q
    coef = mdls["ridge"].coef_
    contrib = test[cols].values * coef

    def drivers(row):
        order = np.argsort(row)
        pos = [(FEATURE_LABEL[cols[i]], float(row[i])) for i in order[::-1][:3] if row[i] > 0]
        neg = [(FEATURE_LABEL[cols[i]], float(row[i])) for i in order[:2] if row[i] < 0]
        return pos + neg
    test["drivers"] = [drivers(r) for r in contrib]

    # importance: |ridge coef| and GBM permutation importance on recent data
    from sklearn.inspection import permutation_importance
    recent = train[train["date"] >= dates[max(0, len(dates) - 26)]]
    pi = permutation_importance(mdls["gbm"], recent[cols].values, recent["target"].values,
                                n_repeats=3, random_state=7, scoring="r2")
    imp = [{"feature": FEATURE_LABEL[c], "key": c, "ridge": float(coef[i]),
            "gbm": float(pi.importances_mean[i])} for i, c in enumerate(cols)]
    mr = max(abs(x["ridge"]) for x in imp) or 1.0
    mg = max(abs(x["gbm"]) for x in imp) or 1.0
    imp.sort(key=lambda d: -(abs(d["ridge"]) / mr + max(d["gbm"], 0) / mg))
    reg = regime(spy_c)
    log(f"regime {reg['state']}: SPY {reg['spy_vs_200d']:+.1%} vs 200d, 20d vol {reg['vol_20d']:.0%}, "
        f"52w drawdown {reg['drawdown_52w']:.1%} -> target exposure {reg['exposure']:.0%}")
    ann_vol = (-feats["vol_63"]).iloc[-1]
    return {
        "scores": test.set_index("symbol")[["ml_pct", "exp_excess", "hit_prob", "quintile", "drivers"]],
        "beta": beta.iloc[-1], "vol": ann_vol,
        "report": {
            "models": ["Ridge regression (alpha=5)", "Histogram gradient boosting (250 trees, 15 leaves, min leaf 150)"],
            "ensemble": "average of cross-sectional ranks",
            "features": [{"key": k, "label": l} for k, l in FEATURES],
            "horizon_days": HORIZON, "retrain_every_months": RETRAIN_EVERY, "min_train_months": MIN_TRAIN_DATES,
            "cost_bps": COST_BPS, "samples": int(len(data)), "prediction_date": str(pd.Timestamp(latest).date()),
            "validation": ev, "importance": imp, "regime": reg,
            "caveats": [
                "Universe is today's index members, so delisted and dropped stocks are missing (survivorship bias makes backtests look better than reality).",
                "Prices are Yahoo adjusted closes, not a point-in-time database.",
                "Only the price/volume model is backtested; the 9 fundamental factors use today's values and cannot be validated with free data.",
                "Past out-of-sample performance does not guarantee future results.",
            ],
        },
    }
