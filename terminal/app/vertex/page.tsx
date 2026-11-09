"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import "./vertex.css";

/* ============================================================ types */
type FactorVal = { value: number | null; pct: number | null };
type Stock = {
  symbol: string; name: string; sector: string; index: string; currency: string;
  market_cap: number | null; price: number; change_1d: number | null; score: number | null;
  signal: string; factor_count: number; fund_score: number | null; ml_score: number | null;
  exp_excess_1m: number | null; hit_prob: number | null; ml_quintile: number | null; drivers: [string, number][];
  vol_1y: number | null; dividend_yield: number | null; earnings_date?: string | null;
  value_score: number | null; growth_score: number | null; quality_score: number | null;
  factors: Record<string, FactorVal>; alpha: number | null; beta_realized: number | null;
  atr: number; atr_pct: number; sma20: number; sma50: number;
  entry: number; stop: number; target: number; rr: number | null; spark: number[];
};
type Holding = { symbol: string; name: string; sector: string; weight: number; value: number; shares: number; price: number; score: number };
type Report = {
  version: string; status: string; generated_utc: string; price_date: string; source: string;
  universe: { requested: number; priced: number; scored: number; by_index: Record<string, number> };
  factors: { key: string; label: string; weight: number; higher_is_better: boolean; group: string }[];
  coverage: Record<string, number>; signals: Record<string, number>;
  methodology: Record<string, string>; stocks: Stock[]; ml: MLReport | null; regime: Regime | null;
  portfolio: { capital: number; cash_weight: number; holdings: Holding[];
    trailing: { period: string; portfolio: number; benchmark: number | null }[]; trailing_note: string };
};
type Candle = [string, number, number, number, number, number];
type Regime = { state: string; exposure: number; spy_vs_200d: number; vol_20d: number; drawdown_52w: number };
type PerfStats = { cagr: number; vol: number; sharpe: number | null; max_dd: number; hit: number; months: number };
type MLReport = {
  models: string[]; ensemble: string; features: { key: string; label: string }[]; horizon_days: number;
  retrain_every_months: number; min_train_months: number; cost_bps: number; samples: number; prediction_date: string;
  importance: { feature: string; key: string; ridge: number; gbm: number }[]; regime: Regime; caveats: string[];
  validation: {
    ic: Record<"ridge" | "gbm" | "ens", { mean: number; std: number; hit: number; t: number }>; ic_t: number; evidence: string;
    quintiles: { q: number; ann: number }[]; long_short_ann: number; top_net: PerfStats; universe: PerfStats; spy: PerfStats;
    excess_t: number; avg_turnover: number | null; calibration: { quintile: number; mean_excess: number; hit_rate: number; n: number }[];
    curve: [string, number, number, number][]; rolling_ic: [string, number][]; ic_series: [string, number][];
    benchmarks?: { name: string; stats: PerfStats; curve: [string, number][] }[];
    first_oos: string | null; last_oos: string | null; oos_months: number;
  };
};
const SIG_CLASS: Record<string, string> = { "STRONG BUY": "s-sb", BUY: "s-b", HOLD: "s-h", REDUCE: "s-r", SELL: "s-s" };
const SIG_SHORT: Record<string, string> = { "STRONG BUY": "Strong buy", BUY: "Buy", HOLD: "Hold", REDUCE: "Reduce", SELL: "Sell" };
const sigCls = (s?: string | null) => SIG_CLASS[s || ""] || "s-na";
const isBuy = (s?: string) => s === "STRONG BUY" || s === "BUY";
const isSell = (s?: string) => s === "SELL" || s === "REDUCE";
type Quote = { price: number; prevClose: number | null; change: number | null; open: number | null;
  high: number | null; low: number | null; volume: number | null; time: number; marketState: string };
type QuoteFeed = { quotes: Record<string, Quote>; asOf: number; latencyMs: number; received: number; requested: number;
  intraday: { symbol: string; points: [number, number][] } | null; error?: string };
const TAPE = ["SPY", "QQQ", "DIA", "IWM", "^VIX", "XIU.TO"];
const TAPE_NAME: Record<string, string> = { SPY: "S&P 500", QQQ: "NASDAQ 100", DIA: "DOW 30", IWM: "RUSSELL 2K", "^VIX": "VIX", "XIU.TO": "TSX 60" };
const REFRESH_S = 15;

/* ============================================================ helpers */
const SECTOR_COLORS: Record<string, string> = {
  "Information Technology": "#4f8cff", Financials: "#22c3ee", "Health Care": "#a78bfa",
  "Consumer Discretionary": "#f59e0b", Industrials: "#34d399", "Communication Services": "#f472b6",
  "Consumer Staples": "#fb923c", Energy: "#facc15", Utilities: "#2dd4bf", "Real Estate": "#818cf8",
  Materials: "#94a3b8",
};
const SECTOR_SHORT: Record<string, string> = {
  "Information Technology": "Tech", Financials: "Financials", "Health Care": "Health Care",
  "Consumer Discretionary": "Cons. Disc.", Industrials: "Industrials", "Communication Services": "Comm. Svcs",
  "Consumer Staples": "Cons. Staples", Energy: "Energy", Utilities: "Utilities", "Real Estate": "Real Estate",
  Materials: "Materials",
};
const SECTOR_CODE: Record<string, string> = {
  "Information Technology": "IT", Financials: "FN", "Health Care": "HC", "Consumer Discretionary": "CD",
  Industrials: "IN", "Communication Services": "CM", "Consumer Staples": "CS", Energy: "EN", Utilities: "UT",
  "Real Estate": "RE", Materials: "MA",
};
const sc = (s: string) => SECTOR_COLORS[s] || "#7b88b3";
const short = (s: string) => SECTOR_SHORT[s] || s;
const tick = (s: string) => s.replace(".TO", "");
const fx = (v: number | null | undefined, d = 2) =>
  v == null || !isFinite(v) ? "—" : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v: number | null | undefined, d = 2, sign = true) =>
  v == null || !isFinite(v) ? "—" : `${sign && v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`;
const money = (v: number, d = 0) => "$" + fx(v, d);
const cls = (...a: (string | false | null | undefined)[]) => a.filter(Boolean).join(" ");

/** Number that animates to its new value and flashes green/red when it changes. */
function Tween({ v, fmt, className, flash = true }: { v: number | null | undefined; fmt: (x: number) => string; className?: string; flash?: boolean }) {
  const [shown, setShown] = useState(v ?? null);
  const [dir, setDir] = useState<"" | "fu" | "fd">("");
  const prev = useRef(v ?? null);
  useEffect(() => {
    const from = prev.current, to = v ?? null;
    prev.current = to;
    if (to == null || from == null || from === to) { setShown(to); return; }
    if (flash) { setDir(to > from ? "fu" : "fd"); }
    const t0 = performance.now(), dur = 700;
    let raf = 0;
    const step = (t: number) => {
      const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      setShown(from + (to - from) * e);
      if (k < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    const clear = setTimeout(() => setDir(""), 1200);
    return () => { cancelAnimationFrame(raf); clearTimeout(clear); };
  }, [v, flash]);
  return <span className={cls("vx-tw", dir, className)}>{shown == null ? "—" : fmt(shown)}</span>;
}

function factorDisplay(key: string, v: number | null) {
  if (v == null) return "n/a";
  if (["revenue_growth", "profit_margin", "roe", "momentum_6m"].includes(key)) return pct(v, 1);
  if (key === "debt_to_equity") return `${v.toFixed(2)}x`;
  if (key === "beta") return v.toFixed(2);
  return v.toFixed(1);
}

function useSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ w: 600, h: 300 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, size] as const;
}

/* ============================================================ icons */
const I = {
  logo: (
    <svg viewBox="0 0 32 32" width="30" height="30" aria-hidden>
      <defs><linearGradient id="vxg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#5aa2ff" /><stop offset="1" stopColor="#2c5bff" /></linearGradient></defs>
      <path d="M16 2 29 9.5v13L16 30 3 22.5v-13z" fill="none" stroke="url(#vxg)" strokeWidth="2" />
      <path d="M9.5 11 16 22l6.5-11" fill="none" stroke="#9cc4ff" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ),
  dash: <svg viewBox="0 0 16 16" width="14" height="14"><rect x="1.5" y="2.5" width="13" height="9" rx="1.5" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="M5 14h6" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>,
  grid: <svg viewBox="0 0 16 16" width="14" height="14"><rect x="1.5" y="1.5" width="7" height="8" rx="1" fill="none" stroke="currentColor" strokeWidth="1.3" /><rect x="10" y="1.5" width="4.5" height="4" rx="1" fill="none" stroke="currentColor" strokeWidth="1.3" /><rect x="10" y="7" width="4.5" height="7.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.3" /><rect x="1.5" y="11" width="7" height="3.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.3" /></svg>,
  term: <svg viewBox="0 0 16 16" width="14" height="14"><path d="M2 12 6 8 2 4M8 12h6" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" /></svg>,
  book: <svg viewBox="0 0 16 16" width="14" height="14"><path d="M3 2.5h7a2 2 0 0 1 2 2v9H5a2 2 0 0 1-2-2z M3 11.5a2 2 0 0 1 2-2h7" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>,
  refresh: <svg viewBox="0 0 16 16" width="14" height="14"><path d="M13.5 8a5.5 5.5 0 1 1-1.6-3.9M13.5 2.5v3h-3" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" /></svg>,
  bell: <svg viewBox="0 0 16 16" width="15" height="15"><path d="M4 11V7a4 4 0 0 1 8 0v4l1 1.5H3zM6.5 14h3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" /></svg>,
  plus: <svg viewBox="0 0 16 16" width="14" height="14"><circle cx="8" cy="8" r="6.5" fill="none" stroke="currentColor" strokeWidth="1.2" /><path d="M8 5v6M5 8h6" stroke="currentColor" strokeWidth="1.2" /></svg>,
  search: <svg viewBox="0 0 16 16" width="13" height="13"><circle cx="7" cy="7" r="4.5" fill="none" stroke="currentColor" strokeWidth="1.4" /><path d="m10.5 10.5 3 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" /></svg>,
  sort: <svg viewBox="0 0 16 16" width="14" height="14"><path d="M4 5h8M4 8h8M4 11h8" stroke="currentColor" strokeWidth="1.3" /><circle cx="6" cy="5" r="1.3" fill="currentColor" /><circle cx="10" cy="8" r="1.3" fill="currentColor" /><circle cx="7" cy="11" r="1.3" fill="currentColor" /></svg>,
  expand: <svg viewBox="0 0 16 16" width="13" height="13"><path d="M3 6V3h3M13 6V3h-3M3 10v3h3M13 10v3h-3" fill="none" stroke="currentColor" strokeWidth="1.4" /></svg>,
  gear: <svg viewBox="0 0 16 16" width="13" height="13"><circle cx="8" cy="8" r="2.2" fill="none" stroke="currentColor" strokeWidth="1.3" /><path d="M8 1.8v2M8 12.2v2M1.8 8h2M12.2 8h2M3.6 3.6l1.4 1.4M11 11l1.4 1.4M3.6 12.4 5 11M11 5l1.4-1.4" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" /></svg>,
};

function Badge({ s, size = 22 }: { s: { symbol: string; sector: string }; size?: number }) {
  const c = sc(s.sector);
  return (
    <span className="vx-badge" style={{ width: size, height: size, background: `${c}22`, borderColor: `${c}88`, color: c, fontSize: size * 0.42 }}>
      {tick(s.symbol).slice(0, 2)}
    </span>
  );
}

/* ============================================================ charts */
function CandleChart({ candles, stock, showLevels, showSma }: { candles: Candle[]; stock: Stock | null; showLevels: boolean; showSma: boolean }) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const pad = { l: 8, r: 64, t: 10, b: 24 };
  const cw = Math.max(50, w - pad.l - pad.r);
  const priceH = Math.max(80, (h - pad.t - pad.b) * 0.82);
  const volTop = pad.t + priceH + 8;
  const volH = Math.max(20, h - pad.b - volTop);

  const geo = useMemo(() => {
    if (!candles.length) return null;
    let lo = Math.min(...candles.map((c) => c[3]));
    let hi = Math.max(...candles.map((c) => c[2]));
    if (stock && showLevels) { lo = Math.min(lo, stock.stop); hi = Math.max(hi, stock.target); }
    const padP = (hi - lo) * 0.06; lo -= padP; hi += padP;
    const vmax = Math.max(...candles.map((c) => c[5])) || 1;
    const step = cw / candles.length;
    const y = (p: number) => pad.t + ((hi - p) / (hi - lo)) * priceH;
    const sma: (number | null)[] = candles.map((_, i) => i < 19 ? null : candles.slice(i - 19, i + 1).reduce((a, c) => a + c[4], 0) / 20);
    return { lo, hi, vmax, step, y, sma };
  }, [candles, cw, priceH, stock, showLevels, pad.t]);

  if (!geo) return <div ref={ref} className="vx-chart vx-empty">No price history loaded</div>;
  const { lo, hi, vmax, step, y, sma } = geo;
  const bw = Math.max(1.5, step * 0.62);
  const ticks = Array.from({ length: 6 }, (_, i) => lo + ((hi - lo) * i) / 5);
  const xTicks = candles.map((c, i) => [c[0], i] as const).filter((_, i) => i % Math.ceil(candles.length / 6) === 3);
  const last = candles[candles.length - 1];
  const hc = hover != null ? candles[hover] : last;
  const smaPath = sma.map((v, i) => v == null ? "" : `${sma[i - 1] == null ? "M" : "L"}${(pad.l + i * step + step / 2).toFixed(1)},${y(v).toFixed(1)}`).join("");
  const levels = stock && showLevels ? [
    { k: "TARGET", v: stock.target, c: "#21d49a" },
    { k: "ENTRY", v: stock.entry, c: "#4f8cff" },
    { k: "STOP", v: stock.stop, c: "#ff4d6d" },
  ] : [];

  return (
    <div ref={ref} className="vx-chart"
      onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const i = Math.floor((e.clientX - r.left - pad.l) / step); setHover(i >= 0 && i < candles.length ? i : null); }}
      onMouseLeave={() => setHover(null)}>
      <div className="vx-ohlc">
        <span>{hc[0]}</span>
        <span>O <b>{fx(hc[1])}</b></span><span>H <b>{fx(hc[2])}</b></span><span>L <b>{fx(hc[3])}</b></span>
        <span>C <b className={hc[4] >= hc[1] ? "up" : "dn"}>{fx(hc[4])}</b></span>
        <span>Vol <b>{(hc[5] / 1e6).toFixed(2)}M</b></span>
      </div>
      <svg width={w} height={h}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={pad.l} x2={pad.l + cw} y1={y(t)} y2={y(t)} className="vx-grid" />
            <text x={pad.l + cw + 8} y={y(t) + 3} className="vx-axis">{fx(t)}</text>
          </g>
        ))}
        {xTicks.map(([d, i]) => (
          <g key={d}>
            <line x1={pad.l + i * step + step / 2} x2={pad.l + i * step + step / 2} y1={pad.t} y2={volTop + volH} className="vx-grid" />
            <text x={pad.l + i * step + step / 2} y={h - 7} textAnchor="middle" className="vx-axis">
              {new Date(d + "T00:00:00").toLocaleDateString("en-US", { day: "2-digit", month: "short" })}
            </text>
          </g>
        ))}
        {candles.map((c, i) => {
          const x = pad.l + i * step + step / 2;
          const up = c[4] >= c[1];
          const top = y(Math.max(c[1], c[4])), bot = y(Math.min(c[1], c[4]));
          const vh = (c[5] / vmax) * volH;
          return (
            <g key={c[0]} className={up ? "c-up" : "c-dn"}>
              <rect x={x - bw / 2} y={volTop + volH - vh} width={bw} height={vh} className="vol" />
              <line x1={x} x2={x} y1={y(c[2])} y2={y(c[3])} />
              <rect x={x - bw / 2} y={top} width={bw} height={Math.max(1, bot - top)} rx={0.5} />
            </g>
          );
        })}
        {showSma && <path d={smaPath} className="vx-sma" />}
        {levels.map((l) => (
          <g key={l.k}>
            <line x1={pad.l} x2={pad.l + cw} y1={y(l.v)} y2={y(l.v)} stroke={l.c} strokeDasharray="5 4" strokeWidth={1.1} opacity={0.85} />
            <text x={pad.l + cw - 6} y={y(l.v) - 5} fill={l.c} textAnchor="end" className="vx-lvl">{l.k}</text>
            <rect x={pad.l + cw + 2} y={y(l.v) - 8} width={58} height={16} rx={3} fill={l.c} />
            <text x={pad.l + cw + 31} y={y(l.v) + 4} textAnchor="middle" className="vx-tag">{fx(l.v)}</text>
          </g>
        ))}
        {stock && showLevels && (
          <>
            <rect x={pad.l} y={y(stock.target)} width={cw} height={y(stock.entry) - y(stock.target)} fill="#21d49a" opacity={0.045} />
            <rect x={pad.l} y={y(stock.entry)} width={cw} height={y(stock.stop) - y(stock.entry)} fill="#ff4d6d" opacity={0.05} />
          </>
        )}
        <g>
          <line x1={pad.l} x2={pad.l + cw} y1={y(last[4])} y2={y(last[4])} stroke="#e6ecff" strokeDasharray="2 3" opacity={0.35} />
          <circle cx={pad.l + (candles.length - 1) * step + step / 2} cy={y(last[4])} r={3} className="vx-pulse-dot" />
          <circle cx={pad.l + (candles.length - 1) * step + step / 2} cy={y(last[4])} r={3} className="vx-pulse-ring" />
          <rect x={pad.l + cw + 2} y={y(last[4]) - 8} width={58} height={16} rx={3} fill="#e6ecff" />
          <text x={pad.l + cw + 31} y={y(last[4]) + 4} textAnchor="middle" className="vx-tag dark">{fx(last[4])}</text>
        </g>
        {hover != null && (
          <line x1={pad.l + hover * step + step / 2} x2={pad.l + hover * step + step / 2} y1={pad.t} y2={volTop + volH} className="vx-cross" />
        )}
      </svg>
    </div>
  );
}

function Spark({ data, w = 140, h = 36, color = "#4f8cff", fill = true }: { data: number[]; w?: number; h?: number; color?: string; fill?: boolean }) {
  const id = "sg" + useId().replace(/[^a-zA-Z0-9]/g, "");
  if (!data?.length) return null;
  const lo = Math.min(...data), hi = Math.max(...data);
  const pts = data.map((v, i) => [(i / (data.length - 1)) * w, h - 2 - ((v - lo) / (hi - lo || 1)) * (h - 4)]);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
  return (
    <svg width={w} height={h} className="vx-spark">
      <defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={color} stopOpacity=".35" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>
      {fill && <path d={`${d}L${w},${h}L0,${h}Z`} fill={`url(#${id})`} />}
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} className="vx-draw" pathLength={1} />
    </svg>
  );
}

function arc(cx: number, cy: number, r: number, a0: number, a1: number) {
  const p = (a: number) => [cx + r * Math.cos(a - Math.PI / 2), cy + r * Math.sin(a - Math.PI / 2)];
  const [x0, y0] = p(a0), [x1, y1] = p(a1);
  return `M${x0},${y0} A${r},${r} 0 ${a1 - a0 > Math.PI ? 1 : 0} 1 ${x1},${y1}`;
}

function Ring({ segs, size, stroke, gap = 0.04, children, icons }: {
  segs: { v: number; c: string; label?: string }[]; size: number; stroke: number; gap?: number; children?: React.ReactNode; icons?: boolean;
}) {
  const tot = segs.reduce((a, s) => a + s.v, 0) || 1;
  const r = size / 2 - stroke / 2 - (icons ? 16 : 2);
  let a = 0;
  return (
    <div className="vx-ring" style={{ width: size, height: size }}>
      <svg width={size} height={size}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#141d3d" strokeWidth={stroke} />
        {segs.map((s, i) => {
          const span = (s.v / tot) * Math.PI * 2;
          const a0 = a + gap / 2, a1 = a + span - gap / 2;
          const mid = a + span / 2;
          a += span;
          const ix = size / 2 + (r + stroke / 2 + 13) * Math.cos(mid - Math.PI / 2);
          const iy = size / 2 + (r + stroke / 2 + 13) * Math.sin(mid - Math.PI / 2);
          return (
            <g key={i}>
              {a1 > a0 && <path d={arc(size / 2, size / 2, r, a0, a1)} stroke={s.c} strokeWidth={stroke} fill="none" strokeLinecap="round" className="vx-seg" />}
              {icons && s.label && (
                <g>
                  <circle cx={ix} cy={iy} r={11} fill="#0b1330" stroke={s.c} strokeWidth={1.4} />
                  <text x={ix} y={iy + 3.5} textAnchor="middle" fontSize="9" fontWeight="700" fill={s.c}>{s.label}</text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
      <div className="vx-ring-c">{children}</div>
    </div>
  );
}

/* ============================================================ trade setup bar */
function SetupBar({ stock, price, live, date, intraday }: {
  stock: Stock; price: number; live: boolean; date?: string; intraday: [number, number][] | null;
}) {
  const zLo = Math.max(stock.stop + 0.15 * stock.atr, stock.entry - 0.25 * stock.atr);
  const zHi = stock.entry + 0.25 * stock.atr;
  const lo = stock.stop, hi = stock.target, span = hi - lo || 1;
  const at = (v: number) => Math.max(0, Math.min(100, ((v - lo) / span) * 100));
  const status =
    price <= stock.stop ? { t: "Below stop — setup invalidated", c: "dn" }
    : price >= stock.target ? { t: "Target reached", c: "up" }
    : price < zLo ? { t: "Below entry zone — near risk level", c: "warn" }
    : price <= zHi ? { t: "Inside entry zone", c: "acc" }
    : { t: "Above entry zone — wait for pullback", c: "mut" };
  const dist = (v: number) => pct(v / price - 1, 1);
  const pts = intraday && intraday.length > 1 ? intraday.map((p) => p[1]) : null;
  return (
    <section className="vx-card vx-setup">
      <div className="vx-setup-h">
        <b>{tick(stock.symbol)} — Trade Setup</b>
        <span className={cls("vx-badge2", live && "live")}>{live ? <><span className="vx-dot ok" />Live</> : `Scan ${date || ""}`}</span>
      </div>
      <div className="vx-setup-k">
        <div><span>Entry zone</span><b className="up">{fx(zLo)}–{fx(zHi)}</b></div>
        <div><span>Target</span><b>${fx(stock.target)}</b><i className="up">{dist(stock.target)}</i></div>
        <div><span>Risk level</span><b className="dn">${fx(stock.stop)}</b><i className="dn">{dist(stock.stop)}</i></div>
      </div>
      <div className="vx-setup-bar">
        <span className="seg r" style={{ left: 0, width: `${at(zLo)}%` }} />
        <span className="seg a" style={{ left: `${at(zLo)}%`, width: `${at(zHi) - at(zLo)}%` }} />
        <span className="seg g" style={{ left: `${at(zHi)}%`, width: `${100 - at(zHi)}%` }} />
        <span className={cls("vx-mark", status.c)} style={{ left: `${at(price)}%` }}>
          <em><Tween v={price} fmt={(x) => fx(x)} /></em>
        </span>
      </div>
      <div className="vx-setup-t">
        <span>${fx(stock.stop)}</span>
        <span style={{ left: `${at(stock.entry)}%` }}>${fx(stock.entry)}</span>
        <span>${fx(stock.target)}</span>
      </div>
      <div className="vx-setup-f">
        <span className={cls("vx-st", status.c)}>{status.t}</span>
        {pts && <Spark data={pts} w={110} h={22} fill={false} color={pts[pts.length - 1] >= pts[0] ? "#21d49a" : "#ff4d6d"} />}
      </div>
      <div className="vx-setup-n">ATR(14) volatility levels from the {date} scan · price {live ? "live, ~15 min delayed" : "from scan"} · not a forecast</div>
    </section>
  );
}

/* ============================================================ console */
const TAG_CLASS: Record<string, string> = {
  INIT: "t-init", UNIVERSE: "t-uni", PRICES: "t-px", FUNDS: "t-fund", FACTORS: "t-fac", RANK: "t-rank",
  SCORE: "t-score", LEVELS: "t-lvl", SIGNAL: "t-sig", PORTF: "t-port", DONE: "t-done", ERROR: "t-err", ML: "t-ml", WARN: "t-err",
};
function LogLine({ line }: { line: string }) {
  const m = line.match(/^\[\s*([\d.]+s)\]\s+(\w+)\s+(.*)$/);
  if (!m) return <div className={cls("vx-log", line.startsWith("$") && "cmd", /error|Traceback/i.test(line) && "err")}>{line || " "}</div>;
  return (
    <div className="vx-log">
      <span className="ts">{m[1]}</span> <span className={cls("tg", TAG_CLASS[m[2]])}>{m[2].padEnd(8, " ")}</span> <span>{m[3]}</span>
    </div>
  );
}

/* ============================================================ market tab */
type IndexData = { symbol: string; price: number; prevClose: number | null; change: number | null; points: [number, number][]; marketState: string };
type MarketFeed = { asOf: number | null; ageMs: number | null; refreshing: boolean; source: string | null; coverage: number;
  quotes: Record<string, [number, number | null]>; indices: Record<string, IndexData> };
type Box = { x: number; y: number; w: number; h: number };
const INDEX_CARDS: [string, string][] = [["^GSPC", "S&P 500"], ["^IXIC", "NASDAQ"], ["^DJI", "DOW"], ["^GSPTSE", "TSX COMPOSITE"]];

/** Squarified treemap (Bruls et al.): lays values into rects with aspect ratios close to 1. */
function squarify<T>(items: { v: number; d: T }[], r: Box): (Box & { d: T })[] {
  const out: (Box & { d: T })[] = [];
  const list = items.filter((i) => i.v > 0).sort((a, b) => b.v - a.v);
  const total = list.reduce((a, i) => a + i.v, 0);
  if (!total || r.w <= 0 || r.h <= 0) return out;
  const scaled = list.map((i) => ({ a: (i.v / total) * r.w * r.h, d: i.d }));
  const rect = { ...r };
  const worst = (row: { a: number }[], side: number) => {
    const s = row.reduce((x, i) => x + i.a, 0);
    let m = 0;
    for (const i of row) m = Math.max(m, (side * side * i.a) / (s * s), (s * s) / (side * side * i.a));
    return m;
  };
  const lay = (row: { a: number; d: T }[]) => {
    const s = row.reduce((x, i) => x + i.a, 0);
    if (rect.w >= rect.h) {
      const w = s / rect.h; let y = rect.y;
      for (const i of row) { const h = i.a / w; out.push({ x: rect.x, y, w, h, d: i.d }); y += h; }
      rect.x += w; rect.w -= w;
    } else {
      const h = s / rect.w; let x = rect.x;
      for (const i of row) { const w = i.a / h; out.push({ x, y: rect.y, w, h, d: i.d }); x += w; }
      rect.y += h; rect.h -= h;
    }
  };
  let row: { a: number; d: T }[] = [];
  for (const it of scaled) {
    const side = Math.min(rect.w, rect.h);
    if (!row.length || worst([...row, it], side) <= worst(row, side)) row.push(it);
    else { lay(row); row = [it]; }
  }
  if (row.length) lay(row);
  return out;
}

function heat(c: number | null | undefined) {
  if (c == null || !isFinite(c)) return "#262c45";
  const k = Math.max(-1, Math.min(1, c / 0.03));
  const n = [64, 69, 84], to = k < 0 ? [246, 53, 56] : [48, 204, 90], t = Math.pow(Math.abs(k), 0.8);
  return `rgb(${n.map((v, i) => Math.round(v + (to[i] - v) * t)).join(",")})`;
}

function IndexCard({ name, d }: { name: string; d?: IndexData }) {
  const [ref, { w, h }] = useSize<HTMLDivElement>();
  const pts = d?.points || [];
  const up = (d?.change ?? 0) >= 0, col = up ? "#21d49a" : "#ff4d6d";
  let path = "", area = "", prevY = 0, lastXY: [number, number] = [0, 0];
  if (pts.length > 1 && w > 0) {
    const vals = pts.map((p) => p[1]).concat(d?.prevClose != null ? [d.prevClose] : []);
    const lo = Math.min(...vals), hi = Math.max(...vals), padv = (hi - lo) * 0.08 || 1;
    const t0 = pts[0][0], t1 = Math.max(pts[pts.length - 1][0], t0 + 6.5 * 3600);
    const X = (t: number) => ((t - t0) / (t1 - t0)) * (w - 52);
    const Y = (v: number) => 6 + ((hi + padv - v) / (hi - lo + 2 * padv)) * (h - 22);
    path = pts.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    area = `${path}L${X(pts[pts.length - 1][0]).toFixed(1)},${h - 16}L0,${h - 16}Z`;
    prevY = d?.prevClose != null ? Y(d.prevClose) : -10;
    lastXY = [X(pts[pts.length - 1][0]), Y(pts[pts.length - 1][1])];
  }
  const id = "ix" + useId().replace(/[^a-zA-Z0-9]/g, "");
  return (
    <section className="vx-card vx-ixc">
      <div className="vx-ixh">
        <b>{name}</b>
        <span className={up ? "up" : "dn"}>{d?.change != null && d.prevClose != null ? `${up ? "+" : ""}${fx(d.price - d.prevClose)} (${pct(d.change)})` : "—"}</span>
      </div>
      <div className="vx-ixp"><Tween v={d?.price} fmt={(x) => fx(x)} /></div>
      <div className="vx-ixch" ref={ref}>
        {pts.length > 1 ? (
          <svg width={w} height={h}>
            <defs><linearGradient id={id} x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={col} stopOpacity=".28" /><stop offset="1" stopColor={col} stopOpacity="0" /></linearGradient></defs>
            <line x1={0} x2={w - 52} y1={prevY} y2={prevY} stroke="#7482ad" strokeDasharray="3 3" opacity={0.6} />
            <path d={area} fill={`url(#${id})`} />
            <path d={path} fill="none" stroke={col} strokeWidth={1.6} className="vx-draw" pathLength={1} />
            <circle cx={lastXY[0]} cy={lastXY[1]} r={3} fill={col} />
            <circle cx={lastXY[0]} cy={lastXY[1]} r={3} fill="none" stroke={col} className="vx-pulse-ring" />
            <text x={w - 48} y={prevY + 3} className="vx-axis">prev</text>
            {["9:30", "12:00", "4:00"].map((t, i) => <text key={t} x={[0, (w - 52) * 0.385, w - 70][i]} y={h - 3} className="vx-axis">{t}</text>)}
          </svg>
        ) : <div className="vx-empty sm">Waiting for intraday data…</div>}
      </div>
    </section>
  );
}

function Breadth({ label, l, r, a, b }: { label: string; l: string; r: string; a: number; b: number }) {
  const t = a + b || 1;
  return (
    <section className="vx-card vx-br">
      <div className="vx-br-h"><span className="up">{l}</span><b>{label}</b><span className="dn">{r}</span></div>
      <div className="vx-br-v"><span className="up">{pct(a / t, 1, false)} ({a})</span><span className="dn">({b}) {pct(b / t, 1, false)}</span></div>
      <div className="vx-br-bar"><span className="g" style={{ width: `${(a / t) * 100}%` }} /><span className="r" style={{ width: `${(b / t) * 100}%` }} /></div>
    </section>
  );
}

function MarketView({ report, onOpen, clock }: { report: Report | null; onOpen: (s: string) => void; clock: string }) {
  const [mk, setMk] = useState<MarketFeed | null>(null);
  const [uni, setUni] = useState<"All" | "S&P 500" | "TSX 60">("S&P 500");
  const [hover, setHover] = useState<{ s: Stock & { lp: number; lc: number | null }; x: number; y: number } | null>(null);
  const [nextIn, setNextIn] = useState(20);
  const [hmRef, { w: hw, h: hh }] = useSize<HTMLDivElement>();

  const poll = useCallback(async (wait = false) => {
    try { setMk(await fetch(`/api/vertex/market${wait ? "?wait=1" : ""}`, { cache: "no-store" }).then((r) => r.json())); } catch {}
    setNextIn(20);
  }, []);
  useEffect(() => {
    poll(true);
    const id = setInterval(() => setNextIn((n) => { if (n <= 1) { poll(); return 20; } return n - 1; }), 1000);
    return () => clearInterval(id);
  }, [poll]);

  const rows = useMemo(() => (report?.stocks || [])
    .filter((s) => uni === "All" || s.index === uni)
    .map((s) => { const lq = mk?.quotes[s.symbol.toUpperCase()]; return { ...s, lp: lq?.[0] ?? s.price, lc: lq ? lq[1] : s.change_1d, live: !!lq }; }), [report, mk, uni]);

  const br = useMemo(() => {
    let adv = 0, dec = 0, a50 = 0, b50 = 0, a20 = 0, b20 = 0, hi = 0, lo = 0;
    for (const s of rows) {
      if ((s.lc ?? 0) > 0) adv++; else if ((s.lc ?? 0) < 0) dec++;
      if (s.lp >= s.sma50) a50++; else b50++;
      if (s.lp >= s.sma20) a20++; else b20++;
      if (s.spark?.length) { if (s.lp >= Math.max(...s.spark)) hi++; else if (s.lp <= Math.min(...s.spark)) lo++; }
    }
    return { adv, dec, a50, b50, a20, b20, hi, lo };
  }, [rows]);

  const tiles = useMemo(() => {
    if (hw < 50 || hh < 50) return [];
    const bySec = new Map<string, typeof rows>();
    for (const s of rows) if (s.market_cap && s.market_cap > 0) { const g = bySec.get(s.sector) || []; g.push(s); bySec.set(s.sector, g); }
    const secs = squarify([...bySec.entries()].map(([k, g]) => ({ v: g.reduce((a, s) => a + (s.market_cap || 0), 0), d: k })), { x: 0, y: 0, w: hw, h: hh });
    return secs.map((sb) => {
      const inner = { x: sb.x + 1, y: sb.y + 15, w: Math.max(0, sb.w - 2), h: Math.max(0, sb.h - 16) };
      return { sector: sb.d, box: sb, cells: squarify((bySec.get(sb.d) || []).map((s) => ({ v: s.market_cap || 0, d: s })), inner) };
    });
  }, [rows, hw, hh]);

  const movers = useMemo(() => {
    const live = rows.filter((s) => s.lc != null).sort((a, b) => (b.lc as number) - (a.lc as number));
    return { up: live.slice(0, 10), dn: live.slice(-10).reverse() };
  }, [rows]);
  const headline = mk?.indices["^GSPC"];
  const isOpen = /REGULAR/.test(headline?.marketState || "");
  const mtable = (list: typeof rows, title: string) => (
    <section className="vx-card vx-mv">
      <h3>{title}</h3>
      <table>
        <thead><tr><th>Ticker</th><th>Last</th><th>Change</th><th>Signal</th><th>Score</th></tr></thead>
        <tbody>
          {list.map((s) => (
            <tr key={s.symbol} onClick={() => onOpen(s.symbol)}>
              <td><span className="vx-li-s"><Badge s={s} size={16} /><b>{tick(s.symbol)}</b></span></td>
              <td><Tween v={s.lp} fmt={(x) => fx(x)} /></td>
              <td className={(s.lc ?? 0) >= 0 ? "up" : "dn"}>{pct(s.lc)}</td>
              <td><span className={cls("vx-sig", sigCls(s.signal))}>{s.signal}</span></td>
              <td>{fx(s.score, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );

  return (
    <main className="vx-mkt">
      <div className="vx-mhead">
        <h1 className="vx-h1">Market Overview</h1>
        <div className="vx-mhead-r">
          <span className={cls("vx-badge2", isOpen && "live")}>{isOpen ? <><span className="vx-dot ok" />Market open</> : "Market closed"}</span>
          <span className="vx-muted sm vx-num">{mk ? `${mk.coverage}/${report?.stocks.length || 0} quotes · ${mk.refreshing ? "refreshing…" : `next ${nextIn}s`}` : "connecting…"} · {clock}</span>
        </div>
      </div>
      <div className="vx-idx">{INDEX_CARDS.map(([s, n]) => <IndexCard key={s} name={n} d={mk?.indices[s]} />)}</div>
      <div className="vx-brs">
        <Breadth label="Advancing / Declining" l="Advancing" r="Declining" a={br.adv} b={br.dec} />
        <Breadth label="60-Day High / Low" l="New High" r="New Low" a={br.hi} b={br.lo} />
        <Breadth label="SMA 20" l="Above" r="Below" a={br.a20} b={br.b20} />
        <Breadth label="SMA 50" l="Above" r="Below" a={br.a50} b={br.b50} />
      </div>
      <div className="vx-mrow">
        <section className="vx-card vx-hm-card">
          <div className="vx-chead">
            <h3>{uni === "All" ? "Universe" : uni} · 1-Day Performance <span className="vx-muted sm">size = market cap</span></h3>
            <div className="vx-mini">
              {(["S&P 500", "TSX 60", "All"] as const).map((u) => <button key={u} className={cls(uni === u && "on")} onClick={() => setUni(u)}>{u}</button>)}
            </div>
          </div>
          <div className="vx-hm" ref={hmRef} onMouseLeave={() => setHover(null)}>
            {tiles.map((t) => (
              <div key={t.sector}>
                <div className="vx-hm-sec" style={{ left: t.box.x, top: t.box.y, width: t.box.w, height: t.box.h }}>
                  <span>{t.box.w > 60 ? t.sector.toUpperCase() : ""}</span>
                </div>
                {t.cells.map((c) => {
                  const area = c.w * c.h, fs = Math.max(8, Math.min(30, Math.sqrt(area) / 4.2));
                  return (
                    <div key={c.d.symbol} className="vx-hm-c" style={{ left: c.x, top: c.y, width: c.w, height: c.h, background: heat(c.d.lc) }}
                      onClick={() => onOpen(c.d.symbol)}
                      onMouseMove={(e) => { const r = (e.currentTarget.parentElement?.parentElement as HTMLElement).getBoundingClientRect(); setHover({ s: c.d, x: e.clientX - r.left, y: e.clientY - r.top }); }}>
                      {c.w > 26 && c.h > 16 && <b style={{ fontSize: fs }}>{tick(c.d.symbol)}</b>}
                      {c.w > 34 && c.h > 30 && <i style={{ fontSize: Math.max(8, fs * 0.5) }}>{pct(c.d.lc)}</i>}
                    </div>
                  );
                })}
              </div>
            ))}
            {hover && (
              <div className="vx-hm-tip" style={{ left: Math.min(hover.x + 14, hw - 230), top: Math.min(hover.y + 14, hh - 130) }}>
                <div className="vx-li-s"><Badge s={hover.s} size={18} /><b>{tick(hover.s.symbol)}</b><span className="vx-muted">{hover.s.name}</span></div>
                <div className="vx-hm-tip-g">
                  <span>Last</span><b>{fx(hover.s.lp)}</b>
                  <span>Change</span><b className={(hover.s.lc ?? 0) >= 0 ? "up" : "dn"}>{pct(hover.s.lc)}</b>
                  <span>Score</span><b>{fx(hover.s.score, 1)} <em className={cls("vx-sig", sigCls(hover.s.signal))}>{hover.s.signal}</em></b>
                  <span>Mkt cap</span><b>{hover.s.market_cap ? `${(hover.s.market_cap / 1e9).toFixed(1)}B ${hover.s.currency}` : "—"}</b>
                </div>
                <div className="vx-muted sm">Click to open in Strategy Terminal</div>
              </div>
            )}
            {!report && <div className="vx-empty">Run a scan first</div>}
          </div>
          <div className="vx-hm-leg">
            {[-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03].map((v) => <span key={v} style={{ background: heat(v) }}>{v > 0 ? "+" : ""}{(v * 100).toFixed(0)}%</span>)}
          </div>
        </section>
        <div className="vx-mside">
          {mtable(movers.up, "Top Gainers")}
          {mtable(movers.dn, "Top Losers")}
        </div>
      </div>
      <div className="vx-disc">Quotes from Yahoo Finance, ~15 min delayed. Breadth and heatmap cover the VertexCapital universe ({uni}), not the full exchange.</div>
    </main>
  );
}

/* ============================================================ shared line chart */
function LineChart({ series, height = 220, fmt, zero, log = false }: {
  series: { name: string; color: string; pts: [string, number][]; dash?: boolean; bars?: boolean; glow?: boolean; area?: boolean; dim?: boolean }[];
  height?: number; fmt: (v: number) => string; zero?: number; log?: boolean;
}) {
  const tf = (v: number) => (log ? Math.log(Math.max(v, 1e-9)) : v);
  const inv = (v: number) => (log ? Math.exp(v) : v);
  const [ref, { w }] = useSize<HTMLDivElement>();
  const [hi, setHi] = useState<number | null>(null);
  const all = series.flatMap((s) => s.pts.map((p) => tf(p[1]))).filter((v) => isFinite(v)).concat(zero != null ? [tf(zero)] : []);
  const n = Math.max(...series.map((s) => s.pts.length), 0);
  if (!n || w < 50) return <div ref={ref} style={{ height }} className="vx-empty sm">No data</div>;
  const lo = Math.min(...all), hiV = Math.max(...all), pad = (hiV - lo) * 0.08 || 1;
  const L = 8, R = 56, T = 8, B = 20, cw = w - L - R, ch = height - T - B;
  const X = (i: number) => L + (n === 1 ? cw / 2 : (i / (n - 1)) * cw);
  const Y = (raw: number) => { const v = tf(raw); return T + ((hiV + pad - v) / (hiV - lo + 2 * pad)) * ch; };
  const ticks = Array.from({ length: 5 }, (_, i) => lo - pad + ((hiV - lo + 2 * pad) * i) / 4);
  const dates = series[0].pts.map((p) => p[0]);
  return (
    <div ref={ref} className="vx-lc" style={{ minHeight: height }}
      onMouseMove={(e) => { const r = e.currentTarget.getBoundingClientRect(); const i = Math.round(((e.clientX - r.left - L) / cw) * (n - 1)); setHi(i >= 0 && i < n ? i : null); }}
      onMouseLeave={() => setHi(null)}>
      <svg width={w} height={height}>
        {ticks.map((t, i) => <g key={i}><line x1={L} x2={L + cw} y1={Y(inv(t))} y2={Y(inv(t))} className="vx-grid" /><text x={L + cw + 6} y={Y(inv(t)) + 3} className="vx-axis">{fmt(inv(t))}</text></g>)}
        {zero != null && <line x1={L} x2={L + cw} y1={Y(zero)} y2={Y(zero)} stroke="var(--mut)" strokeDasharray="3 3" opacity={0.6} />}
        {dates.filter((_, i) => i % Math.max(1, Math.ceil(n / 6)) === 0).map((d) => {
          const i = dates.indexOf(d);
          return <text key={d} x={X(i)} y={height - 5} textAnchor={i === 0 ? "start" : "middle"} className="vx-axis">{d.slice(0, 7)}</text>;
        })}
        {series.map((s) => s.bars
          ? s.pts.map((p, i) => <rect key={i} x={X(i) - Math.max(1, cw / n * 0.35)} width={Math.max(2, cw / n * 0.7)} y={Math.min(Y(p[1]), Y(zero ?? 0))}
              height={Math.abs(Y(p[1]) - Y(zero ?? 0))} fill={p[1] >= (zero ?? 0) ? "var(--up)" : "var(--dn)"} opacity={0.45} />)
          : (() => {
              const d = s.pts.map((p, i) => !isFinite(p[1]) ? "" : `${i && isFinite(s.pts[i - 1][1]) ? "L" : "M"}${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
              const base = Y(zero ?? (log ? Math.exp(lo - pad) : lo - pad));
              const last = s.pts.length - 1;
              return (
                <g key={s.name} className={cls(s.dim && "vx-dimline", s.glow && "vx-hero")}>
                  {s.area && last > 0 && (
                    <>
                      <defs>
                        <linearGradient id="vx-hero-fill" x1="0" x2="0" y1="0" y2="1">
                          <stop offset="0" stopColor="#e8a33d" stopOpacity="0.32" /><stop offset="1" stopColor="#e8a33d" stopOpacity="0" />
                        </linearGradient>
                      </defs>
                      <path d={`${d}L${X(last).toFixed(1)},${base.toFixed(1)}L${X(0).toFixed(1)},${base.toFixed(1)}Z`} fill="url(#vx-hero-fill)" className="vx-hero-area" />
                    </>
                  )}
                  {s.glow && <path d={d} fill="none" stroke={s.color} strokeWidth={14} className="vx-glow2" />}
                  {s.glow && <path d={d} fill="none" stroke={s.color} strokeWidth={7} className="vx-glow" />}
                  <path d={d} fill="none" stroke={s.color} strokeWidth={s.glow ? 3 : s.dash ? 1.2 : 1.3}
                    strokeDasharray={s.dash ? "4 3" : undefined} className="vx-draw" pathLength={1} />
                  {s.glow && last >= 0 && (
                    <>
                      <circle cx={X(last)} cy={Y(s.pts[last][1])} r={5} fill={s.color} className="vx-glow-dot" />
                      <circle cx={X(last)} cy={Y(s.pts[last][1])} r={5} fill="none" stroke={s.color} className="vx-pulse-ring" />
                      <text x={X(last) - 8} y={Y(s.pts[last][1]) - 12} textAnchor="end" className="vx-hero-tag">VertexCapital ${fx(s.pts[last][1], 2)}</text>
                    </>
                  )}
                </g>
              );
            })())}
        {hi != null && <line x1={X(hi)} x2={X(hi)} y1={T} y2={T + ch} className="vx-cross" />}
      </svg>
      <div className="vx-lc-leg">
        {hi != null && <span className="vx-muted">{dates[hi]}</span>}
        {series.map((s) => (
          <span key={s.name}><i style={{ background: s.color }} />{s.name}{hi != null && s.pts[hi] && isFinite(s.pts[hi][1]) ? <b> {fmt(s.pts[hi][1])}</b> : null}</span>
        ))}
      </div>
    </div>
  );
}

/* ============================================================ benchmark comparison */
const BENCH_COLORS = ["#8ab4f8", "#c58af9", "#a8a29e", "#f28b82", "#81c995", "#78d9ec"];
function BenchmarkChart({ ml }: { ml: MLReport }) {
  const v = ml.validation;
  const [logScale, setLogScale] = useState(true);
  const [hidden, setHidden] = useState<Record<string, boolean>>({});
  const ours = { name: "VertexCapital strategy (net)", color: "#f5b54a", glow: true, area: true, stats: v.top_net, pts: v.curve.map((c) => [c[0], c[1]] as [string, number]) };
  const uni = { name: "Equal-weight universe", color: "var(--txt)", dash: true, dim: true, stats: v.universe, pts: v.curve.map((c) => [c[0], c[2]] as [string, number]) };
  const benches = (v.benchmarks || []).map((b, i) => ({ name: b.name, color: BENCH_COLORS[i % BENCH_COLORS.length], dim: true, stats: b.stats, pts: b.curve }));
  const all = [ours, uni, ...benches];
  const shown = all.filter((x) => !hidden[x.name]);
  const years = (v.oos_months * ml.horizon_days) / 252;
  const ranked = [...all].sort((a, b) => (b.stats?.cagr ?? -9) - (a.stats?.cagr ?? -9));
  return (
    <section className="vx-card vx-pad vx-bench">
      <div className="vx-chead">
        <h3>{years.toFixed(1)} years out of sample vs the major markets <span className="vx-muted sm">growth of $1, {v.first_oos} to {v.last_oos}</span></h3>
        <div className="vx-mini">
          <button className={cls(logScale && "on")} onClick={() => setLogScale(true)}>Log</button>
          <button className={cls(!logScale && "on")} onClick={() => setLogScale(false)}>Linear</button>
        </div>
      </div>
      <LineChart height={360} log={logScale} fmt={(x) => `$${x.toFixed(x < 10 ? 2 : 1)}`} zero={1}
        series={[...shown.filter((x) => !("glow" in x)), ...shown.filter((x) => "glow" in x)]} />
      <table className="vx-mtab vx-btab">
        <thead><tr><th></th><th>Line</th><th>$1 became</th><th>CAGR</th><th>Volatility</th><th>Sharpe</th><th>Worst drawdown</th></tr></thead>
        <tbody>
          {ranked.map((x) => (
            <tr key={x.name} className={cls(hidden[x.name] && "off", "glow" in x && "ours")} onClick={() => setHidden({ ...hidden, [x.name]: !hidden[x.name] })}>
              <td><i className="vx-sw" style={{ background: x.color }} /></td>
              <td>{x.name}</td>
              <td>${fx(x.pts[x.pts.length - 1]?.[1], 2)}</td>
              <td className={(x.stats?.cagr ?? 0) >= 0 ? "up" : "dn"}>{pct(x.stats?.cagr, 1)}</td>
              <td>{pct(x.stats?.vol, 1, false)}</td>
              <td>{fx(x.stats?.sharpe, 2)}</td>
              <td className="dn">{pct(x.stats?.max_dd, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="vx-foot vx-left">
        Click a row to hide or show its line. Benchmarks are ETFs with dividends reinvested; TSX is in CAD. The VertexCapital line only uses months the model had not trained on,
        but its universe is today&apos;s index members, which leaves out companies that failed (survivorship bias, often 1 to 3% a year over a decade).
        The fairest comparison is the dashed equal-weight universe, which has the same bias.
      </div>
    </section>
  );
}

/* ============================================================ calendar-year returns */
function yearly(pts: [string, number][]): Record<string, { r: number; partial: boolean }> {
  const out: Record<string, { r: number; partial: boolean }> = {};
  let base = 1, startMonth = pts.length ? Number(pts[0][0].slice(5, 7)) : 1;
  const byYear = new Map<string, [string, number][]>();
  pts.forEach((p) => { const y = p[0].slice(0, 4); byYear.set(y, [...(byYear.get(y) || []), p]); });
  const years = [...byYear.keys()].sort();
  years.forEach((y, i) => {
    const g = byYear.get(y)!;
    const end = g[g.length - 1][1];
    const partial = (i === 0 && startMonth > 2) || (i === years.length - 1 && Number(g[g.length - 1][0].slice(5, 7)) < 11);
    out[y] = { r: end / base - 1, partial };
    base = end;
    startMonth = 1;
  });
  return out;
}

function AnnualReturns({ ml }: { ml: MLReport }) {
  const v = ml.validation;
  const lines = [
    { name: "VertexCapital", color: "#f5b54a", pts: v.curve.map((c) => [c[0], c[1]] as [string, number]), hero: true },
    ...(v.benchmarks || []).map((b, i) => ({ name: b.name.replace(/ \(.*\)/, ""), color: BENCH_COLORS[i % BENCH_COLORS.length], pts: b.curve, hero: false })),
    { name: "Equal-weight universe", color: "var(--txt)", pts: v.curve.map((c) => [c[0], c[2]] as [string, number]), hero: false },
  ];
  const ys = lines.map((l) => ({ ...l, y: yearly(l.pts) }));
  const years = Object.keys(ys[0].y).sort();
  const sp = ys.find((l) => l.name.startsWith("S&P 500"));
  const barLines = ys.filter((l) => l.hero || l.name.startsWith("S&P 500") || l.name.startsWith("Nasdaq") || l.name.startsWith("Equal"));
  const allR = barLines.flatMap((l) => years.map((y) => l.y[y]?.r ?? 0));
  const mx = Math.max(0.05, ...allR.map(Math.abs));
  const fullYears = years.filter((y) => !ys[0].y[y].partial);
  const beatSp = sp ? fullYears.filter((y) => ys[0].y[y].r > (sp.y[y]?.r ?? 0)).length : 0;
  const posYears = fullYears.filter((y) => ys[0].y[y].r > 0).length;
  const cell = (r: number) => {
    const k = Math.max(-1, Math.min(1, r / 0.3));
    return k >= 0 ? `rgba(69, 192, 138, ${0.12 + 0.55 * k})` : `rgba(229, 83, 75, ${0.12 + 0.55 * -k})`;
  };
  return (
    <section className="vx-card vx-pad vx-annual">
      <div className="vx-chead">
        <h3>Returns by calendar year</h3>
        <span className="vx-muted sm">
          VertexCapital was positive in <b className="up">{posYears} of {fullYears.length}</b> full years and beat the S&amp;P 500 in <b className="vx-acc">{beatSp} of {fullYears.length}</b>
        </span>
      </div>
      <div className="vx-ybars">
        {years.map((y) => (
          <div key={y} className="vx-ycol">
            <div className="vx-yplot">
              <div className="vx-yzero" />
              {barLines.map((l, i) => {
                const r = l.y[y]?.r ?? 0, h = (Math.abs(r) / mx) * 50;
                return (
                  <div key={l.name} className={cls("vx-yb", l.hero && "hero", r < 0 && "neg")} title={`${l.name} ${y}: ${pct(r, 1)}`}
                    style={{ height: `${h}%`, left: `${6 + i * (88 / barLines.length)}%`, width: `${(88 / barLines.length) - 4}%`, background: l.color, ...(r >= 0 ? { bottom: "50%" } : { top: "50%" }) }} />
                );
              })}
            </div>
            <b className={ys[0].y[y].r >= 0 ? "up" : "dn"}>{pct(ys[0].y[y].r, 0)}</b>
            <span>{y}{ys[0].y[y].partial ? "*" : ""}</span>
          </div>
        ))}
      </div>
      <div className="vx-lc-leg">{barLines.map((l) => <span key={l.name}><i style={{ background: l.color }} />{l.name}</span>)}</div>
      <div className="vx-ytab-wrap">
        <table className="vx-ytab">
          <thead><tr><th>Line</th>{years.map((y) => <th key={y}>{y}{ys[0].y[y].partial ? "*" : ""}</th>)}</tr></thead>
          <tbody>
            {ys.map((l) => (
              <tr key={l.name} className={cls(l.hero && "hero")}>
                <td><i className="vx-sw" style={{ background: l.color }} />{l.name}</td>
                {years.map((y) => { const r = l.y[y]?.r; return <td key={y} style={{ background: r == null ? undefined : cell(r) }}>{r == null ? "—" : pct(r, 1)}</td>; })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="vx-foot vx-left">* partial year (the test window starts {v.first_oos} and ends {v.last_oos}). Out-of-sample, after costs; same survivorship caveat as above.</div>
    </section>
  );
}

/* ============================================================ model tab */
const EVIDENCE_TEXT: Record<string, string> = {
  strong: "The ranking predicted next-month returns consistently out of sample, and the top quintile beat the universe after costs with statistical significance.",
  moderate: "The ranking showed a real but modest edge out of sample. Useful for tilting a diversified portfolio; not enough on its own to concentrate money.",
  weak: "The out-of-sample edge is small and could be luck. Treat the model as one input, not a reason to trade.",
  none: "No reliable out-of-sample edge. Rankings fall back mainly to fundamentals; do not rely on the model.",
};

function ModelView({ report }: { report: Report | null }) {
  const ml = report?.ml;
  if (!ml) {
    return (
      <main className="vx-model"><h1 className="vx-h1">Model</h1>
        <section className="vx-card vx-pad"><p>No model results yet. Press <b>Run scan</b>: the first V20 scan downloads 5 years of prices and trains the model (a few minutes).</p></section>
      </main>
    );
  }
  const v = ml.validation, e = v.ic.ens, reg = ml.regime;
  const sgn = (x?: number | null) => ((x ?? 0) >= 0 ? "up" : "dn");
  const kpis: [string, string, string, string?][] = [
    ["Out-of-sample months", String(v.oos_months), "vx-acc", `${v.first_oos} to ${v.last_oos}`],
    ["Rank IC (ensemble)", `${e.mean >= 0 ? "+" : ""}${e.mean.toFixed(3)}`, sgn(e.mean), `t-stat ${v.ic_t.toFixed(2)} · positive ${pct(e.hit, 0, false)} of months`],
    ["VertexCapital CAGR, net", pct(v.top_net?.cagr, 1), sgn(v.top_net?.cagr), `after ${ml.cost_bps} bps per trade`],
    ["Equal-weight universe", pct(v.universe?.cagr, 1), "vx-kpi-mut", `SPY ${pct(v.spy?.cagr, 1)}`],
    ["VertexCapital Sharpe", fx(v.top_net?.sharpe, 2), "vx-acc", `universe ${fx(v.universe?.sharpe, 2)}`],
    ["VertexCapital max drawdown", pct(v.top_net?.max_dd, 1), "dn", `monthly hit rate ${pct(v.top_net?.hit, 0, false)}`],
  ];
  const maxImp = Math.max(...ml.importance.map((i) => Math.abs(i.ridge)), 1e-9);
  const maxG = Math.max(...ml.importance.map((i) => Math.max(i.gbm, 0)), 1e-9);
  return (
    <main className="vx-model">
      <div className="vx-mhead">
        <h1 className="vx-h1">Model</h1>
        <span className="vx-muted sm">
          Market regime <b className={reg.state === "RISK-ON" ? "up" : reg.state === "RISK-OFF" ? "dn" : "vx-acc"}>{reg.state.replace("-", " ").toLowerCase()}</b>
          {" "}(SPY {pct(reg.spy_vs_200d, 1)} vs 200-day, exposure {pct(reg.exposure, 0, false)}) · trained on {ml.samples.toLocaleString()} stock-months · predictions as of {ml.prediction_date}
        </span>
      </div>
      <div className="vx-kpis">
        {kpis.map(([k, val, c, sub]) => <section key={k} className={cls("vx-card vx-kpi", c)}><span>{k}</span><b className={c}>{val}</b>{sub && <small>{sub}</small>}</section>)}
      </div>
      <BenchmarkChart ml={ml} />
      <AnnualReturns ml={ml} />
      <div className="vx-mgrid2">

        <section className="vx-card vx-pad">
          <h3>Information coefficient by month</h3>
          <LineChart height={210} fmt={(x) => x.toFixed(2)} zero={0}
            series={[
              { name: "Monthly IC", color: "var(--up)", pts: v.ic_series, bars: true },
              { name: "12-month average", color: "var(--acc)", pts: v.ic_series.map((p) => [p[0], v.rolling_ic.find((r) => r[0] === p[0])?.[1] ?? NaN] as [string, number]) },
            ]} />
          <div className="vx-foot">IC is the rank correlation between predictions and what actually happened next month. Above 0.03 consistently is considered useful in practice.</div>
        </section>
        <section className="vx-card vx-pad">
          <h3>What drives the model</h3>
          <div className="vx-imp">
            <div className="vx-imp-h"><span>Feature</span><span>Linear weight</span><span>Tree importance</span></div>
            {ml.importance.map((i) => (
              <div key={i.key} className="vx-imp-r">
                <span>{i.feature}</span>
                <span className="vx-imp-b"><i className={i.ridge >= 0 ? "pos" : "neg"} style={{ width: `${(Math.abs(i.ridge) / maxImp) * 100}%` }} /></span>
                <span className="vx-imp-b"><i className="tree" style={{ width: `${(Math.max(i.gbm, 0) / maxG) * 100}%` }} /></span>
              </div>
            ))}
          </div>
        </section>
        <section className="vx-card vx-pad">
          <h3>Calibration</h3>
          <table className="vx-mtab">
            <thead><tr><th>Quintile</th><th>Avg 1-month excess</th><th>Beat universe</th><th>Observations</th></tr></thead>
            <tbody>{v.calibration.map((c) => (
              <tr key={c.quintile}><td>Q{c.quintile}</td><td className={c.mean_excess >= 0 ? "up" : "dn"}>{pct(c.mean_excess, 2)}</td><td>{pct(c.hit_rate, 1, false)}</td><td>{c.n.toLocaleString()}</td></tr>
            ))}</tbody>
          </table>
          <div className="vx-foot">These are the numbers behind each stock&apos;s &quot;expected 1-month excess&quot; and &quot;beat the universe&quot; figures.</div>
          <h3 className="vx-mt">Model comparison</h3>
          <table className="vx-mtab">
            <thead><tr><th>Model</th><th>Mean IC</th><th>t-stat</th><th>Positive months</th></tr></thead>
            <tbody>{(["ridge", "gbm", "ens"] as const).map((m) => (
              <tr key={m}><td>{m === "ridge" ? "Ridge (linear)" : m === "gbm" ? "Gradient boosting" : "Ensemble (used)"}</td>
                <td>{v.ic[m].mean.toFixed(3)}</td><td>{v.ic[m].t.toFixed(2)}</td><td>{pct(v.ic[m].hit, 0, false)}</td></tr>
            ))}</tbody>
          </table>
        </section>
        <section className="vx-card vx-pad">
          <h3>How it was tested</h3>
          <ul className="vx-list2">
            <li>{ml.features.length} price and volume features, ranked across stocks each month.</li>
            <li>{ml.models.join("; ")}; {ml.ensemble}.</li>
            <li>Walk-forward: starts after {ml.min_train_months} months of history, retrains every {ml.retrain_every_months} months, never sees the month it predicts.</li>
            <li>A one-month purge gap stops training labels overlapping the test month.</li>
            <li>Costs: {ml.cost_bps} bps each way on every change in the top quintile (average turnover {pct(v.avg_turnover, 0, false)} a month).</li>
          </ul>
          <h3 className="vx-mt">Before using real money</h3>
          <ul className="vx-list2 warn">
            {ml.caveats.map((c) => <li key={c}>{c}</li>)}
            <li>Paper-trade the signals for 3 to 6 months and compare live results with this page before committing money.</li>
          </ul>
        </section>
      </div>
    </main>
  );
}

/* ============================================================ top picks */
type NewsItem = { title: string; publisher: string; link: string; time: number; topics: string[]; tone: "positive" | "negative" | "neutral"; related?: boolean };

function ago(t: number) {
  const s = Date.now() / 1000 - t;
  if (s < 3600) return `${Math.max(1, Math.round(s / 60))}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

function earningsChip(d?: string | null) {
  if (!d) return null;
  const days = Math.round((new Date(d + "T12:00:00").getTime() - Date.now()) / 86400000);
  if (days >= 0 && days <= 60) return <span className={cls("vx-chip", days <= 7 && "hot")}>Earnings {days === 0 ? "today" : `in ${days} days`}</span>;
  if (days < 0 && days >= -21) return <span className="vx-chip">Reported {-days} days ago</span>;
  return null;
}

/** Plain-language reasons with the percentile that earned them (0..1). */
function reasons(s: Stock): [number, string][] {
  const out: [number, string][] = [];
  const f = s.factors || {};
  const peer = s.sector.replace("Information Technology", "tech").replace("Communication Services", "communications").toLowerCase();
  const add = (k: string, txt: (val: number, pc: number) => string) => {
    const pc = f[k]?.pct ?? null, val = f[k]?.value ?? null;
    if (pc != null && val != null && pc >= 0.75) out.push([pc, txt(val, Math.min(99, Math.round(pc * 100)))]);
  };
  if (s.ml_quintile === 5) out.push([0.995, `Top 20% of the machine-learning ranking${s.hit_prob ? `; that group beat the universe in ${Math.round(s.hit_prob * 100)}% of past months` : ""}`]);
  add("revenue_growth", (x, pc) => `Revenue up ${(x * 100).toFixed(0)}% year over year, faster than ${pc}% of ${peer} peers`);
  add("forward_pe", (x, pc) => `Cheaper than ${pc}% of ${peer} peers on forward earnings (${x.toFixed(1)}× P/E)`);
  add("peg", (x, pc) => `PEG of ${x.toFixed(2)}: inexpensive for its growth compared with ${pc}% of peers`);
  add("profit_margin", (x, pc) => `${(x * 100).toFixed(0)}% profit margin, better than ${pc}% of peers`);
  add("roe", (x, pc) => `Return on equity of ${(x * 100).toFixed(0)}%, higher than ${pc}% of peers`);
  add("debt_to_equity", (x, pc) => `Light balance sheet: debt is ${x.toFixed(2)}× equity, lower than ${pc}% of peers`);
  add("momentum_6m", (x, pc) => `Up ${(x * 100).toFixed(0)}% over six months, stronger than ${pc}% of peers`);
  if ((s.alpha ?? 0) > 0.1) out.push([Math.min(0.95, 0.75 + (s.alpha ?? 0) / 2), `Beat the S&P 500 by ${((s.alpha ?? 0) * 100).toFixed(0)}% over the past year after adjusting for risk`]);
  return out.sort((a, b) => b[0] - a[0]).slice(0, 5);
}

function PickChart({ symbol, s, live }: { symbol: string; s: Stock; live: number }) {
  const [ref, { w }] = useSize<HTMLDivElement>();
  const [c, setC] = useState<Candle[]>([]);
  useEffect(() => { fetch(`/api/vertex/candles?symbol=${encodeURIComponent(symbol)}`).then((r) => r.json()).then((j) => setC(j.candles || [])).catch(() => setC([])); }, [symbol]);
  const h = 230, L = 4, R = 96, T = 10, B = 20;
  const pts = c.slice(-126).map((x) => [x[0], x[4]] as [string, number]);
  if (pts.length) pts[pts.length - 1] = [pts[pts.length - 1][0], live];
  if (pts.length < 2 || w < 60) return <div ref={ref} className="vx-pchart" style={{ height: h }} />;
  const vals = pts.map((p) => p[1]).concat([s.stop, s.target]);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.06;
  const cw = w - L - R, X = (i: number) => L + (i / (pts.length - 1)) * cw, Y = (v: number) => T + ((hi + pad - v) / (hi - lo + 2 * pad)) * (h - T - B);
  const d = pts.map((p, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
  const up = live >= pts[0][1];
  const col = up ? "var(--up)" : "var(--dn)";
  const lines: [string, number, string][] = [["Target", s.target, "var(--up)"], ["Entry", s.entry, "var(--acc)"], ["Stop", s.stop, "var(--dn)"]];
  return (
    <div ref={ref} className="vx-pchart" style={{ height: h }}>
      <svg width={w} height={h}>
        <defs><linearGradient id="vx-pfill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor={up ? "#45c08a" : "#e5534b"} stopOpacity="0.25" /><stop offset="1" stopColor={up ? "#45c08a" : "#e5534b"} stopOpacity="0" /></linearGradient></defs>
        <rect x={L} y={Y(s.target)} width={cw} height={Math.max(0, Y(s.entry) - Y(s.target))} fill="var(--up)" opacity={0.05} />
        <rect x={L} y={Y(s.entry)} width={cw} height={Math.max(0, Y(s.stop) - Y(s.entry))} fill="var(--dn)" opacity={0.06} />
        <path d={`${d}L${X(pts.length - 1)},${h - B}L${L},${h - B}Z`} fill="url(#vx-pfill)" />
        <path d={d} fill="none" stroke={col} strokeWidth={1.8} className="vx-draw" pathLength={1} />
        {lines.map(([k, v, cl]) => (
          <g key={k}>
            <line x1={L} x2={L + cw} y1={Y(v)} y2={Y(v)} stroke={cl} strokeDasharray="4 4" opacity={0.8} />
            <text x={L + cw + 6} y={Y(v) + 4} fill={cl} className="vx-plv">{k} {fx(v)}</text>
          </g>
        ))}
        <circle cx={X(pts.length - 1)} cy={Y(live)} r={3.5} fill={col} />
        {[0, Math.floor((pts.length - 1) / 2), pts.length - 1].map((i) => (
          <text key={i} x={X(i)} y={h - 4} textAnchor={i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle"} className="vx-axis">
            {new Date(pts[i][0] + "T00:00:00").toLocaleDateString("en-US", { month: "short", day: "numeric" })}
          </text>
        ))}
      </svg>
    </div>
  );
}

function TopPicks({ report, px, chg, onOpen }: { report: Report | null; px: (s: Stock) => number; chg: (s: Stock) => number | null; onOpen: (s: string) => void }) {
  const [mkt, setMkt] = useState<"All" | "S&P 500" | "TSX 60">("All");
  const [news, setNews] = useState<Record<string, NewsItem[]>>({});
  const [selSym, setSelSym] = useState<string | null>(null);
  const ranked = useMemo(() => [...(report?.stocks || [])].filter((s) => s.score != null).sort((a, b) => (b.score ?? 0) - (a.score ?? 0)), [report]);
  const picks = useMemo(() => {
    const out: Stock[] = [], perSec: Record<string, number> = {};
    for (const s of ranked) {
      if (!isBuy(s.signal) || (mkt !== "All" && s.index !== mkt) || (perSec[s.sector] || 0) >= 2) continue;
      out.push(s); perSec[s.sector] = (perSec[s.sector] || 0) + 1;
      if (out.length >= 10) break;
    }
    return out;
  }, [ranked, mkt]);
  const key = picks.map((p) => p.symbol).join(",");
  useEffect(() => {
    if (!key) return;
    let dead = false;
    const names = picks.map((p) => p.name.replace(/\|/g, " ")).join("|");
    const load = () => fetch(`/api/vertex/news?symbols=${encodeURIComponent(key)}&names=${encodeURIComponent(names)}`).then((r) => r.json())
      .then((j) => { if (!dead) setNews((n) => ({ ...n, ...j.news })); }).catch(() => {});
    load();
    const id = setInterval(load, 10 * 60_000);
    return () => { dead = true; clearInterval(id); };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!report) return <main className="vx-picks"><h1 className="vx-h1">Today&apos;s top picks</h1><section className="vx-card vx-pad">Run a scan first.</section></main>;
  const s = picks.find((p) => p.symbol === selSym) || picks[0];
  const nItems = s ? news[s.symbol.toUpperCase()] : undefined;
  const rankOf = (x: Stock) => ranked.indexOf(x) + 1;
  const sectors = new Set(picks.map((p) => p.sector)).size;
  const strong = picks.filter((p) => p.signal === "STRONG BUY").length;
  return (
    <main className="vx-picks">
      <div className="vx-mhead">
        <div>
          <h1 className="vx-h1">Today&apos;s top picks</h1>
          <p className="vx-sub2">
            {picks.length} highest-rated stocks from the {report.price_date} scan · {strong} strong buys · {sectors} sectors · market regime{" "}
            <b className={report.regime?.state === "RISK-ON" ? "up" : report.regime?.state === "RISK-OFF" ? "dn" : "vx-acc"}>{(report.regime?.state || "n/a").replace("-", " ").toLowerCase()}</b>
          </p>
        </div>
        <div className="vx-mini">
          {(["All", "S&P 500", "TSX 60"] as const).map((m) => <button key={m} className={cls(mkt === m && "on")} onClick={() => { setMkt(m); setSelSym(null); }}>{m}</button>)}
        </div>
      </div>

      <div className="vx-pk">
        <section className="vx-card vx-pk-list">
          <div className="vx-pk-lh"><span>#</span><span>Stock</span><span>Price</span><span>6 months</span><span>Score</span></div>
          {picks.map((p, i) => {
            const c = chg(p), price = px(p);
            const sp = [...(p.spark || []), price];
            return (
              <button key={p.symbol} className={cls("vx-pk-row", p.symbol === s?.symbol && "on")} onClick={() => setSelSym(p.symbol)}>
                <span className="vx-pk-n">{i + 1}</span>
                <span className="vx-pk-id"><b>{tick(p.symbol)}</b><small>{p.name}</small><em>{short(p.sector)} · <span className={cls("vx-sig", sigCls(p.signal))}>{SIG_SHORT[p.signal]}</span></em></span>
                <span className="vx-pk-px"><Tween v={price} fmt={(x) => fx(x)} /><small className={(c ?? 0) >= 0 ? "up" : "dn"}>{pct(c)}</small></span>
                <span className="vx-pk-sp"><Spark data={sp} w={96} h={30} fill={false} color={price >= (sp[0] ?? 0) ? "#45c08a" : "#e5534b"} /></span>
                <span className="vx-pk-sc"><b>{fx(p.score, 1)}</b><i style={{ width: `${p.score ?? 0}%` }} /></span>
              </button>
            );
          })}
          {!picks.length && <div className="vx-empty sm">No Buy-rated stocks in this market.</div>}
        </section>

        {s && (
          <section className="vx-card vx-pk-det" key={s.symbol}>
            <div className="vx-pd-h">
              <div>
                <div className="vx-pd-t"><b>{tick(s.symbol)}</b><span className={cls("vx-sig", sigCls(s.signal))}>{SIG_SHORT[s.signal]}</span>{earningsChip(s.earnings_date)}</div>
                <div className="vx-muted">{s.name} · {s.sector} · {s.index}</div>
              </div>
              <div className="vx-pd-px">
                <b><Tween v={px(s)} fmt={(x) => fx(x)} /></b><span className="vx-muted"> {s.currency}</span>
                <div className={(chg(s) ?? 0) >= 0 ? "up" : "dn"}>{pct(chg(s))} today</div>
              </div>
            </div>

            <div className="vx-pd-stats">
              <div><span>Score</span><b className="vx-acc">{fx(s.score, 1)}</b><small>rank {rankOf(s)} of {ranked.length}</small></div>
              <div><span>Machine learning</span><b>{fx(s.ml_score, 0)}</b><small>{s.ml_quintile ? `quintile ${s.ml_quintile} of 5` : "—"}</small></div>
              <div><span>Fundamentals</span><b>{fx(s.fund_score, 0)}</b><small>{s.factor_count} of 9 factors</small></div>
              <div><span>Expected 1-month excess</span><b className={(s.exp_excess_1m ?? 0) >= 0 ? "up" : "dn"}>{pct(s.exp_excess_1m, 2)}</b><small>historical average</small></div>
            </div>

            <PickChart symbol={s.symbol} s={s} live={px(s)} />

            <div className="vx-pd-cols">
              <div>
                <h4>Why the model likes it</h4>
                <ul className="vx-why2">
                  {reasons(s).map(([pc, t]) => (
                    <li key={t}><span>{t}</span><i><em style={{ width: `${pc * 100}%` }} /></i></li>
                  ))}
                </ul>
              </div>
              <div>
                <h4>Latest news <span className="vx-muted sm">Yahoo Finance</span></h4>
                {!nItems ? <div className="vx-news-empty">Loading headlines…</div> : !nItems.length ? <div className="vx-news-empty">No recent headlines.</div> : (
                  <ul className="vx-news">
                    {nItems.map((n) => (
                      <li key={n.link} className={cls(n.related && "rel")}>
                        <a href={n.link} target="_blank" rel="noopener noreferrer"><span className={cls("vx-tone", n.tone)} aria-label={`${n.tone} headline`} /><span className="vx-news-t">{n.title}</span></a>
                        <div className="vx-news-m">
                          {n.related && <span className="vx-rel">Industry</span>}
                          {n.topics.map((t) => <span key={t} className="vx-topic">{t}</span>)}<span>{n.publisher}</span><span>{ago(n.time)}</span>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
            <div className="vx-pd-f">
              <span className="vx-muted sm">Levels: entry {fx(s.entry)} · stop {fx(s.stop)} · target {fx(s.target)} (volatility-based, from the scan)</span>
              <button className="vx-p2-open" onClick={() => onOpen(s.symbol)}>Open in Strategy Terminal</button>
            </div>
          </section>
        )}
      </div>
      <div className="vx-disc">Ratings come from the VertexCapital model and are research, not advice. Headlines are from Yahoo Finance; topic and tone tags are keyword-based. &quot;Industry&quot; marks stories that don&apos;t name the company.</div>
    </main>
  );
}

/* ============================================================ portfolio builder */
type Answers = { horizon: number; reaction: number; goal: number; tolerance: number; market: number; amount: number; count: number; fractional: boolean };
const DEFAULT_ANSWERS: Answers = { horizon: 2, reaction: 1, goal: 1, tolerance: 1, market: 2, amount: 10000, count: 12, fractional: true };
const QUESTIONS: { key: keyof Answers; q: string; opts: string[] }[] = [
  { key: "horizon", q: "How long will this money stay invested?", opts: ["Under 1 year", "1 to 3 years", "3 to 10 years", "10+ years"] },
  { key: "reaction", q: "If your portfolio dropped 20% in a month, you would…", opts: ["Sell to stop the loss", "Wait it out", "Buy more"] },
  { key: "goal", q: "What matters most?", opts: ["Steady income", "Balanced growth", "Maximum growth"] },
  { key: "tolerance", q: "How much day-to-day swing can you live with?", opts: ["Low", "Medium", "High", "Very high"] },
  { key: "market", q: "Which markets?", opts: ["US only", "Canada only", "Both"] },
];
const DEFENSIVE = new Set(["Utilities", "Consumer Staples", "Health Care", "Real Estate"]);
const GROWTHY = new Set(["Information Technology", "Communication Services", "Consumer Discretionary"]);
const INCOME = new Set(["Utilities", "Real Estate", "Energy", "Financials", "Consumer Staples"]);
const PROFILES = [
  { name: "Conservative", equity: 0.5, maxBeta: 1.0, maxVol: 0.3, defensive: 1.5, growth: 0.75 },
  { name: "Balanced", equity: 0.7, maxBeta: 1.2, maxVol: 0.4, defensive: 1.15, growth: 1.0 },
  { name: "Growth", equity: 0.85, maxBeta: 1.5, maxVol: 0.55, defensive: 0.9, growth: 1.2 },
  { name: "Aggressive", equity: 0.95, maxBeta: 9, maxVol: 9, defensive: 0.7, growth: 1.4 },
];

function useStored<T>(key: string, init: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(init);
  useEffect(() => { try { const r = localStorage.getItem(key); if (r) setV({ ...init, ...JSON.parse(r) }); } catch {} }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const set = (n: T) => { setV(n); try { localStorage.setItem(key, JSON.stringify(n)); } catch {} };
  return [v, set];
}

function PortfolioBuilder({ report, px, onOpen }: { report: Report | null; px: (s: Stock) => number; onOpen: (s: string) => void }) {
  const [a, setA] = useStored<Answers>("vx-builder", DEFAULT_ANSWERS);
  const [picks, setPicks] = useStored<Record<string, boolean>>("vx-builder-picks", {});

  const profile = useMemo(() => {
    const score = Math.max(1, Math.min(10, [2, 4.5, 6.5, 8.5][a.tolerance] + [-1.5, -0.5, 0.5, 1][a.horizon] + [-1.5, 0, 1][a.reaction]));
    const pi = score <= 3.5 ? 0 : score <= 5.5 ? 1 : score <= 7.5 ? 2 : 3;
    const p = PROFILES[pi];
    const equity = Math.min(p.equity, [0.35, 0.65, 1, 1][a.horizon]);
    const short = a.horizon <= 1;
    const w = { ml: short ? 0.65 : 0.35, fund: short ? 0.35 : 0.65, div: a.goal === 0 ? 0.3 : 0, growth: a.goal === 2 ? 0.2 : 0 };
    return { ...p, score, equity, w, short };
  }, [a]);

  const plan = useMemo(() => {
    if (!report) return null;
    const mk = (s: Stock) => a.market === 2 || (a.market === 0 ? s.index !== "TSX 60" : s.index === "TSX 60");
    const pool = report.stocks.filter((s) => mk(s) && s.score != null);
    const divRank = new Map([...pool].sort((x, y) => (x.dividend_yield ?? 0) - (y.dividend_yield ?? 0)).map((s, i) => [s.symbol, (i / Math.max(1, pool.length - 1)) * 100]));
    const pscore = (s: Stock) => {
      const { ml, fund, div, growth } = profile.w;
      const parts: [number | null, number][] = [[s.ml_score, ml], [s.fund_score, fund], [divRank.get(s.symbol) ?? null, div], [s.growth_score, growth]];
      const ok = parts.filter(([v, wt]) => v != null && wt > 0);
      const tw = ok.reduce((t, [, wt]) => t + wt, 0) || 1;
      return ok.reduce((t, [v, wt]) => t + (v as number) * wt, 0) / tw;
    };
    const eligible = pool.filter((s) => !isSell(s.signal) && (s.beta_realized ?? 1) <= profile.maxBeta && (s.vol_1y ?? 0.3) <= profile.maxVol)
      .map((s) => ({ s, ps: pscore(s) }));
    const capBySec = new Map<string, number>();
    pool.forEach((s) => capBySec.set(s.sector, (capBySec.get(s.sector) || 0) + (s.market_cap || 0)));
    const totalCap = [...capBySec.values()].reduce((x, y) => x + y, 0) || 1;
    const bySec = new Map<string, { s: Stock; ps: number }[]>();
    eligible.forEach((e) => { const g = bySec.get(e.s.sector) || []; g.push(e); bySec.set(e.s.sector, g); });
    bySec.forEach((g) => g.sort((x, y) => y.ps - x.ps));
    let alloc = [...bySec.entries()].map(([sec, g]) => {
      const bench = (capBySec.get(sec) || 0) / totalCap;
      const top = g.slice(0, 3).reduce((t, e) => t + e.ps, 0) / Math.min(3, g.length);
      let m = Math.exp(0.6 * (top - 50) / 50);
      if (DEFENSIVE.has(sec)) m *= profile.defensive;
      if (GROWTHY.has(sec)) m *= profile.growth;
      if (a.goal === 0 && INCOME.has(sec)) m *= 1.3;
      return { sector: sec, bench, w: Math.max(bench, 0.01) * m, top };
    });
    for (let it = 0; it < 20; it++) {
      const t = alloc.reduce((x, y) => x + y.w, 0);
      alloc = alloc.map((x) => ({ ...x, w: Math.min(0.3, x.w / t) }));
    }
    const t = alloc.reduce((x, y) => x + y.w, 0);
    alloc = alloc.map((x) => ({ ...x, w: x.w / t })).sort((x, y) => y.w - x.w);
    let nLeft = a.count;
    const slots = alloc.map((x) => { const n = x.w >= 0.03 ? Math.max(1, Math.round(a.count * x.w)) : 0; return { ...x, n }; });
    let diff = slots.reduce((x, y) => x + y.n, 0) - nLeft;
    for (let i = slots.length - 1; diff > 0 && i >= 0; i--) if (slots[i].n > 1) { slots[i].n--; diff--; }
    for (let i = 0; diff < 0 && i < slots.length; i++) { slots[i].n++; diff++; }
    nLeft = 0;
    return slots.map((x) => ({ ...x, cands: (bySec.get(x.sector) || []).slice(0, Math.max(6, x.n + 2)) }));
  }, [report, a, profile]);

  const chosen = (sec: { n: number; cands: { s: Stock }[] }, s: Stock, i: number) => picks[s.symbol] ?? i < sec.n;

  const basket = useMemo(() => {
    if (!plan) return null;
    const equity$ = a.amount * profile.equity;
    const active = plan.map((sec) => ({ ...sec, sel: sec.cands.filter((c, i) => chosen(sec, c.s, i)).map((c) => c.s) })).filter((x) => x.sel.length);
    const tw = active.reduce((t, x) => t + x.w, 0) || 1;
    const lines: { s: Stock; sector: string; target: number; shares: number; cost: number; price: number }[] = [];
    active.forEach((sec) => {
      const sec$ = equity$ * sec.w / tw;
      const iv = sec.sel.map((s) => 1 / Math.max(0.1, s.vol_1y ?? 0.3));
      const ivt = iv.reduce((x, y) => x + y, 0);
      sec.sel.forEach((s, i) => {
        const target = sec$ * iv[i] / ivt, price = px(s);
        const shares = a.fractional ? Math.floor((target / price) * 1000) / 1000 : Math.floor(target / price);
        lines.push({ s, sector: sec.sector, target, shares, cost: shares * price, price });
      });
    });
    const inv = lines.reduce((t, l) => t + l.cost, 0) || 1;
    const wts = lines.map((l) => l.cost / inv);
    const beta = lines.reduce((t, l, i) => t + wts[i] * (l.s.beta_realized ?? 1), 0);
    let varr = 0;
    lines.forEach((li, i) => lines.forEach((lj, j) => {
      const vi = li.s.vol_1y ?? 0.3, vj = lj.s.vol_1y ?? 0.3;
      varr += wts[i] * wts[j] * vi * vj * (i === j ? 1 : 0.35);
    }));
    const yieldW = lines.reduce((t, l, i) => t + wts[i] * (l.s.dividend_yield ?? 0), 0);
    const exp = lines.reduce((t, l, i) => t + wts[i] * (l.s.exp_excess_1m ?? 0), 0);
    const invested = lines.reduce((t, l) => t + l.cost, 0);
    return { lines, invested, reserve: a.amount - invested, beta, vol: Math.sqrt(varr), yieldW, exp, equity$ };
  }, [plan, picks, a, profile, px]); // eslint-disable-line react-hooks/exhaustive-deps

  const exportCsv = () => {
    if (!basket) return;
    const rows = basket.lines.map((l) => [l.s.symbol, `"${l.s.name}"`, l.sector, l.s.signal, l.shares, l.price.toFixed(2), l.cost.toFixed(2)].join(","));
    const url = URL.createObjectURL(new Blob(["symbol,name,sector,signal,shares,price,cost\n" + rows.join("\n")], { type: "text/csv" }));
    const el = document.createElement("a"); el.href = url; el.download = "vertex_portfolio.csv"; el.click(); URL.revokeObjectURL(url);
  };

  if (!report) return <main className="vx-build"><h1 className="vx-h1">Portfolio</h1><section className="vx-card vx-pad">Run a scan first.</section></main>;
  return (
    <main className="vx-build">
      <h1 className="vx-h1">Build your portfolio</h1>
      <div className="vx-b-top">
        <section className="vx-card vx-pad vx-quiz">
          {QUESTIONS.map((q) => (
            <div key={q.key} className="vx-q">
              <label>{q.q}</label>
              <div className="vx-opts2">
                {q.opts.map((o, i) => <button key={o} className={cls(a[q.key] === i && "on")} onClick={() => { setA({ ...a, [q.key]: i }); setPicks({}); }}>{o}</button>)}
              </div>
            </div>
          ))}
          <div className="vx-q vx-q2">
            <div><label>Amount to invest</label>
              <div className="vx-input"><span className="vx-muted">$</span><input type="number" min={1000} step={1000} value={a.amount} onChange={(e) => setA({ ...a, amount: Math.max(0, Number(e.target.value)) })} /></div></div>
            <div><label>Number of stocks</label>
              <div className="vx-opts2">{[8, 12, 20].map((n) => <button key={n} className={cls(a.count === n && "on")} onClick={() => { setA({ ...a, count: n }); setPicks({}); }}>{n}</button>)}</div></div>
            <div><label>Share type</label>
              <div className="vx-opts2">
                <button className={cls(a.fractional && "on")} onClick={() => setA({ ...a, fractional: true })}>Fractional</button>
                <button className={cls(!a.fractional && "on")} onClick={() => setA({ ...a, fractional: false })}>Whole shares</button>
              </div></div>
          </div>
        </section>

        <section className="vx-card vx-pad vx-prof">
          <span className="vx-muted sm">Your profile</span>
          <div className="vx-prof-n">{profile.name}</div>
          <div className="vx-meter"><i style={{ width: `${profile.score * 10}%` }} /></div>
          <div className="vx-muted sm">Risk score {profile.score.toFixed(1)} of 10</div>
          <div className="vx-split">
            <div style={{ width: `${profile.equity * 100}%` }} className="eq"><b>{pct(profile.equity, 0, false)}</b> stocks</div>
            {profile.equity < 1 && <div style={{ width: `${(1 - profile.equity) * 100}%` }} className="rs"><b>{pct(1 - profile.equity, 0, false)}</b> reserve</div>}
          </div>
          <p className="vx-prof-t">
            {profile.equity < 1 ? `Keep ${pct(1 - profile.equity, 0, false)} in cash, a high-interest savings ETF or a bond ETF. ` : ""}
            Picks are ranked {profile.short ? "mostly on the machine-learning model, which looks one month ahead" : "mostly on fundamentals (valuation, profitability, balance sheet), which matter more over years"}
            {a.goal === 0 ? ", with extra weight on dividends" : a.goal === 2 ? ", with extra weight on revenue growth" : ""}.
            {profile.maxBeta < 9 ? ` Stocks with beta above ${profile.maxBeta} or volatility above ${pct(profile.maxVol, 0, false)} are left out.` : ""}
          </p>
          <h3 className="vx-mt">Sector allocation</h3>
          <div className="vx-salloc">
            {plan?.filter((x) => x.w > 0.004).map((x) => (
              <div key={x.sector} className="vx-sa">
                <span>{x.sector}</span>
                <span className="vx-sa-b"><i style={{ width: `${x.w * 100 / 0.3}%`, background: sc(x.sector) }} /><em style={{ left: `${Math.min(100, x.bench * 100 / 0.3)}%` }} title="Index weight" /></span>
                <b>{pct(x.w, 0, false)}</b>
                <small className={Math.abs(x.w - x.bench) < 0.005 ? "vx-muted" : x.w > x.bench ? "up" : "dn"}>{Math.abs(x.w - x.bench) < 0.005 ? "0" : `${x.w > x.bench ? "+" : "−"}${Math.abs((x.w - x.bench) * 100).toFixed(0)}`}</small>
              </div>
            ))}
          </div>
          <div className="vx-foot">Bar = recommended weight; tick = the sector&apos;s weight in the index. Numbers on the right are the over/underweight in percentage points.</div>
        </section>

        <section className="vx-card vx-pad vx-sum">
          <span className="vx-muted sm">Your portfolio</span>
          <div className="vx-sum-v"><Tween v={basket?.invested ?? 0} fmt={(x) => money(x)} /></div>
          <div className="vx-muted sm">in {basket?.lines.length || 0} stocks · {money(Math.max(0, basket?.reserve ?? 0))} reserve</div>
          <div className="vx-sum-g">
            <span>Beta vs market</span><b>{fx(basket?.beta, 2)}</b>
            <span>Estimated volatility</span><b>{pct(basket?.vol, 0, false)}</b>
            <span>Dividend yield</span><b>{pct(basket?.yieldW, 2, false)}</b>
            <span>Model expected 1M excess</span><b className={(basket?.exp ?? 0) >= 0 ? "up" : "dn"}>{pct(basket?.exp, 2)}</b>
          </div>
          <div className="vx-foot">Volatility assumes an average 0.35 correlation between holdings. Expected excess comes from the model&apos;s out-of-sample calibration and is uncertain.</div>
          <div className="vx-actions">
            <button onClick={exportCsv}>Export CSV</button>
            <button onClick={() => setPicks({})}>Reset picks</button>
          </div>
        </section>
      </div>

      <h3 className="vx-h3">Choose your stocks <span className="vx-muted sm">Pre-selected picks fit your profile best; tick or untick to change. Money is split across your picks by inverse volatility.</span></h3>
      <div className="vx-secs">
        {plan?.filter((x) => x.n > 0 || x.w > 0.03).map((sec) => (
          <section key={sec.sector} className="vx-card vx-sec">
            <div className="vx-sec-h"><span className="vx-sq" style={{ background: sc(sec.sector) }} /><b>{sec.sector}</b><span>{pct(sec.w, 0, false)} · {money(a.amount * profile.equity * sec.w)}</span></div>
            {sec.cands.map((c, i) => {
              const on = chosen(sec, c.s, i);
              const line = basket?.lines.find((l) => l.s.symbol === c.s.symbol);
              return (
                <div key={c.s.symbol} className={cls("vx-pick", on && "on")}>
                  <button className="vx-chk" onClick={() => setPicks({ ...picks, [c.s.symbol]: !on })} aria-pressed={on} aria-label={`Include ${c.s.symbol}`}>{on ? "✓" : ""}</button>
                  <div className="vx-pick-n" onClick={() => onOpen(c.s.symbol)}>
                    <b>{tick(c.s.symbol)}</b> <span className="vx-muted">{c.s.name}</span>
                    <div className="vx-pick-m">
                      <span className={cls("vx-sig", sigCls(c.s.signal))}>{SIG_SHORT[c.s.signal]}</span>
                      <span>fit {c.ps.toFixed(0)}</span><span>β {fx(c.s.beta_realized, 2)}</span>
                      {c.s.dividend_yield ? <span>yield {pct(c.s.dividend_yield, 1, false)}</span> : null}
                    </div>
                  </div>
                  <div className="vx-pick-r">
                    <b><Tween v={px(c.s)} fmt={(x) => fx(x)} /></b>
                    {on && line ? <span>{a.fractional ? line.shares.toFixed(3) : line.shares} sh · {money(line.cost)}</span> : <span className="vx-muted">—</span>}
                  </div>
                </div>
              );
            })}
          </section>
        ))}
      </div>
      <div className="vx-disc">Educational tool, not financial advice. Check prices with your broker before placing orders; quotes are delayed.</div>
    </main>
  );
}

/* ============================================================ page */
export default function VertexPage() {
  const [report, setReport] = useState<Report | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [savedLog, setSavedLog] = useState<string[]>([]);
  const [view, setView] = useState<"terminal" | "dashboard" | "market" | "model">("terminal");
  const [universe, setUniverse] = useState("All");
  const [sel, setSel] = useState<string | null>(null);
  const [candles, setCandles] = useState<Candle[]>([]);
  const [range, setRange] = useState(120);
  const [showLevels, setShowLevels] = useState(true);
  const [showSma, setShowSma] = useState(true);
  const [rightTab, setRightTab] = useState<"rank" | "factors">("rank");
  const [rankMode, setRankMode] = useState<"score" | "value_score" | "growth_score" | "quality_score">("score");
  const [query, setQuery] = useState("");
  const [bottomTab, setBottomTab] = useState<"levels" | "portfolio" | "console">("levels");
  const [sigFilter, setSigFilter] = useState("ALL");
  const [sortKey, setSortKey] = useState<string>("score");
  const [account, setAccount] = useState(100000);
  const [risk, setRisk] = useState(1);
  const [live, setLive] = useState<string[]>([]);
  const [scanning, setScanning] = useState(false);
  const [replayN, setReplayN] = useState(0);
  const [perfMode, setPerfMode] = useState<"portfolio" | "benchmark">("portfolio");
  const [topMode, setTopMode] = useState<"score" | "momentum">("score");
  const [showMethod, setShowMethod] = useState(false);
  const [clock, setClock] = useState("");
  const consoleRef = useRef<HTMLDivElement>(null);
  const [theme, setTheme] = useState<"carbon" | "classic">("carbon");
  useEffect(() => {
    try { const t = localStorage.getItem("vx-theme"); if (t === "classic" || t === "carbon") setTheme(t); } catch {}
  }, []);
  const switchTheme = (t: "carbon" | "classic") => {
    setTheme(t);
    try { localStorage.setItem("vx-theme", t); } catch {}
  };
  const [feed, setFeed] = useState<QuoteFeed | null>(null);
  const [nextIn, setNextIn] = useState(REFRESH_S);
  const [liveOn, setLiveOn] = useState(true);

  const load = useCallback(async () => {
    try {
      const r = await fetch("/api/vertex/data", { cache: "no-store" }).then((x) => x.json());
      setSavedLog((r.log || "").split(/\r?\n/).filter(Boolean));
      if (r.ok) {
        setReport(r.report);
        setSel((cur) => cur || r.report.portfolio.holdings[0]?.symbol || r.report.stocks[0]?.symbol || null);
      }
    } finally {
      setLoaded(true);
    }
  }, []);
  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    const t = () => setClock(new Date().toLocaleTimeString("en-US", { hour12: false }));
    t(); const id = setInterval(t, 1000); return () => clearInterval(id);
  }, []);

  useEffect(() => {
    if (!sel) return;
    fetch(`/api/vertex/candles?symbol=${encodeURIComponent(sel)}`).then((r) => r.json()).then((r) => setCandles(r.candles || []));
  }, [sel, report?.generated_utc]);

  // Replay of the last real scan log, so the console is "alive" on load (labelled as replay).
  useEffect(() => {
    if (scanning || !savedLog.length) return;
    setReplayN(0);
    const id = setInterval(() => setReplayN((n) => { if (n >= savedLog.length) { clearInterval(id); return n; } return n + 3; }), 40);
    return () => clearInterval(id);
  }, [savedLog, scanning]);

  const consoleLines = scanning || live.length ? live : savedLog.slice(0, replayN);
  useEffect(() => {
    const el = consoleRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [consoleLines.length, bottomTab]);

  const runScan = async () => {
    if (scanning) return;
    setScanning(true); setLive([]); setBottomTab("console"); setView("terminal");
    try {
      const res = await fetch("/api/vertex/scan", { method: "POST" });
      if (!res.body) throw new Error("no stream");
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split(/\r?\n/);
        buf = parts.pop() || "";
        if (parts.length) setLive((l) => [...l, ...parts].slice(-2500));
      }
      if (buf) setLive((l) => [...l, buf]);
    } catch (e) {
      setLive((l) => [...l, `[error] ${(e as Error).message}`]);
    } finally {
      setScanning(false);
      await load();
    }
  };

  const stocks = useMemo(() => (report?.stocks || []).filter((s) => universe === "All" || s.index === universe), [report, universe]);
  const scored = useMemo(() => stocks.filter((s) => s.score != null), [stocks]);
  const stock = useMemo(() => report?.stocks.find((s) => s.symbol === sel) || null, [report, sel]);
  const q = useCallback((sym: string) => (liveOn ? feed?.quotes[sym] : undefined), [feed, liveOn]);
  const px = useCallback((s: { symbol: string; price: number }) => q(s.symbol)?.price ?? s.price, [q]);
  const chg = useCallback((s: { symbol: string; change_1d?: number | null }) => q(s.symbol)?.change ?? s.change_1d ?? null, [q]);

  // today's candle from the live quote (appended or updated in place)
  const liveCandles = useMemo(() => {
    const lq = sel ? q(sel) : undefined;
    if (!lq || !candles.length) return candles;
    const d = new Date(lq.time * 1000);
    const day = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    const last = candles[candles.length - 1];
    const bar: Candle = [day, lq.open ?? last[4], Math.max(lq.high ?? lq.price, lq.price), Math.min(lq.low ?? lq.price, lq.price), lq.price, lq.volume ?? 0];
    if (day > last[0]) return [...candles, bar];
    if (day === last[0]) return [...candles.slice(0, -1), bar];
    return candles;
  }, [candles, sel, q]);
  const shown = liveCandles.slice(-range);

  const ranked = useMemo(() => {
    const q = query.trim().toUpperCase();
    return scored
      .filter((s) => !q || s.symbol.includes(q) || s.name.toUpperCase().includes(q))
      .sort((a, b) => ((b[rankMode] ?? -1) as number) - ((a[rankMode] ?? -1) as number))
      .slice(0, 80);
  }, [scored, rankMode, query]);

  const table = useMemo(() => {
    const rows = scored.filter((s) => sigFilter === "ALL" || (sigFilter === "BUY" ? isBuy(s.signal) : sigFilter === "SELL" ? isSell(s.signal) : s.signal === sigFilter));
    const k = sortKey as keyof Stock;
    return [...rows].sort((a, b) => {
      const av = a[k] as number | string | null, bv = b[k] as number | string | null;
      if (typeof av === "string" || typeof bv === "string") return String(av).localeCompare(String(bv));
      return ((bv ?? -1e9) as number) - ((av ?? -1e9) as number);
    }).slice(0, 250);
  }, [scored, sigFilter, sortKey]);

  const plan = useMemo(() => {
    if (!stock) return null;
    const perShare = stock.entry - stock.stop;
    const shares = perShare > 0 ? Math.floor((account * risk) / 100 / perShare) : 0;
    const cappedShares = Math.min(shares, Math.floor((account * 0.25) / stock.entry));
    return {
      shares: cappedShares, capped: cappedShares < shares,
      position: cappedShares * stock.entry, maxLoss: cappedShares * perShare,
      gain: cappedShares * (stock.target - stock.entry),
    };
  }, [stock, account, risk]);

  const exportCsv = () => {
    const head = "symbol,name,sector,index,signal,score,price,entry,stop,target,rr,alpha\n";
    const body = table.map((s) => [s.symbol, `"${s.name}"`, s.sector, s.index, s.signal, s.score, s.price, s.entry, s.stop, s.target, s.rr, s.alpha].join(",")).join("\n");
    const url = URL.createObjectURL(new Blob([head + body], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = `vertex_levels_${report?.price_date || "scan"}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  /* ----- portfolio aggregates */
  const port = report?.portfolio;
  const sectors = useMemo(() => {
    const m = new Map<string, { sector: string; weight: number; value: number; items: Holding[] }>();
    for (const h of port?.holdings || []) {
      const e = m.get(h.sector) || { sector: h.sector, weight: 0, value: 0, items: [] };
      e.weight += h.weight; e.value += h.value; e.items.push(h); m.set(h.sector, e);
    }
    return [...m.values()].sort((a, b) => b.weight - a.weight);
  }, [port]);

  /* ----- live quote polling (visible rows + holdings + top ranked + index tape) */
  const topRanked = useMemo(() => [...scored].sort((a, b) => topMode === "score"
    ? (b.score! - a.score!) : ((b.factors.momentum_6m?.value ?? -9) - (a.factors.momentum_6m?.value ?? -9))).slice(0, 8), [scored, topMode]);
  const watch = useMemo(() => {
    const set = new Set<string>();
    if (sel) set.add(sel);
    TAPE.forEach((t) => set.add(t));
    (port?.holdings || []).forEach((h) => set.add(h.symbol));
    topRanked.forEach((s) => set.add(s.symbol));
    (bottomTab === "levels" ? table : []).slice(0, 25).forEach((s) => set.add(s.symbol));
    ranked.slice(0, 12).forEach((s) => set.add(s.symbol));
    return [...set].slice(0, 60);
  }, [sel, port, topRanked, table, ranked, bottomTab]);
  const watchRef = useRef(watch); watchRef.current = watch;
  const selRef = useRef(sel); selRef.current = sel;

  const poll = useCallback(async () => {
    if (!watchRef.current.length) return;
    try {
      const url = `/api/vertex/quotes?symbols=${encodeURIComponent(watchRef.current.join(","))}${selRef.current ? `&intraday=${encodeURIComponent(selRef.current)}` : ""}`;
      const r: QuoteFeed = await fetch(url, { cache: "no-store" }).then((x) => x.json());
      setFeed((prev) => ({ ...r, quotes: { ...(prev?.quotes || {}), ...r.quotes },
        intraday: r.intraday || (prev?.intraday?.symbol === selRef.current ? prev.intraday : null) }));
    } catch (e) {
      setFeed((prev) => prev ? { ...prev, error: (e as Error).message } : prev);
    }
    setNextIn(REFRESH_S);
  }, []);
  useEffect(() => {
    if (!liveOn || !report) return;
    poll();
    const id = setInterval(() => setNextIn((n) => { if (n <= 1) { poll(); return REFRESH_S; } return n - 1; }), 1000);
    return () => clearInterval(id);
  }, [liveOn, report, poll]);
  useEffect(() => { if (liveOn && report) poll(); }, [sel]); // eslint-disable-line react-hooks/exhaustive-deps

  const marketState = feed?.quotes.SPY?.marketState || "";
  const isOpen = /REGULAR/.test(marketState);
  const liveValue = useMemo(() => {
    if (!port) return null;
    const inv = port.holdings.reduce((a, h) => a + h.shares * px(h), 0);
    const cost = port.holdings.reduce((a, h) => a + h.shares * h.price, 0);
    const cash = port.capital - cost;
    const today = port.holdings.reduce((a, h) => { const c = q(h.symbol)?.change; return a + (c != null ? h.shares * px(h) * c / (1 + c) : 0); }, 0);
    return { total: inv + cash, pnl: inv - cost, today, todayPct: today / (inv + cash - today) };
  }, [port, px, q]);

  const lastLine = (scanning ? live : savedLog).filter((l) => /^\[/.test(l)).slice(-1)[0] || "";

  /* ============================================================ render */
  return (
    <div className={cls("vx-root", theme === "carbon" && "th-carbon")}>
      <div className="vx-bg" />
      <header className="vx-nav">
        <div className="vx-brand">{I.logo}<span>Vertex<b>Capital</b></span></div>
        <nav className="vx-tabs">
          <button className={cls(view === "dashboard" && "on")} onClick={() => setView("dashboard")}>{I.dash}Portfolio</button>
          <button className={cls(view === "market" && "on")} onClick={() => setView("market")}>{I.grid}Market</button>
          <button className={cls(view === "terminal" && "on")} onClick={() => setView("terminal")}>{I.term}Strategy Terminal</button>
          <button className={cls(view === "model" && "on")} onClick={() => setView("model")}>{I.book}Model</button>
          <button onClick={() => setShowMethod(true)}>{I.book}Methodology</button>
          <button onClick={runScan} disabled={scanning}>{I.refresh}{scanning ? "Scanning…" : "Run Scan"}</button>
        </nav>
        <div className="vx-user">
          <button className={cls("vx-live", liveOn && feed?.received ? "on" : "")} onClick={() => setLiveOn(!liveOn)} title="Toggle live quotes">
            <span className={cls("vx-dot", scanning ? "busy" : liveOn && feed?.received ? "ok" : "off")} />
            {scanning ? "Engine running" : !liveOn ? "Live off" : feed?.received ? (isOpen ? "LIVE" : "Market closed") : "Connecting…"}
            {liveOn && <span className="vx-ring-t" style={{ ["--k" as string]: `${(nextIn / REFRESH_S) * 100}%` }} />}
          </button>
          <span className="vx-clock">{clock}</span>
          <span className="vx-ic">{I.bell}</span>
          <span className="vx-avatar">VC</span>
          <span className="vx-pill">RESEARCH</span>
          <button className="vx-theme" onClick={() => switchTheme(theme === "carbon" ? "classic" : "carbon")}
            title="Switch between the Carbon and Classic designs">{theme === "carbon" ? "Classic theme" : "Carbon theme"}</button>
        </div>
      </header>

      {loaded && !report && !scanning && (
        <div className="vx-banner">
          No scan data found yet. Click <b>Run Scan</b> to pull prices and fundamentals for the S&amp;P 500 + TSX 60 (first run takes a few minutes; later runs use a 24h cache).
        </div>
      )}

      {view === "model" ? (
        <ModelView report={report} />
      ) : view === "market" ? (
        <MarketView report={report} clock={clock} onOpen={(sym) => { setSel(sym); setView("terminal"); }} />
      ) : view === "terminal" ? (
        <main className="vx-term">
          <div className="vx-title">
            <h1><span className="vx-title-ic">{I.term}</span>Strategy Terminal</h1>
            <div className="vx-title-r">
              <span className="vx-muted">Universe:</span>
              <select value={universe} onChange={(e) => setUniverse(e.target.value)}>
                <option>All</option><option>S&amp;P 500</option><option>TSX 60</option>
              </select>
            </div>
          </div>

          {/* ---- chart */}
          <section className="vx-card vx-a">
            <div className="vx-toolbar">
              <div className="vx-tb-l">
                {["1M", "3M", "6M"].map((r, i) => {
                  const n = [22, 66, 160][i];
                  return <button key={r} className={cls(range === n && "on")} onClick={() => setRange(n)}>{r}</button>;
                })}
                <span className="vx-sep" />
                <button className={cls(showSma && "on")} onClick={() => setShowSma(!showSma)}>SMA 20</button>
                <button className={cls(showLevels && "on")} onClick={() => setShowLevels(!showLevels)}>Trade levels</button>
                <span className="vx-sep" />
                {stock && (
                  <span className="vx-sym">
                    <Badge s={stock} size={18} /> <b>{tick(stock.symbol)}</b> <span className="vx-muted">{stock.name}</span>
                    <span className="vx-muted">· {stock.index} · {stock.currency}</span>
                    <span className={cls("vx-chg", (chg(stock) || 0) >= 0 ? "up" : "dn")}><Tween v={px(stock)} fmt={(x) => fx(x)} /> {pct(chg(stock))}</span>
                  </span>
                )}
              </div>
              <div className="vx-tb-r"><span>{I.gear}</span><span>{I.expand}</span></div>
            </div>
            <CandleChart candles={shown} stock={stock} showLevels={showLevels} showSma={showSma} />
          </section>

          {/* ---- rankings / factors */}
          <section className="vx-card vx-b">
            <div className="vx-seg2">
              <button className={cls(rightTab === "rank" && "on")} onClick={() => setRightTab("rank")}>Rankings</button>
              <button className={cls(rightTab === "factors" && "on")} onClick={() => setRightTab("factors")}>Factors</button>
            </div>
            {rightTab === "rank" ? (
              <>
                <div className="vx-sub">
                  {([["score", "ALL"], ["value_score", "VALUE"], ["growth_score", "GROWTH"], ["quality_score", "QUALITY"]] as const).map(([k, l]) => (
                    <button key={k} className={cls(rankMode === k && "on")} onClick={() => setRankMode(k)}>{l}</button>
                  ))}
                </div>
                <div className="vx-search">
                  <span className="vx-in">{I.search}<input placeholder="Search symbol" value={query} onChange={(e) => setQuery(e.target.value)} /></span>
                  <span className="vx-ic">{I.sort}</span>
                </div>
                <div className="vx-lhead"><span>Symbol</span><span>Signal</span><span>{rankMode === "score" ? "Score" : rankMode.replace("_score", "")}</span></div>
                <div className="vx-list">
                  {ranked.map((s) => (
                    <button key={s.symbol} className={cls("vx-li", s.symbol === sel && "sel")} onClick={() => setSel(s.symbol)}>
                      <span className="vx-li-s"><Badge s={s} size={16} /><b>{tick(s.symbol)}</b><span className="vx-muted">{s.index === "TSX 60" ? "TSX" : "US"}</span></span>
                      <span className={cls("vx-sig", sigCls(s.signal))}>{SIG_SHORT[s.signal] || "—"}</span>
                      <span className="vx-num">{fx(s[rankMode] as number, 1)}<small className={cls("vx-lc", (chg(s) || 0) >= 0 ? "up" : "dn")}>{pct(chg(s), 1)}</small></span>
                    </button>
                  ))}
                  {!ranked.length && <div className="vx-empty sm">No scored stocks</div>}
                </div>
              </>
            ) : stock ? (
              <div className="vx-factors">
                <div className="vx-fhead">
                  <Badge s={stock} size={26} />
                  <div><b>{tick(stock.symbol)}</b><div className="vx-muted">{short(stock.sector)} · {stock.factor_count}/9 factors</div></div>
                  <div className="vx-score">{fx(stock.score, 1)}<small>/100</small></div>
                </div>
                <div className="vx-mlbox">
                  <div className="vx-mlhead"><span className={cls("vx-sig", sigCls(stock.signal))}>{stock.signal}</span>
                    <span className="vx-muted">{stock.ml_quintile ? `Model quintile ${stock.ml_quintile} of 5` : "No model score"}</span></div>
                  {([["Machine-learning score", stock.ml_score], ["Fundamental score", stock.fund_score]] as const).map(([l, v]) => (
                    <div key={l} className="vx-frow">
                      <div className="vx-frow-t"><span>{l}</span><b>{fx(v, 0)}</b></div>
                      <div className="vx-bar"><span style={{ width: `${v ?? 0}%`, opacity: v == null ? 0 : 1 }} /></div>
                    </div>
                  ))}
                  <div className="vx-mlkv">
                    <span>Expected 1-month excess</span><b className={(stock.exp_excess_1m ?? 0) >= 0 ? "up" : "dn"}>{pct(stock.exp_excess_1m, 2)}</b>
                    <span>Beat the universe, historically</span><b>{stock.hit_prob == null ? "—" : pct(stock.hit_prob, 0, false)}</b>
                  </div>
                  {!!stock.drivers?.length && (
                    <div className="vx-drv">
                      {stock.drivers.map(([n, v]) => <span key={n} className={v >= 0 ? "up" : "dn"}>{v >= 0 ? "+" : "−"} {n}</span>)}
                    </div>
                  )}
                  <div className="vx-frow-p">Expected excess and hit rate are this quintile&apos;s averages in out-of-sample testing, not a promise.</div>
                </div>
                <div className="vx-fsub">Fundamental factors</div>
                {report?.factors.map((f) => {
                  const fv = stock.factors[f.key];
                  const p = fv?.pct;
                  return (
                    <div key={f.key} className="vx-frow">
                      <div className="vx-frow-t"><span>{f.label} <i>{Math.round(f.weight * 100)}%</i></span><b>{factorDisplay(f.key, fv?.value ?? null)}</b></div>
                      <div className="vx-bar"><span style={{ width: `${(p ?? 0) * 100}%`, opacity: p == null ? 0 : 1 }} /></div>
                      <div className="vx-frow-p">{p == null ? "missing — excluded" : `${Math.round(p * 100)}th pct in sector`}</div>
                    </div>
                  );
                })}
                <div className="vx-fextra">
                  <div><span>Alpha vs SPY (1y)</span><b className={(stock.alpha || 0) >= 0 ? "up" : "dn"}>{pct(stock.alpha, 1)}</b></div>
                  <div><span>Realised beta</span><b>{fx(stock.beta_realized, 2)}</b></div>
                  <div><span>ATR(14)</span><b>{fx(stock.atr)} ({pct(stock.atr_pct, 1, false)})</b></div>
                </div>
              </div>
            ) : <div className="vx-empty sm">Select a stock</div>}
          </section>

          {/* ---- table / console */}
          <section className="vx-card vx-c">
            <div className="vx-ctabs">
              <div className="vx-ctabs-l">
                <button className={cls(bottomTab === "levels" && "on")} onClick={() => setBottomTab("levels")}>Trade Levels</button>
                <button className={cls(bottomTab === "portfolio" && "on")} onClick={() => setBottomTab("portfolio")}>Model Portfolio</button>
                <button className={cls(bottomTab === "console" && "on")} onClick={() => setBottomTab("console")}>
                  Scan Console {scanning && <span className="vx-dot busy" />}
                </button>
              </div>
              {bottomTab === "levels" && (
                <div className="vx-chips">
                  {["ALL", "BUY", "HOLD", "SELL"].map((f) => (
                    <button key={f} className={cls(sigFilter === f && "on")} onClick={() => setSigFilter(f)}>{f}</button>
                  ))}
                </div>
              )}
              {bottomTab === "console" && <span className="vx-muted sm">{scanning ? "LIVE · python vertex_screener.py" : savedLog.length ? `REPLAY of last real scan · ${report?.generated_utc?.slice(0, 16).replace("T", " ")} UTC` : ""}</span>}
            </div>
            {bottomTab === "levels" && (
              <div className="vx-table">
                <table>
                  <thead>
                    <tr>
                      {[["symbol", "Symbol"], ["sector", "Sector"], ["signal", "Signal"], ["price", "Price"], ["change_1d", "Chg"], ["entry", "Entry"], ["stop", "Stop"], ["target", "Target"], ["rr", "R:R"], ["alpha", "Alpha 1y"], ["score", "Score ▾"]].map(([k, l]) => (
                        <th key={k} onClick={() => setSortKey(k)} className={cls(sortKey === k && "on")}>{l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.map((s) => (
                      <tr key={s.symbol} className={cls(s.symbol === sel && "sel")} onClick={() => setSel(s.symbol)}>
                        <td><span className="vx-li-s"><Badge s={s} size={16} /><b>{tick(s.symbol)}</b></span></td>
                        <td className="vx-muted">{short(s.sector)}</td>
                        <td><span className={cls("vx-sig", sigCls(s.signal))}>{s.signal}</span></td>
                        <td><Tween v={px(s)} fmt={(x) => fx(x)} /></td>
                        <td className={(chg(s) || 0) >= 0 ? "up" : "dn"}>{pct(chg(s), 2)}</td>
                        <td className="c-entry">{fx(s.entry)}</td>
                        <td className="dn">{fx(s.stop)}</td>
                        <td className="up">{fx(s.target)}</td>
                        <td>{fx(s.rr, 2)}</td>
                        <td className={(s.alpha || 0) >= 0 ? "up" : "dn"}>{pct(s.alpha, 1)}</td>
                        <td><b>{fx(s.score, 1)}</b></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {bottomTab === "portfolio" && (
              <div className="vx-table">
                <table>
                  <thead><tr><th>Symbol</th><th>Name</th><th>Sector</th><th>Score</th><th>Price</th><th>Shares</th><th>Value</th><th>Weight</th></tr></thead>
                  <tbody>
                    {port?.holdings.map((h) => (
                      <tr key={h.symbol} onClick={() => setSel(h.symbol)} className={cls(h.symbol === sel && "sel")}>
                        <td><span className="vx-li-s"><Badge s={h} size={16} /><b>{tick(h.symbol)}</b></span></td>
                        <td className="vx-muted">{h.name}</td><td className="vx-muted">{short(h.sector)}</td>
                        <td>{fx(h.score, 1)}</td><td><Tween v={px(h)} fmt={(x) => fx(x)} /></td><td>{h.shares}</td><td><Tween v={h.shares * px(h)} fmt={(x) => money(x)} flash={false} /></td>
                        <td><span className="vx-wbar"><span style={{ width: `${h.weight * 1000}%` }} /></span>{pct(h.weight, 1, false)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            {bottomTab === "console" && (
              <div className="vx-console" ref={consoleRef}>
                {consoleLines.map((l, i) => <LogLine key={i} line={l} />)}
                {(scanning || replayN < savedLog.length) && <div className="vx-log"><span className="vx-caret">▋</span></div>}
                {!consoleLines.length && !scanning && <div className="vx-log vx-muted">No scan log yet. Press RUN SCAN.</div>}
              </div>
            )}
          </section>

          {/* ---- planner */}
          <div className="vx-d">
          {stock && <SetupBar stock={stock} price={px(stock)} live={!!q(stock.symbol)} date={report?.price_date}
            intraday={feed?.intraday?.symbol === stock.symbol ? feed.intraday.points : null} />}
          <section className="vx-card vx-plan-card vx-mix">
            <h3>Signal mix</h3>
            <div className="vx-muted sm">{scored.length} stocks · {universe === "All" ? "S&P 500 + TSX 60" : universe}</div>
            {["STRONG BUY", "BUY", "HOLD", "REDUCE", "SELL"].map((t) => {
              const n = scored.filter((x) => x.signal === t).length;
              return (
                <button key={t} className={cls("vx-mixr", (t === "STRONG BUY" || t === "BUY" ? "BUY" : t === "HOLD" ? "HOLD" : "SELL") === sigFilter && "on")}
                  onClick={() => { setSigFilter(t === "STRONG BUY" || t === "BUY" ? "BUY" : t === "HOLD" ? "HOLD" : "SELL"); setBottomTab("levels"); }}>
                  <span className={cls("vx-sig", sigCls(t))}>{SIG_SHORT[t]}</span>
                  <span className="vx-mixb"><i className={sigCls(t)} style={{ width: `${scored.length ? (n / scored.length) * 100 : 0}%` }} /></span>
                  <b>{n}</b>
                </button>
              );
            })}
            {report?.regime && (
              <div className="vx-mixreg">
                <span>Market regime</span>
                <b className={report.regime.state === "RISK-ON" ? "up" : report.regime.state === "RISK-OFF" ? "dn" : "vx-acc"}>{report.regime.state.replace("-", " ").toLowerCase()}</b>
              </div>
            )}
            <div className="vx-actions">
              <button onClick={exportCsv} disabled={!table.length}>Export CSV</button>
            </div>
          </section>
          </div>

          <footer className="vx-ticker">
            <span className="vx-ticker-k">{isOpen ? "LIVE" : "QUOTES"}</span>
            <div className="vx-tape">
              <div className="vx-tape-in">
                {[0, 1].map((dup) => (
                  <span key={dup} className="vx-tape-set" aria-hidden={dup === 1}>
                    {[...TAPE, ...(port?.holdings.map((h) => h.symbol) || [])].map((t) => {
                      const qq = q(t);
                      return (
                        <span key={t} className="vx-tape-i">
                          <b>{TAPE_NAME[t] || tick(t)}</b>
                          <Tween v={qq?.price} fmt={(x) => fx(x)} />
                          <span className={(qq?.change || 0) >= 0 ? "up" : "dn"}>{qq?.change == null ? "" : `${qq.change >= 0 ? "▲" : "▼"} ${pct(qq.change, 2)}`}</span>
                        </span>
                      );
                    })}
                  </span>
                ))}
              </div>
            </div>
            <span className="vx-ticker-k">FEED</span>
            <span className="vx-ticker-s">{feed ? `${feed.received}/${feed.requested} · ${feed.latencyMs}ms` : "—"}</span>
            <span className="vx-ticker-k">NEXT</span>
            <span className="vx-ticker-s">{liveOn ? `${String(nextIn).padStart(2, "0")}s` : "off"}</span>
            <span className="vx-ticker-k">SCAN</span>
            <span className="vx-ticker-s">{report ? `${report.universe.scored}/${report.universe.requested} · ${report.price_date}` : "—"}</span>
            <span className="vx-ticker-s vx-ticker-clock">{clock}</span>
          </footer>
        </main>
      ) : (
        <PortfolioBuilder report={report} px={px} onOpen={(sym) => { setSel(sym); setView("terminal"); }} />
      )}

      {showMethod && report && (
        <div className="vx-modal" onClick={() => setShowMethod(false)}>
          <div className="vx-card vx-mbody" onClick={(e) => e.stopPropagation()}>
            <div className="vx-chead"><h3>Methodology · Screener v{report.version}</h3><button className="vx-x" onClick={() => setShowMethod(false)}>✕</button></div>
            <div className="vx-mgrid">
              <div>
                <h4>Factor model</h4>
                <table className="vx-mtab">
                  <thead><tr><th>Factor</th><th>Weight</th><th>Better</th><th>Coverage</th></tr></thead>
                  <tbody>
                    {report.factors.map((f) => (
                      <tr key={f.key}><td>{f.label}</td><td>{Math.round(f.weight * 100)}%</td><td>{f.higher_is_better ? "Higher" : "Lower"}</td><td>{report.coverage[f.key]}/{report.universe.priced}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="vx-mtext">
                {Object.entries(report.methodology).map(([k, v]) => (<p key={k}><b>{k[0].toUpperCase() + k.slice(1)}.</b> {v}</p>))}
                <p className="vx-muted">Generated {report.generated_utc.slice(0, 19).replace("T", " ")} UTC · prices as of {report.price_date} · {report.source}</p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
