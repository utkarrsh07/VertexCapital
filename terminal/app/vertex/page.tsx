"use client";

import { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";
import "./vertex.css";

/* ============================================================ types */
type FactorVal = { value: number | null; pct: number | null };
type Stock = {
  symbol: string; name: string; sector: string; index: string; currency: string;
  market_cap: number | null; price: number; change_1d: number | null; score: number | null;
  signal: "LONG" | "WATCH" | "AVOID" | "N/A"; factor_count: number;
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
  methodology: Record<string, string>; stocks: Stock[];
  portfolio: { capital: number; cash_weight: number; holdings: Holding[];
    trailing: { period: string; portfolio: number; benchmark: number | null }[]; trailing_note: string };
};
type Candle = [string, number, number, number, number, number];

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
      <path d={d} fill="none" stroke={color} strokeWidth={1.5} />
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

/* ============================================================ console */
const TAG_CLASS: Record<string, string> = {
  INIT: "t-init", UNIVERSE: "t-uni", PRICES: "t-px", FUNDS: "t-fund", FACTORS: "t-fac", RANK: "t-rank",
  SCORE: "t-score", LEVELS: "t-lvl", SIGNAL: "t-sig", PORTF: "t-port", DONE: "t-done", ERROR: "t-err",
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

/* ============================================================ page */
export default function VertexPage() {
  const [report, setReport] = useState<Report | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [savedLog, setSavedLog] = useState<string[]>([]);
  const [view, setView] = useState<"terminal" | "dashboard">("terminal");
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
  const shown = candles.slice(-range);

  const ranked = useMemo(() => {
    const q = query.trim().toUpperCase();
    return scored
      .filter((s) => !q || s.symbol.includes(q) || s.name.toUpperCase().includes(q))
      .sort((a, b) => ((b[rankMode] ?? -1) as number) - ((a[rankMode] ?? -1) as number))
      .slice(0, 80);
  }, [scored, rankMode, query]);

  const table = useMemo(() => {
    const rows = scored.filter((s) => sigFilter === "ALL" || s.signal === sigFilter);
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

  const lastLine = (scanning ? live : savedLog).filter((l) => /^\[/.test(l)).slice(-1)[0] || "";

  /* ============================================================ render */
  return (
    <div className="vx-root">
      <div className="vx-bg" />
      <header className="vx-nav">
        <div className="vx-brand">{I.logo}<span>Vertex<b>Capital</b></span></div>
        <nav className="vx-tabs">
          <button className={cls(view === "dashboard" && "on")} onClick={() => setView("dashboard")}>{I.dash}Dashboard</button>
          <button className={cls(view === "terminal" && "on")} onClick={() => setView("terminal")}>{I.term}Strategy Terminal</button>
          <button onClick={() => setShowMethod(true)}>{I.book}Methodology</button>
          <button onClick={runScan} disabled={scanning}>{I.refresh}{scanning ? "Scanning…" : "Run Scan"}</button>
        </nav>
        <div className="vx-user">
          <span className={cls("vx-dot", scanning ? "busy" : report ? "ok" : "off")} />
          <span className="vx-muted">{scanning ? "Engine running" : report ? `Data ${report.price_date}` : "No data"}</span>
          <span className="vx-clock">{clock}</span>
          <span className="vx-ic">{I.bell}</span>
          <span className="vx-avatar">VC</span>
          <span className="vx-pill">RESEARCH</span>
        </div>
      </header>

      {loaded && !report && !scanning && (
        <div className="vx-banner">
          No scan data found yet. Click <b>Run Scan</b> to pull prices and fundamentals for the S&amp;P 500 + TSX 60 (first run takes a few minutes; later runs use a 24h cache).
        </div>
      )}

      {view === "terminal" ? (
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
                    <span className={cls("vx-chg", (stock.change_1d || 0) >= 0 ? "up" : "dn")}>{fx(stock.price)} {pct(stock.change_1d)}</span>
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
                      <span className={cls("vx-sig", s.signal)}>{s.signal === "LONG" ? "Long" : s.signal === "WATCH" ? "Watch" : "Avoid"}</span>
                      <span className="vx-num">{fx(s[rankMode] as number, 1)}</span>
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
                  {["ALL", "LONG", "WATCH", "AVOID"].map((f) => (
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
                      {[["symbol", "Symbol"], ["sector", "Sector"], ["signal", "Signal"], ["price", "Price"], ["entry", "Entry"], ["stop", "Stop"], ["target", "Target"], ["rr", "R:R"], ["alpha", "Alpha 1y"], ["score", "Score ▾"]].map(([k, l]) => (
                        <th key={k} onClick={() => setSortKey(k)} className={cls(sortKey === k && "on")}>{l}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {table.map((s) => (
                      <tr key={s.symbol} className={cls(s.symbol === sel && "sel")} onClick={() => setSel(s.symbol)}>
                        <td><span className="vx-li-s"><Badge s={s} size={16} /><b>{tick(s.symbol)}</b></span></td>
                        <td className="vx-muted">{short(s.sector)}</td>
                        <td><span className={cls("vx-sig", s.signal)}>{s.signal}</span></td>
                        <td>{fx(s.price)}</td>
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
                        <td>{fx(h.score, 1)}</td><td>{fx(h.price)}</td><td>{h.shares}</td><td>{money(h.value)}</td>
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
          <section className="vx-card vx-d">
            <h3>Position Planner</h3>
            <label className="vx-lbl">Selected stock</label>
            <div className="vx-select">
              {stock ? (<><Badge s={stock} size={18} /><b>{tick(stock.symbol)}</b><span className="vx-muted">{stock.name}</span></>) : <span className="vx-muted">—</span>}
            </div>
            <div className="vx-opts">
              <div className="on"><span className="vx-dot ok" />Long only</div>
              <div><span className={cls("vx-sig", stock?.signal || "N/A")}>{stock?.signal || "—"}</span>Signal</div>
            </div>
            <label className="vx-lbl">Account size (USD, hypothetical)</label>
            <div className="vx-input">
              <input type="number" value={account} min={1000} step={1000} onChange={(e) => setAccount(Math.max(0, Number(e.target.value)))} />
              <span className="vx-muted">≈ {plan ? `${plan.shares} sh` : "—"}</span>
            </div>
            <label className="vx-lbl">Risk per trade <b className="vx-acc">{risk.toFixed(2)}%</b></label>
            <input className="vx-range" type="range" min={0.25} max={3} step={0.25} value={risk} onChange={(e) => setRisk(Number(e.target.value))}
              style={{ ["--p" as string]: `${((risk - 0.25) / 2.75) * 100}%` }} />
            {stock && plan && (
              <div className="vx-plan">
                <div><span>Entry</span><b className="c-entry">{fx(stock.entry)}</b></div>
                <div><span>Stop</span><b className="dn">{fx(stock.stop)}</b></div>
                <div><span>Target</span><b className="up">{fx(stock.target)}</b></div>
                <div><span>Position</span><b>{money(plan.position)}</b></div>
                <div><span>Max loss</span><b className="dn">-{money(plan.maxLoss)}</b></div>
                <div><span>At target</span><b className="up">+{money(plan.gain)}</b></div>
              </div>
            )}
            {plan?.capped && <div className="vx-note">Position capped at 25% of account.</div>}
            <div className="vx-actions">
              <button onClick={runScan} disabled={scanning}>{scanning ? "SCANNING…" : "RUN SCAN"}</button>
              <button onClick={exportCsv} disabled={!table.length}>EXPORT CSV</button>
            </div>
          </section>

          <footer className="vx-ticker">
            <span className="vx-ticker-k">ENGINE</span>
            <span className="vx-ticker-v">{lastLine || "idle"}</span>
            <span className="vx-ticker-k">UNIVERSE</span>
            <span className="vx-ticker-v">{report ? `${report.universe.scored}/${report.universe.requested} scored` : "—"}</span>
            <span className="vx-ticker-k">SIGNALS</span>
            <span className="vx-ticker-v">{report ? Object.entries(report.signals).map(([k, v]) => `${k} ${v}`).join(" · ") : "—"}</span>
          </footer>
        </main>
      ) : (
        /* ============================================================ dashboard */
        <main className="vx-dash">
          <h1 className="vx-h1">Portfolio Dashboard</h1>
          <div className="vx-row1">
            <section className="vx-card">
              <h3>Overview</h3>
              <div className="vx-ov">
                <Ring size={150} stroke={11} gap={0.06}
                  segs={(() => {
                    const top = sectors.slice(0, 3).map((s, i) => ({ v: s.weight, c: ["#f59e0b", "#22c3ee", "#4f8cff"][i] }));
                    const rest = sectors.slice(3).reduce((a, s) => a + s.weight, 0) + (port?.cash_weight || 0);
                    return [...top, { v: rest, c: "#5b6fb3" }];
                  })()}>
                  <div className="vx-ov-v">{money(port?.capital || 0)}</div>
                  <div className="vx-muted sm">Model portfolio</div>
                </Ring>
                <div className="vx-legend">
                  {sectors.slice(0, 3).map((s, i) => (
                    <div key={s.sector}>
                      <span className="vx-ldot" style={{ background: ["#f59e0b", "#22c3ee", "#4f8cff"][i] }} />
                      <div><div className="vx-lname">{short(s.sector)}</div><div className="vx-lval">{pct(s.weight, 2, false)}</div></div>
                      <span className="vx-lmoney">{money(s.value, 2)}</span>
                    </div>
                  ))}
                  <div>
                    <span className="vx-ldot" style={{ background: "#5b6fb3" }} />
                    <div><div className="vx-lname">Other + cash</div><div className="vx-lval">{pct(sectors.slice(3).reduce((a, s) => a + s.weight, 0) + (port?.cash_weight || 0), 2, false)}</div></div>
                    <span className="vx-lmoney">{money((sectors.slice(3).reduce((a, s) => a + s.weight, 0) + (port?.cash_weight || 0)) * (port?.capital || 0), 2)}</span>
                  </div>
                </div>
              </div>
            </section>

            <section className="vx-card">
              <div className="vx-chead">
                <h3>Trailing Performance</h3>
                <div className="vx-mini">
                  <button className={cls(perfMode === "portfolio" && "on")} onClick={() => setPerfMode("portfolio")}>Basket</button>
                  <button className={cls(perfMode === "benchmark" && "on")} onClick={() => setPerfMode("benchmark")}>SPY</button>
                </div>
              </div>
              <div className="vx-bars">
                {(() => {
                  const tr = port?.trailing || [];
                  const vals = tr.map((t) => (perfMode === "portfolio" ? t.portfolio : t.benchmark) ?? 0);
                  const mx = Math.max(0.0001, ...vals.map(Math.abs));
                  return tr.map((t, i) => {
                    const v = vals[i];
                    const other = perfMode === "portfolio" ? t.benchmark : t.portfolio;
                    return (
                      <div key={t.period} className="vx-barcol">
                        <div className="vx-barwrap">
                          <div className={cls("vx-barv", v < 0 && "neg", i === tr.length - 1 && "hi")} style={{ height: `${(Math.abs(v) / mx) * 100}%` }} />
                        </div>
                        <div className={cls("vx-barl", v >= 0 ? "up" : "dn")}>{pct(v, 1)}</div>
                        <div className="vx-barp">{t.period}</div>
                        <div className="vx-bart">{perfMode === "portfolio" ? "SPY" : "Basket"} {pct(other, 1)}</div>
                      </div>
                    );
                  });
                })()}
              </div>
              <div className="vx-foot">Today&apos;s picks over past windows — hindsight, not a backtest.</div>
            </section>

            <section className="vx-card">
              <div className="vx-chead"><h3>Scanner Status</h3><span className="vx-ic">{I.plus}</span></div>
              {report ? (
                <div className="vx-status">
                  {Object.entries(report.universe.by_index).map(([k, v]) => {
                    const sub = report.stocks.filter((s) => s.index === k);
                    const longs = sub.filter((s) => s.signal === "LONG").length;
                    return (
                      <div key={k} className="vx-st">
                        <div className="vx-st-h"><span className="vx-dot ok" />{k} <span className="vx-muted">({v})</span></div>
                        <div className="vx-st-box">
                          <div><b>9-factor screen</b><div className="vx-muted sm">{sub.filter((s) => s.score != null).length} scored · {report.price_date}</div></div>
                          <div className="vx-st-r"><span className="up">+{longs} LONG</span><div className="vx-muted sm">{sub.length - longs} other</div></div>
                        </div>
                      </div>
                    );
                  })}
                  <button className="vx-more" onClick={() => { setView("terminal"); setBottomTab("console"); }}>SHOW SCAN LOG</button>
                </div>
              ) : <div className="vx-empty sm">No scan yet</div>}
            </section>
          </div>

          <div className="vx-row2">
            <section className="vx-card vx-alloc">
              <div className="vx-alloc-h">
                <h3>Model Portfolio by Sector</h3>
                <div className="vx-cols"><span>HOLDING ▾</span><span>SHARES ▾</span><span>VALUE ▾</span><span>ALLOCATION ▾</span></div>
              </div>
              <div className="vx-alloc-b">
                <div className="vx-alloc-ring">
                  <Ring size={290} stroke={22} gap={0.05} icons
                    segs={sectors.map((s) => ({ v: s.weight, c: sc(s.sector), label: SECTOR_CODE[s.sector] || s.sector.slice(0, 2).toUpperCase() }))}>
                    <div className="vx-big"><b>{port?.holdings.length || 0}</b> STOCKS</div>
                    <div className="vx-muted sm">in</div>
                    <div className="vx-big"><b>{sectors.length}</b> SECTORS</div>
                  </Ring>
                </div>
                <div className="vx-tree">
                  {sectors.map((s) => (
                    <div key={s.sector} className="vx-grp">
                      <div className="vx-grp-h">
                        <span className="vx-li-s"><span className="vx-sq" style={{ background: sc(s.sector) }} /><b>{s.sector}</b></span>
                        <span>{s.items.reduce((a, h) => a + h.shares, 0)}</span><span>{money(s.value, 2)}</span><span>{pct(s.weight, 1, false)}</span>
                      </div>
                      {s.items.map((h) => (
                        <div key={h.symbol} className="vx-grp-i" onClick={() => { setSel(h.symbol); setView("terminal"); }}>
                          <span className="vx-li-s"><span className="vx-elbow" /><Badge s={h} size={16} />{h.name} <span className="vx-muted">({tick(h.symbol)})</span></span>
                          <span>{h.shares}</span><span>{money(h.value, 2)}</span><span>{pct(h.weight, 1, false)}</span>
                        </div>
                      ))}
                    </div>
                  ))}
                  {!!port && <div className="vx-grp-h cash"><span className="vx-li-s"><span className="vx-sq" style={{ background: "#5b6fb3" }} /><b>Cash reserve</b></span><span>—</span><span>{money(port.cash_weight * port.capital, 2)}</span><span>{pct(port.cash_weight, 1, false)}</span></div>}
                </div>
              </div>
            </section>

            <section className="vx-card vx-best">
              <div className="vx-chead">
                <h3>Top Ranked</h3>
                <div className="vx-mini">
                  <button className={cls(topMode === "score" && "on")} onClick={() => setTopMode("score")}>Score</button>
                  <button className={cls(topMode === "momentum" && "on")} onClick={() => setTopMode("momentum")}>Momentum</button>
                </div>
              </div>
              <div className="vx-best-l">
                {[...scored]
                  .sort((a, b) => topMode === "score" ? (b.score! - a.score!) : ((b.factors.momentum_6m?.value ?? -9) - (a.factors.momentum_6m?.value ?? -9)))
                  .slice(0, 8)
                  .map((s) => (
                    <div key={s.symbol} className="vx-bc" onClick={() => { setSel(s.symbol); setView("terminal"); }}>
                      <div className="vx-bc-h">
                        <span className="vx-li-s"><Badge s={s} size={20} /><b>{s.name}</b> <span className="vx-muted">({tick(s.symbol)})</span></span>
                        <span className={(s.change_1d || 0) >= 0 ? "up" : "dn"}>{pct(s.change_1d)}</span>
                      </div>
                      <div className="vx-bc-b">
                        <div className="vx-muted sm">{topMode === "score" ? `Score ${fx(s.score, 1)}` : `6M ${pct(s.factors.momentum_6m?.value, 1)}`}<br />Price</div>
                        <Spark data={s.spark} w={120} h={34} color={(s.spark?.[s.spark.length - 1] ?? 0) >= (s.spark?.[0] ?? 0) ? "#4f8cff" : "#ff4d6d"} />
                        <div className="vx-bc-p"><span className="vx-muted sm">{s.currency}</span><b>{fx(s.price)}</b></div>
                      </div>
                    </div>
                  ))}
              </div>
            </section>
          </div>
          <div className="vx-disc">
            Research &amp; demonstration tool. Hypothetical $100k model portfolio built from a transparent factor ranking — not investment advice, no orders placed.
            Data: {report?.source || "Yahoo Finance"}.
          </div>
        </main>
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
