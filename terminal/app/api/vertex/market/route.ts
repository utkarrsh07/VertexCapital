/**
 * Whole-universe market snapshot for the Market tab (heatmap, breadth, movers, index charts).
 * GET /api/vertex/market -> { asOf, ageMs, refreshing, coverage, quotes: { SYM: [price, change] }, indices: {...} }
 * Universe quotes refresh in the background at most every 45s (stale-while-revalidate),
 * so the page always gets an instant answer and Yahoo isn't hammered.
 */
import fs from "fs/promises";
import path from "path";
import { dataDir } from "../_root";
import { chart, num, pool, quote, spark } from "../_yahoo";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const INDICES = ["^GSPC", "^IXIC", "^DJI", "^GSPTSE"];
const UNIVERSE_TTL = 45_000;
const INDEX_TTL = 20_000;

type Snap = { at: number; quotes: Record<string, [number, number | null]>; source: string };
type IndexData = { symbol: string; price: number; prevClose: number | null; change: number | null; points: [number, number][]; marketState: string };

let snap: Snap | null = null;
let refreshing: Promise<void> | null = null;
let indexCache: { at: number; data: Record<string, IndexData> } | null = null;

async function universe(): Promise<string[]> {
  try {
    const r = JSON.parse(await fs.readFile(path.join(dataDir(), "screener.json"), "utf8"));
    return (r.stocks || []).map((s: { symbol: string }) => s.symbol);
  } catch { return []; }
}

async function refresh() {
  const syms = await universe();
  if (!syms.length) return;
  const quotes: Record<string, [number, number | null]> = {};
  const bulk = await spark(syms);
  for (const [s, v] of Object.entries(bulk)) quotes[s] = [v.price, v.change];
  const missing = syms.filter((s) => !quotes[s.toUpperCase()]);
  if (missing.length) {
    const qs = await pool(missing, 10, (s) => quote(s, UNIVERSE_TTL));
    missing.forEach((s, i) => { const q = qs[i]; if (q) quotes[s.toUpperCase()] = [q.price, q.change]; });
  }
  if (Object.keys(quotes).length) {
    snap = { at: Date.now(), quotes, source: Object.keys(bulk).length ? "spark" : "chart" };
  }
}

async function indices(): Promise<Record<string, IndexData>> {
  if (indexCache && Date.now() - indexCache.at < INDEX_TTL) return indexCache.data;
  const data: Record<string, IndexData> = {};
  await pool(INDICES, 4, async (sym) => {
    try {
      const res = await chart(sym, "1d", "5m");
      const m = res.meta || {};
      const ts: number[] = res.timestamp || [];
      const c: (number | null)[] = res.indicators?.quote?.[0]?.close || [];
      const points = ts.map((t, i) => [t, c[i]] as [number, number | null]).filter((p): p is [number, number] => p[1] != null);
      const price = num(m.regularMarketPrice) ?? points[points.length - 1]?.[1];
      const prev = num(m.chartPreviousClose) ?? num(m.previousClose);
      const now = Date.now() / 1000, reg = m.currentTradingPeriod?.regular;
      if (price != null) data[sym] = {
        symbol: sym, price, prevClose: prev, change: prev ? price / prev - 1 : null, points,
        marketState: String(m.marketState || (reg && now >= reg.start && now <= reg.end ? "REGULAR" : "CLOSED")),
      };
    } catch { if (indexCache?.data[sym]) data[sym] = indexCache.data[sym]; }
  });
  indexCache = { at: Date.now(), data };
  return data;
}

export async function GET(req: Request) {
  const wait = new URL(req.url).searchParams.get("wait") === "1";
  if ((!snap || Date.now() - snap.at > UNIVERSE_TTL) && !refreshing) {
    refreshing = refresh().catch(() => {}).finally(() => { refreshing = null; });
  }
  if (wait && refreshing && !snap) await Promise.race([refreshing, new Promise((r) => setTimeout(r, 25_000))]);
  const idx = await indices();
  return Response.json({
    asOf: snap?.at ?? null, ageMs: snap ? Date.now() - snap.at : null, refreshing: !!refreshing,
    source: snap?.source ?? null, coverage: snap ? Object.keys(snap.quotes).length : 0,
    quotes: snap?.quotes ?? {}, indices: idx,
  });
}
