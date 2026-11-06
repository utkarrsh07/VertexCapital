/** Shared Yahoo Finance helpers (public chart/spark endpoints, ~15 min delayed). */
export type Quote = {
  price: number; prevClose: number | null; change: number | null; open: number | null;
  high: number | null; low: number | null; volume: number | null; time: number; marketState: string;
};

const BASES = (process.env.VERTEX_YAHOO_BASE || "https://query1.finance.yahoo.com,https://query2.finance.yahoo.com").split(",");
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36";

export const num = (v: unknown) => (typeof v === "number" && isFinite(v) ? v : null);

async function getJson(path: string) {
  let lastErr: unknown;
  for (const base of BASES) {
    try {
      const ctrl = new AbortController();
      const timer = setTimeout(() => ctrl.abort(), 7000);
      const r = await fetch(base + path, { headers: { "User-Agent": UA, Accept: "application/json" }, cache: "no-store", signal: ctrl.signal });
      clearTimeout(timer);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return await r.json();
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export async function chart(symbol: string, range: string, interval: string): Promise<any> {
  const j = await getJson(`/v8/finance/chart/${encodeURIComponent(symbol)}?range=${range}&interval=${interval}&includePrePost=false`);
  const res = j?.chart?.result?.[0];
  if (!res) throw new Error("empty result");
  return res;
}

const TTL = 10_000;
const cache = new Map<string, { at: number; q: Quote | null }>();

export async function quote(symbol: string, ttl = TTL): Promise<Quote | null> {
  const hit = cache.get(symbol);
  if (hit && Date.now() - hit.at < ttl) return hit.q;
  let q: Quote | null = null;
  try {
    const res = await chart(symbol, "5d", "1d");
    const m = res.meta || {};
    const ts: number[] = res.timestamp || [];
    const qq = res.indicators?.quote?.[0] || {};
    const i = ts.length - 1;
    const price = num(m.regularMarketPrice) ?? num(qq.close?.[i]);
    const prev = ts.length >= 2 ? num(qq.close?.[i - 1]) : num(m.previousClose) ?? num(m.chartPreviousClose);
    if (price != null) {
      const now = Date.now() / 1000, reg = m.currentTradingPeriod?.regular;
      q = {
        price, prevClose: prev, change: prev ? price / prev - 1 : null,
        open: num(qq.open?.[i]), high: num(m.regularMarketDayHigh) ?? num(qq.high?.[i]),
        low: num(m.regularMarketDayLow) ?? num(qq.low?.[i]), volume: num(m.regularMarketVolume) ?? num(qq.volume?.[i]),
        time: num(m.regularMarketTime) ?? ts[i] ?? Math.floor(now),
        marketState: String(m.marketState || (reg && now >= reg.start && now <= reg.end ? "REGULAR" : "CLOSED")),
      };
    }
  } catch {
    q = hit?.q ?? null;
  }
  cache.set(symbol, { at: Date.now(), q });
  return q;
}

/** Many symbols in one request (20 per call). Returns {} for any chunk Yahoo refuses. */
export async function spark(symbols: string[]): Promise<Record<string, { price: number; change: number | null }>> {
  const out: Record<string, { price: number; change: number | null }> = {};
  const chunks: string[][] = [];
  for (let i = 0; i < symbols.length; i += 20) chunks.push(symbols.slice(i, i + 20));
  await pool(chunks, 4, async (c) => {
    try {
      const j = await getJson(`/v7/finance/spark?symbols=${encodeURIComponent(c.join(","))}&range=1d&interval=1d`);
      for (const r of j?.spark?.result || []) {
        const m = r?.response?.[0]?.meta || {};
        const price = num(m.regularMarketPrice);
        const prev = num(m.chartPreviousClose) ?? num(m.previousClose);
        if (r?.symbol && price != null) out[String(r.symbol).toUpperCase()] = { price, change: prev ? price / prev - 1 : null };
      }
    } catch { /* fall back to per-symbol below */ }
  });
  return out;
}

export async function pool<T, R>(items: T[], n: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let k = 0;
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, async () => {
    while (k < items.length) { const i = k++; out[i] = await fn(items[i]); }
  }));
  return out;
}
