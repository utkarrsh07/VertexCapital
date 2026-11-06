/**
 * Live (delayed) quotes from Yahoo Finance's public chart endpoint.
 * GET /api/vertex/quotes?symbols=AAPL,MU,RY.TO&intraday=MU
 *  -> { ok, asOf, latencyMs, quotes: { SYM: { price, prevClose, change, open, high, low, volume, time, marketState } },
 *       intraday?: { symbol, points: [unixSec, close][] } }
 * Results are cached per symbol for 10s so many open tabs don't hammer Yahoo.
 */
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

import { chart, pool, quote, type Quote } from "../_yahoo";

const MAX_SYMBOLS = 60;

export async function GET(req: Request) {
  const t0 = Date.now();
  const sp = new URL(req.url).searchParams;
  const symbols = [...new Set((sp.get("symbols") || "").split(",").map((s) => s.trim().toUpperCase()).filter((s) => /^[A-Z0-9.^=-]{1,15}$/.test(s)))].slice(0, MAX_SYMBOLS);
  const intradaySym = (sp.get("intraday") || "").toUpperCase();

  const [qs, intraday] = await Promise.all([
    pool(symbols, 8, quote),
    /^[A-Z0-9.^=-]{1,15}$/.test(intradaySym)
      ? chart(intradaySym, "1d", "5m").then((res) => {
          const ts: number[] = res.timestamp || [];
          const c: (number | null)[] = res.indicators?.quote?.[0]?.close || [];
          return { symbol: intradaySym, points: ts.map((t, i) => [t, c[i]] as [number, number | null]).filter((p) => p[1] != null) };
        }).catch(() => null)
      : Promise.resolve(null),
  ]);

  const quotes: Record<string, Quote> = {};
  symbols.forEach((s, i) => { if (qs[i]) quotes[s] = qs[i] as Quote; });
  return Response.json({
    ok: Object.keys(quotes).length > 0, asOf: Date.now(), latencyMs: Date.now() - t0,
    requested: symbols.length, received: Object.keys(quotes).length, quotes, intraday,
  });
}
