/**
 * Latest headlines per ticker from Yahoo Finance's public search endpoint, tagged by topic and tone.
 * GET /api/vertex/news?symbols=MU,NVDA  -> { asOf, news: { MU: [{ title, publisher, link, time, topics, tone }] } }
 * Tags come from transparent keyword rules (no AI summarisation). Cached 10 minutes per symbol.
 */
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BASES = (process.env.VERTEX_YAHOO_BASE || "https://query1.finance.yahoo.com,https://query2.finance.yahoo.com").split(",");
const UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130 Safari/537.36";
const TTL = 10 * 60_000;
const cache = new Map<string, { at: number; items: Item[] }>();

type Item = { title: string; publisher: string; link: string; time: number; topics: string[]; tone: "positive" | "negative" | "neutral" };


const TOPICS: [string, RegExp][] = [
  ["Earnings", /\b(earnings|eps|quarterly|quarter|q[1-4]\b|results|revenue|profit|sales|beat|miss(es|ed)?)\b/i],
  ["Guidance", /\b(guidance|outlook|forecast|raises?|lowers?|cuts? (its )?(outlook|forecast))\b/i],
  ["Tariffs & trade", /\b(tariffs?|trade (war|deal|talks)|export (ban|controls?)|import|sanctions?|customs|duties)\b/i],
  ["Analyst", /\b(upgrades?|downgrades?|price target|rating|analysts?|overweight|underweight|outperform|buy rating|sell rating)\b/i],
  ["Deal", /\b(acquir\w*|acquisition|merger|merge|buyout|takeover|stake|spin-?off|deal)\b/i],
  ["Legal & regulation", /\b(lawsuit|sues?|sued|sec\b|ftc|doj|probe|investigation|antitrust|regulators?|court|fine[ds]?|settle\w*)\b/i],
  ["Product & AI", /\b(launch\w*|unveil\w*|product|chips?|\bai\b|artificial intelligence|model|platform|data cent(er|re)s?)\b/i],
  ["Dividend & buyback", /\b(dividend|buyback|repurchase)\b/i],
  ["Leadership", /\b(ceo|cfo|chief executive|resign\w*|appoint\w*|steps? down)\b/i],
];
const POS = /\b(beats?|surg\w*|soar\w*|jump\w*|rall\w*|gains?|record|upgrade[ds]?|raises?|strong|tops?|boost\w*|approv\w*|wins?|outperform\w*|rebound\w*|climb\w*|higher)\b/i;
const NEG = /\b(miss(es|ed)?|plung\w*|fall\w*|fell|drops?|dropped|slump\w*|downgrade[ds]?|cuts?|lawsuit|probe|weak\w*|warn\w*|tumbl\w*|sink\w*|sank|slid|slides?|lower|loss(es)?|recall\w*|delay\w*)\b/i;

function tag(title: string): Pick<Item, "topics" | "tone"> {
  const topics = TOPICS.filter(([, re]) => re.test(title)).map(([t]) => t).slice(0, 2);
  const p = POS.test(title), n = NEG.test(title);
  return { topics: topics.length ? topics : ["General"], tone: p && !n ? "positive" : n && !p ? "negative" : "neutral" };
}

async function fetchNews(symbol: string): Promise<Item[]> {
  const hit = cache.get(symbol);
  if (hit && Date.now() - hit.at < TTL) return hit.items;
  for (const base of BASES) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 7000);
      const r = await fetch(`${base}/v1/finance/search?q=${encodeURIComponent(symbol)}&quotesCount=0&newsCount=8&enableFuzzyQuery=false`,
        { headers: { "User-Agent": UA, Accept: "application/json" }, cache: "no-store", signal: ctrl.signal });
      clearTimeout(t);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const items: Item[] = (j?.news || []).filter((n: any) => n?.title && n?.link).map((n: any) => ({
        title: String(n.title), publisher: String(n.publisher || ""), link: String(n.link),
        time: Number(n.providerPublishTime || 0), ...tag(String(n.title)),
      })).sort((a: Item, b: Item) => b.time - a.time).slice(0, 6);
      cache.set(symbol, { at: Date.now(), items });
      return items;
    } catch { /* try next base */ }
  }
  return hit?.items ?? [];
}

export async function GET(req: Request) {
  const symbols = [...new Set((new URL(req.url).searchParams.get("symbols") || "").split(",")
    .map((s) => s.trim().toUpperCase()).filter((s) => /^[A-Z0-9.^=-]{1,15}$/.test(s)))].slice(0, 20);
  const out: Record<string, Item[]> = {};
  let k = 0;
  await Promise.all(Array.from({ length: Math.min(5, symbols.length) }, async () => {
    while (k < symbols.length) { const s = symbols[k++]; out[s] = await fetchNews(s); }
  }));
  return Response.json({ asOf: Date.now(), news: out });
}
