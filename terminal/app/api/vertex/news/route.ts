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

type Item = { title: string; publisher: string; link: string; time: number; topics: string[]; tone: "positive" | "negative" | "neutral"; related: boolean };

const GENERIC = new Set(["american", "general", "united", "first", "national", "western", "international", "global", "the", "bank",
  "royal", "canadian", "southern", "northern", "public", "republic", "texas", "pacific", "atlantic", "new", "consolidated"]);
/** Words that identify this company in a headline: ticker plus a cleaned company name. */
function matchers(symbol: string, name: string): RegExp[] {
  const base = symbol.replace(/\.TO$/, "").replace(/-[A-Z]$/, "");
  const clean = name.replace(/\(.*?\)/g, "").replace(/\b(inc|corp|corporation|company|co|group|holdings?|plc|ltd|limited|the|class [ab]|n\.v\.|s\.a\.)\b\.?/gi, "")
    .replace(/[^A-Za-z0-9&' -]/g, " ").replace(/\s+/g, " ").trim();
  const words = clean.split(" ").filter(Boolean);
  const out: RegExp[] = [];
  if (base.length >= 2) out.push(new RegExp(`\\b${base}\\b`));
  if (words.length) {
    const first = words[0];
    if (first.length >= 4 && !GENERIC.has(first.toLowerCase())) out.push(new RegExp(`\\b${first}`, "i"));
    else if (words.length >= 2) out.push(new RegExp(`\\b${words[0]} ${words[1]}`, "i"));
  }
  return out;
}

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

async function fetchNews(symbol: string, name: string): Promise<Item[]> {
  const hit = cache.get(symbol);
  // Yahoo search understands US tickers; for Toronto listings search by company name instead.
  const query = symbol.endsWith(".TO") && name ? name.replace(/\(.*?\)/g, "").trim() : symbol;
  const ms = matchers(symbol, name);
  if (hit && Date.now() - hit.at < TTL) return hit.items;
  for (const base of BASES) {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 7000);
      const r = await fetch(`${base}/v1/finance/search?q=${encodeURIComponent(query)}&quotesCount=0&newsCount=12&enableFuzzyQuery=false`,
        { headers: { "User-Agent": UA, Accept: "application/json" }, cache: "no-store", signal: ctrl.signal });
      clearTimeout(t);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const items: Item[] = (j?.news || []).filter((n: any) => n?.title && n?.link).map((n: any) => ({
        title: String(n.title), publisher: String(n.publisher || ""), link: String(n.link),
        time: Number(n.providerPublishTime || 0), ...tag(String(n.title)),
        related: !ms.some((re) => re.test(String(n.title))),
      })).sort((a: Item, b: Item) => (Number(a.related) - Number(b.related)) || b.time - a.time).slice(0, 6);
      cache.set(symbol, { at: Date.now(), items });
      return items;
    } catch { /* try next base */ }
  }
  return hit?.items ?? [];
}

export async function GET(req: Request) {
  const sp = new URL(req.url).searchParams;
  const rawSyms = (sp.get("symbols") || "").split(",").map((s) => s.trim().toUpperCase());
  const rawNames = (sp.get("names") || "").split("|");
  const names: Record<string, string> = {};
  rawSyms.forEach((s, i) => { names[s] = (rawNames[i] || "").slice(0, 80); });
  const symbols = [...new Set(rawSyms.filter((s) => /^[A-Z0-9.^=-]{1,15}$/.test(s)))].slice(0, 20);
  const out: Record<string, Item[]> = {};
  let k = 0;
  await Promise.all(Array.from({ length: Math.min(5, symbols.length) }, async () => {
    while (k < symbols.length) { const s = symbols[k++]; out[s] = await fetchNews(s, names[s] || ""); }
  }));
  return Response.json({ asOf: Date.now(), news: out });
}
