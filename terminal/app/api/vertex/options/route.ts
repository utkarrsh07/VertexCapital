/**
 * Option chain for one US symbol, fetched by backend/vertex_options.py (yfinance handles Yahoo's crumb).
 * GET /api/vertex/options?symbol=MU[&expiry=2026-12-18][&ref=120]
 * Cached 5 minutes per symbol+expiry. Set VERTEX_OPTIONS_MOCK=1 for a synthetic chain (offline demos/tests).
 */
import { spawn } from "child_process";
import path from "path";
import { projectRoot, pythonExe } from "../_root";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const TTL = 5 * 60_000;
const cache = new Map<string, { at: number; body: string }>();

function run(args: string[]): Promise<string> {
  return new Promise((resolve) => {
    const root = projectRoot();
    const child = spawn(pythonExe(), ["-u", path.join(root, "backend", "vertex_options.py"), ...args],
      { cwd: root, env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
    let out = "";
    const timer = setTimeout(() => { child.kill(); resolve(JSON.stringify({ error: "Option chain request timed out." })); }, 40_000);
    child.stdout.on("data", (d) => { out += d.toString(); });
    child.on("error", (e) => { clearTimeout(timer); resolve(JSON.stringify({ error: `Could not start Python: ${e.message}` })); });
    child.on("close", () => {
      clearTimeout(timer);
      const start = out.indexOf("{");
      resolve(start >= 0 ? out.slice(start) : JSON.stringify({ error: "Empty response from options script." }));
    });
  });
}

export async function GET(req: Request) {
  const sp = new URL(req.url).searchParams;
  const symbol = (sp.get("symbol") || "").toUpperCase();
  const expiry = sp.get("expiry") || "";
  const ref = Number(sp.get("ref") || "0");
  if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(symbol)) return Response.json({ error: "Invalid symbol." }, { status: 400 });
  if (expiry && !/^\d{4}-\d{2}-\d{2}$/.test(expiry)) return Response.json({ error: "Invalid expiry." }, { status: 400 });
  const key = `${symbol}|${expiry}`;
  const hit = cache.get(key);
  if (hit && Date.now() - hit.at < TTL) return new Response(hit.body, { headers: { "Content-Type": "application/json" } });
  const args = [symbol];
  if (expiry) args.push("--expiry", expiry);
  if (process.env.VERTEX_OPTIONS_MOCK === "1") args.push("--mock", "--spot", String(ref > 0 ? ref : 100));
  const body = await run(args);
  try {
    if (!JSON.parse(body).error) cache.set(key, { at: Date.now(), body });
  } catch {
    return Response.json({ error: "Malformed option chain response." });
  }
  return new Response(body, { headers: { "Content-Type": "application/json" } });
}
