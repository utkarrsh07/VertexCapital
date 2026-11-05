import fs from "fs/promises";
import path from "path";
import { dataDir } from "../_root";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

let cache: { mtime: number; data: Record<string, unknown[]> } | null = null;

export async function GET(req: Request) {
  const symbol = new URL(req.url).searchParams.get("symbol") || "SPY";
  const file = path.join(dataDir(), "candles.json");
  try {
    const st = await fs.stat(file);
    if (!cache || cache.mtime !== st.mtimeMs) {
      cache = { mtime: st.mtimeMs, data: JSON.parse(await fs.readFile(file, "utf8")) };
    }
    return Response.json({ ok: true, symbol, candles: cache.data[symbol] || [] });
  } catch {
    return Response.json({ ok: false, symbol, candles: [] });
  }
}
