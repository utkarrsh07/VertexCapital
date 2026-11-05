import { spawn } from "child_process";
import path from "path";
import { projectRoot, pythonExe } from "../_root";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

let running = false;

export async function GET() {
  return Response.json({ running });
}

/** Runs the real Python screener and streams its stdout/stderr line by line. */
export async function POST(req: Request) {
  if (running) return new Response("A scan is already running.\n", { status: 409 });
  const limit = new URL(req.url).searchParams.get("limit");
  const root = projectRoot();
  const args = ["-u", path.join(root, "backend", "vertex_screener.py"), "--root", root];
  if (limit && /^\d+$/.test(limit)) args.push("--limit", limit);

  running = true;
  const enc = new TextEncoder();
  const stream = new ReadableStream({
    start(controller) {
      let closed = false;
      const send = (s: string) => {
        if (!closed) {
          try { controller.enqueue(enc.encode(s)); } catch { closed = true; }
        }
      };
      const finish = (msg: string) => {
        running = false;
        send(msg);
        if (!closed) { closed = true; try { controller.close(); } catch {} }
      };
      send(`$ ${path.basename(pythonExe())} ${args.slice(1).map((a) => path.basename(a)).join(" ")}\n`);
      const child = spawn(pythonExe(), args, { cwd: root, env: { ...process.env, PYTHONIOENCODING: "utf-8" } });
      child.stdout.on("data", (d) => send(d.toString()));
      child.stderr.on("data", (d) => {
        const t = d.toString();
        if (!/FutureWarning|DeprecationWarning|warnings\.warn/.test(t)) send(t);
      });
      child.on("error", (e) => finish(`[error] could not start Python: ${e.message}\n`));
      child.on("close", (code) => finish(`\n[process exited with code ${code}]\n`));
    },
  });
  return new Response(stream, {
    headers: { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store", "X-Accel-Buffering": "no" },
  });
}
