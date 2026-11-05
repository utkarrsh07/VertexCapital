import path from "path";
import fs from "fs";

/** Project root = parent of the Next.js app folder (terminal/). Override with VERTEX_ROOT. */
export function projectRoot(): string {
  return process.env.VERTEX_ROOT || path.resolve(process.cwd(), "..");
}

export function dataDir(): string {
  return path.join(projectRoot(), "data", "vertex");
}

export function pythonExe(): string {
  const root = projectRoot();
  const candidates = [
    path.join(root, ".venv", "Scripts", "python.exe"),
    path.join(root, ".venv", "bin", "python"),
  ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
  return process.platform === "win32" ? "python" : "python3";
}
