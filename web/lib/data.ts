import type { ExportData, RunIndex } from "./types";

// Resolve a path under the app's basePath/public so it works on Vercel.
function publicUrl(path: string): string {
  const clean = path.startsWith("/") ? path : `/${path}`;
  return clean;
}

export async function loadRunIndex(): Promise<RunIndex> {
  const res = await fetch(publicUrl("/data/index.json"), { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load run index: ${res.status}`);
  }
  return res.json();
}

export async function loadRun(runId: string): Promise<ExportData> {
  const res = await fetch(publicUrl(`/data/${runId}.json`), { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load run ${runId}: ${res.status}`);
  }
  return res.json();
}
