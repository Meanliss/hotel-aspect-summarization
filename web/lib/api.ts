// Client for the analyze backend (FastAPI service, implemented later).
// The base URL is configured via NEXT_PUBLIC_API_URL. When unset, the Analyze
// tab shows a "backend not connected" banner instead of calling anything.

import type { AnalyzeRequest, AnalyzeResponse } from "./types";

export function getApiBaseUrl(): string | null {
  const url = process.env.NEXT_PUBLIC_API_URL;
  if (!url || !url.trim()) return null;
  return url.replace(/\/+$/, "");
}

export async function analyzeReviews(
  req: AnalyzeRequest,
): Promise<AnalyzeResponse> {
  const base = getApiBaseUrl();
  if (!base) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not configured. Connect a backend to use Analyze.",
    );
  }
  const res = await fetch(`${base}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Analyze request failed (${res.status} ${res.statusText}): ${text}`,
    );
  }
  return (await res.json()) as AnalyzeResponse;
}
