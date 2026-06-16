"use client";

import { useState } from "react";
import type { Entity } from "@/lib/types";
import { analyzeReviews, getApiBaseUrl } from "@/lib/api";
import { AspectTree } from "@/components/AspectTree";

const SAMPLE = `The room was spotless and the bed was incredibly comfortable.
Breakfast was cold and the coffee tasted stale.
Staff at the front desk were friendly and checked us in quickly.
The wifi kept disconnecting in the evening.
Great location, walking distance to the beach.`;

export function AnalyzeView() {
  const apiConfigured = getApiBaseUrl() !== null;
  const [text, setText] = useState<string>(SAMPLE);
  const [entityName, setEntityName] = useState<string>("My Hotel");
  const [result, setResult] = useState<Entity | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  async function onSubmit() {
    setError("");
    setResult(null);
    const reviews = text
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (reviews.length === 0) {
      setError("Please enter at least one review sentence.");
      return;
    }
    setLoading(true);
    try {
      const res = await analyzeReviews({
        reviews,
        entity_name: entityName,
        options: { sentiment_backend: "bert", split_sentiment: true },
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-4">
      {!apiConfigured ? (
        <div className="rounded-md border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          <strong>Backend not connected.</strong> Set{" "}
          <code className="rounded bg-amber-100 px-1">NEXT_PUBLIC_API_URL</code>{" "}
          to a running analyze service to enable live analysis. See{" "}
          <code className="rounded bg-amber-100 px-1">web/API_CONTRACT.md</code>.
          The Explore tab works without a backend.
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-3 md:grid-cols-[1fr_240px]">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Reviews (one sentence per line)
          </label>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={10}
            className="w-full rounded-md border border-slate-300 bg-white p-3 font-mono text-sm"
          />
        </div>
        <div className="space-y-3">
          <div>
            <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
              Hotel name
            </label>
            <input
              value={entityName}
              onChange={(e) => setEntityName(e.target.value)}
              className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
            />
          </div>
          <button
            type="button"
            onClick={onSubmit}
            disabled={loading || !apiConfigured}
            className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          >
            {loading ? "Analyzing…" : "Analyze"}
          </button>
          {!apiConfigured ? (
            <p className="text-xs text-slate-400">
              Button is disabled until a backend is connected.
            </p>
          ) : null}
        </div>
      </div>

      {error ? (
        <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      ) : null}

      {result ? <AspectTree entity={result} /> : null}
    </div>
  );
}
