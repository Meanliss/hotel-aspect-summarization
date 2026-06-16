"use client";

import { useEffect, useMemo, useState } from "react";
import type { ExportData, Entity } from "@/lib/types";
import { loadRunIndex, loadRun } from "@/lib/data";
import { AspectTree } from "@/components/AspectTree";

export function ExploreView() {
  const [runs, setRuns] = useState<string[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [data, setData] = useState<ExportData | null>(null);
  const [entityId, setEntityId] = useState<string>("");
  const [query, setQuery] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  // Load the run index once.
  useEffect(() => {
    loadRunIndex()
      .then((idx) => {
        setRuns(idx.runs);
        if (idx.runs.length > 0) setRunId(idx.runs[0]);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Load the selected run.
  useEffect(() => {
    if (!runId) return;
    setLoading(true);
    setError("");
    loadRun(runId)
      .then((d) => {
        setData(d);
        if (d.entities.length > 0) setEntityId(d.entities[0].entity_id);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [runId]);

  const entities = data?.entities ?? [];
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entities;
    return entities.filter(
      (e) =>
        e.entity_name.toLowerCase().includes(q) ||
        e.entity_id.toLowerCase().includes(q),
    );
  }, [entities, query]);

  const entity: Entity | undefined = useMemo(
    () => entities.find((e) => e.entity_id === entityId),
    [entities, entityId],
  );

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[320px_1fr]">
      <aside className="space-y-4">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Run
          </label>
          <select
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
          >
            {runs.map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Search hotels
          </label>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="name or id…"
            className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
          />
        </div>

        <div className="max-h-[60vh] overflow-y-auto rounded-md border border-slate-200 bg-white">
          {filtered.map((e) => (
            <button
              key={e.entity_id}
              type="button"
              onClick={() => setEntityId(e.entity_id)}
              className={`block w-full border-b border-slate-100 px-3 py-2 text-left text-sm last:border-b-0 ${
                e.entity_id === entityId
                  ? "bg-indigo-50 font-medium text-indigo-700"
                  : "text-slate-700 hover:bg-slate-50"
              }`}
            >
              <div className="truncate">{e.entity_name}</div>
              <div className="text-[11px] text-slate-400">
                {e.entity_id} · {e.split}
              </div>
            </button>
          ))}
          {filtered.length === 0 ? (
            <div className="px-3 py-2 text-sm text-slate-400">No matches.</div>
          ) : null}
        </div>
      </aside>

      <main>
        {error ? (
          <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
            {error}
          </div>
        ) : loading ? (
          <div className="text-sm text-slate-500">Loading…</div>
        ) : entity ? (
          <AspectTree entity={entity} />
        ) : (
          <div className="text-sm text-slate-500">Select a hotel to begin.</div>
        )}
      </main>
    </div>
  );
}
