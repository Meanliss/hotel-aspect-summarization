"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  COLOR_BAR,
  COLOR_BG_LIGHT,
  COLOR_RING,
  COLOR_TEXT,
  METHOD_IDS,
  loadSpaceData,
  type MethodId,
  type SpaceData,
  type SpaceEntity,
} from "@/lib/space";

function methodMeta(data: SpaceData, mid: MethodId) {
  return data.method_meta?.[mid] ?? {
    label: mid.toUpperCase(),
    short: mid.toUpperCase(),
    desc: "",
    color: "slate",
  };
}

function sampleLabel(index: number) {
  return `Property Sample ${String(index + 1).padStart(2, "0")}`;
}

function GoldPanel({ entity }: { entity: SpaceEntity }) {
  const [open, setOpen] = useState(false);
  const overall = entity.gold?.overall ?? [];
  if (overall.length === 0) return null;
  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium text-amber-800"
      >
        <span>Human gold - overall ({overall.length} references)</span>
        <span className="text-xs text-amber-600">{open ? "hide" : "show"}</span>
      </button>
      {open ? (
        <div className="space-y-2 border-t border-amber-200 px-3 py-2">
          {overall.map((ref, i) => (
            <p key={i} className="text-xs leading-relaxed text-amber-900">
              <span className="mr-1 font-mono text-amber-500">#{i + 1}</span>
              {ref}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function AspectCard({
  data,
  entity,
  method,
  aspect,
}: {
  data: SpaceData;
  entity: SpaceEntity;
  method: MethodId;
  aspect: string;
}) {
  const cell = entity.methods[method]?.aspects?.[aspect as never] as
    | { overall?: string; positive?: string; negative?: string }
    | undefined;
  const goldRefs = entity.gold?.aspects?.[aspect as never] as
    | string[]
    | undefined;
  const [showGold, setShowGold] = useState(false);

  const hasContent =
    cell && (cell.overall || cell.positive || cell.negative);

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-sm font-semibold capitalize text-slate-800">
          {ASPECT_LABEL[aspect] ?? aspect}
        </h3>
        {goldRefs && goldRefs.length > 0 ? (
          <button
            type="button"
            onClick={() => setShowGold((v) => !v)}
            className="text-[11px] font-medium text-amber-600 hover:underline"
          >
            {showGold ? "hide gold" : `gold (${goldRefs.length})`}
          </button>
        ) : null}
      </div>

      {!hasContent ? (
        <p className="text-xs italic text-slate-400">No output for this aspect.</p>
      ) : cell?.overall !== undefined ? (
        <p className="text-sm leading-relaxed text-slate-700">{cell.overall}</p>
      ) : (
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
          <div className="rounded-md bg-emerald-50 p-2 ring-1 ring-emerald-100">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-emerald-600">
              Positive
            </div>
            <p className="text-xs leading-relaxed text-slate-700">
              {cell?.positive || (
                <span className="italic text-slate-400">none</span>
              )}
            </p>
          </div>
          <div className="rounded-md bg-rose-50 p-2 ring-1 ring-rose-100">
            <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-rose-600">
              Negative
            </div>
            <p className="text-xs leading-relaxed text-slate-700">
              {cell?.negative || (
                <span className="italic text-slate-400">none</span>
              )}
            </p>
          </div>
        </div>
      )}

      {showGold && goldRefs ? (
        <div className="mt-3 space-y-1 rounded-md border border-amber-200 bg-amber-50 p-2">
          {goldRefs.map((ref, i) => (
            <p key={i} className="text-[11px] leading-relaxed text-amber-900">
              <span className="mr-1 font-mono text-amber-500">#{i + 1}</span>
              {ref}
            </p>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function SpaceExplore() {
  const [data, setData] = useState<SpaceData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [entityId, setEntityId] = useState("");
  const [method, setMethod] = useState<MethodId>("m3");
  const [query, setQuery] = useState("");

  useEffect(() => {
    loadSpaceData()
      .then((d) => {
        setData(d);
        if (d.entities.length > 0) setEntityId(d.entities[0].entity_id);
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, []);

  const entities = data?.entities ?? [];
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return entities;
    return entities.filter(
      (e) =>
        sampleLabel(entities.indexOf(e)).toLowerCase().includes(q) ||
        e.entity_id.toLowerCase().includes(q),
    );
  }, [entities, query]);

  const entity = useMemo(
    () => entities.find((e) => e.entity_id === entityId),
    [entities, entityId],
  );

  if (loading) return <div className="text-sm text-slate-500">Loading…</div>;
  if (error)
    return (
      <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
        {error}
      </div>
    );
  if (!data) return null;

  const meta = methodMeta(data, method);

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-[300px_1fr]">
      <aside className="space-y-4">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Search samples
          </label>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="sample or id..."
            className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
          />
        </div>
        <div className="max-h-[60vh] overflow-y-auto rounded-md border border-slate-200 bg-white">
          {filtered.map((e) => {
            const index = Math.max(0, entities.findIndex((item) => item.entity_id === e.entity_id));
            return (
            <button
              key={e.entity_id}
              type="button"
              onClick={() => setEntityId(e.entity_id)}
              className={`block w-full border-b border-slate-100 px-3 py-2 text-left text-sm last:border-b-0 ${
                e.entity_id === entityId
                  ? "bg-[var(--surface-container)] font-medium text-[var(--primary)]"
                  : "text-slate-700 hover:bg-[var(--surface-container-low)]"
              }`}
            >
              <div className="truncate">{sampleLabel(index)}</div>
              <div className="text-[11px] text-slate-400">
                Anonymous group - {e.split}
              </div>
            </button>
            );
          })}
          {filtered.length === 0 ? (
            <div className="px-3 py-2 text-sm text-slate-400">No matches.</div>
          ) : null}
        </div>
      </aside>

      <main className="space-y-5">
        {entity ? (
          <>
            <div>
              <h2 className="text-xl font-bold text-slate-900">
                {sampleLabel(Math.max(0, entities.findIndex((item) => item.entity_id === entity.entity_id)))}
              </h2>
              <p className="text-xs text-slate-500">
                Anonymous benchmark group - {entity.split}
              </p>
            </div>

            <div>
              <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
                Method
              </label>
              <div className="inline-flex flex-wrap gap-1 rounded-md border border-slate-300 bg-white p-0.5">
                {METHOD_IDS.map((m) => {
                  const mm = methodMeta(data, m);
                  return (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setMethod(m)}
                      className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                        method === m
                          ? "bg-[var(--primary)] text-white"
                          : "text-slate-600 hover:bg-[var(--surface-container)]"
                      }`}
                      title={mm.desc}
                    >
                      {mm.short}
                    </button>
                  );
                })}
              </div>
              <p className="mt-1.5 text-xs text-slate-500">{meta.desc}</p>
            </div>

            <div
              className={`rounded-xl border p-4 ring-1 ${COLOR_BG_LIGHT[meta.color]} ${COLOR_RING[meta.color]}`}
            >
              <div className="mb-1 flex items-center gap-2">
                <span
                  className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
                />
                <span
                  className={`text-xs font-semibold uppercase tracking-wide ${COLOR_TEXT[meta.color]}`}
                >
                  {meta.short} overall summary
                </span>
              </div>
              <p className="text-sm leading-relaxed text-slate-800">
                {entity.methods[method]?.overall || (
                  <span className="italic text-slate-400">
                    No overall summary.
                  </span>
                )}
              </p>
            </div>

            <GoldPanel entity={entity} />

            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {data.aspects.map((a) => (
                <AspectCard
                  key={a}
                  data={data}
                  entity={entity}
                  method={method}
                  aspect={a}
                />
              ))}
            </div>
          </>
        ) : (
          <div className="text-sm text-slate-500">Select a sample to begin.</div>
        )}
      </main>
    </div>
  );
}
