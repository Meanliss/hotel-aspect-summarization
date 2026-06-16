"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  DATASET_META,
  METRIC_LABEL,
  METHOD_META,
  type Dataset,
  type RougeResults,
  type Split,
} from "@/lib/results";

const METHODS = ["m1", "m2", "m3", "m4"] as const;
const METRICS = ["rouge1", "rouge2", "rougeL"] as const;

function fmt(v: number | undefined): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return v.toFixed(4);
}

function pct(v: number | undefined): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

const COLOR_BAR: Record<string, string> = {
  slate: "bg-slate-500",
  sky: "bg-sky-500",
  emerald: "bg-emerald-500",
  violet: "bg-violet-500",
};

const COLOR_TEXT: Record<string, string> = {
  slate: "text-slate-600",
  sky: "text-sky-600",
  emerald: "text-emerald-600",
  violet: "text-violet-600",
};

const COLOR_BG_LIGHT: Record<string, string> = {
  slate: "bg-slate-50",
  sky: "bg-sky-50",
  emerald: "bg-emerald-50",
  violet: "bg-violet-50",
};

const COLOR_RING: Record<string, string> = {
  slate: "ring-slate-300",
  sky: "ring-sky-300",
  emerald: "ring-emerald-300",
  violet: "ring-violet-300",
};

export function ResultsView() {
  const [dataset, setDataset] = useState<Dataset>("space");
  const [split, setSplit] = useState<Split>("all");
  const [data, setData] = useState<RougeResults | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    setData(null);
    fetch(`/data/${DATASET_META[dataset].file}`, { cache: "no-store" })
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load (${r.status})`);
        return r.json();
      })
      .then((j) => {
        if (!cancelled) setData(j);
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  const macro = useMemo(() => {
    if (!data) return null;
    return METHODS.map((m) => {
      const cell = data[m].by_split[split]?.["MACRO"];
      return {
        method: m,
        rouge1: cell?.rouge1 ?? NaN,
        rouge2: cell?.rouge2 ?? NaN,
        rougeL: cell?.rougeL ?? NaN,
      };
    });
  }, [data, split]);

  const bestPerMetric = useMemo(() => {
    if (!macro) return null;
    const out: Record<string, string> = {};
    for (const metric of METRICS) {
      let best = -Infinity;
      let winner = "";
      for (const row of macro) {
        if (row[metric] > best) {
          best = row[metric];
          winner = row.method;
        }
      }
      out[metric] = winner;
    }
    return out;
  }, [macro]);

  const perAspect = useMemo(() => {
    if (!data) return null;
    const aspects = DATASET_META[dataset].aspects;
    return aspects.map((a) => {
      const row: Record<string, number> = {};
      for (const m of METHODS) {
        row[m] = data[m].by_split[split]?.[a]?.rouge1 ?? NaN;
      }
      return { aspect: a, scores: row };
    });
  }, [data, dataset, split]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Dataset
          </label>
          <div className="inline-flex rounded-md border border-slate-300 bg-white p-0.5">
            {(["space", "hasos"] as Dataset[]).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDataset(d)}
                className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                  dataset === d
                    ? "bg-indigo-600 text-white"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {DATASET_META[d].label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
            Split
          </label>
          <div className="inline-flex rounded-md border border-slate-300 bg-white p-0.5">
            {(["dev", "test", "all"] as Split[]).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSplit(s)}
                className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                  split === s
                    ? "bg-indigo-600 text-white"
                    : "text-slate-600 hover:bg-slate-100"
                }`}
              >
                {s.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
      </div>

      {error ? (
        <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
          {error}
        </div>
      ) : null}
      {loading ? (
        <div className="text-sm text-slate-500">Loading…</div>
      ) : null}

      {macro && bestPerMetric && perAspect ? (
        <>
          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-1 text-lg font-semibold text-slate-900">
              Macro ROUGE F1 — {DATASET_META[dataset].label} ({split})
            </h2>
            <p className="mb-4 text-xs text-slate-500">
              Mean ROUGE over {DATASET_META[dataset].aspects.length} aspects,
              against human gold summaries (ROUGE-1.5.5 via pyrouge).
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-200 text-left text-xs uppercase tracking-wide text-slate-500">
                    <th className="py-2 pr-3 font-semibold">Method</th>
                    {METRICS.map((m) => (
                      <th key={m} className="py-2 pr-3 text-right font-semibold">
                        {METRIC_LABEL[m]}
                      </th>
                    ))}
                    <th className="py-2 text-right font-semibold">Δ R1 vs M1</th>
                  </tr>
                </thead>
                <tbody>
                  {macro.map((row) => {
                    const meta = METHOD_META[row.method];
                    const isM1 = row.method === "m1";
                    const delta = macro[0].rouge1 ? row.rouge1 - macro[0].rouge1 : 0;
                    return (
                      <tr
                        key={row.method}
                        className="border-b border-slate-100 last:border-b-0"
                      >
                        <td className="py-2 pr-3">
                          <div className="flex items-center gap-2">
                            <span
                              className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
                            />
                            <div>
                              <div className="font-medium text-slate-800">
                                {meta.short}
                              </div>
                              <div className="text-[11px] text-slate-500">
                                {meta.desc}
                              </div>
                            </div>
                          </div>
                        </td>
                        {METRICS.map((metric) => {
                          const isBest = bestPerMetric[metric] === row.method;
                          return (
                            <td
                              key={metric}
                              className={`py-2 pr-3 text-right font-mono ${
                                isBest
                                  ? `font-semibold ${COLOR_TEXT[meta.color]}`
                                  : "text-slate-700"
                              }`}
                            >
                              {fmt(row[metric])}
                              {isBest ? (
                                <span
                                  className={`ml-1 text-[10px] uppercase ${COLOR_TEXT[meta.color]}`}
                                >
                                  best
                                </span>
                              ) : null}
                            </td>
                          );
                        })}
                        <td
                          className={`py-2 text-right font-mono ${
                            isM1
                              ? "text-slate-400"
                              : delta > 0
                              ? "text-emerald-600"
                              : delta < 0
                              ? "text-rose-600"
                              : "text-slate-500"
                          }`}
                        >
                          {isM1 ? "—" : `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-1 text-lg font-semibold text-slate-900">
              Per-aspect ROUGE-1 — {DATASET_META[dataset].label} ({split})
            </h2>
            <p className="mb-4 text-xs text-slate-500">
              F1 on each aspect, 4 methods overlaid.
            </p>
            <div className="space-y-4">
              {perAspect.map((row) => {
                const values = METHODS.map(
                  (m) => row.scores[m] ?? NaN,
                );
                const max = Math.max(...values.filter((v) => !Number.isNaN(v)));
                return (
                  <div key={row.aspect}>
                    <div className="mb-1 flex items-baseline justify-between">
                      <div className="text-sm font-semibold capitalize text-slate-700">
                        {ASPECT_LABEL[row.aspect] ?? row.aspect}
                      </div>
                      <div className="text-[11px] text-slate-400">
                        max {fmt(max)}
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      {METHODS.map((m, i) => {
                        const v = row.scores[m] ?? NaN;
                        const width = max > 0 ? (v / max) * 100 : 0;
                        const meta = METHOD_META[m];
                        return (
                          <div
                            key={m}
                            className="flex items-center gap-2 text-xs"
                          >
                            <div className="w-8 text-right font-mono text-slate-500">
                              {meta.short}
                            </div>
                            <div className="relative h-5 flex-1 overflow-hidden rounded bg-slate-100">
                              <div
                                className={`h-full ${COLOR_BAR[meta.color]} transition-all`}
                                style={{ width: `${width}%` }}
                              />
                              <div className="absolute inset-0 flex items-center px-2 font-mono text-[11px] text-slate-700">
                                {Number.isNaN(v) ? "—" : fmt(v)}
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="mb-3 text-lg font-semibold text-slate-900">
              Methods
            </h2>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
              {METHODS.map((m) => {
                const meta = METHOD_META[m];
                return (
                  <div
                    key={m}
                    className={`rounded-lg border p-3 ring-1 ${COLOR_BG_LIGHT[meta.color]} ${COLOR_RING[meta.color]}`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
                      />
                      <div
                        className={`text-sm font-semibold ${COLOR_TEXT[meta.color]}`}
                      >
                        {meta.label}
                      </div>
                    </div>
                    <p className="mt-1 text-xs text-slate-600">{meta.desc}</p>
                  </div>
                );
              })}
            </div>
          </section>

          <section className="rounded-xl border border-slate-200 bg-slate-100 p-4 text-xs text-slate-600">
            <strong>Setup.</strong> ROUGE-1.5.5 via pyrouge + Strawberry Perl.
            SPACE gold: 6 flat aspects × 3 refs. HASOS gold: 4 parent aspects
            aggregated from 29 sub-aspects. All 4 methods share identical SemAE
            sentence selection; they differ only in how the selected evidence
            is rendered. M3 and M4 concatenate positive + negative generated
            summaries per aspect before scoring.
          </section>
        </>
      ) : null}
    </div>
  );
}
