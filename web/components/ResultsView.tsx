"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  COLOR_BAR,
  COLOR_BG_LIGHT,
  COLOR_TEXT,
  HASOS_ASPECTS,
  METHOD_IDS,
  METHOD_META,
  METRIC_LABEL,
  SPACE_ASPECTS,
  type MethodId,
  type RougeComparison,
} from "@/lib/space";

const METRICS = ["rouge1", "rouge2", "rougeL"] as const;
type Split = "dev" | "test" | "all";
type Dataset = "space" | "hasos";

const DATASET_META: Record<
  Dataset,
  {
    label: string;
    file: string;
    aspects: readonly string[];
    note: string;
    setup: string;
    maxBar: number;
  }
> = {
  space: {
    label: "SPACE",
    file: "/data/rouge_space.json",
    aspects: SPACE_ASPECTS,
    note:
      "Mean ROUGE over 6 flat aspects, against human gold summaries (ROUGE-1.5.5 via pyrouge).",
    setup:
      "SPACE gold: 6 flat aspects x 3 refs, plus a non-aspectual general overall summary.",
    maxBar: 0.4,
  },
  hasos: {
    label: "HASOS",
    file: "/data/rouge_hasos.json",
    aspects: HASOS_ASPECTS,
    note:
      "Mean ROUGE over 4 HASOS parent aspects aggregated from the hotel taxonomy, against human gold summaries (ROUGE-1.5.5 via pyrouge).",
    setup:
      "HASOS gold: 4 parent aspects aggregated from 29 hotel sub-aspects.",
    maxBar: 0.25,
  },
};

function fmt(v: number | undefined): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "-";
  return v.toFixed(4);
}

function methodMeta(mid: MethodId) {
  return METHOD_META[mid];
}

export function ResultsView() {
  const [dataset, setDataset] = useState<Dataset>("space");
  const [split, setSplit] = useState<Split>("all");
  const [data, setData] = useState<RougeComparison | null>(null);
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const datasetMeta = DATASET_META[dataset];

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch(datasetMeta.file, { cache: "no-store" })
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
  }, [datasetMeta.file]);

  const cell = (mid: MethodId, key: string) =>
    data?.[mid]?.by_split?.[split]?.[key];

  const macro = useMemo(() => {
    if (!data) return null;
    return METHOD_IDS.map((m) => {
      const c = cell(m, "MACRO");
      return {
        method: m,
        rouge1: c?.rouge1 ?? NaN,
        rouge2: c?.rouge2 ?? NaN,
        rougeL: c?.rougeL ?? NaN,
      };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, split]);

  const general = useMemo(() => {
    if (!data) return null;
    const rows = METHOD_IDS.map((m) => {
      const c = cell(m, "GENERAL");
      return {
        method: m,
        rouge1: c?.rouge1 ?? NaN,
        rouge2: c?.rouge2 ?? NaN,
        rougeL: c?.rougeL ?? NaN,
        n: c?.n,
      };
    });
    return rows.some((r) => !Number.isNaN(r.rouge1)) ? rows : null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
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
    return datasetMeta.aspects.map((a) => {
      const row: Record<string, number> = {};
      for (const m of METHOD_IDS) {
        row[m] = cell(m, a)?.rouge1 ?? NaN;
      }
      return { aspect: a, scores: row };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, split, datasetMeta.aspects]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[var(--on-surface-variant)]">
            Dataset
          </label>
          <div className="inline-flex rounded-md border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-0.5">
            {(["space", "hasos"] as Dataset[]).map((d) => (
              <button
                key={d}
                type="button"
                onClick={() => setDataset(d)}
                className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                  dataset === d
                    ? "bg-[var(--primary)] text-white"
                    : "text-[var(--on-surface-variant)] hover:bg-[var(--surface-container)]"
                }`}
              >
                {DATASET_META[d].label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[var(--on-surface-variant)]">
            Split
          </label>
          <div className="inline-flex rounded-md border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-0.5">
            {(["dev", "test", "all"] as Split[]).map((s) => (
              <button
                key={s}
                type="button"
                onClick={() => setSplit(s)}
                className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                  split === s
                    ? "bg-[var(--primary)] text-white"
                    : "text-[var(--on-surface-variant)] hover:bg-[var(--surface-container)]"
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
      {loading ? <div className="text-sm text-slate-500">Loading...</div> : null}

      {macro && bestPerMetric && perAspect ? (
        <>
          <section className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {METHOD_IDS.map((m) => {
              const meta = methodMeta(m);
              const c = cell(m, "MACRO");
              const g = cell(m, "GENERAL");
              const vals = [c?.rouge1, c?.rouge2, c?.rougeL];
              return (
                <div
                  key={m}
                  className={`rounded-xl border border-[var(--outline-variant)] p-4 shadow-[var(--shadow-soft)] ${COLOR_BG_LIGHT[meta.color]}`}
                >
                  <div className="mb-2 flex items-center gap-2">
                    <span
                      className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
                    />
                    <span className={`text-sm font-semibold ${COLOR_TEXT[meta.color]}`}>
                      {meta.short}
                    </span>
                  </div>
                  <p className="mb-3 text-[11px] leading-snug text-slate-600">
                    {meta.desc}
                  </p>
                  <div className="space-y-1.5">
                    {METRICS.map((metric, i) => {
                      const v = vals[i] ?? NaN;
                      const width = Number.isNaN(v)
                        ? 0
                        : Math.min(100, (v / datasetMeta.maxBar) * 100);
                      return (
                        <div key={metric} className="text-[11px]">
                          <div className="mb-0.5 flex justify-between text-slate-500">
                            <span>{METRIC_LABEL[metric]}</span>
                            <span className="font-mono">{fmt(v)}</span>
                          </div>
                          <div className="h-2 overflow-hidden rounded bg-white/70">
                            <div
                              className={`h-full ${COLOR_BAR[meta.color]}`}
                              style={{ width: `${width}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  {g ? (
                    <div className="mt-3 border-t border-slate-200/70 pt-2 text-[11px] text-slate-600">
                      <div className="flex justify-between">
                        <span className="font-medium">Overall R1</span>
                        <span className="font-mono">{fmt(g.rouge1)}</span>
                      </div>
                    </div>
                  ) : null}
                </div>
              );
            })}
          </section>

          <section className="rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-5 shadow-[var(--shadow-soft)]">
            <h2 className="mb-1 text-lg font-semibold text-[var(--primary)]">
              Macro ROUGE F1 - {datasetMeta.label} ({split})
            </h2>
            <p className="mb-4 text-xs text-[var(--on-surface-variant)]">
              {datasetMeta.note}
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[var(--outline-variant)] text-left text-xs uppercase tracking-wide text-[var(--on-surface-variant)]">
                    <th className="py-2 pr-3 font-semibold">Method</th>
                    {METRICS.map((m) => (
                      <th key={m} className="py-2 pr-3 text-right font-semibold">
                        {METRIC_LABEL[m]}
                      </th>
                    ))}
                    <th className="py-2 text-right font-semibold">R1 vs M1</th>
                  </tr>
                </thead>
                <tbody>
                  {macro.map((row) => {
                    const meta = methodMeta(row.method);
                    const isM1 = row.method === "m1";
                    const delta = macro[0].rouge1
                      ? row.rouge1 - macro[0].rouge1
                      : 0;
                    return (
                      <tr
                        key={row.method}
                        className="border-b border-[var(--surface-container)] last:border-b-0"
                      >
                        <td className="py-2 pr-3">
                          <div className="flex items-center gap-2">
                            <span
                              className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
                            />
                            <div>
                              <div className="font-medium text-[var(--primary)]">
                                {meta.short}
                              </div>
                              <div className="text-[11px] text-[var(--on-surface-variant)]">
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
                                  : "text-[var(--on-surface)]"
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
                          {isM1
                            ? "-"
                            : `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>

          {general ? (
            <section className="rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-5 shadow-[var(--shadow-soft)]">
              <h2 className="mb-1 text-lg font-semibold text-[var(--primary)]">
                Overall summary ROUGE F1 - {datasetMeta.label} ({split})
              </h2>
              <p className="mb-4 text-xs text-[var(--on-surface-variant)]">
                Entity-level overall summary scored against the {datasetMeta.label}{" "}
                <code className="rounded bg-[var(--surface-container)] px-1">general</code> gold
                reference. There is no sentiment-level gold, so positive /
                negative summaries are shown in Explore but not scored.
              </p>
              <div className="space-y-3">
                {METRICS.map((metric) => {
                  const vals = general.map((r) => r[metric]);
                  const max = Math.max(
                    ...vals.filter((v) => !Number.isNaN(v)),
                    0.0001,
                  );
                  return (
                    <div key={metric}>
                      <div className="mb-1 text-sm font-semibold text-[var(--primary)]">
                        {METRIC_LABEL[metric]}
                      </div>
                      <div className="space-y-1.5">
                        {general.map((r) => {
                          const meta = methodMeta(r.method);
                          const v = r[metric];
                          const width = Number.isNaN(v) ? 0 : (v / max) * 100;
                          return (
                            <div
                              key={r.method}
                              className="flex items-center gap-2 text-xs"
                            >
                              <div className="w-8 text-right font-mono text-[var(--on-surface-variant)]">
                                {meta.short}
                              </div>
                              <div className="relative h-5 flex-1 overflow-hidden rounded bg-[var(--surface-container)]">
                                <div
                                  className={`h-full ${COLOR_BAR[meta.color]}`}
                                  style={{ width: `${width}%` }}
                                />
                                <div className="absolute inset-0 flex items-center px-2 font-mono text-[11px] text-[var(--on-surface)]">
                                  {fmt(v)}
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
          ) : null}

          <section className="rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-5 shadow-[var(--shadow-soft)]">
            <h2 className="mb-1 text-lg font-semibold text-[var(--primary)]">
              Per-aspect ROUGE-1 - {datasetMeta.label} ({split})
            </h2>
            <p className="mb-4 text-xs text-[var(--on-surface-variant)]">
              F1 on each aspect, 4 methods overlaid.
            </p>
            <div className="space-y-4">
              {perAspect.map((row) => {
                const values = METHOD_IDS.map((m) => row.scores[m] ?? NaN);
                const max = Math.max(
                  ...values.filter((v) => !Number.isNaN(v)),
                  0.0001,
                );
                return (
                  <div key={row.aspect}>
                    <div className="mb-1 flex items-baseline justify-between">
                      <div className="text-sm font-semibold capitalize text-[var(--primary)]">
                        {ASPECT_LABEL[row.aspect] ?? row.aspect}
                      </div>
                      <div className="text-[11px] text-[var(--on-surface-variant)]">
                        max {fmt(max)}
                      </div>
                    </div>
                    <div className="space-y-1.5">
                      {METHOD_IDS.map((m) => {
                        const v = row.scores[m] ?? NaN;
                        const width = max > 0 ? (v / max) * 100 : 0;
                        const meta = methodMeta(m);
                        return (
                          <div
                            key={m}
                            className="flex items-center gap-2 text-xs"
                          >
                            <div className="w-8 text-right font-mono text-[var(--on-surface-variant)]">
                              {meta.short}
                            </div>
                            <div className="relative h-5 flex-1 overflow-hidden rounded bg-[var(--surface-container)]">
                              <div
                                className={`h-full ${COLOR_BAR[meta.color]} transition-all`}
                                style={{ width: `${width}%` }}
                              />
                              <div className="absolute inset-0 flex items-center px-2 font-mono text-[11px] text-[var(--on-surface)]">
                                {Number.isNaN(v) ? "-" : fmt(v)}
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

          <section className="rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-container-low)] p-4 text-xs text-[var(--on-surface-variant)]">
            <strong>Setup.</strong> ROUGE-1.5.5 via pyrouge + Strawberry Perl.{" "}
            {datasetMeta.setup} All 4 methods share identical SemAE sentence
            selection; they differ only in how the selected evidence is rendered.
            M3 and M4 concatenate positive + negative generated summaries per
            aspect before scoring.
          </section>
        </>
      ) : null}
    </div>
  );
}
