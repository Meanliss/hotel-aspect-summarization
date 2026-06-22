"use client";

import { useEffect, useMemo, useState } from "react";
import { COLOR_BAR, COLOR_TEXT, METHOD_IDS, type MethodId } from "@/lib/space";
import {
  type Dataset,
  type MethodSweep,
  type Phase,
  type SweepData,
  type SweepPoint,
  VERDICT_LABEL,
} from "@/lib/sweep";

const DATASETS: Dataset[] = ["space", "hasos"];
const PHASES: Phase[] = ["threshold", "tokabs"];

function fmt(v: number | null | undefined, nd = 4): string {
  if (v === undefined || v === null || Number.isNaN(v)) return "-";
  return v.toFixed(nd);
}

function fmtValue(v: number | string): string {
  return typeof v === "number" ? String(v) : v;
}

const VERDICT_STYLE: Record<string, string> = {
  default_optimal: "border-emerald-200 bg-emerald-50 text-emerald-800",
  switch: "border-amber-200 bg-amber-50 text-amber-800",
  keep_default_coverage_artifact:
    "border-slate-200 bg-slate-50 text-slate-700",
};

function MethodBlock({
  method,
  sweep,
  maxBar,
}: {
  method: MethodId;
  sweep: MethodSweep;
  maxBar: number;
}) {
  const meta = METHOD_META_LOOKUP[method];
  const verdict = sweep.verdict;
  return (
    <div className="rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-4 shadow-[var(--shadow-soft)]">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
          />
          <span className={`text-sm font-semibold ${COLOR_TEXT[meta.color]}`}>
            {meta.label}
          </span>
        </div>
        {verdict ? (
          <span
            className={`rounded-full border px-2.5 py-0.5 text-[11px] font-medium ${VERDICT_STYLE[verdict.status]}`}
          >
            {VERDICT_LABEL[verdict.status]}
          </span>
        ) : null}
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[var(--outline-variant)] text-left text-xs uppercase tracking-wide text-[var(--on-surface-variant)]">
              <th className="py-2 pr-3 font-semibold">Value</th>
              <th className="py-2 pr-3 text-right font-semibold">R1</th>
              <th className="py-2 pr-3 text-right font-semibold">R2</th>
              <th className="py-2 pr-3 text-right font-semibold">RL</th>
              <th className="py-2 pr-3 text-right font-semibold">Coverage</th>
              <th className="py-2 text-right font-semibold">ΔR1 vs default</th>
            </tr>
          </thead>
          <tbody>
            {sweep.points.map((p: SweepPoint) => {
              const defR1 = sweep.points.find((x) => x.is_default)?.rouge1;
              const delta =
                defR1 !== undefined && !p.is_default
                  ? p.rouge1 - defR1
                  : null;
              const barW = Math.min(100, (p.rouge1 / maxBar) * 100);
              return (
                <tr
                  key={fmtValue(p.value)}
                  className={`border-b border-[var(--surface-container)] last:border-b-0 ${
                    p.is_best ? "bg-emerald-50/40" : ""
                  }`}
                >
                  <td className="py-2 pr-3 font-mono">
                    {fmtValue(p.value)}
                    {p.is_default ? (
                      <span className="ml-1 text-[10px] uppercase text-[var(--on-surface-variant)]">
                        default
                      </span>
                    ) : null}
                    {p.is_best ? (
                      <span className="ml-1 text-[10px] font-semibold uppercase text-emerald-700">
                        best
                      </span>
                    ) : null}
                  </td>
                  <td className="py-2 pr-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      <div className="h-2 w-16 overflow-hidden rounded bg-[var(--surface-container)]">
                        <div
                          className={`h-full ${COLOR_BAR[meta.color]}`}
                          style={{ width: `${barW}%` }}
                        />
                      </div>
                      <span
                        className={`font-mono ${p.is_best ? "font-semibold text-emerald-700" : ""}`}
                      >
                        {fmt(p.rouge1, 5)}
                      </span>
                    </div>
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-[var(--on-surface-variant)]">
                    {fmt(p.rouge2, 5)}
                  </td>
                  <td className="py-2 pr-3 text-right font-mono text-[var(--on-surface-variant)]">
                    {fmt(p.rougeL, 5)}
                  </td>
                  <td className="py-2 pr-3 text-right font-mono">
                    {p.coverage === null ? (
                      "-"
                    ) : (
                      <span
                        className={
                          p.coverage < 0.9
                            ? "text-rose-600"
                            : "text-[var(--on-surface)]"
                        }
                      >
                        {(p.coverage * 100).toFixed(0)}%
                      </span>
                    )}
                  </td>
                  <td
                    className={`py-2 text-right font-mono ${
                      delta === null
                        ? "text-slate-400"
                        : delta > 0
                          ? "text-emerald-600"
                          : delta < 0
                            ? "text-rose-600"
                            : "text-slate-500"
                    }`}
                  >
                    {delta === null
                      ? p.is_default
                        ? "-"
                        : ""
                      : `${delta >= 0 ? "+" : ""}${delta.toFixed(5)}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {verdict ? (
        <p className="mt-3 text-[11px] leading-snug text-[var(--on-surface-variant)]">
          {verdict.status === "default_optimal" ? (
            <>
              Default <strong>{fmtValue(verdict.default)}</strong> has the
              highest macro ROUGE-1 ({fmt(verdict.default_r1, 5)}) in the grid —
              no swept value beats it.
            </>
          ) : verdict.status === "switch" ? (
            <>
              <strong>{fmtValue(verdict.best)}</strong> beats default{" "}
              <strong>{fmtValue(verdict.default)}</strong> by ΔR1{" "}
              {verdict.delta >= 0 ? "+" : ""}
              {verdict.delta.toFixed(5)} at equal coverage (
              {verdict.best_cov === null || verdict.best_cov === undefined
                ? "-"
                : `${(verdict.best_cov * 100).toFixed(0)}%`}{" "}
              vs{" "}
              {verdict.default_cov === null ||
              verdict.default_cov === undefined
                ? "-"
                : `${(verdict.default_cov * 100).toFixed(0)}%`}
              ) → recommend switching.
            </>
          ) : (
            <>
              {fmtValue(verdict.best)} scores higher raw, but only by dropping
              coverage (
              {verdict.best_cov === null || verdict.best_cov === undefined
                ? "-"
                : `${(verdict.best_cov * 100).toFixed(0)}%`}{" "}
              vs default{" "}
              {verdict.default_cov === null ||
              verdict.default_cov === undefined
                ? "-"
                : `${(verdict.default_cov * 100).toFixed(0)}%`}
              ) → the gain is a coverage artifact, keep default{" "}
              <strong>{fmtValue(verdict.default)}</strong>.
            </>
          )}
        </p>
      ) : null}
    </div>
  );
}

// Local lookup so this component does not depend on METHOD_META export shape.
const METHOD_META_LOOKUP: Record<MethodId, { label: string; color: string }> = {
  m1: { label: "M1 — Extractive (SemAE)", color: "slate" },
  m2: { label: "M2 — Abstractive (no sentiment)", color: "sky" },
  m3: { label: "M3 — Sentiment split · Keyword", color: "emerald" },
  m4: { label: "M4 — Sentiment split · BERT-ABSA", color: "violet" },
};

export function OptimalityView() {
  const [data, setData] = useState<SweepData | null>(null);
  const [dataset, setDataset] = useState<Dataset>("space");
  const [phase, setPhase] = useState<Phase>("threshold");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    fetch("/data/sweep.json", { cache: "no-store" })
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
  }, []);

  const phaseBlock = data?.datasets?.[dataset]?.[phase];
  const phaseMeta = data?.phase_meta?.[phase];
  const datasetMeta = data?.dataset_meta?.[dataset];
  const maxBar = datasetMeta?.maxBar ?? 0.4;

  const availableMethods = useMemo(() => {
    if (!phaseBlock) return [];
    return METHOD_IDS.filter((m) => phaseBlock.methods[m]);
  }, [phaseBlock]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[var(--on-surface-variant)]">
            Dataset
          </label>
          <div className="inline-flex rounded-md border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-0.5">
            {DATASETS.map((d) => (
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
                {data?.dataset_meta?.[d]?.label ?? d.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[var(--on-surface-variant)]">
            Parameter
          </label>
          <div className="inline-flex rounded-md border border-[var(--outline-variant)] bg-[var(--surface-bright)] p-0.5">
            {PHASES.map((p) => (
              <button
                key={p}
                type="button"
                onClick={() => setPhase(p)}
                className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                  phase === p
                    ? "bg-[var(--primary)] text-white"
                    : "text-[var(--on-surface-variant)] hover:bg-[var(--surface-container)]"
                }`}
              >
                {data?.phase_meta?.[p]?.label ?? p}
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

      {data && phaseMeta ? (
        <>
          <section className="rounded-xl border border-[var(--outline-variant)] bg-[var(--surface-container-low)] p-4 text-xs text-[var(--on-surface-variant)]">
            <div className="mb-1 text-sm font-semibold text-[var(--primary)]">
              {phaseMeta.label}{" "}
              <code className="rounded bg-[var(--surface-container)] px-1 font-mono text-[11px]">
                {phaseMeta.param}
              </code>
            </div>
            <p className="leading-snug">{phaseMeta.note}</p>
            <p className="mt-2">
              <strong>Decision metric:</strong> {data.decision_metric}.
            </p>
          </section>

          {availableMethods.length === 0 ? (
            <div className="rounded-md border border-[var(--outline-variant)] bg-[var(--surface-bright)] px-4 py-6 text-center text-sm text-[var(--on-surface-variant)]">
              No sweep cells for this dataset/parameter yet.
            </div>
          ) : (
            <div className="space-y-4">
              {availableMethods.map((m) => (
                <MethodBlock
                  key={m}
                  method={m}
                  sweep={phaseBlock!.methods[m]!}
                  maxBar={maxBar}
                />
              ))}
            </div>
          )}
        </>
      ) : null}
    </div>
  );
}
