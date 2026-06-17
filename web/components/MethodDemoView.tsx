"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  COLOR_BAR,
  COLOR_BG_LIGHT,
  COLOR_RING,
  COLOR_TEXT,
  type MethodId,
} from "@/lib/space";

type DemoAspect = "building" | "food" | "rooms" | "service";
type DemoMethodId = MethodId;
type EvidenceMode = "base" | "keyword" | "bert";

interface DemoEvidence {
  sentence: string;
  rank?: number;
  score?: number;
  source_review_id?: string;
  source_sentence_index?: number;
  matched_aspect_seed: string[];
  sentiment_label: string;
  matched_sentiment_keywords: string[];
  was_truncated: boolean;
}

interface DemoSynthesis {
  summary: string;
  status: string;
  evidence_count: number;
  evidence_used: number;
  copied_from_evidence: boolean;
  evidence: DemoEvidence[];
}

interface DemoMethodMeta {
  label: string;
  short: string;
  desc: string;
  color: string;
}

interface DemoAspectBlock {
  aspect: DemoAspect;
  gold: string[];
  shared_evidence: DemoEvidence[];
  keyword_evidence: DemoEvidence[];
  bert_evidence: DemoEvidence[];
  methods: {
    m1: { output: string; evidence: DemoEvidence[] };
    m2: { output: string; synthesis: DemoSynthesis };
    m3: {
      positive: string;
      negative: string;
      positive_synthesis: DemoSynthesis;
      negative_synthesis: DemoSynthesis;
    };
    m4: {
      positive: string;
      negative: string;
      positive_synthesis: DemoSynthesis;
      negative_synthesis: DemoSynthesis;
    };
  };
}

interface DemoData {
  entity: {
    entity_id: string;
    entity_name: string;
    split: string;
    review_count: number;
    sample_reviews: Array<{
      review_id: string;
      rating: number;
      sentences: string[];
    }>;
    gold_overall: string[];
  };
  method_meta: Record<DemoMethodId, DemoMethodMeta>;
  overall: Record<DemoMethodId, string>;
  aspects: DemoAspectBlock[];
}

interface Stage {
  title: string;
  input: string;
  operation: string;
  output: string;
}

const METHOD_IDS: DemoMethodId[] = ["m1", "m2", "m3", "m4"];

function fmtScore(value?: number) {
  if (value === undefined || value === null || Number.isNaN(value)) return "";
  return value.toFixed(6);
}

function methodMeta(data: DemoData, method: DemoMethodId) {
  return data.method_meta[method];
}

function evidenceModeFor(method: DemoMethodId): EvidenceMode {
  if (method === "m3") return "keyword";
  if (method === "m4") return "bert";
  return "base";
}

function EmptyText({ label = "none" }: { label?: string }) {
  return <span className="italic text-slate-400">{label}</span>;
}

function MethodBadge({
  data,
  method,
}: {
  data: DemoData;
  method: DemoMethodId;
}) {
  const meta = methodMeta(data, method);
  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${COLOR_BAR[meta.color]}`}
      />
      <span
        className={`text-xs font-semibold uppercase tracking-wide ${COLOR_TEXT[meta.color]}`}
      >
        {meta.short}
      </span>
    </div>
  );
}

function pipelineStages(method: DemoMethodId, aspectLabel: string): Stage[] {
  const shared = [
    {
      title: "Raw review pool",
      input: "100 River Hotel reviews",
      operation: "Flatten review sentences and keep source ids for audit.",
      output: "Sentence pool with review_id and sentence index.",
    },
    {
      title: "SemAE aspect ranking",
      input: `${aspectLabel} seeds + sentence embeddings`,
      operation:
        "Score candidate sentences against the aspect prototype with KL-distance.",
      output: "Ranked evidence shared by all four methods.",
    },
  ];

  if (method === "m1") {
    return [
      ...shared,
      {
        title: "Extractive write",
        input: "Top SemAE sentences",
        operation: "Keep the selected text unchanged.",
        output: "Readable evidence list, but no rewrite or polarity split.",
      },
    ];
  }

  if (method === "m2") {
    return [
      ...shared,
      {
        title: "FLAN-T5 rewrite",
        input: "Top SemAE sentences",
        operation:
          "Compress evidence into one abstractive summary without sentiment labels.",
        output: "Cleaner text, but positive and negative signals are mixed.",
      },
    ];
  }

  if (method === "m3") {
    return [
      ...shared,
      {
        title: "Keyword polarity gate",
        input: "Ranked evidence",
        operation:
          "Use sentiment keywords to bucket evidence into positive and negative groups.",
        output: "Fast split, but generic words can mislabel neutral sentences.",
      },
      {
        title: "Rewrite each bucket",
        input: "Positive bucket + negative bucket",
        operation: "Run FLAN-T5 separately per polarity.",
        output: "Two summaries that expose contrast in the same aspect.",
      },
    ];
  }

  return [
    ...shared,
    {
      title: "BERT-ABSA polarity gate",
      input: "Ranked evidence + aspect context",
      operation:
        "Use an aspect-aware classifier to label each evidence sentence.",
      output: "More contextual positive, neutral, and negative buckets.",
    },
    {
      title: "Rewrite each bucket",
      input: "BERT positive bucket + BERT negative bucket",
      operation: "Run FLAN-T5 separately per polarity.",
      output: "A cleaner contrast when keyword rules are too broad.",
    },
  ];
}

function MethodTabs({
  data,
  value,
  onChange,
}: {
  data: DemoData;
  value: DemoMethodId;
  onChange: (method: DemoMethodId) => void;
}) {
  return (
    <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
      {METHOD_IDS.map((method) => {
        const meta = methodMeta(data, method);
        const active = method === value;
        return (
          <button
            key={method}
            type="button"
            onClick={() => onChange(method)}
            className={`min-h-24 rounded-lg border p-3 text-left transition ${
              active
                ? `${COLOR_BG_LIGHT[meta.color]} ${COLOR_RING[meta.color]} ring-2`
                : "border-slate-200 bg-white hover:bg-slate-50"
            }`}
          >
            <MethodBadge data={data} method={method} />
            <div className="mt-2 text-sm font-semibold text-slate-900">
              {meta.label.replace(`${meta.short} - `, "")}
            </div>
            <p className="mt-1 text-xs leading-snug text-slate-500">
              {meta.desc}
            </p>
          </button>
        );
      })}
    </div>
  );
}

function PipelineDiagram({
  data,
  method,
  aspect,
}: {
  data: DemoData;
  method: DemoMethodId;
  aspect: DemoAspectBlock;
}) {
  const meta = methodMeta(data, method);
  const stages = pipelineStages(
    method,
    ASPECT_LABEL[aspect.aspect] ?? aspect.aspect,
  );

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            How {meta.short} actually runs
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            The first two stages are shared. The method-specific branch begins
            after SemAE selects the evidence.
          </p>
        </div>
        <span
          className={`rounded px-2 py-1 text-xs font-semibold ring-1 ${COLOR_BG_LIGHT[meta.color]} ${COLOR_TEXT[meta.color]} ${COLOR_RING[meta.color]}`}
        >
          {ASPECT_LABEL[aspect.aspect] ?? aspect.aspect}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[repeat(4,minmax(0,1fr))]">
        {stages.map((stage, index) => (
          <div key={`${method}-${stage.title}`} className="relative">
            <div
              className={`pipeline-stage h-full rounded-lg border border-slate-200 bg-slate-50 p-3`}
              style={{ animationDelay: `${index * 110}ms` }}
            >
              <div className="mb-2 flex items-center justify-between gap-2">
                <span className="font-mono text-[11px] text-slate-400">
                  S{index + 1}
                </span>
                <span
                  className={`h-2 w-2 rounded-full ${COLOR_BAR[meta.color]}`}
                />
              </div>
              <h3 className="text-sm font-semibold text-slate-900">
                {stage.title}
              </h3>
              <dl className="mt-3 space-y-2 text-xs leading-relaxed">
                <div>
                  <dt className="font-semibold text-slate-500">Input</dt>
                  <dd className="text-slate-700">{stage.input}</dd>
                </div>
                <div>
                  <dt className="font-semibold text-slate-500">Operation</dt>
                  <dd className="text-slate-700">{stage.operation}</dd>
                </div>
                <div>
                  <dt className="font-semibold text-slate-500">Output</dt>
                  <dd className="text-slate-700">{stage.output}</dd>
                </div>
              </dl>
            </div>
            {index < stages.length - 1 ? (
              <div
                className={`pipeline-connector hidden xl:block ${COLOR_BAR[meta.color]}`}
              />
            ) : null}
          </div>
        ))}
      </div>
    </section>
  );
}

function EvidenceTable({
  rows,
  mode,
}: {
  rows: DemoEvidence[];
  mode: EvidenceMode;
}) {
  if (rows.length === 0) {
    return <p className="text-sm text-slate-400">No evidence rows.</p>;
  }

  return (
    <div className="overflow-x-auto rounded-md border border-slate-200 bg-white">
      <table className="w-full min-w-[780px] text-left text-xs">
        <thead className="border-b border-slate-200 bg-slate-50 text-[11px] uppercase tracking-wide text-slate-500">
          <tr>
            <th className="w-14 px-3 py-2 font-semibold">Rank</th>
            <th className="px-3 py-2 font-semibold">Evidence sentence</th>
            <th className="w-28 px-3 py-2 font-semibold">Score</th>
            <th className="w-32 px-3 py-2 font-semibold">Source</th>
            <th className="w-32 px-3 py-2 font-semibold">Seed</th>
            {mode !== "base" ? (
              <th className="w-36 px-3 py-2 font-semibold">Polarity</th>
            ) : null}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr
              key={`${row.source_review_id}-${row.rank}-${index}`}
              className="border-b border-slate-100 last:border-b-0"
            >
              <td className="px-3 py-2 font-mono text-slate-500">
                {row.rank ?? index + 1}
              </td>
              <td className="px-3 py-2 leading-relaxed text-slate-700">
                {row.sentence}
                {row.was_truncated ? (
                  <span className="ml-2 rounded bg-amber-50 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 ring-1 ring-amber-200">
                    truncated
                  </span>
                ) : null}
              </td>
              <td className="px-3 py-2 font-mono text-slate-500">
                {fmtScore(row.score)}
              </td>
              <td className="px-3 py-2 font-mono text-slate-500">
                {row.source_review_id}
                {row.source_sentence_index !== undefined
                  ? `:${row.source_sentence_index}`
                  : ""}
              </td>
              <td className="px-3 py-2 text-slate-500">
                {row.matched_aspect_seed.length ? (
                  row.matched_aspect_seed.join(", ")
                ) : (
                  <EmptyText />
                )}
              </td>
              {mode !== "base" ? (
                <td className="px-3 py-2">
                  <span
                    className={`rounded px-1.5 py-0.5 text-[11px] font-semibold ${
                      row.sentiment_label === "pos"
                        ? "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200"
                        : row.sentiment_label === "neg"
                          ? "bg-rose-50 text-rose-700 ring-1 ring-rose-200"
                          : "bg-slate-50 text-slate-600 ring-1 ring-slate-200"
                    }`}
                  >
                    {row.sentiment_label || "neu"}
                  </span>
                  {row.matched_sentiment_keywords.length ? (
                    <div className="mt-1 text-[10px] text-slate-400">
                      {row.matched_sentiment_keywords.join(", ")}
                    </div>
                  ) : null}
                </td>
              ) : null}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function TextPanel({
  title,
  tone,
  text,
}: {
  title: string;
  tone: "neutral" | "positive" | "negative";
  text: string;
}) {
  const toneClass =
    tone === "positive"
      ? "bg-emerald-50 text-emerald-700 ring-emerald-200"
      : tone === "negative"
        ? "bg-rose-50 text-rose-700 ring-rose-200"
        : "bg-slate-50 text-slate-700 ring-slate-200";

  return (
    <div className={`rounded-md p-3 ring-1 ${toneClass}`}>
      <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide">
        {title}
      </div>
      <p className="text-sm leading-relaxed text-slate-800">
        {text || <EmptyText />}
      </p>
    </div>
  );
}

function SelectedMethodOutput({
  data,
  aspect,
  method,
}: {
  data: DemoData;
  aspect: DemoAspectBlock;
  method: DemoMethodId;
}) {
  const meta = methodMeta(data, method);
  const status =
    method === "m2"
      ? aspect.methods.m2.synthesis.status
      : method === "m3"
        ? [
            aspect.methods.m3.positive_synthesis.status,
            aspect.methods.m3.negative_synthesis.status,
          ]
            .filter(Boolean)
            .join(" / ")
        : method === "m4"
          ? [
              aspect.methods.m4.positive_synthesis.status,
              aspect.methods.m4.negative_synthesis.status,
            ]
              .filter(Boolean)
              .join(" / ")
          : "direct evidence";

  return (
    <section
      className={`rounded-lg border p-4 ring-1 ${COLOR_BG_LIGHT[meta.color]} ${COLOR_RING[meta.color]}`}
    >
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <MethodBadge data={data} method={method} />
          <h2 className="mt-1 text-lg font-semibold text-slate-900">
            Current output after the pipeline
          </h2>
        </div>
        <span className="rounded bg-white/70 px-2 py-1 text-[11px] text-slate-500 ring-1 ring-slate-200">
          {status || "n/a"}
        </span>
      </div>

      {method === "m1" ? (
        <TextPanel
          title="Extractive evidence output"
          tone="neutral"
          text={aspect.methods.m1.output}
        />
      ) : method === "m2" ? (
        <TextPanel
          title="Single abstractive summary"
          tone="neutral"
          text={aspect.methods.m2.output}
        />
      ) : method === "m3" ? (
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <TextPanel
            title="Keyword positive bucket"
            tone="positive"
            text={aspect.methods.m3.positive}
          />
          <TextPanel
            title="Keyword negative bucket"
            tone="negative"
            text={aspect.methods.m3.negative}
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
          <TextPanel
            title="BERT positive bucket"
            tone="positive"
            text={aspect.methods.m4.positive}
          />
          <TextPanel
            title="BERT negative bucket"
            tone="negative"
            text={aspect.methods.m4.negative}
          />
        </div>
      )}
    </section>
  );
}

function ReviewInput({ data }: { data: DemoData }) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-3">
        <h2 className="text-lg font-semibold text-slate-900">
          {data.entity.entity_name}
        </h2>
        <p className="text-xs text-slate-500">
          {data.entity.entity_id} | {data.entity.split} |{" "}
          {data.entity.review_count} reviews
        </p>
      </div>
      <div className="space-y-3">
        {data.entity.sample_reviews.slice(0, 2).map((review) => (
          <div
            key={review.review_id}
            className="border-t border-slate-100 pt-3 first:border-t-0 first:pt-0"
          >
            <div className="mb-1 flex items-center justify-between text-[11px] text-slate-500">
              <span className="font-mono">{review.review_id}</span>
              <span>rating {review.rating}</span>
            </div>
            <ul className="space-y-1">
              {review.sentences.slice(0, 4).map((sentence, index) => (
                <li
                  key={`${review.review_id}-${index}`}
                  className="text-xs leading-relaxed text-slate-700"
                >
                  {sentence}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  );
}

function DifferencePanel({
  data,
  aspect,
}: {
  data: DemoData;
  aspect: DemoAspectBlock;
}) {
  const rows = [
    {
      method: "m1" as DemoMethodId,
      shape: "No generation",
      behavior: "Faithful to selected text, but often choppy.",
      output: aspect.methods.m1.output,
    },
    {
      method: "m2" as DemoMethodId,
      shape: "One rewrite",
      behavior: "More readable, but sentiment contrast is collapsed.",
      output: aspect.methods.m2.output,
    },
    {
      method: "m3" as DemoMethodId,
      shape: "Keyword split",
      behavior: "Cheap polarity split; can overreact to generic sentiment words.",
      output: `Pos: ${aspect.methods.m3.positive || "none"} | Neg: ${
        aspect.methods.m3.negative || "none"
      }`,
    },
    {
      method: "m4" as DemoMethodId,
      shape: "BERT-ABSA split",
      behavior: "Aspect-aware polarity split; better when context matters.",
      output: `Pos: ${aspect.methods.m4.positive || "none"} | Neg: ${
        aspect.methods.m4.negative || "none"
      }`,
    },
  ];

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <h2 className="mb-3 text-lg font-semibold text-slate-900">
        What changes between methods
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[820px] text-left text-xs">
          <thead className="border-b border-slate-200 text-[11px] uppercase tracking-wide text-slate-500">
            <tr>
              <th className="w-20 py-2 pr-3 font-semibold">Method</th>
              <th className="w-32 py-2 pr-3 font-semibold">Shape</th>
              <th className="w-56 py-2 pr-3 font-semibold">Behavior</th>
              <th className="py-2 font-semibold">Current aspect output</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.method} className="border-b border-slate-100 last:border-b-0">
                <td className="py-2 pr-3">
                  <MethodBadge data={data} method={row.method} />
                </td>
                <td className="py-2 pr-3 text-slate-500">{row.shape}</td>
                <td className="py-2 pr-3 leading-relaxed text-slate-600">
                  {row.behavior}
                </td>
                <td className="py-2 leading-relaxed text-slate-700">
                  {row.output}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function MethodDemoView() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [aspectName, setAspectName] = useState<DemoAspect>("building");
  const [method, setMethod] = useState<DemoMethodId>("m1");

  useEffect(() => {
    let cancelled = false;
    fetch("/data/space_method_demo.json", { cache: "no-store" })
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load demo data (${res.status})`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) {
          setData(json);
          if (json.aspects?.[0]?.aspect) setAspectName(json.aspects[0].aspect);
        }
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

  const aspect = useMemo(
    () => data?.aspects.find((item) => item.aspect === aspectName),
    [data, aspectName],
  );

  if (loading) return <div className="text-sm text-slate-500">Loading...</div>;
  if (error) {
    return (
      <div className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">
        {error}
      </div>
    );
  }
  if (!data || !aspect) return null;

  const evidenceMode = evidenceModeFor(method);
  const evidenceRows =
    evidenceMode === "base"
      ? aspect.shared_evidence
      : evidenceMode === "keyword"
        ? aspect.keyword_evidence
        : aspect.bert_evidence;

  return (
    <div className="space-y-6">
      <MethodTabs data={data} value={method} onChange={setMethod} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[330px_1fr]">
        <aside className="space-y-4">
          <ReviewInput data={data} />
          <section className="rounded-lg border border-amber-200 bg-amber-50 p-4">
            <h2 className="mb-2 text-sm font-semibold text-amber-900">
              Human gold signal
            </h2>
            <p className="text-xs leading-relaxed text-amber-900">
              {data.entity.gold_overall[0]}
            </p>
          </section>
        </aside>

        <main className="space-y-6">
          <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 className="text-lg font-semibold text-slate-900">
                  Pick an aspect, then follow the method branch
                </h2>
                <p className="text-xs text-slate-500">
                  The rows below are the actual evidence moving through the
                  selected pipeline.
                </p>
              </div>
              <div className="inline-flex flex-wrap gap-1 rounded-md border border-slate-300 bg-white p-0.5">
                {data.aspects.map((item) => (
                  <button
                    key={item.aspect}
                    type="button"
                    onClick={() => setAspectName(item.aspect)}
                    className={`rounded px-3 py-1.5 text-sm font-medium transition ${
                      aspectName === item.aspect
                        ? "bg-indigo-600 text-white"
                        : "text-slate-600 hover:bg-slate-100"
                    }`}
                  >
                    {ASPECT_LABEL[item.aspect] ?? item.aspect}
                  </button>
                ))}
              </div>
            </div>
            <EvidenceTable rows={evidenceRows} mode={evidenceMode} />
          </section>

          <PipelineDiagram data={data} method={method} aspect={aspect} />
          <SelectedMethodOutput data={data} method={method} aspect={aspect} />
          <DifferencePanel data={data} aspect={aspect} />
        </main>
      </div>
    </div>
  );
}
