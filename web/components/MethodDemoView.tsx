"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  COLOR_BAR,
  COLOR_BG_LIGHT,
  COLOR_RING,
  COLOR_TEXT,
  anonymizePropertyText,
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

function sanitizeDemoData<T extends { entity?: { entity_name?: string } }>(
  payload: T,
): T {
  const propertyName = payload.entity?.entity_name;
  const visit = (value: unknown): unknown => {
    if (typeof value === "string") return anonymizePropertyText(value, propertyName);
    if (Array.isArray(value)) return value.map(visit);
    if (value && typeof value === "object") {
      return Object.fromEntries(
        Object.entries(value).map(([key, item]) => [key, visit(item)]),
      );
    }
    return value;
  };
  return visit(payload) as T;
}

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
      input: "100 anonymized review samples",
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

// Renders the real data that flows through a given stage. Keyed by the stable
// stage.title strings produced in pipelineStages().
function StageEvidence({
  data,
  method,
  aspect,
  stageTitle,
}: {
  data: DemoData;
  method: DemoMethodId;
  aspect: DemoAspectBlock;
  stageTitle: string;
}) {
  if (stageTitle === "Raw review pool") {
    const reviews = data.entity.sample_reviews.slice(0, 2);
    return (
      <div className="space-y-3">
        <p className="text-xs text-slate-500">
          Raw input: review sentences flattened into the entity pool (showing a
          sample).
        </p>
        {reviews.map((review) => (
          <div
            key={review.review_id}
            className="rounded-md border border-slate-200 bg-white p-3"
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
    );
  }

  if (stageTitle === "SemAE aspect ranking") {
    return <EvidenceTable rows={aspect.shared_evidence} mode="base" />;
  }

  if (stageTitle === "Keyword polarity gate") {
    return <EvidenceTable rows={aspect.keyword_evidence} mode="keyword" />;
  }

  if (stageTitle === "BERT-ABSA polarity gate") {
    return <EvidenceTable rows={aspect.bert_evidence} mode="bert" />;
  }

  if (stageTitle === "Extractive write") {
    return (
      <TextPanel
        title="Extractive evidence output"
        tone="neutral"
        text={aspect.methods.m1.output}
      />
    );
  }

  if (stageTitle === "FLAN-T5 rewrite") {
    return (
      <TextPanel
        title="Single abstractive summary"
        tone="neutral"
        text={aspect.methods.m2.output}
      />
    );
  }

  if (stageTitle === "Rewrite each bucket") {
    const branch = method === "m3" ? aspect.methods.m3 : aspect.methods.m4;
    return (
      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
        <TextPanel title="Positive bucket" tone="positive" text={branch.positive} />
        <TextPanel title="Negative bucket" tone="negative" text={branch.negative} />
      </div>
    );
  }

  return null;
}

function StageIOPanel({
  data,
  method,
  aspect,
  stage,
  panelId,
}: {
  data: DemoData;
  method: DemoMethodId;
  aspect: DemoAspectBlock;
  stage: Stage;
  panelId: string;
}) {
  return (
    <div id={panelId} className="stage-io-panel mt-3">
      <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <h3 className="text-sm font-semibold text-slate-900">{stage.title}</h3>
        <dl className="mt-3 grid grid-cols-1 gap-3 text-xs leading-relaxed md:grid-cols-3">
          <div className="stage-io-item" style={{ animationDelay: "40ms" }}>
            <dt className="font-semibold uppercase tracking-wide text-slate-400">
              Input
            </dt>
            <dd className="mt-1 text-slate-700">{stage.input}</dd>
          </div>
          <div className="stage-io-item" style={{ animationDelay: "120ms" }}>
            <dt className="font-semibold uppercase tracking-wide text-slate-400">
              Operation
            </dt>
            <dd className="mt-1 text-slate-700">{stage.operation}</dd>
          </div>
          <div className="stage-io-item" style={{ animationDelay: "200ms" }}>
            <dt className="font-semibold uppercase tracking-wide text-slate-400">
              Output
            </dt>
            <dd className="mt-1 text-slate-700">{stage.output}</dd>
          </div>
        </dl>
        <div
          className="stage-io-evidence mt-4 border-t border-slate-100 pt-4"
          style={{ animationDelay: "280ms" }}
        >
          <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-slate-400">
            Real data at this stage
          </div>
          <StageEvidence
            data={data}
            method={method}
            aspect={aspect}
            stageTitle={stage.title}
          />
        </div>
      </div>
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

  const [activeStage, setActiveStage] = useState<number | null>(0);

  // Reset the open stage whenever the method or aspect changes, since the
  // number of stages differs (M1/M2 = 3, M3/M4 = 4).
  useEffect(() => {
    setActiveStage(0);
  }, [method, aspect.aspect]);

  const active = activeStage !== null ? stages[activeStage] : undefined;

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-slate-900">
            How {meta.short} actually runs
          </h2>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            The first two stages are shared. Click any stage to see its real
            input and output.
          </p>
        </div>
        <span
          className={`rounded px-2 py-1 text-xs font-semibold ring-1 ${COLOR_BG_LIGHT[meta.color]} ${COLOR_TEXT[meta.color]} ${COLOR_RING[meta.color]}`}
        >
          {ASPECT_LABEL[aspect.aspect] ?? aspect.aspect}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[repeat(4,minmax(0,1fr))]">
        {stages.map((stage, index) => {
          const isActive = index === activeStage;
          const isPast = activeStage !== null && index <= activeStage;
          const panelId = `stage-panel-${method}-${index}`;
          return (
            <div key={`${method}-${stage.title}`} className="relative">
              <button
                type="button"
                onClick={() =>
                  setActiveStage((current) =>
                    current === index ? null : index,
                  )
                }
                aria-expanded={isActive}
                aria-controls={panelId}
                className={`pipeline-stage flex h-full w-full flex-col rounded-lg border p-3 text-left ${
                  isActive
                    ? `pipeline-stage-active ${COLOR_BG_LIGHT[meta.color]} ${COLOR_RING[meta.color]} ring-2`
                    : "border-slate-200 bg-slate-50 hover:bg-white"
                }`}
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
                <p className="mt-2 text-xs leading-relaxed text-slate-500">
                  {stage.operation}
                </p>
              </button>
              {index < stages.length - 1 ? (
                <div
                  className={`pipeline-connector hidden xl:block ${
                    isPast ? COLOR_BAR[meta.color] : "bg-slate-300"
                  }`}
                />
              ) : null}
            </div>
          );
        })}
      </div>

      {active ? (
        <StageIOPanel
          data={data}
          method={method}
          aspect={aspect}
          stage={active}
          panelId={`stage-panel-${method}-${activeStage}`}
        />
      ) : null}
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

function EntityStoryHero({ data }: { data: DemoData }) {
  return (
    <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="grid grid-cols-1 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="border-b border-[var(--outline-variant)] bg-[var(--primary)] p-6 text-white lg:border-b-0 lg:border-r lg:border-[var(--outline-variant)]">
          <div className="text-xs font-semibold uppercase tracking-[0.28em] text-[var(--secondary-fixed)]">
            Entity input
          </div>
          <h2 className="mt-3 text-3xl font-bold tracking-tight">
            Representative property sample
          </h2>
          <div className="mt-4 flex flex-wrap gap-2 text-xs">
            <span className="rounded-full bg-white/10 px-3 py-1 ring-1 ring-white/15">
              ID {data.entity.entity_id}
            </span>
            <span className="rounded-full bg-white/10 px-3 py-1 ring-1 ring-white/15">
              {data.entity.split} split
            </span>
            <span className="rounded-full bg-white/10 px-3 py-1 ring-1 ring-white/15">
              {data.entity.review_count} reviews
            </span>
          </div>
          <div className="mt-6 rounded-xl bg-white/10 p-4 ring-1 ring-white/15">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-wide text-[var(--secondary-fixed)]">
              Human reference signal
            </div>
            <p className="text-sm leading-relaxed text-slate-100">
              {data.entity.gold_overall[0]}
            </p>
          </div>
        </div>

        <div className="bg-gradient-to-br from-slate-50 to-white p-6">
          <div className="mb-3 flex items-center justify-between gap-3">
            <div>
              <div className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Review samples
              </div>
              <p className="text-sm text-slate-500">
                These sentences are the raw material every method starts from.
              </p>
            </div>
            <span className="rounded-full bg-[var(--secondary-fixed)] px-3 py-1 text-xs font-semibold text-[var(--primary)] ring-1 ring-[var(--outline-variant)]">
              Scroll ↓ follow the pipeline
            </span>
          </div>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            {data.entity.sample_reviews.slice(0, 3).map((review) => (
              <article
                key={review.review_id}
                className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm"
              >
                <div className="mb-2 flex items-center justify-between text-[11px] text-slate-500">
                  <span className="font-mono">{review.review_id}</span>
                  <span>★ {review.rating}</span>
                </div>
                <ul className="space-y-2">
                  {review.sentences.slice(0, 3).map((sentence, index) => (
                    <li
                      key={`${review.review_id}-${index}`}
                      className="text-xs leading-relaxed text-slate-700"
                    >
                      {sentence}
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function StoryControls({
  data,
  method,
  onMethodChange,
  aspectName,
  onAspectChange,
}: {
  data: DemoData;
  method: DemoMethodId;
  onMethodChange: (method: DemoMethodId) => void;
  aspectName: DemoAspect;
  onAspectChange: (aspect: DemoAspect) => void;
}) {
  return (
    <section className="sticky top-0 z-10 -mx-4 border-y border-slate-200 bg-white/95 px-4 py-3 shadow-sm backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div className="flex flex-wrap gap-2">
          {METHOD_IDS.map((item) => {
            const meta = methodMeta(data, item);
            const active = item === method;
            return (
              <button
                key={item}
                type="button"
                onClick={() => onMethodChange(item)}
                className={`rounded-full px-4 py-2 text-sm font-semibold transition ${
                  active
                    ? `${COLOR_BG_LIGHT[meta.color]} ${COLOR_TEXT[meta.color]} ${COLOR_RING[meta.color]} ring-2`
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200"
                }`}
              >
                {meta.short}
                <span className="ml-2 hidden font-normal sm:inline">
                  {meta.label.replace(`${meta.short} - `, "")}
                </span>
              </button>
            );
          })}
        </div>
        <div className="flex flex-wrap gap-1 rounded-full bg-slate-100 p-1">
          {data.aspects.map((item) => (
            <button
              key={item.aspect}
              type="button"
              onClick={() => onAspectChange(item.aspect)}
              className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                aspectName === item.aspect
                  ? "bg-[var(--primary)] text-white shadow-sm"
                  : "text-slate-600 hover:bg-white"
              }`}
            >
              {ASPECT_LABEL[item.aspect] ?? item.aspect}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
}

function MethodStoryCard({
  data,
  method,
  aspect,
}: {
  data: DemoData;
  method: DemoMethodId;
  aspect: DemoAspectBlock;
}) {
  const meta = methodMeta(data, method);
  const aspectLabel = ASPECT_LABEL[aspect.aspect] ?? aspect.aspect;
  const mode = evidenceModeFor(method);
  const evidenceRows =
    mode === "base"
      ? aspect.shared_evidence
      : mode === "keyword"
        ? aspect.keyword_evidence
        : aspect.bert_evidence;

  return (
    <section className="grid grid-cols-1 gap-6 lg:grid-cols-[280px_1fr]">
      <aside className="lg:sticky lg:top-28 lg:self-start">
        <div
          className={`rounded-2xl border p-5 shadow-sm ${COLOR_BG_LIGHT[meta.color]} ${COLOR_RING[meta.color]} ring-1`}
        >
          <MethodBadge data={data} method={method} />
          <h2 className="mt-3 text-2xl font-bold text-slate-900">
            {meta.label.replace(`${meta.short} - `, "")}
          </h2>
          <p className="mt-2 text-sm leading-relaxed text-slate-600">
            {meta.desc}
          </p>
          <div className="mt-5 space-y-3 text-xs text-slate-600">
            <div className="rounded-lg bg-white/70 p-3 ring-1 ring-white/80">
              <div className="font-semibold text-slate-900">Current aspect</div>
              <div>{aspectLabel}</div>
            </div>
            <div className="rounded-lg bg-white/70 p-3 ring-1 ring-white/80">
              <div className="font-semibold text-slate-900">Evidence rows</div>
              <div>{evidenceRows.length} ranked sentences available</div>
            </div>
          </div>
        </div>
      </aside>

      <main className="space-y-6">
        <PipelineDiagram data={data} method={method} aspect={aspect} />
        <SelectedMethodOutput data={data} method={method} aspect={aspect} />
      </main>
    </section>
  );
}

function MethodGallery({
  data,
  aspect,
  activeMethod,
  onMethodChange,
}: {
  data: DemoData;
  aspect: DemoAspectBlock;
  activeMethod: DemoMethodId;
  onMethodChange: (method: DemoMethodId) => void;
}) {
  const rows = METHOD_IDS.map((method) => ({
    method,
    meta: methodMeta(data, method),
    output:
      method === "m1"
        ? aspect.methods.m1.output
        : method === "m2"
          ? aspect.methods.m2.output
          : method === "m3"
            ? `Positive: ${aspect.methods.m3.positive || "none"}\nNegative: ${
                aspect.methods.m3.negative || "none"
              }`
            : `Positive: ${aspect.methods.m4.positive || "none"}\nNegative: ${
                aspect.methods.m4.negative || "none"
              }`,
  }));

  return (
    <section className="rounded-2xl border border-[var(--outline-variant)] bg-[var(--primary)] p-5 text-white shadow-sm">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.24em] text-[var(--secondary-fixed)]">
            M1 → M4 outputs
          </div>
          <h2 className="mt-2 text-xl font-bold">Compare the final shape</h2>
        </div>
        <p className="max-w-xl text-xs leading-relaxed text-slate-300">
          The pipeline panel above explains how the selected method reaches its
          output. These cards show the four possible final outputs side by side.
        </p>
      </div>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {rows.map(({ method, meta, output }) => (
          <button
            key={method}
            type="button"
            onClick={() => onMethodChange(method)}
            className={`rounded-xl border p-4 text-left transition ${
              method === activeMethod
                ? "border-white bg-white text-slate-950"
                : "border-white/10 bg-white/5 text-slate-100 hover:bg-white/10"
            }`}
          >
            <div className="mb-2 text-xs font-semibold uppercase tracking-wide">
              {meta.short}
            </div>
            <p className="line-clamp-6 whitespace-pre-line text-xs leading-relaxed">
              {output}
            </p>
          </button>
        ))}
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
          setData(sanitizeDemoData(json));
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

  return (
    <div className="space-y-8">
      <EntityStoryHero data={data} />
      <StoryControls
        data={data}
        method={method}
        onMethodChange={setMethod}
        aspectName={aspectName}
        onAspectChange={setAspectName}
      />
      <MethodStoryCard data={data} method={method} aspect={aspect} />
      <MethodGallery
        data={data}
        aspect={aspect}
        activeMethod={method}
        onMethodChange={setMethod}
      />
    </div>
  );
}
