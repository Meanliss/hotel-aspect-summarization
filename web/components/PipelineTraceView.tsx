"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ASPECT_LABEL,
  METHOD_IDS,
  anonymizePropertyText,
  loadMethodDemoData,
  type MethodId,
} from "@/lib/space";

type DemoAspect = "building" | "food" | "rooms" | "service";
type SentimentFilter = "all" | "pos" | "neg" | "neu";

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

interface DemoAspectBlock {
  aspect: DemoAspect;
  gold: string[];
  shared_evidence: DemoEvidence[];
  keyword_evidence: DemoEvidence[];
  bert_evidence: DemoEvidence[];
  methods: {
    m1: { output: string; evidence: DemoEvidence[] };
    m2: { output: string; synthesis: { evidence: DemoEvidence[] } };
    m3: {
      positive: string;
      negative: string;
      positive_synthesis: { evidence: DemoEvidence[] };
      negative_synthesis: { evidence: DemoEvidence[] };
    };
    m4: {
      positive: string;
      negative: string;
      positive_synthesis: { evidence: DemoEvidence[] };
      negative_synthesis: { evidence: DemoEvidence[] };
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
  overall: Record<MethodId, string>;
  aspects: DemoAspectBlock[];
}

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

function sentimentClass(label: string) {
  if (label === "pos") return "bg-emerald-50 text-emerald-700 ring-emerald-200";
  if (label === "neg") return "bg-rose-50 text-rose-700 ring-rose-200";
  return "bg-slate-50 text-slate-600 ring-slate-200";
}

function sentenceMatches(text: string, query: string) {
  return !query || text.toLowerCase().includes(query);
}

function StageShell({
  number,
  title,
  subtitle,
  count,
  children,
}: {
  number: number;
  title: string;
  subtitle: string;
  count: number;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-100 px-4 py-3">
        <div>
          <div className="mb-1 font-mono text-[11px] text-slate-400">
            STAGE {number}
          </div>
          <h2 className="text-base font-semibold text-slate-900">{title}</h2>
          <p className="mt-1 text-xs leading-relaxed text-slate-500">
            {subtitle}
          </p>
        </div>
        <span className="rounded bg-slate-50 px-2 py-1 text-xs font-medium text-slate-600 ring-1 ring-slate-200">
          {count} rows
        </span>
      </div>
      <div className="max-h-[520px] overflow-y-auto p-4">{children}</div>
    </section>
  );
}

function RawSentenceList({
  data,
  query,
}: {
  data: DemoData;
  query: string;
}) {
  const rows = data.entity.sample_reviews
    .map((review) => ({
      ...review,
      sentences: review.sentences.filter((sentence) =>
        sentenceMatches(sentence, query),
      ),
    }))
    .filter((review) => review.sentences.length > 0);

  return (
    <div className="space-y-3">
      {rows.map((review) => (
        <div key={review.review_id} className="rounded-md bg-slate-50 p-3">
          <div className="mb-2 flex flex-wrap justify-between gap-2 text-[11px] text-slate-500">
            <span className="font-mono">{review.review_id}</span>
            <span>rating {review.rating}</span>
          </div>
          <ol className="space-y-1">
            {review.sentences.map((sentence, index) => (
              <li
                key={`${review.review_id}-${index}`}
                className="text-sm leading-relaxed text-slate-700"
              >
                <span className="mr-2 font-mono text-[11px] text-slate-400">
                  S{index + 1}
                </span>
                {sentence}
              </li>
            ))}
          </ol>
        </div>
      ))}
    </div>
  );
}

function EvidenceList({
  rows,
  query,
  sentiment,
}: {
  rows: DemoEvidence[];
  query: string;
  sentiment: SentimentFilter;
}) {
  const filtered = rows.filter((row) => {
    if (!sentenceMatches(row.sentence, query)) return false;
    if (sentiment !== "all" && (row.sentiment_label || "neu") !== sentiment) {
      return false;
    }
    return true;
  });

  return (
    <div className="space-y-2">
      {filtered.map((row, index) => (
        <article
          key={`${row.source_review_id}-${row.source_sentence_index}-${index}`}
          className="rounded-md border border-slate-100 bg-slate-50 p-3"
        >
          <div className="mb-2 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
            <span className="font-mono">rank {row.rank ?? index + 1}</span>
            {row.score !== undefined ? (
              <span className="font-mono">score {fmtScore(row.score)}</span>
            ) : null}
            <span className="font-mono">
              {row.source_review_id}:{row.source_sentence_index}
            </span>
            <span
              className={`rounded px-1.5 py-0.5 font-semibold ring-1 ${sentimentClass(
                row.sentiment_label,
              )}`}
            >
              {row.sentiment_label || "neu"}
            </span>
            {row.was_truncated ? (
              <span className="rounded bg-amber-50 px-1.5 py-0.5 font-medium text-amber-700 ring-1 ring-amber-200">
                truncated
              </span>
            ) : null}
          </div>
          <p className="text-sm leading-relaxed text-slate-800">
            {row.sentence}
          </p>
          <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-500">
            {row.matched_aspect_seed.length ? (
              <span>seed: {row.matched_aspect_seed.join(", ")}</span>
            ) : null}
            {row.matched_sentiment_keywords.length ? (
              <span>polarity: {row.matched_sentiment_keywords.join(", ")}</span>
            ) : null}
          </div>
        </article>
      ))}
      {filtered.length === 0 ? (
        <p className="text-sm text-slate-400">No matching sentences.</p>
      ) : null}
    </div>
  );
}

function OutputGrid({
  aspect,
  query,
}: {
  aspect: DemoAspectBlock;
  query: string;
}) {
  const outputs = [
    ["M1", aspect.methods.m1.output],
    ["M2", aspect.methods.m2.output],
    ["M3 positive", aspect.methods.m3.positive],
    ["M3 negative", aspect.methods.m3.negative],
    ["M4 positive", aspect.methods.m4.positive],
    ["M4 negative", aspect.methods.m4.negative],
  ].filter(([, text]) => sentenceMatches(text, query));

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      {outputs.map(([label, text]) => (
        <div key={label} className="rounded-md bg-slate-50 p-3 ring-1 ring-slate-100">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
            {label}
          </div>
          <p className="text-sm leading-relaxed text-slate-800">
            {text || <span className="italic text-slate-400">none</span>}
          </p>
        </div>
      ))}
    </div>
  );
}

export function PipelineTraceView() {
  const [data, setData] = useState<DemoData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [aspectName, setAspectName] = useState<DemoAspect>("building");
  const [query, setQuery] = useState("");
  const [sentiment, setSentiment] = useState<SentimentFilter>("all");

  useEffect(() => {
    let cancelled = false;
    loadMethodDemoData<DemoData>()
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

  const normalizedQuery = query.trim().toLowerCase();
  const aspect = useMemo(
    () => data?.aspects.find((item) => item.aspect === aspectName),
    [data, aspectName],
  );

  const rawSentenceCount = useMemo(() => {
    return (
      data?.entity.sample_reviews.reduce(
        (total, review) => total + review.sentences.length,
        0,
      ) ?? 0
    );
  }, [data]);

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
    <div className="space-y-5">
      <section className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm">
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-[1fr_220px_180px]">
          <label>
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
              Search any sentence
            </span>
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="room, service, breakfast..."
              className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
            />
          </label>
          <label>
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
              Aspect
            </span>
            <select
              value={aspectName}
              onChange={(event) => setAspectName(event.target.value as DemoAspect)}
              className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              {data.aspects.map((item) => (
                <option key={item.aspect} value={item.aspect}>
                  {ASPECT_LABEL[item.aspect] ?? item.aspect}
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">
              Sentiment
            </span>
            <select
              value={sentiment}
              onChange={(event) => setSentiment(event.target.value as SentimentFilter)}
              className="w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm"
            >
              <option value="all">All</option>
              <option value="pos">Positive</option>
              <option value="neg">Negative</option>
              <option value="neu">Neutral</option>
            </select>
          </label>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-2 text-xs text-slate-600 md:grid-cols-5">
          <div className="rounded bg-slate-50 p-2 ring-1 ring-slate-100">
            Raw sentences: {rawSentenceCount}
          </div>
          <div className="rounded bg-slate-50 p-2 ring-1 ring-slate-100">
            SemAE: {aspect.shared_evidence.length}
          </div>
          <div className="rounded bg-slate-50 p-2 ring-1 ring-slate-100">
            Keyword: {aspect.keyword_evidence.length}
          </div>
          <div className="rounded bg-slate-50 p-2 ring-1 ring-slate-100">
            BERT: {aspect.bert_evidence.length}
          </div>
          <div className="rounded bg-slate-50 p-2 ring-1 ring-slate-100">
            Methods: {METHOD_IDS.length}
          </div>
        </div>
      </section>

      <StageShell
        number={1}
        title="Raw review sentences"
        subtitle="All review sentences exported for the selected anonymized sample before any model step."
        count={rawSentenceCount}
      >
        <RawSentenceList data={data} query={normalizedQuery} />
      </StageShell>

      <StageShell
        number={2}
        title="SemAE aspect evidence"
        subtitle="Ranked sentences selected by the aspect-matching stage before sentiment branching."
        count={aspect.shared_evidence.length}
      >
        <EvidenceList rows={aspect.shared_evidence} query={normalizedQuery} sentiment={sentiment} />
      </StageShell>

      <StageShell
        number={3}
        title="Keyword sentiment split"
        subtitle="The same evidence after rule/keyword polarity assignment."
        count={aspect.keyword_evidence.length}
      >
        <EvidenceList rows={aspect.keyword_evidence} query={normalizedQuery} sentiment={sentiment} />
      </StageShell>

      <StageShell
        number={4}
        title="BERT-ABSA sentiment split"
        subtitle="The same evidence after aspect-aware sentiment classification."
        count={aspect.bert_evidence.length}
      >
        <EvidenceList rows={aspect.bert_evidence} query={normalizedQuery} sentiment={sentiment} />
      </StageShell>

      <StageShell
        number={5}
        title="Generated method outputs"
        subtitle="Final text produced by each method from the stage-level evidence."
        count={6}
      >
        <OutputGrid aspect={aspect} query={normalizedQuery} />
      </StageShell>
    </div>
  );
}
