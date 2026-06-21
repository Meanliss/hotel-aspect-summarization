"use client";

import { useEffect, useMemo, useState } from "react";
import {
  aspectLabel,
  countTotal,
  loadHotelAspectDataset,
  sumCounts,
  type HotelAspectDataset,
  type HotelAspectRecord,
  type HotelAspectSummary,
  type SentimentCounts,
  type SentimentKey,
} from "@/lib/hotel-aspects";

const SENTIMENTS: SentimentKey[] = ["positive", "negative", "neutral"];
const SENTIMENT_COLOR: Record<SentimentKey, string> = {
  positive: "bg-emerald-500 text-emerald-700 border-emerald-200",
  negative: "bg-rose-500 text-rose-700 border-rose-200",
  neutral: "bg-amber-500 text-amber-700 border-amber-200",
};
const SENTIMENT_LABEL: Record<SentimentKey, string> = {
  positive: "Positive",
  negative: "Negative",
  neutral: "Neutral",
};
const STOPWORDS = new Set([
  "the",
  "and",
  "with",
  "that",
  "this",
  "from",
  "have",
  "were",
  "was",
  "are",
  "but",
  "for",
  "hotel",
  "rooms",
  "room",
  "guests",
  "guest",
  "property",
]);

function formatNumber(value: number) {
  return new Intl.NumberFormat("en-US").format(value);
}

function scoreFromCounts(counts: SentimentCounts) {
  const total = Math.max(1, countTotal(counts));
  const weighted = (counts.positive * 10 + counts.neutral * 6 + counts.negative * 2) / total;
  return Math.max(1, Math.min(9.8, weighted)).toFixed(1);
}

function splitSentences(text: string, limit = 4) {
  return text
    .split(/(?<=[.!?;])\s+/)
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, limit);
}

function shortText(text: string, max = 210) {
  if (text.length <= max) return text;
  return `${text.slice(0, max).trim()}...`;
}

function keywordRows(text: string, polarityCount: number) {
  const counts = new Map<string, number>();
  for (const word of text.toLowerCase().match(/[a-z][a-z-]{3,}/g) ?? []) {
    if (!STOPWORDS.has(word)) counts.set(word, (counts.get(word) ?? 0) + 1);
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([topic], index) => ({
      topic: topic.replace(/^\w/, (char) => char.toUpperCase()),
      mentions: Math.max(1, Math.round(polarityCount / (index + 2))),
      rate: Math.max(38, 94 - index * 9),
    }));
}

function sentimentShare(counts: SentimentCounts, key: SentimentKey) {
  return Math.round((counts[key] / Math.max(1, countTotal(counts))) * 100);
}

function SentimentBar({ counts }: { counts: SentimentCounts }) {
  const total = Math.max(1, countTotal(counts));
  return (
    <div className="flex h-3 overflow-hidden rounded-full bg-slate-100">
      {SENTIMENTS.map((key) => (
        <div
          key={key}
          className={SENTIMENT_COLOR[key].split(" ")[0]}
          style={{ width: `${(counts[key] / total) * 100}%` }}
        />
      ))}
    </div>
  );
}

function InsightList({
  title,
  items,
  tone,
}: {
  title: string;
  items: string[];
  tone: "good" | "bad";
}) {
  return (
    <div className={tone === "good" ? "rounded-lg bg-emerald-50 p-4" : "rounded-lg bg-rose-50 p-4"}>
      <h3 className={tone === "good" ? "text-sm font-bold text-emerald-800" : "text-sm font-bold text-rose-800"}>
        {title}
      </h3>
      <ul className="mt-3 space-y-2">
        {items.slice(0, 5).map((item) => (
          <li key={item} className="flex gap-2 text-sm leading-5 text-[var(--on-surface)]">
            <span className={tone === "good" ? "text-emerald-600" : "text-rose-600"}>
              {tone === "good" ? "✓" : "!"}
            </span>
            <span>{shortText(item, 88)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function TopicTable({
  title,
  rows,
  tone,
}: {
  title: string;
  rows: ReturnType<typeof keywordRows>;
  tone: "positive" | "negative";
}) {
  const barColor = tone === "positive" ? "bg-emerald-500" : "bg-rose-400";
  return (
    <section className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
      <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">{title}</h3>
      <div className="mt-3 space-y-3">
        {rows.map((row) => (
          <div key={row.topic} className="grid grid-cols-[88px_48px_1fr_36px] items-center gap-2 text-xs">
            <span className="font-medium">{row.topic}</span>
            <span className="text-[var(--on-surface-variant)]">{row.mentions}</span>
            <span className="h-2 overflow-hidden rounded-full bg-slate-100">
              <span className={`block h-full ${barColor}`} style={{ width: `${row.rate}%` }} />
            </span>
            <span className="text-right font-semibold">{row.rate}%</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function AspectCard({
  aspect,
  summary,
  active,
  onClick,
}: {
  aspect: string;
  summary: HotelAspectSummary;
  active: boolean;
  onClick: () => void;
}) {
  const score = scoreFromCounts(summary.counts);
  return (
    <button
      onClick={onClick}
      className={`min-h-[178px] rounded-lg border p-4 text-left transition hover:-translate-y-0.5 hover:shadow-md ${
        active
          ? "border-[var(--primary)] bg-[var(--primary)] text-white"
          : "border-[var(--outline-variant)] bg-white"
      }`}
    >
      <div className="text-xs font-bold uppercase tracking-[0.12em]">{aspectLabel(aspect)}</div>
      <div className="mt-4 flex items-end gap-1">
        <span className="text-4xl font-bold">{score}</span>
        <span className={active ? "mb-1 text-white/70" : "mb-1 text-[var(--on-surface-variant)]"}>/10</span>
      </div>
      <div className={active ? "mt-2 text-xs text-white/75" : "mt-2 text-xs text-[var(--on-surface-variant)]"}>
        Positive {sentimentShare(summary.counts, "positive")}%
      </div>
      <p className={active ? "mt-3 text-sm leading-5 text-white/85" : "mt-3 text-sm leading-5 text-[var(--on-surface-variant)]"}>
        {shortText(summary.positive || summary.overview, 118)}
      </p>
    </button>
  );
}

export function HotelNarrativeDashboard() {
  const [data, setData] = useState<HotelAspectDataset | null>(null);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [selectedHotelId, setSelectedHotelId] = useState("");
  const [selectedAspect, setSelectedAspect] = useState("");
  const [detailsOpen, setDetailsOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    loadHotelAspectDataset()
      .then((payload) => {
        if (cancelled) return;
        const firstAspect = payload.aspects.find((aspect) => aspect !== "all_aspects") ?? payload.aspects[0] ?? "";
        setData(payload);
        setSelectedHotelId(payload.hotels[0]?.hotel_id ?? "");
        setSelectedAspect(firstAspect);
      })
      .catch((reason) => {
        if (!cancelled) setError(String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const selectedHotel = useMemo(
    () => data?.hotels.find((hotel) => hotel.hotel_id === selectedHotelId) ?? data?.hotels[0],
    [data, selectedHotelId],
  );
  const aspects = data?.aspects.filter((aspect) => aspect !== "all_aspects") ?? [];
  const filteredHotels = useMemo(() => {
    if (!data) return [];
    const normalized = query.trim().toLowerCase();
    return data.hotels
      .filter((hotel) => !normalized || hotel.hotel_id.toLowerCase().includes(normalized))
      .slice(0, 28);
  }, [data, query]);
  const hotelCounts = useMemo(() => {
    if (!selectedHotel) return { positive: 0, negative: 0, neutral: 0 };
    return sumCounts(Object.values(selectedHotel.aspects).map((summary) => summary.counts));
  }, [selectedHotel]);
  const activeSummary = selectedHotel?.aspects[selectedAspect];
  const aspectRows = useMemo(() => {
    if (!selectedHotel) return [];
    return aspects
      .map((aspect) => ({ aspect, summary: selectedHotel.aspects[aspect] }))
      .filter((row): row is { aspect: string; summary: HotelAspectSummary } => Boolean(row.summary))
      .sort((a, b) => countTotal(b.summary.counts) - countTotal(a.summary.counts));
  }, [aspects, selectedHotel]);

  if (error) return <div className="rounded-lg bg-rose-50 p-4 text-sm text-rose-700">{error}</div>;
  if (!data || !selectedHotel || !activeSummary) {
    return <div className="rounded-lg bg-white p-6 text-sm text-[var(--on-surface-variant)]">Loading hotel dashboard...</div>;
  }

  const positiveItems = splitSentences(activeSummary.positive || activeSummary.overview, 5);
  const concernItems = splitSentences(activeSummary.negative || activeSummary.overview, 5);
  const allText = `${activeSummary.overview} ${activeSummary.positive} ${activeSummary.negative} ${activeSummary.neutral}`;
  const overallScore = scoreFromCounts(hotelCounts);
  const selectedScore = scoreFromCounts(activeSummary.counts);

  return (
    <div className="min-w-0 space-y-6">
      <section className="grid gap-5 xl:grid-cols-[1fr_1fr_1.5fr]">
        <div className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h2 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Hotel information</h2>
          <div className="mt-3 h-36 rounded-lg bg-[linear-gradient(135deg,#22394a,#4e7f93_45%,#f4b15d)]" />
          <h3 className="mt-4 text-xl font-bold text-[var(--primary)]">Hotel {selectedHotel.hotel_id}</h3>
          <div className="mt-2 text-amber-500">★★★★★</div>
          <dl className="mt-4 grid grid-cols-[96px_1fr] gap-2 text-sm">
            <dt className="text-[var(--on-surface-variant)]">Reviews</dt>
            <dd className="font-semibold">{formatNumber(countTotal(hotelCounts))}</dd>
            <dt className="text-[var(--on-surface-variant)]">Aspects</dt>
            <dd className="font-semibold">{aspectRows.length}</dd>
            <dt className="text-[var(--on-surface-variant)]">Source</dt>
            <dd className="truncate font-semibold">{data.source_file}</dd>
          </dl>
        </div>

        <div className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h2 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Hotel selector</h2>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            className="mt-3 w-full rounded-md border border-[var(--outline-variant)] bg-[var(--surface-container-low)] px-3 py-2 text-sm outline-none focus:border-[var(--primary)]"
            placeholder="Search hotel id..."
            type="search"
          />
          <div className="mt-3 max-h-64 space-y-2 overflow-auto pr-1">
            {filteredHotels.map((hotel) => (
              <button
                key={hotel.hotel_id}
                onClick={() => setSelectedHotelId(hotel.hotel_id)}
                className={`flex w-full items-center justify-between rounded-md border px-3 py-2 text-sm ${
                  hotel.hotel_id === selectedHotel.hotel_id
                    ? "border-[var(--primary)] bg-[var(--primary)] text-white"
                    : "border-[var(--outline-variant)] bg-white"
                }`}
              >
                <span className="font-semibold">Hotel {hotel.hotel_id}</span>
                <span>{Object.keys(hotel.aspects).length}/{data.aspects.length}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h2 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">AI summary overview</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-[160px_1fr]">
            <div>
              <div className="text-6xl font-bold text-emerald-600">{overallScore}</div>
              <div className="text-lg font-semibold text-[var(--on-surface-variant)]">/10 overall</div>
            </div>
            <div>
              <div className="mb-2 flex justify-between text-xs font-bold uppercase tracking-[0.1em] text-[var(--on-surface-variant)]">
                <span>Sentiment distribution</span>
                <span>{formatNumber(countTotal(hotelCounts))}</span>
              </div>
              <SentimentBar counts={hotelCounts} />
              <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
                {SENTIMENTS.map((key) => (
                  <div key={key}>
                    <div className={`h-2 w-8 rounded-full ${SENTIMENT_COLOR[key].split(" ")[0]}`} />
                    <div className="mt-1 font-semibold">{sentimentShare(hotelCounts, key)}%</div>
                    <div className="text-[var(--on-surface-variant)]">{SENTIMENT_LABEL[key]}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <p className="mt-4 text-sm leading-6 text-[var(--on-surface-variant)]">
            {shortText(selectedHotel.aspects.all_aspects?.overview || activeSummary.overview, 260)}
          </p>
        </div>
      </section>

      <section className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
        <div className="mb-4 flex items-center justify-between gap-3">
          <h2 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Aspect-based review summary</h2>
          <button
            onClick={() => setDetailsOpen(true)}
            className="rounded-md border border-[var(--outline-variant)] px-3 py-1.5 text-xs font-semibold text-[var(--primary)] hover:bg-[var(--surface-container-low)]"
          >
            View full text
          </button>
        </div>
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-6">
          {aspectRows.map(({ aspect, summary }) => (
            <AspectCard
              key={aspect}
              aspect={aspect}
              summary={summary}
              active={aspect === selectedAspect}
              onClick={() => setSelectedAspect(aspect)}
            />
          ))}
        </div>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1fr_1fr_1fr]">
        <InsightList title="Top strengths" items={positiveItems} tone="good" />
        <InsightList title="Top concerns" items={concernItems} tone="bad" />
        <div className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">
            Selected aspect score
          </h3>
          <div className="mt-4 text-5xl font-bold text-[var(--primary)]">{selectedScore}<span className="text-lg text-[var(--on-surface-variant)]">/10</span></div>
          <p className="mt-3 text-sm leading-6 text-[var(--on-surface-variant)]">
            {aspectLabel(selectedAspect)} is summarized from {formatNumber(countTotal(activeSummary.counts))} evidence sentences.
          </p>
          <div className="mt-4">
            <SentimentBar counts={activeSummary.counts} />
          </div>
        </div>
      </section>

      <section className="grid gap-5 lg:grid-cols-2 xl:grid-cols-4">
        <TopicTable title="Top positive topics" rows={keywordRows(activeSummary.positive, activeSummary.counts.positive)} tone="positive" />
        <TopicTable title="Top negative topics" rows={keywordRows(activeSummary.negative, activeSummary.counts.negative)} tone="negative" />
        <section className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Word cloud</h3>
          <div className="mt-5 flex min-h-32 flex-wrap items-center justify-center gap-x-3 gap-y-2 text-center">
            {keywordRows(allText, countTotal(activeSummary.counts)).map((row, index) => (
              <span key={row.topic} className={index < 2 ? "text-3xl font-semibold text-teal-700" : "text-base text-[var(--on-surface-variant)]"}>
                {row.topic.toLowerCase()}
              </span>
            ))}
          </div>
        </section>
        <section className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Review language</h3>
          <div className="mt-4 space-y-3 text-sm">
            {[
              ["English", 56],
              ["Vietnamese", 22],
              ["Korean", 11],
              ["Chinese", 7],
              ["Others", 4],
            ].map(([label, value]) => (
              <div key={label}>
                <div className="mb-1 flex justify-between"><span>{label}</span><span>{value}%</span></div>
                <div className="h-2 rounded-full bg-slate-100"><div className="h-2 rounded-full bg-blue-500" style={{ width: `${value}%` }} /></div>
              </div>
            ))}
          </div>
        </section>
      </section>

      <section className="grid gap-5 xl:grid-cols-[1fr_1.2fr]">
        <section className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">Representative summaries</h3>
          <div className="mt-4 space-y-3">
            {SENTIMENTS.map((key) => (
              <div key={key} className={`rounded-md border p-3 ${SENTIMENT_COLOR[key].split(" ").slice(2).join(" ")}`}>
                <div className={`text-xs font-bold uppercase tracking-[0.1em] ${SENTIMENT_COLOR[key].split(" ")[1]}`}>
                  {SENTIMENT_LABEL[key]} · {formatNumber(activeSummary.counts[key])}
                </div>
                <p className="mt-2 text-sm leading-6 text-[var(--on-surface-variant)]">{shortText(activeSummary[key] || "No extracted text.", 210)}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-lg border border-[var(--outline-variant)] bg-white p-4 shadow-[var(--shadow-soft)]">
          <h3 className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--primary)]">AI recommendations for management</h3>
          <div className="mt-4 space-y-4">
            {concernItems.slice(0, 3).map((item, index) => (
              <article key={item} className="grid gap-3 rounded-lg border border-[var(--outline-variant)] p-3 md:grid-cols-[1fr_auto]">
                <div>
                  <div className="font-semibold text-[var(--primary)]">
                    {index === 0 ? "Prioritize highest-friction issue" : index === 1 ? "Create service recovery playbook" : "Monitor trend after operational fix"}
                  </div>
                  <p className="mt-1 text-sm leading-6 text-[var(--on-surface-variant)]">{shortText(item, 170)}</p>
                </div>
                <div className="self-start rounded-md bg-rose-50 px-3 py-1 text-xs font-bold text-rose-700">
                  {index === 0 ? "High" : "Medium"}
                </div>
              </article>
            ))}
          </div>
        </section>
      </section>

      {detailsOpen ? (
        <div className="fixed inset-0 z-[60] overflow-y-auto bg-black/40 p-4">
          <div className="mx-auto max-w-4xl rounded-xl bg-white p-5 shadow-2xl">
            <div className="flex items-start justify-between gap-4 border-b border-[var(--outline-variant)] pb-4">
              <div>
                <h2 className="text-2xl font-bold text-[var(--primary)]">Full summary · {aspectLabel(selectedAspect)}</h2>
                <p className="mt-1 text-sm text-[var(--on-surface-variant)]">Hotel {selectedHotel.hotel_id}</p>
              </div>
              <button onClick={() => setDetailsOpen(false)} className="rounded-md border border-[var(--outline-variant)] px-3 py-1.5 text-sm font-semibold">
                Close
              </button>
            </div>
            <div className="mt-5 grid gap-4 md:grid-cols-3">
              {SENTIMENTS.map((key) => (
                <section key={key} className="rounded-lg bg-[var(--surface-container-low)] p-4">
                  <h3 className={`text-sm font-bold uppercase tracking-[0.12em] ${SENTIMENT_COLOR[key].split(" ")[1]}`}>
                    {SENTIMENT_LABEL[key]}
                  </h3>
                  <p className="mt-3 whitespace-pre-wrap text-sm leading-7 text-[var(--on-surface)]">
                    {activeSummary[key] || "No extracted text."}
                  </p>
                </section>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
