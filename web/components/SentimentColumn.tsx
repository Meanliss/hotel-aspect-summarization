import type { EvidenceItem, Sentiment } from "@/lib/types";
import { EvidenceList } from "./EvidenceList";

const STYLES: Record<
  Sentiment,
  { box: string; label: string; title: string }
> = {
  positive: {
    box: "border-emerald-200 bg-emerald-50",
    label: "text-emerald-700",
    title: "Positive",
  },
  negative: {
    box: "border-rose-200 bg-rose-50",
    label: "text-rose-700",
    title: "Negative",
  },
  neutral: {
    box: "border-slate-200 bg-slate-100",
    label: "text-slate-600",
    title: "Neutral",
  },
};

export function SentimentColumn({
  sentiment,
  summary,
  evidence,
}: {
  sentiment: Sentiment;
  summary: string;
  evidence: EvidenceItem[];
}) {
  const s = STYLES[sentiment];
  const hasContent = Boolean(summary) || (evidence && evidence.length > 0);
  return (
    <div className={`flex-1 rounded-lg border p-3 ${s.box}`}>
      <div className={`text-xs font-semibold uppercase tracking-wide ${s.label}`}>
        {s.title}
      </div>
      {hasContent ? (
        <>
          {summary ? (
            <p className="mt-1 text-sm text-slate-700">{summary}</p>
          ) : (
            <p className="mt-1 text-sm italic text-slate-400">
              No {s.title.toLowerCase()} summary generated.
            </p>
          )}
          <EvidenceList items={evidence} />
        </>
      ) : (
        <p className="mt-1 text-sm italic text-slate-400">No signal.</p>
      )}
    </div>
  );
}
