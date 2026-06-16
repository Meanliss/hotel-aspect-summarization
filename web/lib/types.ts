// Shared types for the SemAE HASOS sentiment-summarization web app.
// These mirror 1:1 the JSON shape produced by scripts/export_web_data.py and
// the response of POST /analyze (see API_CONTRACT.md). Keep them in sync.

export type Sentiment = "positive" | "negative" | "neutral";

export interface EvidenceItem {
  sentence: string;
  score: number | null;
  rank: number | null;
  review_id: string | null;
}

export interface ChildSummaries {
  positive: string;
  negative: string;
  neutral: string;
}

export interface ChildEvidence {
  positive: EvidenceItem[];
  negative: EvidenceItem[];
  neutral: EvidenceItem[];
}

export interface ChildAspect {
  code: string; // e.g. "FAC_ROOM"
  scale: string; // human label e.g. "Room, Bed & Sleep Quality"
  description: string;
  summaries: ChildSummaries;
  evidence: ChildEvidence;
}

export interface ParentAspect {
  code: string; // e.g. "FACILITY"
  summary: string; // optional parent-level abstractive summary
  children: ChildAspect[];
}

export interface Entity {
  entity_id: string;
  entity_name: string;
  split: string;
  overall_summary: string;
  parents: ParentAspect[];
}

export interface TaxonomyEntry {
  code: string;
  group: string;
  scale: string;
  description: string;
}

export interface ExportData {
  run_id: string;
  generated_at: string;
  aspect_taxonomy: Record<string, TaxonomyEntry>;
  parent_order: string[];
  entities: Entity[];
}

export interface RunIndex {
  runs: string[];
}

// ---- Analyze API (backend implemented later) ----

export interface AnalyzeOptions {
  sentiment_backend?: "keyword" | "bert";
  split_sentiment?: boolean;
  max_tokens?: number;
}

export interface AnalyzeRequest {
  reviews: string[];
  entity_name?: string;
  options?: AnalyzeOptions;
}

// The analyze response is a single Entity (same shape used by Explore).
export type AnalyzeResponse = Entity;
