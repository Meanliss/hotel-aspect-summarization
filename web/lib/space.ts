// Types + metadata for the SPACE 4-method export produced by
// scripts/export_space_4method_web_data.py. Consumed by the Explore view
// (SpaceExplore.tsx) and the Results view (ResultsView.tsx).

export const SPACE_ASPECTS = [
  "building",
  "cleanliness",
  "food",
  "location",
  "rooms",
  "service",
] as const;

export type SpaceAspect = (typeof SPACE_ASPECTS)[number];
export type MethodId = "m1" | "m2" | "m3" | "m4";

export const METHOD_IDS: MethodId[] = ["m1", "m2", "m3", "m4"];

// One aspect cell per method. Extractive/flat methods carry a single `overall`
// string; sentiment-split methods (m3/m4) carry positive/negative strings.
export interface AspectCell {
  overall?: string;
  positive?: string;
  negative?: string;
}

export interface MethodOutput {
  overall: string;
  aspects: Partial<Record<SpaceAspect, AspectCell>>;
}

export interface EntityGold {
  overall: string[];
  aspects: Partial<Record<SpaceAspect, string[]>>;
}

export interface SpaceEntity {
  entity_id: string;
  entity_name: string;
  split: string;
  gold: EntityGold;
  methods: Record<MethodId, MethodOutput>;
}

export interface MethodMeta {
  label: string;
  short: string;
  desc: string;
  color: string;
}

// ROUGE comparison blob (same shape as reports/rouge_comparison_space.json):
// { m1: { by_split: { all: { <aspect>: {rouge1,..}, MACRO: {..}, GENERAL: {..} } } }, ... }
export interface RougeCell {
  rouge1: number;
  rouge2: number;
  rougeL: number;
  n?: number;
}
export interface RougeMethodBlock {
  method: string;
  by_split: Record<string, Record<string, RougeCell>>;
}
export type RougeComparison = Record<MethodId, RougeMethodBlock>;

export interface SpaceData {
  dataset: string;
  title: string;
  generated_at: string;
  aspects: SpaceAspect[];
  methods: MethodId[];
  method_meta: Record<MethodId, MethodMeta>;
  rouge: RougeComparison | null;
  entities: SpaceEntity[];
}

export const ASPECT_LABEL: Record<string, string> = {
  building: "Building",
  cleanliness: "Cleanliness",
  food: "Food",
  location: "Location",
  rooms: "Rooms",
  service: "Service",
};

export const METRIC_LABEL: Record<string, string> = {
  rouge1: "ROUGE-1",
  rouge2: "ROUGE-2",
  rougeL: "ROUGE-L",
};

// Fallback method metadata if the export omits it.
export const METHOD_META: Record<MethodId, MethodMeta> = {
  m1: {
    label: "M1 — Extractive (SemAE)",
    short: "M1",
    desc: "Raw top-ranked SemAE sentences, joined as-is.",
    color: "slate",
  },
  m2: {
    label: "M2 — Abstractive (no sentiment)",
    short: "M2",
    desc: "FLAN-T5 rewrites the selected evidence; no sentiment split.",
    color: "sky",
  },
  m3: {
    label: "M3 — Sentiment split · Keyword",
    short: "M3",
    desc: "Keyword sentiment split, then FLAN-T5 rewrites each polarity.",
    color: "emerald",
  },
  m4: {
    label: "M4 — Sentiment split · BERT-ABSA",
    short: "M4",
    desc: "BERT-ABSA sentiment split, then FLAN-T5 rewrites each polarity.",
    color: "violet",
  },
};

export const COLOR_BAR: Record<string, string> = {
  slate: "bg-slate-500",
  sky: "bg-sky-500",
  emerald: "bg-emerald-500",
  violet: "bg-violet-500",
};
export const COLOR_TEXT: Record<string, string> = {
  slate: "text-slate-600",
  sky: "text-sky-600",
  emerald: "text-emerald-600",
  violet: "text-violet-600",
};
export const COLOR_BG_LIGHT: Record<string, string> = {
  slate: "bg-slate-50",
  sky: "bg-sky-50",
  emerald: "bg-emerald-50",
  violet: "bg-violet-50",
};
export const COLOR_RING: Record<string, string> = {
  slate: "ring-slate-300",
  sky: "ring-sky-300",
  emerald: "ring-emerald-300",
  violet: "ring-violet-300",
};

export async function loadSpaceData(): Promise<SpaceData> {
  const res = await fetch("/data/space_4method.json", { cache: "no-store" });
  if (!res.ok) throw new Error(`Failed to load SPACE data (${res.status})`);
  return res.json();
}
