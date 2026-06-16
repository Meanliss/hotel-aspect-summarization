export interface MethodBlock {
  method: string;
  by_split: Record<
    string,
    Record<string, { rouge1: number; rouge2: number; rougeL: number; n: number }>
  >;
  coverage: Record<string, unknown>;
}

export interface RougeResults {
  m1: MethodBlock;
  m2: MethodBlock;
  m3: MethodBlock;
  m4: MethodBlock;
}

export type Dataset = "space" | "hasos";
export type Split = "dev" | "test" | "all";

export const METHOD_META: Record<
  string,
  { label: string; short: string; desc: string; color: string }
> = {
  m1: {
    label: "M1 — SemAE gốc (extractive)",
    short: "M1",
    desc: "Raw top-k sentences selected by SemAE, joined as-is.",
    color: "slate",
  },
  m2: {
    label: "M2 — Trước sentiment (abstractive)",
    short: "M2",
    desc: "FLAN-T5 rewrites the selected evidence; no sentiment split.",
    color: "sky",
  },
  m3: {
    label: "M3 — Sau sentiment · Keyword",
    short: "M3",
    desc: "Sentiment split with a keyword lexicon, then FLAN-T5 rewrites each polarity.",
    color: "emerald",
  },
  m4: {
    label: "M4 — Sau sentiment · BERT-ABSA",
    short: "M4",
    desc: "Sentiment split with a BERT-ABSA classifier, then FLAN-T5 rewrites each polarity.",
    color: "violet",
  },
};

export const DATASET_META: Record<
  Dataset,
  { label: string; file: string; aspects: string[] }
> = {
  space: {
    label: "SPACE",
    file: "rouge_space.json",
    aspects: ["building", "cleanliness", "food", "location", "rooms", "service"],
  },
  hasos: {
    label: "HASOS",
    file: "rouge_hasos.json",
    aspects: ["facility", "amenity", "service", "experience"],
  },
};

export const ASPECT_LABEL: Record<string, string> = {
  building: "Building",
  cleanliness: "Cleanliness",
  food: "Food",
  location: "Location",
  rooms: "Rooms",
  service: "Service",
  facility: "Facility",
  amenity: "Amenity",
  experience: "Experience",
};

export const METRIC_LABEL: Record<string, string> = {
  rouge1: "ROUGE-1",
  rouge2: "ROUGE-2",
  rougeL: "ROUGE-L",
};
