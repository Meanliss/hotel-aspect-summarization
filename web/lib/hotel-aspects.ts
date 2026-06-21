export type SentimentKey = "positive" | "negative" | "neutral";

export interface SentimentCounts {
  positive: number;
  negative: number;
  neutral: number;
}

export interface HotelAspectSummary {
  counts: SentimentCounts;
  overview: string;
  positive: string;
  negative: string;
  neutral: string;
}

export interface HotelAspectRecord {
  hotel_id: string;
  aspects: Record<string, HotelAspectSummary>;
}

export interface HotelAspectDataset {
  dataset: string;
  source_file: string;
  generated_at: string;
  aspects: string[];
  hotels: HotelAspectRecord[];
}

export const ASPECT_LABELS: Record<string, string> = {
  all_aspects: "All aspects",
  facility: "Facility",
  service: "Service",
  amenity: "Amenity",
  experience: "Experience",
  loyalty: "Loyalty",
  branding: "Branding",
};

export const SENTIMENT_LABELS: Record<SentimentKey, string> = {
  positive: "Positive",
  negative: "Negative",
  neutral: "Neutral",
};

async function fetchJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Failed to load ${path} (${response.status})`);
  }
  return response.json();
}

export async function loadHotelAspectDataset(): Promise<HotelAspectDataset> {
  return fetchJson<HotelAspectDataset>("/data/hotel_aspect_summary.json");
}

export function sumCounts(values: SentimentCounts[]): SentimentCounts {
  return values.reduce(
    (total, item) => ({
      positive: total.positive + item.positive,
      negative: total.negative + item.negative,
      neutral: total.neutral + item.neutral,
    }),
    { positive: 0, negative: 0, neutral: 0 },
  );
}

export function countTotal(counts: SentimentCounts): number {
  return counts.positive + counts.negative + counts.neutral;
}

export function aspectLabel(aspect: string): string {
  return ASPECT_LABELS[aspect] ?? aspect.replaceAll("_", " ");
}
