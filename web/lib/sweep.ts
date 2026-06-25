// Types + metadata for the hyperparameter-sweep export produced by
// scripts/export_sweep_web_data.py. Consumed by the Optimality view
// (OptimalityView.tsx).
//
// Types + metadata for available sweep cells. Threshold and token-budget cells
// identify the recommended value while still showing the previous code default.
// The decision metric is macro ROUGE-1 F1 (split=all) scored with a FIXED
// denominator: every gold-bearing (aspect, entity) stays in the mean and empty
// system outputs count as ROUGE 0, so a tighter threshold that simply answers
// fewer instances is penalised, not flattered. Coverage is reported next to
// every point so any sparsity is visible.

import type { MethodId } from "@/lib/space";

export type Dataset = "space" | "hasos";
export type Phase = "threshold" | "tokabs";

export interface SweepPoint {
  value: number | string;
  rouge1: number;
  rouge2: number;
  rougeL: number;
  coverage: number | null;
  n_aspects: number;
  is_default: boolean;
  is_best: boolean;
}

export type VerdictStatus =
  | "default_optimal"
  | "switch"
  | "keep_default_coverage_artifact";

export interface SweepVerdict {
  status: VerdictStatus;
  default: number | string;
  best: number | string;
  best_r1: number;
  default_r1: number;
  best_cov?: number | null;
  default_cov?: number | null;
  delta: number;
}

export interface MethodSweep {
  default: number | string | null;
  best: number | string | null;
  points: SweepPoint[];
  verdict: SweepVerdict | null;
}

export interface PhaseSweep {
  methods: Partial<Record<MethodId, MethodSweep>>;
}

export type DatasetSweep = Partial<Record<Phase, PhaseSweep>>;

export interface SweepMethodMeta {
  short: string;
  label: string;
  color: string;
}

export interface SweepPhaseMeta {
  label: string;
  param: string;
  note: string;
}

export interface SweepDatasetMeta {
  label: string;
  maxBar: number;
}

export interface SweepData {
  generated_at: string;
  decision_metric: string;
  method_meta: Record<MethodId, SweepMethodMeta>;
  phase_meta: Record<Phase, SweepPhaseMeta>;
  dataset_meta: Record<Dataset, SweepDatasetMeta>;
  datasets: Partial<Record<Dataset, DatasetSweep>>;
}

export const VERDICT_LABEL: Record<VerdictStatus, string> = {
  default_optimal: "Default is optimal",
  switch: "Use recommended value",
  keep_default_coverage_artifact: "Keep default (low coverage)",
};
