"""ROUGE for generated SPACE global summaries.

Input is a CSV with one row per entity and a ``global_summary`` column generated
by the pipeline after aspect summaries are created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import evaluate_space_global_combined_rouge as combined
import evaluate_space_pipeline_official_rouge as official


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = WORKSPACE / "results" / "space_pipeline" / "space_qwen_native_full_20260626"
DEFAULT_RUN_NAME = "space_qwen_native_full_20260626"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated global SPACE summaries with ROUGE.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--global-summary-csv", default="")
    parser.add_argument("--gold-dir", default=str(official.SPACE_GOLD_DIR))
    parser.add_argument("--split-source-dir", default=str(official.SPACE_SPLIT_SOURCE))
    parser.add_argument("--rouge-home", default=str(official.ROUGE_HOME))
    parser.add_argument("--summary-field", default="global_summary")
    parser.add_argument("--out-prefix", default="")
    return parser.parse_args()


def clean_id(value: Any) -> str:
    text = official.clean_text(value)
    return text[len("space_") :] if text.startswith("space_") else text


def load_global_systems(path: Path, summary_field: str) -> dict[str, str]:
    df = pd.read_csv(path).fillna("")
    required = {"hotel_id", summary_field}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    systems = {}
    for _, row in df.iterrows():
        entity_id = clean_id(row.get("hotel_id"))
        summary = official.clean_text(row.get(summary_field))
        if entity_id and summary:
            systems[entity_id] = summary
    return systems


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SPACE Generated Global Summary ROUGE",
        "",
        "Metric: pyrouge / ROUGE-1.5.5 F1.",
        "",
        "System summary = generated global summary per entity.",
        "Gold reference = concatenated six SPACE gold aspect summaries per entity.",
        "",
        f"Source CSV: `{payload['source_csv']}`",
        "",
        "| Split | ROUGE-1 | ROUGE-2 | ROUGE-L | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("dev", "test", "all"):
        row = payload["by_split"].get(split, {})
        lines.append(
            f"| {split} | {fmt(row.get('rouge1'))} | {fmt(row.get('rouge2'))} | "
            f"{fmt(row.get('rougeL'))} | {row.get('n', 0)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_name = args.run_name
    global_csv = Path(args.global_summary_csv) if args.global_summary_csv else results_dir / f"{run_name}_global_summary.csv"
    out_prefix = args.out_prefix or f"{run_name}_generated_global_summary_rouge"
    out_json = results_dir / f"{out_prefix}.json"
    out_md = results_dir / f"{out_prefix}.md"

    rouge_home = Path(args.rouge_home)
    official.apply_rouge_patch(rouge_home)
    split_map = official.load_split_map(Path(args.split_source_dir))
    aspect_refs = official.load_gold_refs(Path(args.gold_dir))
    refs = combined.combine_refs(aspect_refs)
    systems = load_global_systems(global_csv, args.summary_field)
    results = combined.evaluate_global(systems, refs, split_map, rouge_home)

    payload = {
        "metric_type": "space_generated_global_summary_rouge",
        "source_csv": str(global_csv.resolve()),
        "gold_dir": str(Path(args.gold_dir).resolve()),
        "rouge_home": str(rouge_home.resolve()),
        "summary_field": args.summary_field,
        "system_entities": len(systems),
        "gold_entities": len(refs),
        "by_split": results,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_md, payload)
    print(json.dumps({"json": str(out_json), "md": str(out_md), "all": results["all"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
