#!/usr/bin/env python3
"""Score SPACE LLM-judge metrics and write paper-ready tables."""
from __future__ import annotations

import argparse
from pathlib import Path

import score_concrete_metrics as scm

REPO = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO / "reports" / "metrics"

DEFAULT_DATASET = METRICS_DIR / "space_metric_dataset.jsonl"
DEFAULT_JUDGMENTS = METRICS_DIR / "space_metric_judgments.jsonl"
DEFAULT_JSON = METRICS_DIR / "space_metrics.json"
DEFAULT_MD = METRICS_DIR / "space_metrics.md"
DEFAULT_TABLES = METRICS_DIR / "space_paper_tables.md"


def build_markdown(payload: dict, detailed: bool) -> str:
    metrics = payload["methods"]
    usage = payload["deepseek_usage"]
    lines = [
        "# Concrete Metric Suite - SPACE",
        "",
        "SPACE comparison uses the original 6 aspect groups "
        "(building, cleanliness, food, location, rooms, service). M1 is the "
        "extractive SemAE baseline; M2 rewrites evidence without sentiment split; "
        "M3 and M4 evaluate keyword and BERT-ABSA sentiment splits.",
        "",
    ]
    if payload["judged_rows"] == 0:
        lines += [
            "> LLM judge metrics are not populated yet. Run "
            "`scripts/run_concrete_metric_judge.py` with `DEEPSEEK_API_KEY` set.",
            "",
        ]
    lines += scm.table_evidence_faithfulness(metrics)
    lines += [""]
    lines += scm.table_utility(metrics)
    lines += [""]
    lines += scm.table_production(metrics, usage)
    if detailed:
        lines += [
            "",
            "## Notes",
            "",
            "- SPACE judge uses aspect-level and sentiment-level outputs only; entity-level overall summaries are not included in this table.",
            "- M2 cached outputs are counted as generated outputs because they are restored generated summaries, not extractive fallbacks.",
            "- SPACE has ROUGE references for the six aspect groups, but not gold sentiment-level labels for ABSA F1 or sentiment-split F1.",
        ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--judgments", default=str(DEFAULT_JUDGMENTS))
    parser.add_argument("--json-out", default=str(DEFAULT_JSON))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--tables-out", default=str(DEFAULT_TABLES))
    args = parser.parse_args()

    dataset = scm.read_jsonl(Path(args.dataset), required=True)
    judgments = scm.read_jsonl(Path(args.judgments), required=False)
    auto = scm.automatic_metrics(dataset)
    judge, usage = scm.judge_metrics(judgments)
    space_rouge = scm.load_space_rouge()
    methods = scm.merge_metrics(auto, judge, space_rouge)
    payload = {
        "dataset": "space",
        "rubric_version": "concrete-v1",
        "source_rows": len(dataset),
        "judged_rows": len(judgments),
        "methods": methods,
        "deepseek_usage": usage,
        "deepseek_pricing_usd_per_1m": {
            "default_input": scm.DEFAULT_PRICE_INPUT_PER_1M,
            "default_output": scm.DEFAULT_PRICE_OUTPUT_PER_1M,
            "by_model": {
                model: {"input": prices[0], "output": prices[1]}
                for model, prices in sorted(scm.MODEL_PRICING_USD_PER_1M.items())
            },
        },
    }
    scm.write_json(Path(args.json_out), payload)
    scm.write_text(Path(args.md_out), build_markdown(payload, detailed=True))
    scm.write_text(Path(args.tables_out), build_markdown(payload, detailed=False))
    print(f"written -> {Path(args.json_out).relative_to(REPO)}")
    print(f"written -> {Path(args.md_out).relative_to(REPO)}")
    print(f"written -> {Path(args.tables_out).relative_to(REPO)}")


if __name__ == "__main__":
    main()
