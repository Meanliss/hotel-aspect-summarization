#!/usr/bin/env python3
"""Score concrete HASOS metrics and write paper-ready tables.

The scorer combines:
  * automatic metrics from the normalized HASOS dataset,
  * DeepSeek judge outputs when available,
  * existing HASOS/SPACE ROUGE artifacts.

Usage:
    python scripts/score_concrete_metrics.py
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO / "reports" / "metrics"

DEFAULT_DATASET = METRICS_DIR / "concrete_metric_dataset.jsonl"
DEFAULT_JUDGMENTS = METRICS_DIR / "concrete_metric_judgments.jsonl"
DEFAULT_JSON = METRICS_DIR / "concrete_metrics_hasos.json"
DEFAULT_MD = METRICS_DIR / "concrete_metrics_hasos.md"
DEFAULT_TABLES = METRICS_DIR / "paper_tables.md"

HASOS_ROUGE = {
    "m1": REPO / "reports" / "rouge_m1_hasos.json",
    "m2": REPO / "reports" / "sweep" / "rouge_m2_hasos_tokabs_128.json",
    "m3": REPO / "reports" / "sweep" / "rouge_m3_hasos_tokabs_96.json",
    "m4": REPO / "reports" / "sweep" / "rouge_m4_hasos_tokabs_96.json",
}
SPACE_ROUGE = REPO / "reports" / "rouge_comparison_space.json"

DEFAULT_PRICE_INPUT_PER_1M = 0.435
DEFAULT_PRICE_OUTPUT_PER_1M = 0.87
MODEL_PRICING_USD_PER_1M = {
    "deepseek-v4-pro": (0.435, 0.87),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-chat": (0.14, 0.28),
}
WORD_RE = re.compile(r"\S+")


def read_jsonl(path: Path, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise SystemExit(f"missing file: {path}")
        return []
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text.rstrip() + "\n")
    tmp.replace(path)


def mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None
    return sum(values) / len(values)


def rate(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return num / den


def fmt(value: Any, nd: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def norm_sentence(text: str) -> str:
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def load_hasos_rouge() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for method, path in HASOS_ROUGE.items():
        if not path.exists():
            continue
        data = json.load(io.open(path, encoding="utf-8"))
        macro = data.get("by_split", {}).get("all", {}).get("MACRO", {})
        all_split = data.get("by_split", {}).get("all", {})
        covs = [
            v.get("coverage")
            for k, v in all_split.items()
            if isinstance(v, dict) and k not in {"MACRO", "GENERAL"} and v.get("coverage") is not None
        ]
        out[method] = {
            "rouge1": macro.get("rouge1"),
            "rouge2": macro.get("rouge2"),
            "rougeL": macro.get("rougeL"),
            "mean_coverage": mean(covs),
            "source": str(path.relative_to(REPO)),
        }
    return out


def load_space_rouge() -> dict[str, dict[str, Any]]:
    if not SPACE_ROUGE.exists():
        return {}
    data = json.load(io.open(SPACE_ROUGE, encoding="utf-8"))
    out = {}
    for key, cell in data.items():
        macro = cell.get("by_split", {}).get("all", {}).get("MACRO", {})
        out[key] = {
            "method": cell.get("method", key),
            "rouge1": macro.get("rouge1"),
            "rouge2": macro.get("rouge2"),
            "rougeL": macro.get("rougeL"),
        }
    return out


def automatic_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    result: dict[str, dict[str, Any]] = {}
    for method, bucket in sorted(by_method.items()):
        n = len(bucket)
        status_counts: dict[str, int] = defaultdict(int)
        evidence_counts = []
        evidence_topk_counts = []
        summary_words = []
        compression_ratios = []
        duplicate_ratios = []
        copied = 0
        for row in bucket:
            status = row.get("status") or ""
            status_counts[status] += 1
            copied += 1 if row.get("copied_from_evidence") else 0
            evidence_counts.append(float(row.get("evidence_count") or 0))
            evidence = row.get("evidence") or []
            evidence_topk_counts.append(float(len(evidence)))
            sw = float(row.get("summary_word_count") or word_count(row.get("summary", "")))
            ew = float(
                row.get("compression_evidence_word_count")
                or row.get("evidence_topk_word_count")
                or word_count(" ".join(e.get("sentence", "") for e in evidence))
            )
            summary_words.append(sw)
            if ew > 0:
                compression_ratios.append(sw / ew)
            if evidence:
                normalized = [norm_sentence(e.get("sentence", "")) for e in evidence]
                unique = len(set(normalized))
                duplicate_ratios.append(1.0 - unique / len(normalized))

        fallback_n = sum(v for k, v in status_counts.items() if k.startswith("fallback"))
        generated_n = status_counts.get("generated", 0) + status_counts.get("cached_existing_output", 0)
        is_extractive = method == "m1" or status_counts.get("extractive", 0) == n
        result[method] = {
            "method_label": bucket[0].get("method_label", method),
            "rows": n,
            "aspects": len({r.get("aspect") for r in bucket}),
            "sentiments": sorted({r.get("sentiment", "") for r in bucket}),
            "status_counts": dict(sorted(status_counts.items())),
            "generated_rate": None if is_extractive else rate(generated_n, n),
            "fallback_rate": None if is_extractive else rate(fallback_n, n),
            "copied_from_evidence_rate": rate(copied, n),
            "avg_evidence_count": mean(evidence_counts),
            "avg_evidence_topk_count": mean(evidence_topk_counts),
            "avg_summary_words": mean(summary_words),
            "avg_compression_ratio": mean(compression_ratios),
            "topk_duplicate_rate": mean(duplicate_ratios),
        }
    return result


def pass_rule(judgment: dict[str, Any]) -> bool:
    scores = judgment["scores"]
    flags = judgment["flags"]
    return (
        scores["evidence_support"] >= 4
        and scores["aspect_correctness"] >= 4
        and scores["sentiment_alignment"] >= 4
        and scores["coverage"] >= 3
        and not flags["unsupported_claim_present"]
        and not flags["sentiment_flip_present"]
    )


def judge_metrics(judgments: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judgments:
        by_method[row["method"]].append(row)

    result: dict[str, dict[str, Any]] = {}
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_usd": 0.0,
        "by_model": {},
    }
    for row in judgments:
        usage = row.get("usage") or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        model = row.get("model") or "unknown"
        input_price, output_price = MODEL_PRICING_USD_PER_1M.get(
            model, (DEFAULT_PRICE_INPUT_PER_1M, DEFAULT_PRICE_OUTPUT_PER_1M)
        )
        estimated = prompt / 1_000_000 * input_price + completion / 1_000_000 * output_price
        usage_totals["prompt_tokens"] += prompt
        usage_totals["completion_tokens"] += completion
        usage_totals["total_tokens"] += int(usage.get("total_tokens") or prompt + completion)
        usage_totals["estimated_usd"] += estimated
        by_model = usage_totals["by_model"].setdefault(
            model,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_usd": 0.0,
            },
        )
        by_model["prompt_tokens"] += prompt
        by_model["completion_tokens"] += completion
        by_model["total_tokens"] += int(usage.get("total_tokens") or prompt + completion)
        by_model["estimated_usd"] += estimated

    for method, bucket in sorted(by_method.items()):
        n = len(bucket)
        score_lists: dict[str, list[float]] = defaultdict(list)
        flag_counts: dict[str, int] = defaultdict(int)
        pass_count = 0
        evidence_total = 0
        relevant_count = 0
        support_count = 0
        cross_aspect_leak_count = 0
        sentiment_leak_count = 0
        verdict_counts: dict[str, int] = defaultdict(int)
        for row in bucket:
            judgment = row["judgment"]
            for key, value in judgment["scores"].items():
                score_lists[key].append(float(value))
            for key, value in judgment["flags"].items():
                if value:
                    flag_counts[key] += 1
            if pass_rule(judgment):
                pass_count += 1
            verdict_counts[judgment.get("verdict", "")] += 1
            for label in judgment.get("evidence_labels") or []:
                evidence_total += 1
                relevant_count += 1 if label.get("relevant") else 0
                support_count += 1 if label.get("supports_summary_claims") else 0
                cross_aspect_leak_count += 1 if label.get("cross_aspect_leakage") else 0
                sentiment_leak_count += 1 if label.get("sentiment_leakage") else 0

        result[method] = {
            "judged_rows": n,
            "avg_scores": {key: mean(vals) for key, vals in sorted(score_lists.items())},
            "pass_rate": rate(pass_count, n),
            "verdict_counts": dict(sorted(verdict_counts.items())),
            "unsupported_claim_rate": rate(flag_counts["unsupported_claim_present"], n),
            "aspect_mismatch_rate": rate(flag_counts["aspect_mismatch_present"], n),
            "sentiment_flip_rate": rate(flag_counts["sentiment_flip_present"], n),
            "major_theme_missing_rate": rate(flag_counts["major_theme_missing"], n),
            "major_theme_recall": None
            if rate(flag_counts["major_theme_missing"], n) is None
            else 1.0 - rate(flag_counts["major_theme_missing"], n),
            "generic_summary_rate": rate(flag_counts["generic_summary"], n),
            "evidence_p_at_5": rate(relevant_count, evidence_total),
            "evidence_support_at_5": rate(support_count, evidence_total),
            "cross_aspect_leakage_rate": rate(cross_aspect_leak_count, evidence_total),
            "sentiment_leakage_rate": rate(sentiment_leak_count, evidence_total),
        }
    return result, usage_totals


def merge_metrics(
    auto: dict[str, dict[str, Any]],
    judge: dict[str, dict[str, Any]],
    rouge: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    methods = sorted(set(auto) | set(judge) | set(rouge))
    merged = {}
    for method in methods:
        merged[method] = {
            "automatic": auto.get(method, {}),
            "judge": judge.get(method, {}),
            "rouge": rouge.get(method, {}),
        }
    return merged


def table_evidence_faithfulness(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Evidence And Faithfulness",
        "",
        "| Method | Judged n | Evidence P@5 | Support@5 | Cross-aspect leakage | Sentiment leakage | Claim support | Unsupported claims | Sentiment flip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, cell in metrics.items():
        j = cell.get("judge", {})
        scores = j.get("avg_scores", {})
        lines.append(
            f"| {method.upper()} | {fmt(j.get('judged_rows'), 0)} | "
            f"{fmt(j.get('evidence_p_at_5'))} | {fmt(j.get('evidence_support_at_5'))} | "
            f"{fmt(j.get('cross_aspect_leakage_rate'))} | {fmt(j.get('sentiment_leakage_rate'))} | "
            f"{fmt(scores.get('evidence_support'))} | {fmt(j.get('unsupported_claim_rate'))} | "
            f"{fmt(j.get('sentiment_flip_rate'))} |"
        )
    return lines


def table_utility(metrics: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Summary Utility By LLM Judge",
        "",
        "| Method | Pass rate | Aspect correctness | Sentiment alignment | Coverage | Specificity | Usefulness | Major theme recall | Generic rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, cell in metrics.items():
        j = cell.get("judge", {})
        scores = j.get("avg_scores", {})
        lines.append(
            f"| {method.upper()} | {fmt(j.get('pass_rate'))} | "
            f"{fmt(scores.get('aspect_correctness'))} | {fmt(scores.get('sentiment_alignment'))} | "
            f"{fmt(scores.get('coverage'))} | {fmt(scores.get('specificity'))} | "
            f"{fmt(scores.get('usefulness'))} | {fmt(j.get('major_theme_recall'))} | "
            f"{fmt(j.get('generic_summary_rate'))} |"
        )
    return lines


def table_production(metrics: dict[str, dict[str, Any]], usage: dict[str, Any]) -> list[str]:
    lines = [
        "## Production Readiness",
        "",
        "| Method | Rows | Generated rate | Fallback rate | Copied-from-evidence | Avg evidence | Compression | Duplicate@top5 | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, cell in metrics.items():
        a = cell.get("automatic", {})
        r = cell.get("rouge", {})
        lines.append(
            f"| {method.upper()} | {fmt(a.get('rows'), 0)} | {fmt(a.get('generated_rate'))} | "
            f"{fmt(a.get('fallback_rate'))} | {fmt(a.get('copied_from_evidence_rate'))} | "
            f"{fmt(a.get('avg_evidence_count'))} | {fmt(a.get('avg_compression_ratio'))} | "
            f"{fmt(a.get('topk_duplicate_rate'))} | {fmt(r.get('rouge1'))} | "
            f"{fmt(r.get('rouge2'))} | {fmt(r.get('rougeL'))} |"
        )
    lines += [
        "",
        "Generated/fallback rates are not applicable to M1 because it is an extractive SemAE baseline.",
        "",
        f"DeepSeek usage: prompt tokens `{usage.get('prompt_tokens', 0)}`, "
        f"completion tokens `{usage.get('completion_tokens', 0)}`, "
        f"estimated cost `${usage.get('estimated_usd', 0.0):.4f}`.",
    ]
    return lines


def table_space_appendix(space: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## SPACE ROUGE Appendix",
        "",
        "| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in sorted(space):
        cell = space[key]
        lines.append(
            f"| {cell.get('method', key)} | {fmt(cell.get('rouge1'))} | "
            f"{fmt(cell.get('rouge2'))} | {fmt(cell.get('rougeL'))} |"
        )
    return lines


def build_markdown(payload: dict[str, Any], detailed: bool) -> str:
    metrics = payload["methods"]
    usage = payload["deepseek_usage"]
    space = payload["space_rouge_appendix"]
    lines = [
        "# Concrete Metric Suite - HASOS",
        "",
        "Primary HASOS comparison uses M1 extractive baseline "
        "(SemAE epoch-20, B=40 words; no generative decoder) and the optimized "
        "synthesis bases selected by the sweep: M2 T=0.0075/B=128, M3 T=0.0055/B=96, "
        "M4 T=0.005/B=96.",
        "",
    ]
    if payload["judged_rows"] == 0:
        lines.append(
            "> LLM judge metrics are not populated yet. Run "
            "`scripts/run_concrete_metric_judge.py` with `DEEPSEEK_API_KEY` set."
        )
        lines.append("")
    lines += table_evidence_faithfulness(metrics)
    lines += [""]
    lines += table_utility(metrics)
    lines += [""]
    lines += table_production(metrics, usage)
    lines += [""]
    lines += table_space_appendix(space)
    if detailed:
        lines += [
            "",
            "## Notes",
            "",
            "- ABSA gold F1 is not reported because the available HASOS artifact does not expose sentence-level gold aspect-sentiment labels.",
            "- Judge-based metrics are reported separately from ROUGE to avoid treating semantic judgments as reference-summary overlap.",
            "- Production p95/cache/retry/JSON-repair metrics require explicit runtime instrumentation and are not inferred from incomplete logs.",
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

    dataset = read_jsonl(Path(args.dataset), required=True)
    judgments = read_jsonl(Path(args.judgments), required=False)
    auto = automatic_metrics(dataset)
    judge, usage = judge_metrics(judgments)
    hasos_rouge = load_hasos_rouge()
    space_rouge = load_space_rouge()
    methods = merge_metrics(auto, judge, hasos_rouge)
    payload = {
        "dataset": "hasos",
        "rubric_version": "concrete-v1",
        "source_rows": len(dataset),
        "judged_rows": len(judgments),
        "methods": methods,
        "deepseek_usage": usage,
        "deepseek_pricing_usd_per_1m": {
            "default_input": DEFAULT_PRICE_INPUT_PER_1M,
            "default_output": DEFAULT_PRICE_OUTPUT_PER_1M,
            "by_model": {
                model: {"input": prices[0], "output": prices[1]}
                for model, prices in sorted(MODEL_PRICING_USD_PER_1M.items())
            },
        },
        "space_rouge_appendix": space_rouge,
    }
    write_json(Path(args.json_out), payload)
    write_text(Path(args.md_out), build_markdown(payload, detailed=True))
    write_text(Path(args.tables_out), build_markdown(payload, detailed=False))
    print(f"written -> {Path(args.json_out).relative_to(REPO)}")
    print(f"written -> {Path(args.md_out).relative_to(REPO)}")
    print(f"written -> {Path(args.tables_out).relative_to(REPO)}")


if __name__ == "__main__":
    main()
