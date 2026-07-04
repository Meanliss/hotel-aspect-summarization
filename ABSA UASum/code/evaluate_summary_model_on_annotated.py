"""Evaluate final-summary generation on the manually annotated hotel ABSA set.

This script isolates the summary model from aspect extraction. It groups the
gold ABSA quads by hotel, aspect, and sentiment, calls the same final-summary
writer used by hotel_aspect_sentiment_pipeline.py, then scores the summaries
against the gold evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI

from hotel_aspect_sentiment_pipeline import (
    ASPECT_NAMES,
    SENTIMENTS,
    FinalSummaryAggregate,
    build_final_summary_metric_rows,
    build_final_summary_rows,
    clean_text,
    normalize_sentiment,
    strip_final_summary_internal_columns,
    write_csv,
    write_final_summary_json,
    write_final_summary_metrics_json,
    write_summary_metrics_csv,
)


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INPUT = WORKSPACE / "data_annotated_22hotels_460reviews_3594quad_20260518.csv"
DEFAULT_OUT_DIR = WORKSPACE / "annotated_summary_model_eval"
DEFAULT_QWEN_BASE_URL = "http://localhost:8000/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-9B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline final-summary quality on annotated ABSA gold quads."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-api-key", default="")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--timeout-sec", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--final-summary-batch-size", type=int, default=10)
    parser.add_argument("--final-summary-workers", type=int, default=3)
    parser.add_argument("--final-summary-max-output-tokens", type=int, default=7000)
    parser.add_argument("--final-summary-samples-per-sentiment", type=int, default=40)
    parser.add_argument("--final-summary-sample-chars", type=int, default=260)
    parser.add_argument("--final-summary-progress-every-batches", type=int, default=5)
    parser.add_argument(
        "--judge-batch-size",
        type=int,
        default=1,
        help="Default 1 keeps Qwen judge from collapsing several rows into one combined verdict.",
    )
    parser.add_argument("--judge-max-samples", type=int, default=12)
    parser.add_argument("--judge-max-sample-chars", type=int, default=240)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--skip-qwen", action="store_true", help="Use deterministic local summaries.")
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--disable-judge", action="store_true")
    parser.add_argument("--disable-bertscore", action="store_true")
    parser.add_argument("--bertscore-language", choices=["vi", "en"], default="en")
    args = parser.parse_args()
    if not args.qwen_api_key:
        args.qwen_api_key = os.getenv("QWEN_API_KEY", "local-dev-key")
    return args


def normalize_gold_aspect(value: Any) -> str:
    aspect = clean_text(value).lower()
    return aspect if aspect in ASPECT_NAMES else ""


def choose_evidence(row: dict[str, str]) -> str:
    for column in ("evidence", "opinion_term", "text", "normalized_text", "reviews"):
        value = clean_text(row.get(column, ""))
        if value and value.lower() not in {"text", "nan", "none", "null"}:
            return value
    return ""


def load_gold_rows(path: Path, max_rows: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if max_rows and len(rows) >= max_rows:
                break
            aspect = normalize_gold_aspect(raw.get("aspect_category", ""))
            sentiment_raw = clean_text(raw.get("sentiment", ""))
            sentiment = normalize_sentiment(sentiment_raw)
            evidence = choose_evidence(raw)
            hotel_key = clean_text(raw.get("hotel_key", ""))
            if not aspect:
                skipped["missing_aspect"] += 1
                continue
            if not sentiment_raw:
                skipped["missing_sentiment"] += 1
                continue
            if not evidence:
                skipped["missing_evidence"] += 1
                continue
            if not hotel_key:
                skipped["missing_hotel_key"] += 1
                continue
            rows.append(
                {
                    "hotel_key": hotel_key,
                    "review_id": clean_text(raw.get("review_id", "")),
                    "sub_ref_id": clean_text(raw.get("sub_ref_id", "")),
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "evidence": evidence,
                    "aspect_term": clean_text(raw.get("aspect_term", "")),
                    "opinion_term": clean_text(raw.get("opinion_term", "")),
                }
            )
    return rows, {"loaded_rows": len(rows), "skipped": dict(skipped)}


def build_gold_aggregate(
    rows: list[dict[str, Any]],
    sample_limit: int,
    sample_char_limit: int,
) -> FinalSummaryAggregate:
    aggregate = FinalSummaryAggregate(sample_limit=sample_limit, sample_char_limit=sample_char_limit)
    for row in rows:
        aggregate.add(
            entity_id=row["hotel_key"],
            data_source="annotated_gold",
            hotel_id=row["hotel_key"],
            source_file="data_annotated",
            aspect=row["aspect"],
            sentiment=row["sentiment"],
            text=row["evidence"],
            confidence=1.0,
            keep_reference_text=True,
            reroute_aspect=False,
        )
    return aggregate


def collect_reference_samples(rows: list[dict[str, Any]], limit: int, max_chars: int) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"counts": Counter(), "samples": defaultdict(list)}
    )
    for row in rows:
        key = (row["hotel_key"], row["aspect"])
        sentiment = row["sentiment"]
        grouped[key]["counts"][sentiment] += 1
        if len(grouped[key]["samples"][sentiment]) < limit:
            grouped[key]["samples"][sentiment].append(clean_text(row["evidence"])[:max_chars])
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in grouped.items():
        out[key] = {
            "counts": {sentiment: int(value["counts"][sentiment]) for sentiment in SENTIMENTS},
            "samples": {sentiment: list(value["samples"][sentiment]) for sentiment in SENTIMENTS},
        }
    return out


def summary_text(row: dict[str, Any]) -> str:
    parts = [
        clean_text(row.get("positive_summary", "")),
        clean_text(row.get("negative_summary", "")),
        clean_text(row.get("neutral_summary", "")),
    ]
    return " ".join(part for part in parts if part)


def parse_judge_items(content: str, expected: int) -> list[dict[str, Any]]:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        content = content.removesuffix("```").strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        payload = json.loads(content[start : end + 1])
    raw_items = payload.get("items", [])
    if not raw_items and expected == 1:
        raw_items = [{**payload, "id": 0}]
    by_id = {int(item.get("id", -1)): item for item in raw_items if isinstance(item, dict)}
    out = []
    for idx in range(expected):
        item = by_id.get(idx, {})
        out.append(
            {
                "id": idx,
                "faithfulness": clamp_score(item.get("faithfulness")),
                "coverage": clamp_score(item.get("coverage")),
                "specificity": clamp_score(item.get("specificity")),
                "overall": clamp_score(item.get("overall")),
                "main_issue": clean_text(item.get("main_issue", ""))[:300],
            }
        )
    return out


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except Exception:
        return 0
    return max(0, min(5, score))


def judge_summary_batches(
    rows: list[dict[str, Any]],
    references: dict[tuple[str, str], dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    client = OpenAI(base_url=args.qwen_base_url, api_key=args.qwen_api_key, timeout=args.timeout_sec)
    out: list[dict[str, Any]] = []
    batches = [rows[start : start + args.judge_batch_size] for start in range(0, len(rows), args.judge_batch_size)]
    system = (
        "You evaluate Vietnamese hotel insight summaries against gold ABSA evidence. Return strict JSON only. "
        "Evidence can be English or Vietnamese. Score 1-5: faithfulness means no unsupported or wrong claim; "
        "coverage means the summary captures the main recurring evidence themes; specificity means it is concrete "
        "and useful for hotel business users. Overall should balance the three scores."
    )
    for batch_idx, batch in enumerate(batches, start=1):
        payload_items = []
        for idx, row in enumerate(batch):
            key = (clean_text(row.get("hotel_id", "")), clean_text(row.get("aspect", "")))
            ref = references.get(key, {"counts": {}, "samples": {}})
            payload_items.append(
                {
                    "id": idx,
                    "hotel_id": row.get("hotel_id", ""),
                    "aspect": row.get("aspect", ""),
                    "gold_counts": ref.get("counts", {}),
                    "summary": summary_text(row)[:1400],
                    "gold_evidence_samples": ref.get("samples", {}),
                }
            )
        user = (
            "Return exactly this schema:\n"
            '{"items":[{"id":0,"faithfulness":5,"coverage":4,"specificity":4,"overall":4,'
            '"main_issue":"short Vietnamese note"}]}\n\n'
            f"Payload:\n{json.dumps({'items': payload_items}, ensure_ascii=False)}"
        )
        last_error = ""
        judged: list[dict[str, Any]] | None = None
        for attempt in range(args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                rsp = client.chat.completions.create(
                    model=args.qwen_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=2500,
                    extra_body=extra_body,
                )
                judged = parse_judge_items(rsp.choices[0].message.content or "", len(batch))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < args.max_retries:
                    time.sleep(1.5 * (attempt + 1))
        if judged is None:
            judged = [
                {
                    "id": idx,
                    "faithfulness": 0,
                    "coverage": 0,
                    "specificity": 0,
                    "overall": 0,
                    "main_issue": f"judge_failed: {last_error[:220]}",
                }
                for idx in range(len(batch))
            ]
        for row, item in zip(batch, judged):
            out.append(
                {
                    "hotel_id": row.get("hotel_id", ""),
                    "aspect": row.get("aspect", ""),
                    "faithfulness": item["faithfulness"],
                    "coverage": item["coverage"],
                    "specificity": item["specificity"],
                    "overall": item["overall"],
                    "main_issue": item["main_issue"],
                }
            )
        if batch_idx == 1 or batch_idx % 5 == 0 or batch_idx == len(batches):
            print(f"[judge] completed {batch_idx}/{len(batches)} batches", flush=True)
    return out


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score_fields = ["faithfulness", "coverage", "specificity", "overall"]
    by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_aspect[str(row.get("aspect", ""))].append(row)

    def stats(items: list[dict[str, Any]], field: str) -> float:
        values = [float(item.get(field, 0) or 0) for item in items]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "rows": len(rows),
        "overall": {field: stats(rows, field) for field in score_fields},
        "by_aspect": {
            aspect: {"rows": len(items), **{field: stats(items, field) for field in score_fields}}
            for aspect, items in sorted(by_aspect.items())
        },
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    gold_rows, load_stats = load_gold_rows(Path(args.input), args.max_rows)
    aggregate = build_gold_aggregate(
        gold_rows,
        args.final_summary_samples_per_sentiment,
        args.final_summary_sample_chars,
    )
    print(f"[load] rows={len(gold_rows):,} skipped={load_stats['skipped']}", flush=True)
    summary_rows_with_keys = build_final_summary_rows(aggregate, args)
    metric_rows = build_final_summary_metric_rows(
        aggregate,
        summary_rows_with_keys,
        args.bertscore_language,
        not args.disable_bertscore,
    )
    summary_rows = [dict(row) for row in summary_rows_with_keys]
    strip_final_summary_internal_columns(summary_rows)

    references = collect_reference_samples(gold_rows, args.judge_max_samples, args.judge_max_sample_chars)
    judge_rows: list[dict[str, Any]] = []
    judge_stats: dict[str, Any] = {}
    if not args.disable_judge and not args.skip_qwen:
        judge_rows = judge_summary_batches(summary_rows, references, args)
        judge_stats = aggregate_scores(judge_rows)

    stats = {
        "input": str(Path(args.input).resolve()),
        "elapsed_sec": round(time.time() - start, 3),
        "load_stats": load_stats,
        "summary_rows": len(summary_rows),
        "summary_model": "local_fallback" if args.skip_qwen else args.qwen_model,
        "judge_stats": judge_stats,
        "metric_note": (
            "ROUGE/coverage are lexical and can be low when Vietnamese summaries are compared "
            "with English gold evidence. Qwen judge scores are the main summary-quality signal."
        ),
    }

    write_csv(out_dir / "gold_qwen_final_summaries.csv", summary_rows)
    write_final_summary_json(out_dir / "gold_qwen_final_summaries.json", summary_rows, stats)
    write_summary_metrics_csv(out_dir / "gold_qwen_final_summary_metrics.csv", metric_rows)
    write_final_summary_metrics_json(out_dir / "gold_qwen_final_summary_metrics.json", metric_rows, stats)
    if judge_rows:
        write_csv(out_dir / "gold_qwen_final_summary_judge.csv", judge_rows)
    (out_dir / "summary_eval_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir.resolve()), **stats}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
