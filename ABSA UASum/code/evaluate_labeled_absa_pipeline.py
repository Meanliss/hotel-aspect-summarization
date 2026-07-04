"""Evaluate the hotel ABSA pipeline against manually annotated quadruples.

This runner treats data_annotated_*.csv as a labeled test set.  Each row is one
gold ABSA unit; the pipeline predicts aspect + sentiment from the evidence text,
then we write evaluation reports and summary files for gold and predicted labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hotel_aspect_sentiment_pipeline import (
    ASPECT_NAMES,
    SENTIMENTS,
    FinalSummaryAggregate,
    QwenClassifier,
    build_final_summary_rows,
    business_final_sentiment_summary,
    clean_text,
    fallback_classify,
    normalize_sentiment,
    write_csv,
    write_final_summary_json,
)


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INPUT = WORKSPACE / "data_annotated_22hotels_460reviews_3594quad_20260518.csv"
DEFAULT_OUT_DIR = WORKSPACE / "labeled_absa_eval"
DEFAULT_QWEN_BASE_URL = "http://localhost:8000/v1"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-9B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pipeline labels and summaries on manually annotated ABSA data."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--mode", choices=["qwen", "local"], default="qwen")
    parser.add_argument(
        "--text-column",
        choices=["evidence", "text", "normalized_text", "opinion_term"],
        default="evidence",
        help="Annotated row field used as the classifier input.",
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-api-key", default="")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--timeout-sec", type=float, default=45.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=2200)
    parser.add_argument("--max-sentence-chars", type=int, default=700)
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--summary-samples-per-sentiment", type=int, default=40)
    parser.add_argument("--summary-sample-chars", type=int, default=220)
    args = parser.parse_args()
    if not args.qwen_api_key:
        args.qwen_api_key = os.getenv("QWEN_API_KEY", "local-dev-key")
    return args


def normalize_gold_aspect(value: Any) -> str:
    aspect = clean_text(value).lower()
    return aspect if aspect in ASPECT_NAMES else ""


def choose_text(row: dict[str, str], preferred: str) -> str:
    for column in [preferred, "evidence", "text", "normalized_text", "opinion_term"]:
        value = clean_text(row.get(column, ""))
        if value and value.lower() not in {"text", "nan", "none", "null"}:
            return value
    return ""


def load_gold_rows(path: Path, text_column: str, max_rows: int = 0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if max_rows and len(rows) >= max_rows:
                break
            text = choose_text(raw, text_column)
            gold_aspect = normalize_gold_aspect(raw.get("aspect_category", ""))
            raw_sentiment = clean_text(raw.get("sentiment", ""))
            gold_sentiment = normalize_sentiment(raw_sentiment)
            if not text:
                skipped["empty_text"] += 1
                continue
            if not gold_aspect:
                skipped["missing_aspect"] += 1
                continue
            if not raw_sentiment:
                skipped["missing_sentiment"] += 1
                continue
            rows.append(
                {
                    "ref_id": clean_text(raw.get("ref_id", "")),
                    "sub_ref_id": clean_text(raw.get("sub_ref_id", "")),
                    "review_id": clean_text(raw.get("review_id", "")),
                    "hotel_key": clean_text(raw.get("hotel_key", "")),
                    "text_for_prediction": text,
                    "gold_aspect": gold_aspect,
                    "gold_sentiment": gold_sentiment,
                    "gold_aspect_raw": clean_text(raw.get("aspect_category", "")),
                    "gold_sentiment_raw": raw_sentiment,
                    "aspect_term": clean_text(raw.get("aspect_term", "")),
                    "opinion_term": clean_text(raw.get("opinion_term", "")),
                    "evidence": clean_text(raw.get("evidence", "")),
                }
            )
    stats = {"skipped": dict(skipped), "loaded_rows": len(rows)}
    return rows, stats


def qwen_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        skip_qwen=False,
        qwen_base_url=args.qwen_base_url,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        max_output_tokens=args.max_output_tokens,
        max_sentence_chars=args.max_sentence_chars,
        qwen_enable_thinking=args.qwen_enable_thinking,
        batch_size=args.batch_size,
    )


def predict_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    if args.mode == "local":
        for row in rows:
            pred = fallback_classify(row["text_for_prediction"])
            row["pred_aspect"] = pred["aspect"]
            row["pred_sentiment"] = pred["sentiment"]
            row["pred_confidence"] = pred["confidence"]
            row["pred_reason"] = pred.get("reason_short", "")
        return

    classifier = QwenClassifier(qwen_args(args))
    total_batches = (len(rows) + args.batch_size - 1) // args.batch_size
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        batch_num = (start // args.batch_size) + 1
        if batch_num == 1 or batch_num % 10 == 0 or batch_num == total_batches:
            print(f"[eval-qwen] batch {batch_num}/{total_batches} rows={len(batch)}", flush=True)
        predictions = classifier.classify_batch([row["text_for_prediction"] for row in batch])
        for row, pred in zip(batch, predictions):
            row["pred_aspect"] = pred["aspect"]
            row["pred_sentiment"] = pred["sentiment"]
            row["pred_confidence"] = pred["confidence"]
            row["pred_reason"] = pred.get("reason_short", "")


def pct(value: int, total: int) -> float:
    return round(value / total, 6) if total else 0.0


def metrics_for(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    aspect_correct = sum(1 for row in rows if row["gold_aspect"] == row.get("pred_aspect"))
    sentiment_correct = sum(1 for row in rows if row["gold_sentiment"] == row.get("pred_sentiment"))
    both_correct = sum(
        1
        for row in rows
        if row["gold_aspect"] == row.get("pred_aspect")
        and row["gold_sentiment"] == row.get("pred_sentiment")
    )
    by_aspect: dict[str, dict[str, Any]] = {}
    for aspect in ASPECT_NAMES:
        subset = [row for row in rows if row["gold_aspect"] == aspect]
        if not subset:
            continue
        by_aspect[aspect] = {
            "support": len(subset),
            "aspect_accuracy": pct(sum(1 for row in subset if row.get("pred_aspect") == aspect), len(subset)),
            "sentiment_accuracy": pct(
                sum(1 for row in subset if row.get("pred_sentiment") == row["gold_sentiment"]),
                len(subset),
            ),
            "joint_accuracy": pct(
                sum(
                    1
                    for row in subset
                    if row.get("pred_aspect") == aspect
                    and row.get("pred_sentiment") == row["gold_sentiment"]
                ),
                len(subset),
            ),
        }

    return {
        "total_rows": total,
        "aspect_correct": aspect_correct,
        "sentiment_correct": sentiment_correct,
        "joint_correct": both_correct,
        "aspect_accuracy": pct(aspect_correct, total),
        "sentiment_accuracy": pct(sentiment_correct, total),
        "joint_accuracy": pct(both_correct, total),
        "gold_aspect_counts": dict(Counter(row["gold_aspect"] for row in rows)),
        "pred_aspect_counts": dict(Counter(row.get("pred_aspect", "") for row in rows)),
        "gold_sentiment_counts": dict(Counter(row["gold_sentiment"] for row in rows)),
        "pred_sentiment_counts": dict(Counter(row.get("pred_sentiment", "") for row in rows)),
        "by_gold_aspect": by_aspect,
    }


def confusion_rows(rows: list[dict[str, Any]], field: str, labels: list[str]) -> list[dict[str, Any]]:
    matrix = defaultdict(Counter)
    for row in rows:
        matrix[row[f"gold_{field}"]][row.get(f"pred_{field}", "")] += 1
    out = []
    for gold in labels:
        item = {"gold": gold}
        for pred in labels:
            item[pred] = matrix[gold][pred]
        extra = sum(count for pred, count in matrix[gold].items() if pred not in labels)
        if extra:
            item["other"] = extra
        out.append(item)
    return out


def mismatch_rows(rows: list[dict[str, Any]], limit: int = 250) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        aspect_ok = row["gold_aspect"] == row.get("pred_aspect")
        sentiment_ok = row["gold_sentiment"] == row.get("pred_sentiment")
        if aspect_ok and sentiment_ok:
            continue
        out.append(
            {
                "sub_ref_id": row["sub_ref_id"],
                "hotel_key": row["hotel_key"],
                "gold_aspect": row["gold_aspect"],
                "pred_aspect": row.get("pred_aspect", ""),
                "gold_sentiment": row["gold_sentiment"],
                "pred_sentiment": row.get("pred_sentiment", ""),
                "pred_confidence": row.get("pred_confidence", ""),
                "aspect_term": row["aspect_term"],
                "opinion_term": row["opinion_term"],
                "text_for_prediction": row["text_for_prediction"],
            }
        )
        if len(out) >= limit:
            break
    return out


def prediction_output_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "sub_ref_id": row["sub_ref_id"],
                "ref_id": row["ref_id"],
                "review_id": row["review_id"],
                "hotel_key": row["hotel_key"],
                "gold_aspect": row["gold_aspect"],
                "pred_aspect": row.get("pred_aspect", ""),
                "aspect_match": row["gold_aspect"] == row.get("pred_aspect"),
                "gold_sentiment": row["gold_sentiment"],
                "pred_sentiment": row.get("pred_sentiment", ""),
                "sentiment_match": row["gold_sentiment"] == row.get("pred_sentiment"),
                "joint_match": (
                    row["gold_aspect"] == row.get("pred_aspect")
                    and row["gold_sentiment"] == row.get("pred_sentiment")
                ),
                "pred_confidence": row.get("pred_confidence", ""),
                "aspect_term": row["aspect_term"],
                "opinion_term": row["opinion_term"],
                "text_for_prediction": row["text_for_prediction"],
            }
        )
    return out


def build_summary(rows: list[dict[str, Any]], label_prefix: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregate = FinalSummaryAggregate(
        sample_limit=args.summary_samples_per_sentiment,
        sample_char_limit=args.summary_sample_chars,
    )
    for row in rows:
        aspect = row[f"{label_prefix}_aspect"]
        sentiment = row[f"{label_prefix}_sentiment"]
        aggregate.add(
            entity_id=row["hotel_key"],
            data_source="annotated",
            hotel_id=row["hotel_key"],
            source_file=Path(args.input).name,
            aspect=aspect,
            sentiment=sentiment,
            text=row["text_for_prediction"],
            confidence=float(row.get("pred_confidence", 1.0) or 1.0) if label_prefix == "pred" else 1.0,
            keep_reference_text=False,
        )

    # Use deterministic local summaries for evaluation repeatability. Qwen-based
    # summary generation can be run later by calling the main pipeline writer.
    rows_out = []
    for group_key in aggregate.sorted_group_keys():
        group_meta = aggregate.group_metadata.get(group_key, {})
        for aspect in ASPECT_NAMES:
            buckets = aggregate.buckets[group_key][aspect]
            total = sum(buckets[sentiment].count for sentiment in SENTIMENTS)
            if total <= 0:
                continue
            output_row: dict[str, Any] = {
                "hotel_id": group_meta.get("hotel_id", ""),
                "aspect": aspect,
                "overall_aspect_summary": "",
            }
            summaries = []
            for sentiment in SENTIMENTS:
                bucket = buckets[sentiment]
                output_row[f"{sentiment}_count"] = bucket.count
                output_row[f"{sentiment}_avg_confidence"] = round(bucket.average_confidence(), 6)
                if bucket.count:
                    summary = business_final_sentiment_summary(
                        {
                            "hotel_id": group_meta.get("hotel_id", ""),
                            "aspect": aspect,
                            "sentiment": sentiment,
                            "count": bucket.count,
                            "avg_confidence": round(bucket.average_confidence(), 6),
                            "samples": bucket.samples,
                        }
                    )
                else:
                    summary = ""
                output_row[f"{sentiment}_summary"] = summary
                if summary:
                    summaries.append(summary.rstrip("."))
            output_row["overall_aspect_summary"] = ". ".join(summaries) + ("." if summaries else "")
            rows_out.append(output_row)

    stats = {
        "summary_type": f"{label_prefix}_labels_final_aspect_sentiment_summary",
        "rows": len(rows_out),
    }
    return rows_out, stats


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    rows, load_stats = load_gold_rows(input_path, args.text_column, args.max_rows)
    print(f"[load] rows={len(rows):,} skipped={load_stats['skipped']}", flush=True)
    predict_rows(rows, args)

    metrics = metrics_for(rows)
    stats = {
        "input": str(input_path.resolve()),
        "mode": args.mode,
        "text_column": args.text_column,
        "elapsed_sec": round(time.time() - start, 3),
        "load_stats": load_stats,
        "metrics": metrics,
    }

    (out_dir / "evaluation_metrics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "predictions_vs_gold.csv", prediction_output_rows(rows))
    write_csv(out_dir / "aspect_confusion.csv", confusion_rows(rows, "aspect", ASPECT_NAMES))
    write_csv(out_dir / "sentiment_confusion.csv", confusion_rows(rows, "sentiment", SENTIMENTS))
    write_csv(out_dir / "mismatches_sample.csv", mismatch_rows(rows))

    gold_summary_rows, gold_summary_stats = build_summary(rows, "gold", args)
    pred_summary_rows, pred_summary_stats = build_summary(rows, "pred", args)
    write_csv(out_dir / "gold_label_summaries.csv", gold_summary_rows)
    write_csv(out_dir / "predicted_label_summaries.csv", pred_summary_rows)
    write_final_summary_json(out_dir / "gold_label_summaries.json", gold_summary_rows, gold_summary_stats)
    write_final_summary_json(out_dir / "predicted_label_summaries.json", pred_summary_rows, pred_summary_stats)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir.resolve()),
                "mode": args.mode,
                "rows": len(rows),
                "aspect_accuracy": metrics["aspect_accuracy"],
                "sentiment_accuracy": metrics["sentiment_accuracy"],
                "joint_accuracy": metrics["joint_accuracy"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
