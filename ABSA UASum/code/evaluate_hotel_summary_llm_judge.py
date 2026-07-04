#!/usr/bin/env python3
"""LLM-as-judge audit for hotel ABSA summaries.

The judge compares each hotel/aspect/sentiment summary against the compact
cluster evidence already produced by the pipeline. This is designed for cases
where raw ROUGE recall is misleading because the source corpus is much longer
than the reader-facing summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RESULTS = WORKSPACE / "results"
DEFAULT_RUN_NAME = "hotel_review1_vi_100plus_llm_v3_full"
ASPECTS = ("facility", "amenity", "service", "experience", "branding", "loyalty")
SENTIMENTS = ("positive", "negative", "neutral")
SCORE_FIELDS = (
    "evidence_support",
    "coverage",
    "sentiment_alignment",
    "aspect_alignment",
    "specificity",
    "readability",
    "overall",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--qwen-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--qwen-api-key", default="")
    parser.add_argument("--qwen-model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--judge-batch-size", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-summary-chars", type=int, default=1200)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--max-samples-per-cluster", type=int, default=3)
    parser.add_argument("--max-sample-chars", type=int, default=220)
    parser.add_argument("--max-output-tokens", type=int, default=3000)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only build judge candidates; do not call the LLM.")
    args = parser.parse_args()
    if not args.qwen_api_key:
        args.qwen_api_key = os.getenv("QWEN_API_KEY", "local-dev-key")
    args.judge_batch_size = max(1, int(args.judge_batch_size))
    return args


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return " ".join(text.split())


def key_string(row: dict[str, Any]) -> str:
    return f"{row['hotel_id']}::{row['aspect']}::{row['sentiment']}"


def clamp_score(value: Any) -> int:
    try:
        score = int(round(float(value)))
    except Exception:
        return 0
    return max(0, min(5, score))


def parse_clusters(value: Any, max_clusters: int, max_samples: int, max_sample_chars: int) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    clusters = []
    for cluster in parsed:
        if not isinstance(cluster, dict):
            continue
        try:
            count = int(float(cluster.get("count", 0) or 0))
        except ValueError:
            count = 0
        clusters.append((count, cluster))

    out: list[dict[str, Any]] = []
    for count, cluster in sorted(clusters, key=lambda item: -item[0])[:max_clusters]:
        samples = [clean_text(sample)[:max_sample_chars] for sample in cluster.get("samples", []) if clean_text(sample)]
        out.append(
            {
                "label": clean_text(cluster.get("label") or cluster.get("cluster_label")),
                "code": clean_text(cluster.get("code") or cluster.get("cluster_code")),
                "measurement_scale": clean_text(cluster.get("measurement_scale")),
                "count": count,
                "descriptors": [clean_text(item) for item in cluster.get("descriptors", []) if clean_text(item)][:12],
                "samples": samples[:max_samples],
            }
        )
    return out


def build_candidate_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    results_dir = Path(args.results_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else results_dir / f"{args.run_name}_final_summary.csv"
    df = pd.read_csv(summary_csv, dtype=str).fillna("")
    rows: list[dict[str, Any]] = []

    for record in df.to_dict("records"):
        hotel_id = clean_text(record.get("hotel_id"))
        aspect = clean_text(record.get("aspect")).lower()
        if not hotel_id or aspect not in ASPECTS:
            continue
        for sentiment in SENTIMENTS:
            summary = clean_text(record.get(f"{sentiment}_summary"))[: args.max_summary_chars]
            count_raw = clean_text(record.get(f"{sentiment}_count"))
            try:
                evidence_count = int(float(count_raw)) if count_raw else 0
            except ValueError:
                evidence_count = 0
            clusters = parse_clusters(
                record.get(f"{sentiment}_clusters"),
                args.max_clusters,
                args.max_samples_per_cluster,
                args.max_sample_chars,
            )
            if not summary and evidence_count <= 0 and not clusters:
                continue
            rows.append(
                {
                    "row_id": len(rows),
                    "hotel_id": hotel_id,
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "summary": summary,
                    "evidence_count": evidence_count,
                    "cluster_count": len(clusters),
                    "cluster_evidence_json": json.dumps(clusters, ensure_ascii=False),
                }
            )
            if args.max_rows and len(rows) >= args.max_rows:
                return rows
    return rows


def parse_judge_response(content: str, expected: int) -> list[dict[str, Any]]:
    content = content.strip()
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").strip()
        content = content.removesuffix("```").strip()
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(content[start : end + 1])

    items = payload.get("items", [])
    if not items and expected == 1 and isinstance(payload, dict):
        items = [{**payload, "id": 0}]

    by_id = {int(item.get("id", -1)): item for item in items if isinstance(item, dict)}
    parsed = []
    for idx in range(expected):
        item = by_id.get(idx)
        if item is None and idx < len(items) and isinstance(items[idx], dict):
            item = items[idx]
        if item is None:
            item = {}
        parsed.append(
            {
                "id": idx,
                **{field: clamp_score(item.get(field)) for field in SCORE_FIELDS},
                "main_issue": clean_text(item.get("main_issue"))[:500],
                "unsupported_claims": clean_text(item.get("unsupported_claims"))[:500],
                "missing_evidence_themes": clean_text(item.get("missing_evidence_themes"))[:500],
                "recommended_fix": clean_text(item.get("recommended_fix"))[:500],
            }
        )
    return parsed


def judge_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    client = OpenAI(base_url=args.qwen_base_url, api_key=args.qwen_api_key, timeout=args.timeout_sec)
    batches = [rows[i : i + args.judge_batch_size] for i in range(0, len(rows), args.judge_batch_size)]
    judged_rows: list[dict[str, Any]] = []
    system = (
        "You are a strict evaluator for hotel aspect-based review summaries. "
        "Use only the provided cluster evidence: labels, descriptors, counts, and samples. "
        "The provided evidence_count is authoritative for count claims in the requested sentiment bucket. "
        "Judge only the requested sentiment bucket; do not expect positive summaries to cover negative or neutral evidence, "
        "and do not expect negative or neutral summaries to cover other sentiment buckets. "
        "Penalize unsupported claims, wrong aspect, wrong sentiment bucket, generic wording, "
        "raw descriptor dumping, and summaries that miss high-count clusters. "
        "Return strict JSON only."
    )

    for batch_idx, batch in enumerate(batches, start=1):
        payload_items = []
        for idx, row in enumerate(batch):
            payload_items.append(
                {
                    "id": idx,
                    "hotel_id": row["hotel_id"],
                    "aspect": row["aspect"],
                    "sentiment": row["sentiment"],
                    "summary": row["summary"],
                    "evidence_count": row["evidence_count"],
                    "cluster_evidence": json.loads(row["cluster_evidence_json"]),
                }
            )
        user = (
            "Score each item from 1 to 5.\n"
            "- evidence_support: summary claims are directly supported by evidence.\n"
            "- coverage: summary covers the important high-count clusters for this sentiment.\n"
            "- sentiment_alignment: summary stays within the requested sentiment bucket.\n"
            "- aspect_alignment: summary stays within the requested aspect.\n"
            "- specificity: summary is concrete, not generic, and not a raw descriptor dump.\n"
            "- readability: summary is concise, readable Vietnamese for a business user.\n"
            "- overall: balanced final score; unsupported claims cap overall at 2; wrong aspect/sentiment caps overall at 2.\n\n"
            "Important: evidence_count supports count statements such as '96 positive sentences'. "
            "Do not penalize a summary for omitting clusters outside the requested sentiment.\n\n"
            "Return exactly this schema:\n"
            '{"items":[{"id":0,"evidence_support":4,"coverage":3,"sentiment_alignment":5,'
            '"aspect_alignment":5,"specificity":4,"readability":4,"overall":4,'
            '"main_issue":"short note","unsupported_claims":"short phrase",'
            '"missing_evidence_themes":"short phrase","recommended_fix":"short phrase"}]}\n\n'
            f"Payload:\n{json.dumps({'items': payload_items}, ensure_ascii=False)}"
        )

        parsed = None
        last_error = ""
        for attempt in range(args.max_retries + 1):
            try:
                extra_body: dict[str, Any] = {"top_k": 20}
                if not args.qwen_enable_thinking:
                    extra_body["chat_template_kwargs"] = {"enable_thinking": False}
                response = client.chat.completions.create(
                    model=args.qwen_model,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=args.max_output_tokens,
                    extra_body=extra_body,
                )
                parsed = parse_judge_response(response.choices[0].message.content or "", len(batch))
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < args.max_retries:
                    time.sleep(1.5 * (attempt + 1))

        if parsed is None:
            parsed = [
                {
                    "id": idx,
                    **{field: 0 for field in SCORE_FIELDS},
                    "main_issue": f"judge_failed: {last_error[:300]}",
                    "unsupported_claims": "",
                    "missing_evidence_themes": "",
                    "recommended_fix": "",
                }
                for idx in range(len(batch))
            ]

        for row, judge in zip(batch, parsed):
            judged_rows.append(
                {
                    **row,
                    **{field: judge[field] for field in SCORE_FIELDS},
                    "main_issue": judge["main_issue"],
                    "unsupported_claims": judge["unsupported_claims"],
                    "missing_evidence_themes": judge["missing_evidence_themes"],
                    "recommended_fix": judge["recommended_fix"],
                }
            )
        if batch_idx == 1 or batch_idx % args.progress_every == 0 or batch_idx == len(batches):
            print(f"[hotel-judge] completed {batch_idx}/{len(batches)} batches rows={len(judged_rows)}", flush=True)
    return judged_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(items: list[dict[str, Any]], field: str) -> float:
        values = [float(row.get(field, 0) or 0) for row in items]
        return round(statistics.mean(values), 4) if values else 0.0

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"rows": len(items), **{field: mean(items, field) for field in SCORE_FIELDS}}

    by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_sentiment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_aspect[str(row.get("aspect", ""))].append(row)
        by_sentiment[str(row.get("sentiment", ""))].append(row)

    return {
        "rows": len(rows),
        "overall": stats(rows),
        "by_aspect": {key: stats(items) for key, items in sorted(by_aspect.items())},
        "by_sentiment": {key: stats(items) for key, items in sorted(by_sentiment.items())},
        "score_fields": list(SCORE_FIELDS),
        "score_distribution": {
            field: dict(Counter(str(row.get(field, 0)) for row in rows))
            for field in SCORE_FIELDS
        },
        "notes": {
            "unit": "hotel_id + aspect + sentiment",
            "reference": "top cluster labels/descriptors/counts/samples embedded in final summary CSV",
            "scale": "1-5 for LLM judge scores; 0 indicates judge failure.",
        },
    }


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / f"{args.run_name}_llm_judge"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_csv = out_dir / "hotel_summary_judge_candidates.csv"
    detail_csv = out_dir / "hotel_summary_llm_judge.csv"
    stats_json = out_dir / "hotel_summary_llm_judge_stats.json"

    rows = build_candidate_rows(args)
    write_csv(candidate_csv, rows)
    if args.dry_run:
        print(json.dumps({"candidate_csv": str(candidate_csv), "candidate_rows": len(rows)}, ensure_ascii=False, indent=2))
        return 0

    already_scored: dict[str, dict[str, Any]] = {}
    if args.resume and detail_csv.exists() and detail_csv.stat().st_size > 0:
        for row in pd.read_csv(detail_csv, dtype=str).fillna("").to_dict("records"):
            already_scored[key_string(row)] = row
    pending = [row for row in rows if key_string(row) not in already_scored]
    judged = list(already_scored.values())
    if pending:
        judged.extend(judge_rows(pending, args))
    judged = sorted(judged, key=lambda row: (str(row["hotel_id"]), str(row["aspect"]), str(row["sentiment"])))
    write_csv(detail_csv, judged)

    stats = {
        "run_name": args.run_name,
        "summary_csv": str((Path(args.summary_csv) if args.summary_csv else results_dir / f"{args.run_name}_final_summary.csv").resolve()),
        "candidate_rows": len(rows),
        "judged_rows": len(judged),
        "qwen_base_url": args.qwen_base_url,
        "qwen_model": args.qwen_model,
        "judge_batch_size": args.judge_batch_size,
        "aggregate": aggregate_scores(judged),
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats["aggregate"]["overall"], ensure_ascii=False, indent=2), flush=True)
    print(f"[done] detail_csv={detail_csv}", flush=True)
    print(f"[done] stats_json={stats_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
