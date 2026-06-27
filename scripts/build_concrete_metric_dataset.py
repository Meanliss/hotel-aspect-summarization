#!/usr/bin/env python3
"""Build the normalized HASOS metric-judge dataset.

The input synthesis JSONL files already contain the optimized HASOS outputs and
their ranked evidence. This script converts them into a stable, compact JSONL
contract for automatic metrics and LLM judging.

Usage:
    python scripts/build_concrete_metric_dataset.py
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "reports" / "metrics"

OPTIMIZED_RUNS = [
    {
        "method": "m2",
        "method_label": "M2 abstractive",
        "threshold": 0.0075,
        "max_new_tokens": 128,
        "path": REPO
        / "outputs"
        / "sweep_hasos_m2_tokabs_128_thr_0p0075_synthesis_lines.jsonl",
    },
    {
        "method": "m3",
        "method_label": "M3 kw-sentiment",
        "threshold": 0.0055,
        "max_new_tokens": 96,
        "path": REPO
        / "outputs"
        / "sweep_hasos_m3_tokabs_96_thr_0p0055_synthesis_lines.jsonl",
    },
    {
        "method": "m4",
        "method_label": "M4 bert-sentiment",
        "threshold": 0.005,
        "max_new_tokens": 96,
        "path": REPO
        / "outputs"
        / "sweep_hasos_m4_tokabs_96_thr_0p005_synthesis_lines.jsonl",
    },
]

WORD_RE = re.compile(r"\S+")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\t", " ")).strip()


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compact_evidence(raw: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    evidence = []
    for ev in raw[:limit]:
        evidence.append(
            {
                "rank": ev.get("rank"),
                "score": ev.get("score"),
                "sentence": clean_text(ev.get("sentence", "")),
                "source_review_id": ev.get("source_review_id"),
                "source_sentence_index": ev.get("source_sentence_index"),
                "matched_aspect_seed": ev.get("matched_aspect_seed") or [],
                "sentiment_label": ev.get("sentiment_label") or "",
                "matched_sentiment_keywords": ev.get("matched_sentiment_keywords") or [],
            }
        )
    return evidence


def normalize_row(raw: dict[str, Any], run: dict[str, Any], evidence_limit: int) -> dict[str, Any]:
    summary = clean_text(raw.get("summary", ""))
    evidence = compact_evidence(raw.get("evidence") or [], evidence_limit)
    evidence_text = " ".join(ev["sentence"] for ev in evidence)
    identity = {
        "method": run["method"],
        "entity_id": str(raw.get("entity_id", "")),
        "split": raw.get("split", ""),
        "aspect": raw.get("aspect", ""),
        "sentiment": raw.get("sentiment") or "",
        "summary": summary,
        "evidence": evidence,
        "rubric_version": "concrete-v1",
    }
    item_id = stable_hash(identity)[:24]
    return {
        "item_id": item_id,
        "rubric_version": "concrete-v1",
        "dataset": "hasos",
        "method": run["method"],
        "method_label": run["method_label"],
        "optimized_threshold": run["threshold"],
        "optimized_max_new_tokens": run["max_new_tokens"],
        "source_run_id": raw.get("source_run_id") or raw.get("run_id"),
        "run_id": raw.get("run_id"),
        "split": raw.get("split", ""),
        "entity_id": str(raw.get("entity_id", "")),
        "level": raw.get("level", ""),
        "aspect": raw.get("aspect", ""),
        "parent_aspect": raw.get("parent_aspect") or "",
        "sentiment": raw.get("sentiment") or "",
        "summary": summary,
        "evidence": evidence,
        "evidence_count": int(raw.get("evidence_count") or len(raw.get("evidence") or [])),
        "evidence_used": int(raw.get("evidence_used") or len(evidence)),
        "status": raw.get("status") or "",
        "copied_from_evidence": bool(raw.get("copied_from_evidence")),
        "selection_mode": raw.get("selection_mode") or "",
        "output_path": raw.get("output_path") or "",
        "source_file": os.path.relpath(str(run["path"]), str(REPO)),
        "summary_word_count": word_count(summary),
        "evidence_topk_word_count": word_count(evidence_text),
        "judge_evidence_k": evidence_limit,
    }


def iter_jsonl(path: Path):
    with io.open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def build_dataset(evidence_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for run in OPTIMIZED_RUNS:
        path = Path(run["path"])
        if not path.exists():
            raise SystemExit(f"missing optimized synthesis file: {path}")
        for _lineno, raw in iter_jsonl(path):
            row = normalize_row(raw, run, evidence_limit)
            if row["item_id"] in seen:
                continue
            seen.add(row["item_id"])
            rows.append(row)
    rows.sort(key=lambda r: (r["method"], r["split"], r["aspect"], r["sentiment"], r["entity_id"]))
    return rows


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_method: dict[str, dict[str, Any]] = {}
    for row in rows:
        cell = by_method.setdefault(
            row["method"],
            {
                "method_label": row["method_label"],
                "rows": 0,
                "aspects": set(),
                "sentiments": set(),
                "statuses": {},
                "evidence_count_total": 0,
                "summary_words_total": 0,
            },
        )
        cell["rows"] += 1
        cell["aspects"].add(row["aspect"])
        cell["sentiments"].add(row["sentiment"])
        cell["statuses"][row["status"]] = cell["statuses"].get(row["status"], 0) + 1
        cell["evidence_count_total"] += row["evidence_count"]
        cell["summary_words_total"] += row["summary_word_count"]

    serializable = {}
    for method, cell in by_method.items():
        rows_n = cell["rows"] or 1
        serializable[method] = {
            "method_label": cell["method_label"],
            "rows": cell["rows"],
            "aspects": len(cell["aspects"]),
            "sentiments": sorted(cell["sentiments"]),
            "statuses": dict(sorted(cell["statuses"].items())),
            "avg_evidence_count": cell["evidence_count_total"] / rows_n,
            "avg_summary_words": cell["summary_words_total"] / rows_n,
        }
    payload = {
        "dataset": "hasos",
        "rubric_version": "concrete-v1",
        "judge_evidence_k": rows[0]["judge_evidence_k"] if rows else None,
        "total_rows": len(rows),
        "by_method": serializable,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR / "concrete_metric_dataset.jsonl"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "concrete_metric_dataset_summary.json"))
    parser.add_argument("--evidence-limit", type=int, default=5)
    args = parser.parse_args()

    if args.evidence_limit < 1:
        raise SystemExit("--evidence-limit must be >= 1")

    out = Path(args.out)
    summary_out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dataset(args.evidence_limit)
    write_jsonl(out, rows)
    write_summary(summary_out, rows)
    print(f"written {len(rows)} rows -> {out.relative_to(REPO)}")
    print(f"written summary -> {summary_out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
