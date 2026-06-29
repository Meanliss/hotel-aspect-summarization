#!/usr/bin/env python3
"""Build the normalized SPACE metric-judge dataset.

SPACE outputs are stored as one synthesis JSONL per method. M2--M4 already
carry evidence in those files. M1 is the extractive SemAE baseline, so it is
rebuilt by grouping the threshold evidence rows by split/entity/aspect.
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
RUBRIC_VERSION = "concrete-v1"
ASPECTS = ["building", "cleanliness", "food", "location", "rooms", "service"]

M1_RUN = {
    "method": "m1",
    "method_label": "M1 extractive",
    "threshold": 0.0082,
    "max_tokens": 40,
    "run_id": "space_eval_4method",
    "path": REPO / "outputs" / "space_eval_4method_threshold_evidence.jsonl",
}

SYNTHESIS_RUNS = [
    {
        "method": "m2",
        "method_label": "M2 abstractive",
        "threshold": 0.0082,
        "max_new_tokens": 192,
        "path": REPO / "outputs" / "space_eval_4method_m2_synthesis_lines.jsonl",
    },
    {
        "method": "m3",
        "method_label": "M3 kw-sentiment",
        "threshold": 0.0082,
        "max_new_tokens": 192,
        "path": REPO / "outputs" / "space_eval_4method_m3_kw_synthesis_lines.jsonl",
    },
    {
        "method": "m4",
        "method_label": "M4 bert-sentiment",
        "threshold": 0.0082,
        "max_new_tokens": 192,
        "path": REPO / "outputs" / "space_eval_4method_m4_bert_synthesis_lines.jsonl",
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


def identity_for(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "method": row["method"],
        "entity_id": row["entity_id"],
        "split": row["split"],
        "aspect": row["aspect"],
        "sentiment": row.get("sentiment", ""),
        "summary": row["summary"],
        "evidence": row["evidence"],
        "rubric_version": row["rubric_version"],
    }


def finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    row["item_id"] = stable_hash(identity_for(row))[:24]
    row["summary_word_count"] = word_count(row["summary"])
    evidence_text = " ".join(ev["sentence"] for ev in row["evidence"])
    row["evidence_topk_word_count"] = word_count(evidence_text)
    row.setdefault("compression_evidence_word_count", row["evidence_topk_word_count"])
    return row


def sort_evidence(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            int(r.get("summary_sentence_index") or r.get("rank") or 0),
            clean_text(r.get("sentence", "")),
        ),
    )


def build_m1_rows(evidence_limit: int) -> list[dict[str, Any]]:
    path = Path(M1_RUN["path"])
    if not path.exists():
        raise SystemExit(f"missing M1 evidence file: {path}")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for _lineno, raw in iter_jsonl(path):
        if raw.get("aspect") not in ASPECTS:
            continue
        if not clean_text(raw.get("sentence", "")):
            continue
        key = (raw.get("split", ""), str(raw.get("entity_id", "")), raw.get("aspect", ""))
        groups.setdefault(key, []).append(raw)

    rows = []
    for split, entity_id, aspect in sorted(groups):
        raw_evidence = sort_evidence(groups[(split, entity_id, aspect)])
        evidence = compact_evidence(raw_evidence, evidence_limit)
        summary = clean_text(" ".join(ev["sentence"] for ev in raw_evidence))
        row = {
            "rubric_version": RUBRIC_VERSION,
            "dataset": "space",
            "method": M1_RUN["method"],
            "method_label": M1_RUN["method_label"],
            "optimized_threshold": M1_RUN["threshold"],
            "optimized_max_tokens": M1_RUN["max_tokens"],
            "optimized_max_new_tokens": None,
            "source_run_id": M1_RUN["run_id"],
            "run_id": M1_RUN["run_id"],
            "split": split,
            "entity_id": entity_id,
            "level": "child",
            "aspect": aspect,
            "parent_aspect": "",
            "sentiment": "",
            "summary": summary,
            "evidence": evidence,
            "evidence_count": len(raw_evidence),
            "evidence_used": len(raw_evidence),
            "status": "extractive",
            "copied_from_evidence": True,
            "selection_mode": "score_threshold_extractive",
            "output_path": str((REPO / "outputs" / "space_eval_4method" / aspect / f"{split}_{entity_id}")),
            "source_file": os.path.relpath(str(path), str(REPO)),
            "compression_evidence_word_count": word_count(summary),
            "judge_evidence_k": evidence_limit,
        }
        rows.append(finalize_row(row))
    return rows


def normalize_synthesis_row(raw: dict[str, Any], run: dict[str, Any], evidence_limit: int) -> dict[str, Any]:
    evidence = compact_evidence(raw.get("evidence") or [], evidence_limit)
    row = {
        "rubric_version": RUBRIC_VERSION,
        "dataset": "space",
        "method": run["method"],
        "method_label": run["method_label"],
        "optimized_threshold": run["threshold"],
        "optimized_max_tokens": None,
        "optimized_max_new_tokens": raw.get("max_new_tokens") or run["max_new_tokens"],
        "source_run_id": raw.get("source_run_id") or raw.get("run_id"),
        "run_id": raw.get("run_id"),
        "split": raw.get("split", ""),
        "entity_id": str(raw.get("entity_id", "")),
        "level": raw.get("level", "child"),
        "aspect": raw.get("aspect", ""),
        "parent_aspect": raw.get("parent_aspect") or "",
        "sentiment": raw.get("sentiment") or "",
        "summary": clean_text(raw.get("summary", "")),
        "evidence": evidence,
        "evidence_count": int(raw.get("evidence_count") or len(raw.get("evidence") or [])),
        "evidence_used": int(raw.get("evidence_used") or len(evidence)),
        "status": raw.get("status") or "",
        "copied_from_evidence": bool(raw.get("copied_from_evidence")),
        "selection_mode": raw.get("selection_mode") or "",
        "output_path": raw.get("output_path") or "",
        "source_file": os.path.relpath(str(run["path"]), str(REPO)),
        "judge_evidence_k": evidence_limit,
    }
    return finalize_row(row)


def build_dataset(evidence_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in build_m1_rows(evidence_limit):
        if row["item_id"] not in seen:
            seen.add(row["item_id"])
            rows.append(row)
    for run in SYNTHESIS_RUNS:
        path = Path(run["path"])
        if not path.exists():
            raise SystemExit(f"missing synthesis file: {path}")
        for _lineno, raw in iter_jsonl(path):
            if raw.get("level") != "child":
                continue
            if raw.get("aspect") not in ASPECTS:
                continue
            if not clean_text(raw.get("summary", "")):
                continue
            row = normalize_synthesis_row(raw, run, evidence_limit)
            if row["item_id"] not in seen:
                seen.add(row["item_id"])
                rows.append(row)
    rows.sort(key=lambda r: (r["method"], r["split"], r["aspect"], r["sentiment"], r["entity_id"]))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


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
        "dataset": "space",
        "rubric_version": RUBRIC_VERSION,
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
    parser.add_argument("--out", default=str(OUT_DIR / "space_metric_dataset.jsonl"))
    parser.add_argument("--summary-out", default=str(OUT_DIR / "space_metric_dataset_summary.json"))
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
