#!/usr/bin/env python3
"""Build the normalized HASOS metric-judge dataset.

The abstractive synthesis JSONL files already contain the optimized HASOS
outputs and their ranked evidence. The M1 extractive baseline is stored as
sentence-level SemAE outputs, so this script groups those sentences into the
same compact JSONL contract used for automatic metrics and LLM judging.

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

M1_RUN = {
    "method": "m1",
    "method_label": "M1 extractive",
    "threshold": None,
    "max_tokens": 40,
    "run_id": "space_hasos_full_e20",
    "lines_path": REPO / "outputs" / "space_hasos_full_e20_lines.jsonl",
    "provenance_path": REPO / "outputs" / "space_hasos_full_e20_provenance.jsonl",
}

OPTIMIZED_SYNTHESIS_RUNS = [
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

V2_SUFFIX = "_v2_faithful"
V3_SUFFIX = "_v3_polarity"

TAXONOMY_PATH = REPO / "data" / "hasos" / "aspect_taxonomy.json"

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


def load_parent_aspects() -> dict[str, str]:
    if not TAXONOMY_PATH.exists():
        return {}
    with io.open(TAXONOMY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {row["code"]: row.get("group", "") for row in data.get("aspects", [])}


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
        "optimized_max_tokens": None,
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
        "compression_evidence_word_count": word_count(evidence_text),
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


def m1_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row.get("split", ""), str(row.get("entity_id", "")), row.get("aspect", ""))


def sort_m1_lines(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            int(r.get("sentence_index") or r.get("summary_sentence_index") or 0),
            int(r.get("rank") or 0),
            clean_text(r.get("sentence", "")),
        ),
    )


def normalize_m1_group(
    key: tuple[str, str, str],
    line_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
    parent_aspects: dict[str, str],
    evidence_limit: int,
) -> dict[str, Any]:
    split, entity_id, aspect = key
    line_rows = sort_m1_lines(line_rows)
    provenance_rows = sort_m1_lines(provenance_rows or line_rows)
    summary = clean_text(" ".join(row.get("sentence", "") for row in line_rows))
    evidence = compact_evidence(provenance_rows, evidence_limit)
    evidence_text = " ".join(row.get("sentence", "") for row in provenance_rows)
    summary_words = word_count(summary)
    identity = {
        "method": M1_RUN["method"],
        "entity_id": entity_id,
        "split": split,
        "aspect": aspect,
        "sentiment": "",
        "summary": summary,
        "evidence": evidence,
        "rubric_version": "concrete-v1",
    }
    item_id = stable_hash(identity)[:24]
    first = line_rows[0] if line_rows else {}
    return {
        "item_id": item_id,
        "rubric_version": "concrete-v1",
        "dataset": "hasos",
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
        "parent_aspect": parent_aspects.get(aspect, ""),
        "sentiment": "",
        "summary": summary,
        "evidence": evidence,
        "evidence_count": len(provenance_rows),
        "evidence_used": len(provenance_rows),
        "status": "extractive",
        "copied_from_evidence": True,
        "selection_mode": "semae_ranked_extractive",
        "output_path": first.get("output_path") or "",
        "source_file": os.path.relpath(str(M1_RUN["lines_path"]), str(REPO)),
        "source_provenance_file": os.path.relpath(str(M1_RUN["provenance_path"]), str(REPO)),
        "summary_word_count": summary_words,
        "evidence_topk_word_count": word_count(" ".join(ev["sentence"] for ev in evidence)),
        # M1 is itself the selected extractive evidence, so compression is 1.0.
        "compression_evidence_word_count": summary_words,
        "judge_evidence_k": evidence_limit,
    }


def build_m1_rows(evidence_limit: int) -> list[dict[str, Any]]:
    lines_path = Path(M1_RUN["lines_path"])
    provenance_path = Path(M1_RUN["provenance_path"])
    if not lines_path.exists():
        raise SystemExit(f"missing M1 lines file: {lines_path}")
    if not provenance_path.exists():
        raise SystemExit(f"missing M1 provenance file: {provenance_path}")

    line_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    provenance_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for _lineno, row in iter_jsonl(lines_path):
        if not clean_text(row.get("sentence", "")):
            continue
        line_groups.setdefault(m1_group_key(row), []).append(row)
    for _lineno, row in iter_jsonl(provenance_path):
        if not clean_text(row.get("sentence", "")):
            continue
        provenance_groups.setdefault(m1_group_key(row), []).append(row)

    parent_aspects = load_parent_aspects()
    rows = [
        normalize_m1_group(key, line_groups[key], provenance_groups.get(key, []), parent_aspects, evidence_limit)
        for key in sorted(line_groups)
    ]
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _variant_synthesis_runs(variant: str) -> list[dict[str, Any]]:
    """Return the synthesis-run list for the requested variant.

    `original` keeps the historical sweep outputs. `v2_faithful` points at the
    re-synthesized outputs that apply the 5 faithfulness levers (evidence
    alignment, faithfulness prompt, consistency filter, sentiment consistency,
    deduped fallback). `v3_polarity` additionally polarity-filters the
    evidence used for splicing/fallback in M3/M4 sentiment-split mode so the
    fallback never introduces a polarity reversal.
    """
    runs: list[dict[str, Any]] = []
    for run in OPTIMIZED_SYNTHESIS_RUNS:
        run = dict(run)
        if variant in {"v2_faithful", "v3_polarity"}:
            # M2 has no sentiment split, so the v3 polarity filter does not
            # apply to it; reuse the v2_faithful output for M2 under v3.
            if variant == "v3_polarity" and run["method"] == "m2":
                suffix = V2_SUFFIX
            else:
                suffix = V2_SUFFIX if variant == "v2_faithful" else V3_SUFFIX
            p = Path(run["path"])
            run["path"] = p.with_name(
                p.name.replace("_synthesis_lines.jsonl",
                               f"{suffix}_synthesis_lines.jsonl"))
        runs.append(run)
    return runs


def build_dataset(evidence_limit: int, variant: str = "original") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in build_m1_rows(evidence_limit):
        if row["item_id"] in seen:
            continue
        seen.add(row["item_id"])
        rows.append(row)
    for run in _variant_synthesis_runs(variant):
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
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--evidence-limit", type=int, default=5)
    parser.add_argument("--variant", choices=["original", "v2_faithful", "v3_polarity"],
                        default="original",
                        help="Which synthesis outputs to package for judging.")
    args = parser.parse_args()

    if args.evidence_limit < 1:
        raise SystemExit("--evidence-limit must be >= 1")

    suffix_map = {"v2_faithful": V2_SUFFIX, "v3_polarity": V3_SUFFIX}
    suffix = suffix_map.get(args.variant, "")
    out = Path(args.out) if args.out else OUT_DIR / f"concrete_metric_dataset{suffix}.jsonl"
    summary_out = (Path(args.summary_out) if args.summary_out
                   else OUT_DIR / f"concrete_metric_dataset{suffix}_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = build_dataset(args.evidence_limit, args.variant)
    write_jsonl(out, rows)
    write_summary(summary_out, rows)
    print(f"variant={args.variant} written {len(rows)} rows -> {out.relative_to(REPO)}")
    print(f"written summary -> {summary_out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
