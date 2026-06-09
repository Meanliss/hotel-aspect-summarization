#!/usr/bin/env python3
"""Export SemAE HASOS aspect outputs into line-oriented JSONL/TSV files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
SPLIT_RE = re.compile(r"\t+|\n+")


def split_summary(text: str) -> list[str]:
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]


def parse_entity_file(path: Path) -> tuple[str, str]:
    name = path.name
    if "_" not in name:
        return "unknown", name
    split, entity_id = name.split("_", 1)
    return split, entity_id


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        fout.write("\t".join(fields) + "\n")
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field, "")
                if value is None:
                    value = ""
                values.append(str(value).replace("\t", " ").replace("\n", " "))
            fout.write("\t".join(values) + "\n")


def aspect_rows(run_id: str) -> list[dict]:
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        raise SystemExit(f"Missing aspect output dir: {run_dir}")
    rows = []
    for aspect_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        aspect = aspect_dir.name
        for entity_path in sorted(p for p in aspect_dir.iterdir() if p.is_file()):
            split, entity_id = parse_entity_file(entity_path)
            sentences = split_summary(entity_path.read_text(encoding="utf-8", errors="replace"))
            if not sentences:
                rows.append({
                    "run_id": run_id,
                    "aspect": aspect,
                    "split": split,
                    "entity_id": entity_id,
                    "sentence_index": "",
                    "sentence": "",
                    "output_path": str(entity_path),
                })
            for idx, sentence in enumerate(sentences, 1):
                rows.append({
                    "run_id": run_id,
                    "aspect": aspect,
                    "split": split,
                    "entity_id": entity_id,
                    "sentence_index": idx,
                    "sentence": sentence,
                    "output_path": str(entity_path),
                })
    return rows


def sentiment_rows(run_id: str) -> list[dict]:
    sent_dir = OUTPUTS_DIR / f"{run_id}_sentiment"
    if not sent_dir.exists():
        return []
    rows = []
    for bucket_dir in sorted(p for p in sent_dir.iterdir() if p.is_dir()):
        if "__" in bucket_dir.name:
            aspect, sentiment = bucket_dir.name.rsplit("__", 1)
        else:
            aspect, sentiment = bucket_dir.name, "unknown"
        for entity_path in sorted(p for p in bucket_dir.iterdir() if p.is_file()):
            split, entity_id = parse_entity_file(entity_path)
            sentences = split_summary(entity_path.read_text(encoding="utf-8", errors="replace"))
            if not sentences:
                rows.append({
                    "run_id": run_id,
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "split": split,
                    "entity_id": entity_id,
                    "sentence_index": "",
                    "sentence": "",
                    "output_path": str(entity_path),
                })
            for idx, sentence in enumerate(sentences, 1):
                rows.append({
                    "run_id": run_id,
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "split": split,
                    "entity_id": entity_id,
                    "sentence_index": idx,
                    "sentence": sentence,
                    "output_path": str(entity_path),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    args = parser.parse_args()

    aspect = aspect_rows(args.run_id)
    sentiment = sentiment_rows(args.run_id)

    aspect_jsonl = OUTPUTS_DIR / f"{args.run_id}_aspect_lines.jsonl"
    aspect_tsv = OUTPUTS_DIR / f"{args.run_id}_aspect_lines.tsv"
    aspect_jsonl_alias = OUTPUTS_DIR / f"{args.run_id}_lines.jsonl"
    aspect_tsv_alias = OUTPUTS_DIR / f"{args.run_id}_lines.tsv"
    sent_jsonl = OUTPUTS_DIR / f"{args.run_id}_aspect_sentiment_lines.jsonl"
    sent_tsv = OUTPUTS_DIR / f"{args.run_id}_aspect_sentiment_lines.tsv"

    write_jsonl(aspect_jsonl, aspect)
    write_tsv(aspect_tsv, aspect, [
        "run_id", "aspect", "split", "entity_id", "sentence_index",
        "sentence", "output_path"
    ])
    write_jsonl(aspect_jsonl_alias, aspect)
    write_tsv(aspect_tsv_alias, aspect, [
        "run_id", "aspect", "split", "entity_id", "sentence_index",
        "sentence", "output_path"
    ])
    write_jsonl(sent_jsonl, sentiment)
    write_tsv(sent_tsv, sentiment, [
        "run_id", "aspect", "sentiment", "split", "entity_id",
        "sentence_index", "sentence", "output_path"
    ])

    print(f"aspect_rows={len(aspect)} -> {aspect_jsonl}")
    print(f"aspect_rows={len(aspect)} -> {aspect_jsonl_alias}")
    print(f"sentiment_rows={len(sentiment)} -> {sent_jsonl}")


if __name__ == "__main__":
    main()
