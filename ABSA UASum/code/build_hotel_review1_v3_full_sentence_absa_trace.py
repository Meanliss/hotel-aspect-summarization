#!/usr/bin/env python3
"""Build an audit-ready sentence -> processing -> ABSA trace file.

The historical segmentation_trace file contains useful split/pre-segmentation
fields, but its ABSA/cluster fields can be stale after taxonomy and anchor
repairs. This script merges the stable split trace with the authoritative
processed_sentences output and writes one row per ABSA segment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import pandas as pd


RUN_PREFIX = "hotel_review1_vi_100plus_llm_v3_full"
RESULTS_DIR = Path("results")
SEGMENTATION_TRACE_PATH = RESULTS_DIR / f"{RUN_PREFIX}_segmentation_trace.csv"
PROCESSED_SENTENCES_PATH = RESULTS_DIR / f"{RUN_PREFIX}_processed_sentences.csv"
OUTPUT_PATH = RESULTS_DIR / f"{RUN_PREFIX}_sentence_absa_processing_trace.csv"
STATS_PATH = RESULTS_DIR / f"{RUN_PREFIX}_sentence_absa_processing_trace_stats.json"

CHUNK_SIZE = 100_000
KEY_COLUMNS = [
    "source_file",
    "entity_id",
    "data_source",
    "hotel_id",
    "review_index",
    "sentence_id",
    "aspect_segment_id",
]

TRACE_COLUMNS = [
    "trace_row_id",
    "source_file",
    "entity_id",
    "data_source",
    "hotel_id",
    "review_index",
    "sentence_id",
    "aspect_segment_id",
    "source_review",
    "source_text",
    "source_sentence",
    "preseg_sentence",
    "shortened_sentence",
    "processed_sentence",
    "aspect_segment_text",
    "segment_text",
    "normalized_text_vi",
    "normalized_text_en",
    "classification_text",
    "summary_text",
    "aspect",
    "sentiment",
    "sentiment_confidence",
    "cluster_code",
    "cluster_label",
    "cluster_descriptors",
    "cluster_assignment_confidence",
    "cluster_assignment_source",
    "pre_segment_confidence",
    "semantic_source_precision",
    "extraction_confidence",
    "sentence_changed_by_shortening",
    "sentence_changed_by_processing",
    "has_cluster_assignment",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required input file(s): {missing}")


def equivalent_keys(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    for column in KEY_COLUMNS:
        if not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def changed(left: pd.Series, right: pd.Series) -> pd.Series:
    left_norm = left.fillna("").astype(str).str.strip()
    right_norm = right.fillna("").astype(str).str.strip()
    return left_norm.ne(right_norm)


def build_trace() -> dict[str, object]:
    require_files([SEGMENTATION_TRACE_PATH, PROCESSED_SENTENCES_PATH])

    if OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    seg_reader = pd.read_csv(SEGMENTATION_TRACE_PATH, chunksize=CHUNK_SIZE)
    processed_reader = pd.read_csv(PROCESSED_SENTENCES_PATH, chunksize=CHUNK_SIZE)

    total_rows = 0
    chunk_count = 0
    sentiment_counts: dict[str, int] = {}
    aspect_counts: dict[str, int] = {}
    cluster_assigned_rows = 0
    stale_cluster_rows_in_original_trace = 0

    for chunk_count, (seg_chunk, processed_chunk) in enumerate(
        zip(seg_reader, processed_reader, strict=True),
        start=1,
    ):
        seg_chunk = seg_chunk.reset_index(drop=True)
        processed_chunk = processed_chunk.reset_index(drop=True)

        if len(seg_chunk) != len(processed_chunk):
            raise ValueError(
                f"Chunk {chunk_count} row-count mismatch: "
                f"seg={len(seg_chunk)} processed={len(processed_chunk)}"
            )

        if not equivalent_keys(seg_chunk, processed_chunk):
            raise ValueError(f"Chunk {chunk_count} key mismatch between inputs")

        out = pd.DataFrame()
        out["trace_row_id"] = range(total_rows + 1, total_rows + len(processed_chunk) + 1)

        for column in KEY_COLUMNS:
            out[column] = processed_chunk[column]

        out["source_review"] = processed_chunk["source_review"]
        out["source_text"] = processed_chunk["source_text"]
        out["source_sentence"] = processed_chunk["source_sentence"]
        out["preseg_sentence"] = seg_chunk["preseg_sentence"]
        out["shortened_sentence"] = processed_chunk["shortened_sentence"]
        out["processed_sentence"] = processed_chunk["processed_sentence"]
        out["aspect_segment_text"] = processed_chunk["aspect_segment_text"]
        out["segment_text"] = processed_chunk["segment_text"]
        out["normalized_text_vi"] = processed_chunk["normalized_text_vi"]
        out["normalized_text_en"] = processed_chunk["normalized_text_en"]
        out["classification_text"] = processed_chunk["classification_text"]
        out["summary_text"] = processed_chunk["summary_text"]
        out["aspect"] = processed_chunk["aspect"]
        out["sentiment"] = processed_chunk["sentiment"]
        out["sentiment_confidence"] = processed_chunk["sentiment_confidence"]
        out["cluster_code"] = processed_chunk["cluster_code"]
        out["cluster_label"] = processed_chunk["cluster_label"]
        out["cluster_descriptors"] = processed_chunk["cluster_descriptors"]
        out["cluster_assignment_confidence"] = processed_chunk[
            "cluster_assignment_confidence"
        ]
        out["cluster_assignment_source"] = processed_chunk["cluster_assignment_source"]
        out["pre_segment_confidence"] = processed_chunk["pre_segment_confidence"]
        out["semantic_source_precision"] = processed_chunk["semantic_source_precision"]
        out["extraction_confidence"] = processed_chunk["extraction_confidence"]

        out["sentence_changed_by_shortening"] = changed(
            out["source_sentence"], out["shortened_sentence"]
        )
        out["sentence_changed_by_processing"] = changed(
            out["shortened_sentence"], out["processed_sentence"]
        )
        out["has_cluster_assignment"] = out["cluster_code"].fillna("").astype(str).str.len() > 0

        out = out[TRACE_COLUMNS]
        out.to_csv(OUTPUT_PATH, mode="a", index=False, header=chunk_count == 1)

        total_rows += len(out)
        cluster_assigned_rows += int(out["has_cluster_assignment"].sum())
        stale_cluster_rows_in_original_trace += int(
            seg_chunk["cluster_code"].isna().sum()
            if "cluster_code" in seg_chunk.columns
            else len(seg_chunk)
        )

        for value, count in out["sentiment"].value_counts(dropna=False).items():
            sentiment_counts[str(value)] = sentiment_counts.get(str(value), 0) + int(count)
        for value, count in out["aspect"].value_counts(dropna=False).items():
            aspect_counts[str(value)] = aspect_counts.get(str(value), 0) + int(count)

        logging.info("Wrote chunk %s, total rows=%s", chunk_count, total_rows)

    stats = {
        "run_prefix": RUN_PREFIX,
        "output_file": str(OUTPUT_PATH),
        "source_segmentation_trace": str(SEGMENTATION_TRACE_PATH),
        "source_processed_sentences": str(PROCESSED_SENTENCES_PATH),
        "row_count": total_rows,
        "chunk_count": chunk_count,
        "cluster_assigned_rows": cluster_assigned_rows,
        "cluster_assigned_rate": round(cluster_assigned_rows / total_rows, 6)
        if total_rows
        else 0,
        "original_segmentation_trace_blank_cluster_code_rows": stale_cluster_rows_in_original_trace,
        "sentiment_counts": sentiment_counts,
        "aspect_counts": aspect_counts,
        "columns": TRACE_COLUMNS,
        "notes": [
            "One row represents one final ABSA segment.",
            "Split/preseg fields come from segmentation_trace.csv.",
            "ABSA, sentiment and cluster fields come from processed_sentences.csv after repair.",
            "This file is intended to replace raw segmentation_trace.csv for final end-to-end audit.",
        ],
    }

    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    setup_logging()
    stats = build_trace()
    logging.info("Done: %s", json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
