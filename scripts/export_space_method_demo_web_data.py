#!/usr/bin/env python3
"""Export one stage-by-stage SPACE 4-method demo for the web app."""

from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime, timezone


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTITY_ID = "250023"
SELECTED_ASPECTS = ["building", "food", "rooms", "service"]


METHOD_META = {
    "m1": {
        "label": "M1 - Extractive (SemAE)",
        "short": "M1",
        "desc": "Raw SemAE-selected sentences joined as evidence.",
        "color": "slate",
    },
    "m2": {
        "label": "M2 - Abstractive",
        "short": "M2",
        "desc": "FLAN-T5 rewrite of the same selected evidence, without sentiment split.",
        "color": "sky",
    },
    "m3": {
        "label": "M3 - Keyword sentiment split",
        "short": "M3",
        "desc": "Keyword polarity split before FLAN-T5 rewrites positive and negative buckets.",
        "color": "emerald",
    },
    "m4": {
        "label": "M4 - BERT-ABSA sentiment split",
        "short": "M4",
        "desc": "BERT-ABSA polarity split before FLAN-T5 rewrites positive and negative buckets.",
        "color": "violet",
    },
}


def clean_text(text: str) -> str:
    text = str(text).replace("\t", " ").replace("\r", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_text(path: str) -> str:
    if not os.path.isfile(path):
        return ""
    with io.open(path, encoding="utf-8", errors="replace") as f:
        return clean_text(f.read())


def read_json(path: str):
    with io.open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.isfile(path):
        return rows
    with io.open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def short_evidence(row: dict) -> dict:
    return {
        "sentence": clean_text(row.get("sentence", "")),
        "rank": row.get("rank"),
        "score": row.get("score"),
        "source_review_id": row.get("source_review_id"),
        "source_sentence_index": row.get("source_sentence_index"),
        "matched_aspect_seed": row.get("matched_aspect_seed") or [],
        "sentiment_label": row.get("sentiment_label", ""),
        "matched_sentiment_keywords": row.get("matched_sentiment_keywords") or [],
        "was_truncated": bool(row.get("was_truncated", False)),
    }


def find_entity() -> dict:
    data = read_json(os.path.join(REPO, "data/space/json/space_summ.json"))
    for entity in data:
        if str(entity.get("entity_id")) == ENTITY_ID:
            return entity
    raise RuntimeError(f"Entity {ENTITY_ID} not found")


def rows_for(rows: list[dict], aspect: str, sentiment: str | None = None) -> list[dict]:
    out = [
        row
        for row in rows
        if str(row.get("entity_id")) == ENTITY_ID and row.get("aspect") == aspect
    ]
    if sentiment is not None:
        out = [row for row in out if row.get("sentiment", "") == sentiment]
    return out


def first_summary(rows: list[dict], aspect: str, sentiment: str | None = None) -> dict:
    found = rows_for(rows, aspect, sentiment)
    return found[0] if found else {}


def synthesis_block(row: dict) -> dict:
    evidence = row.get("evidence") or []
    return {
        "summary": clean_text(row.get("summary", "")),
        "status": row.get("status", ""),
        "evidence_count": row.get("evidence_count", len(evidence)),
        "evidence_used": row.get("evidence_used", len(evidence)),
        "copied_from_evidence": bool(row.get("copied_from_evidence", False)),
        "evidence": [short_evidence(ev) for ev in evidence[:6]],
    }


def aspect_payload(aspect: str, split: str, rows: dict[str, list[dict]]) -> dict:
    file_name = f"{split}_{ENTITY_ID}"
    m1_text = read_text(
        os.path.join(REPO, "outputs/space_eval_4method", aspect, file_name)
    )
    m2_text = read_text(
        os.path.join(REPO, "outputs/space_eval_4method_m2", aspect, file_name)
    )
    m3_pos = read_text(
        os.path.join(REPO, "outputs/space_eval_4method_m3_kw", aspect, "pos", file_name)
    )
    m3_neg = read_text(
        os.path.join(REPO, "outputs/space_eval_4method_m3_kw", aspect, "neg", file_name)
    )
    m4_pos = read_text(
        os.path.join(REPO, "outputs/space_eval_4method_m4_bert", aspect, "pos", file_name)
    )
    m4_neg = read_text(
        os.path.join(REPO, "outputs/space_eval_4method_m4_bert", aspect, "neg", file_name)
    )

    return {
        "aspect": aspect,
        "gold": [],
        "shared_evidence": [short_evidence(row) for row in rows_for(rows["base"], aspect)[:6]],
        "keyword_evidence": [short_evidence(row) for row in rows_for(rows["kw"], aspect)[:6]],
        "bert_evidence": [short_evidence(row) for row in rows_for(rows["bert"], aspect)[:6]],
        "methods": {
            "m1": {
                "output": m1_text,
                "evidence": [short_evidence(row) for row in rows_for(rows["base"], aspect)[:6]],
            },
            "m2": {
                "output": m2_text,
                "synthesis": synthesis_block(first_summary(rows["m2"], aspect)),
            },
            "m3": {
                "positive": m3_pos,
                "negative": m3_neg,
                "positive_synthesis": synthesis_block(first_summary(rows["m3"], aspect, "pos")),
                "negative_synthesis": synthesis_block(first_summary(rows["m3"], aspect, "neg")),
            },
            "m4": {
                "positive": m4_pos,
                "negative": m4_neg,
                "positive_synthesis": synthesis_block(first_summary(rows["m4"], aspect, "pos")),
                "negative_synthesis": synthesis_block(first_summary(rows["m4"], aspect, "neg")),
            },
        },
    }


def main() -> None:
    entity = find_entity()
    split = "test"
    reviews = entity.get("reviews") or []
    summaries = entity.get("summaries") or {}
    rows = {
        "base": read_jsonl(os.path.join(REPO, "outputs/space_eval_4method_threshold_evidence.jsonl")),
        "kw": read_jsonl(os.path.join(REPO, "outputs/space_eval_4method_kw_threshold_evidence.jsonl")),
        "bert": read_jsonl(os.path.join(REPO, "outputs/space_eval_4method_bert_threshold_evidence.jsonl")),
        "m2": read_jsonl(os.path.join(REPO, "outputs/space_eval_4method_m2_synthesis_lines.jsonl")),
        "m3": read_jsonl(os.path.join(REPO, "outputs/space_eval_4method_m3_kw_synthesis_lines.jsonl")),
        "m4": read_jsonl(os.path.join(REPO, "outputs/space_eval_4method_m4_bert_synthesis_lines.jsonl")),
    }

    aspects = [aspect_payload(aspect, split, rows) for aspect in SELECTED_ASPECTS]
    for aspect_block in aspects:
        aspect = aspect_block["aspect"]
        aspect_block["gold"] = [clean_text(s) for s in summaries.get(aspect, [])]

    file_name = f"{split}_{ENTITY_ID}"
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "space",
        "entity": {
            "entity_id": ENTITY_ID,
            "entity_name": entity.get("entity_name", ENTITY_ID),
            "split": split,
            "review_count": len(reviews),
            "sample_reviews": [
                {
                    "review_id": review.get("review_id"),
                    "rating": review.get("rating"),
                    "sentences": [clean_text(s) for s in (review.get("sentences") or [])[:5]],
                }
                for review in reviews[:3]
            ],
            "gold_overall": [clean_text(s) for s in summaries.get("general", [])],
        },
        "method_meta": METHOD_META,
        "pipeline": {
            "shared_selection": "All methods start from the same SemAE ranked evidence.",
            "m1": "Write selected sentences directly.",
            "m2": "Rewrite selected sentences once with FLAN-T5.",
            "m3": "Classify selected evidence with keyword sentiment, then rewrite each polarity bucket.",
            "m4": "Classify selected evidence with BERT-ABSA, then rewrite each polarity bucket.",
        },
        "overall": {
            "m1": read_text(os.path.join(REPO, "outputs/space_eval_4method", "building", file_name)),
            "m2": read_text(os.path.join(REPO, "outputs/space_eval_4method_m2_entity", file_name)),
            "m3": read_text(os.path.join(REPO, "outputs/space_eval_4method_m3_kw_entity", file_name)),
            "m4": read_text(os.path.join(REPO, "outputs/space_eval_4method_m4_bert_entity", file_name)),
        },
        "aspects": aspects,
    }

    out_path = os.path.join(REPO, "web/public/data/space_method_demo.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
