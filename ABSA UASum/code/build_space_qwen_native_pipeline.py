"""Build SPACE-native summaries from existing hotel-pipeline preseg output.

The hotel pipeline uses a different taxonomy from SPACE. This adapter reuses a
previous ``*_processed_sentences.csv`` file, reroutes sentence-level evidence to
SPACE's six aspects, and optionally asks an OpenAI-compatible Qwen server to
write short benchmark-style summaries.

It writes a final-summary CSV compatible with evaluate_space_pipeline_official_rouge.py
using direct SPACE aspects: building/cleanliness/food/location/rooms/service.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_PRESEG_CSV = (
    WORKSPACE
    / "results"
    / "space_pipeline"
    / "full_absa_rulebase_20260614"
    / "space_full_absa_rulebase_20260614_processed_sentences.csv"
)
DEFAULT_OUT_DIR = WORKSPACE / "results" / "space_pipeline" / "space_qwen_native_20260626"
DEFAULT_QWEN_BASE_URL = "http://localhost:8000/v1"
DEFAULT_QWEN_API_KEY = "local-dev-key"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3.5-9B"

SPACE_ASPECTS = ("building", "cleanliness", "food", "location", "rooms", "service")
SENTIMENTS = ("positive", "negative", "neutral")

DESCRIPTOR_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "was",
    "were",
    "that",
    "this",
    "there",
    "their",
    "they",
    "had",
    "has",
    "have",
    "you",
    "your",
    "are",
    "but",
    "not",
    "from",
    "very",
    "really",
    "just",
    "our",
    "out",
    "all",
    "can",
    "could",
    "would",
    "also",
    "hotel",
    "room",
    "rooms",
}

ASPECT_KEYWORDS = {
    "building": (
        "hotel",
        "property",
        "building",
        "lobby",
        "decor",
        "decoration",
        "architecture",
        "interior",
        "exterior",
        "hallway",
        "elevator",
        "common area",
        "atmosphere",
        "design",
        "modern",
        "old",
        "renovated",
        "boutique",
        "facilities",
        "facility",
        "pool",
        "gym",
    ),
    "cleanliness": (
        "clean",
        "cleanliness",
        "spotless",
        "dirty",
        "filthy",
        "stain",
        "stained",
        "smell",
        "odor",
        "mold",
        "mildew",
        "housekeeping",
        "tidy",
        "dust",
        "bathroom clean",
        "room clean",
    ),
    "food": (
        "food",
        "breakfast",
        "restaurant",
        "dinner",
        "lunch",
        "meal",
        "coffee",
        "bar",
        "wine",
        "drink",
        "menu",
        "snack",
        "vending",
        "room service",
        "buffet",
        "cafe",
        "dessert",
    ),
    "location": (
        "location",
        "located",
        "close",
        "near",
        "nearby",
        "walk",
        "walking",
        "distance",
        "downtown",
        "airport",
        "market",
        "shopping",
        "restaurant",
        "street",
        "convenient",
        "transport",
        "bus",
        "train",
        "subway",
        "neighborhood",
        "centre",
        "center",
    ),
    "rooms": (
        "room",
        "suite",
        "bed",
        "beds",
        "bathroom",
        "shower",
        "tub",
        "pillow",
        "mattress",
        "furniture",
        "spacious",
        "small",
        "large",
        "view",
        "window",
        "quiet",
        "noise",
        "noisy",
        "air conditioning",
        "ac",
        "toilet",
    ),
    "service": (
        "staff",
        "service",
        "front desk",
        "reception",
        "concierge",
        "manager",
        "employee",
        "helpful",
        "friendly",
        "rude",
        "professional",
        "check in",
        "check-in",
        "checkout",
        "check out",
        "room service",
        "valet",
        "housekeeping",
        "shuttle",
    ),
}

HOTEL_TO_SPACE_FALLBACK = {
    "amenity": ("food", "building", "rooms"),
    "branding": ("building",),
    "experience": ("location", "building"),
    "facility": ("rooms", "cleanliness", "building", "location"),
    "loyalty": ("service",),
    "service": ("service",),
}


@dataclass
class EvidenceItem:
    entity_id: str
    review_index: str
    sentence_id: str
    aspect: str
    text: str
    sentiment: str
    score: float
    source_order: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPACE-native Qwen summaries from preseg output.")
    parser.add_argument("--preseg-csv", default=str(DEFAULT_PRESEG_CSV))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--run-name", default="space_qwen_native_20260626")
    parser.add_argument("--mode", choices=["qwen", "extractive"], default="qwen")
    parser.add_argument(
        "--reroute-mode",
        choices=["keyword", "qwen"],
        default="keyword",
        help="keyword is fast rule routing; qwen asks the model to classify preseg segments into SPACE aspects.",
    )
    parser.add_argument("--qwen-base-url", default=DEFAULT_QWEN_BASE_URL)
    parser.add_argument("--qwen-api-key", default="")
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-reroute-batch-size", type=int, default=20)
    parser.add_argument("--qwen-reroute-workers", type=int, default=1)
    parser.add_argument("--qwen-reroute-min-confidence", type=float, default=0.35)
    parser.add_argument(
        "--qwen-reroute-missing-policy",
        choices=["qwen", "keyword", "skip"],
        default="qwen",
        help="How to handle preseg records missing from an existing Qwen reroute cache.",
    )
    parser.add_argument("--qwen-summary-workers", type=int, default=24)
    parser.add_argument("--qwen-global-workers", type=int, default=24)
    parser.add_argument(
        "--summary-style",
        choices=["standard", "copy_anchored"],
        default="standard",
        help="standard writes natural summaries; copy_anchored asks Qwen to compress while preserving exact evidence wording.",
    )
    parser.add_argument(
        "--reroute-cache-jsonl",
        default="",
        help="JSONL cache for Qwen reroute results. Defaults to <out-dir>/<run-name>_qwen_reroute_cache.jsonl.",
    )
    parser.add_argument("--max-evidence", type=int, default=12)
    parser.add_argument("--target-words", type=int, default=45)
    parser.add_argument("--global-target-words", type=int, default=90)
    parser.add_argument("--max-sentences", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=2.0)
    parser.add_argument("--request-sleep", type=float, default=0.0)
    parser.add_argument("--limit-entities", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def cluster_code(aspect: str, sentiment: str) -> str:
    return f"SPACE_{aspect.upper()}_{sentiment.upper()}"


def descriptor_terms(items: list[EvidenceItem], limit: int = 20) -> list[str]:
    counts: Counter[str] = Counter()
    phrase_counts: Counter[str] = Counter()
    for item in items:
        lowered = item.text.lower()
        for keyword in ASPECT_KEYWORDS.get(item.aspect, ()):
            if keyword in lowered:
                phrase_counts[keyword] += 3 if " " in keyword else 1
        for token in words(item.text):
            if len(token) < 3 or token in DESCRIPTOR_STOPWORDS:
                continue
            counts[token] += 1
    descriptors: list[str] = []
    for term, _count in phrase_counts.most_common(limit):
        if term not in descriptors:
            descriptors.append(term)
    for term, _count in counts.most_common(limit):
        if term not in descriptors:
            descriptors.append(term)
        if len(descriptors) >= limit:
            break
    return descriptors[:limit]


def build_space_clusters(aspect: str, sentiment: str, items: list[EvidenceItem], max_samples: int = 8) -> list[dict[str, Any]]:
    if not items:
        return []
    ordered = sorted(items, key=lambda item: (-item.score, item.source_order))
    return [
        {
            "label": f"{aspect.title()} {sentiment.title()} Evidence",
            "code": cluster_code(aspect, sentiment),
            "measurement_scale": aspect,
            "count": len(items),
            "avg_confidence": avg([min(max(item.score / 10.0, 0.0), 1.0) for item in items]),
            "descriptors": descriptor_terms(items),
            "samples": [item.text for item in ordered[:max_samples]],
        }
    ]


def choose_text(row: pd.Series) -> str:
    for col in ("summary_text", "segment_text", "aspect_segment_text", "processed_sentence", "source_text", "source_sentence"):
        value = clean(row.get(col, ""))
        if value and value.lower() not in {"nan", "none", "null"}:
            return value
    return ""


def keyword_score(text: str, aspect: str) -> float:
    lowered = f" {text.lower()} "
    token_set = set(words(text))
    score = 0.0
    for keyword in ASPECT_KEYWORDS[aspect]:
        if " " in keyword:
            if keyword in lowered:
                score += 3.0
        elif keyword in token_set:
            score += 2.0
    if aspect == "cleanliness" and re.search(r"\b(not|very|really|so)\s+clean\b", lowered):
        score += 2.0
    if aspect == "location" and re.search(r"\b(walking distance|close to|near|nearby)\b", lowered):
        score += 2.0
    if aspect == "service" and re.search(r"\b(staff|front desk|concierge|reception)\b", lowered):
        score += 2.0
    return score


def fallback_aspect_bonus(hotel_aspect: str, space_aspect: str) -> float:
    return 1.25 if space_aspect in HOTEL_TO_SPACE_FALLBACK.get(hotel_aspect, ()) else 0.0


def parse_json_payload(text: str) -> Any:
    cleaned = clean(text)
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
    if match:
        cleaned = match.group(1)
    return json.loads(cleaned)


def chat_text(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
) -> str:
    """Call Qwen in no-thinking mode and return assistant content."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except TypeError:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
        )
    return clean(response.choices[0].message.content or "")


def qwen_route_batch(client: OpenAI, model: str, batch: list[dict[str, str]]) -> list[dict[str, Any]]:
    payload = [
        {
            "id": item["id"],
            "text": item["text"],
            "previous_hotel_aspect": item.get("hotel_aspect", ""),
        }
        for item in batch
    ]
    prompt = f"""/no_think
Classify each hotel review segment into SPACE benchmark aspects.

SPACE aspects:
- building: property, building, lobby, decor, design, facilities, pool/gym/common areas
- cleanliness: clean/dirty rooms, bathrooms, housekeeping, smells, stains, mold, dust
- food: breakfast, restaurant, bar, drinks, meals, room service food
- location: location, nearby places, airport, downtown, transport, walking distance
- rooms: room, bed, bathroom, shower, furniture, view, noise, air conditioning, room size
- service: staff, front desk, reception, concierge, check-in/out, shuttle, valet, service attitude

Rules:
- Assign 0 to 2 aspects per segment.
- Use [] if the segment is not useful for any SPACE aspect.
- Sentiment must be one of: positive, negative, neutral.
- Keep IDs exactly unchanged.
- Return strict JSON only, no markdown.

Input:
{json.dumps(payload, ensure_ascii=False)}

Return JSON array with objects:
[
  {{"id":"...", "aspects":["rooms"], "sentiment":"positive", "confidence":0.0}}
]"""
    content = chat_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise SPACE hotel-review ABSA classifier."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=3000,
    )
    parsed = parse_json_payload(content or "[]")
    if not isinstance(parsed, list):
        raise ValueError("Qwen reroute response is not a JSON list")
    return parsed


def load_reroute_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            item_id = clean(item.get("id"))
            if item_id:
                cached[item_id] = item
    return cached


def append_reroute_cache(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None or not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def add_keyword_evidence(
    evidence: dict[tuple[str, str], list[EvidenceItem]],
    seen: set[tuple[str, str, str]],
    *,
    entity_id: str,
    review_index: str,
    sentence_id: str,
    text: str,
    hotel_aspect: str,
    sentiment: str,
    order: int,
    min_score: float,
) -> None:
    for space_aspect in SPACE_ASPECTS:
        score = keyword_score(text, space_aspect) + fallback_aspect_bonus(hotel_aspect, space_aspect)
        if score < min_score:
            continue
        dedupe_key = (entity_id, space_aspect, text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        evidence[(entity_id, space_aspect)].append(
            EvidenceItem(
                entity_id=entity_id,
                review_index=review_index,
                sentence_id=sentence_id,
                aspect=space_aspect,
                text=text,
                sentiment=sentiment,
                score=score,
                source_order=int(order),
            )
        )


def add_qwen_evidence(
    evidence: dict[tuple[str, str], list[EvidenceItem]],
    seen: set[tuple[str, str, str]],
    *,
    entity_id: str,
    review_index: str,
    sentence_id: str,
    text: str,
    sentiment: str,
    order: int,
    aspects: list[str],
    confidence: float,
    min_confidence: float,
) -> None:
    if confidence < min_confidence:
        return
    for aspect in aspects[:2]:
        space_aspect = clean(aspect).lower()
        if space_aspect not in SPACE_ASPECTS:
            continue
        dedupe_key = (entity_id, space_aspect, text.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        evidence[(entity_id, space_aspect)].append(
            EvidenceItem(
                entity_id=entity_id,
                review_index=review_index,
                sentence_id=sentence_id,
                aspect=space_aspect,
                text=text,
                sentiment=sentiment if sentiment in {"positive", "negative", "neutral"} else "neutral",
                score=max(2.0, confidence * 5.0),
                source_order=int(order),
            )
        )


def load_preseg_records(preseg_csv: Path, limit_entities: int = 0) -> list[dict[str, Any]]:
    df = pd.read_csv(preseg_csv).fillna("")
    required = {"entity_id", "aspect"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {preseg_csv}: {sorted(missing)}")

    records = []
    allowed_entities: set[str] = set()
    for order, row in df.iterrows():
        entity_id = clean(row.get("hotel_id")) or clean(row.get("entity_id"))
        if entity_id.startswith("space_"):
            entity_id = entity_id[len("space_") :]
        if not entity_id:
            continue
        if limit_entities > 0 and entity_id not in allowed_entities:
            if len(allowed_entities) >= limit_entities:
                continue
            allowed_entities.add(entity_id)
        text = choose_text(row)
        if len(words(text)) < 4:
            continue
        records.append(
            {
                "order": int(order),
                "entity_id": entity_id,
                "review_index": clean(row.get("review_index")),
                "sentence_id": clean(row.get("sentence_id")) or f"{entity_id}_{order}",
                "text": text,
                "hotel_aspect": clean(row.get("aspect")).lower(),
                "sentiment": clean(row.get("sentiment")).lower() or "neutral",
            }
        )
    return records


def load_evidence(
    preseg_csv: Path,
    min_score: float,
    *,
    reroute_mode: str = "keyword",
    client: OpenAI | None = None,
    model: str = DEFAULT_QWEN_MODEL,
    qwen_batch_size: int = 20,
    qwen_workers: int = 1,
    qwen_min_confidence: float = 0.35,
    qwen_missing_policy: str = "qwen",
    limit_entities: int = 0,
    reroute_cache_jsonl: Path | None = None,
) -> dict[tuple[str, str], list[EvidenceItem]]:
    evidence: dict[tuple[str, str], list[EvidenceItem]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    records = load_preseg_records(preseg_csv, limit_entities=limit_entities)

    if reroute_mode == "qwen":
        if client is None:
            raise ValueError("Qwen reroute requires an OpenAI-compatible client")
        cached_routes = load_reroute_cache(reroute_cache_jsonl)
        if cached_routes:
            logging.info("Loaded %s cached Qwen reroute records from %s", len(cached_routes), reroute_cache_jsonl)
        id_to_record: dict[str, dict[str, Any]] = {}

        def add_routed_items(routed_items: list[dict[str, Any]]) -> None:
            for item in routed_items:
                if not isinstance(item, dict):
                    continue
                record = id_to_record.get(clean(item.get("id")))
                if record is None:
                    continue
                aspects = item.get("aspects", [])
                if not isinstance(aspects, list):
                    aspects = []
                sentiment = clean(item.get("sentiment")).lower() or record["sentiment"]
                try:
                    confidence = float(item.get("confidence", 0.0) or 0.0)
                except (TypeError, ValueError):
                    confidence = 0.0
                add_qwen_evidence(
                    evidence,
                    seen,
                    entity_id=record["entity_id"],
                    review_index=record["review_index"],
                    sentence_id=record["sentence_id"],
                    text=record["text"],
                    sentiment=sentiment,
                    order=record["order"],
                    aspects=aspects,
                    confidence=confidence,
                    min_confidence=qwen_min_confidence,
                )

        def add_keyword_fallback(batch_items: list[dict[str, str]], start: int, exc: Exception) -> None:
            logging.warning("Qwen reroute failed for batch %s-%s: %s; using keyword fallback", start, start + len(batch_items), exc)
            for item in batch_items:
                record = id_to_record.get(item["id"])
                if record is None:
                    continue
                add_keyword_evidence(
                    evidence,
                    seen,
                    entity_id=record["entity_id"],
                    review_index=record["review_index"],
                    sentence_id=record["sentence_id"],
                    text=record["text"],
                    hotel_aspect=record["hotel_aspect"],
                    sentiment=record["sentiment"],
                    order=record["order"],
                    min_score=min_score,
                )

        batch_jobs: list[tuple[int, int, list[dict[str, str]], list[dict[str, Any]]]] = []
        for start in range(0, len(records), qwen_batch_size):
            batch_records = records[start : start + qwen_batch_size]
            batch = []
            routed: list[dict[str, Any]] = []
            for offset, record in enumerate(batch_records):
                item_id = f"seg_{start + offset}"
                id_to_record[item_id] = record
                if item_id in cached_routes:
                    routed.append(cached_routes[item_id])
                else:
                    batch.append(
                        {
                            "id": item_id,
                            "text": record["text"],
                            "hotel_aspect": record["hotel_aspect"],
                        }
                    )
            batch_jobs.append((start, len(batch_records), batch, routed))

        completed_records = 0
        pending_jobs = [(start, size, batch) for start, size, batch, routed in batch_jobs if batch]
        for _start, size, _batch, routed in batch_jobs:
            if routed:
                add_routed_items(routed)
                completed_records += size
                if completed_records % max(qwen_batch_size * 25, 1) == 0 or completed_records == len(records):
                    logging.info("Qwen rerouted %s/%s preseg records", completed_records, len(records))

        if pending_jobs:
            if qwen_missing_policy == "keyword":
                missing_items = sum(len(batch) for _start, _size, batch in pending_jobs)
                logging.info(
                    "Qwen reroute cache missing %s records; using keyword fallback by policy",
                    missing_items,
                )
                for start, size, batch in pending_jobs:
                    for item in batch:
                        record = id_to_record.get(item["id"])
                        if record is None:
                            continue
                        add_keyword_evidence(
                            evidence,
                            seen,
                            entity_id=record["entity_id"],
                            review_index=record["review_index"],
                            sentence_id=record["sentence_id"],
                            text=record["text"],
                            hotel_aspect=record["hotel_aspect"],
                            sentiment=record["sentiment"],
                            order=record["order"],
                            min_score=min_score,
                        )
                    completed_records += size
                    if completed_records % max(qwen_batch_size * 25, 1) == 0 or completed_records == len(records):
                        logging.info("Qwen rerouted %s/%s preseg records", completed_records, len(records))
                pending_jobs = []
            elif qwen_missing_policy == "skip":
                missing_items = sum(len(batch) for _start, _size, batch in pending_jobs)
                logging.info("Qwen reroute cache missing %s records; skipping by policy", missing_items)
                completed_records += sum(size for _start, size, _batch in pending_jobs)
                pending_jobs = []

        if pending_jobs:
            max_workers = max(1, int(qwen_workers))
            logging.info("Qwen reroute pending batches: %s with %s workers", len(pending_jobs), max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_job = {
                    executor.submit(qwen_route_batch, client, model, batch): (start, size, batch)
                    for start, size, batch in pending_jobs
                }
                for future in as_completed(future_to_job):
                    start, size, batch = future_to_job[future]
                    try:
                        new_routed = future.result()
                        append_reroute_cache(reroute_cache_jsonl, new_routed)
                        for item in new_routed:
                            item_id = clean(item.get("id")) if isinstance(item, dict) else ""
                            if item_id:
                                cached_routes[item_id] = item
                        add_routed_items(new_routed)
                    except Exception as exc:
                        add_keyword_fallback(batch, start, exc)
                    completed_records += size
                    if completed_records % max(qwen_batch_size * 25, 1) == 0 or completed_records == len(records):
                        logging.info("Qwen rerouted %s/%s preseg records", completed_records, len(records))
    else:
        for record in records:
            add_keyword_evidence(
                evidence,
                seen,
                entity_id=record["entity_id"],
                review_index=record["review_index"],
                sentence_id=record["sentence_id"],
                text=record["text"],
                hotel_aspect=record["hotel_aspect"],
                sentiment=record["sentiment"],
                order=record["order"],
                min_score=min_score,
            )

    for key, items in evidence.items():
        items.sort(key=lambda item: (-item.score, item.source_order))
    return dict(evidence)


def build_extractive_summary(items: list[EvidenceItem], target_words: int, max_sentences: int) -> str:
    selected: list[EvidenceItem] = []
    total_words = 0
    for item in items:
        n_words = len(words(item.text))
        if n_words < 4:
            continue
        if selected and total_words + n_words > target_words:
            continue
        selected.append(item)
        total_words += n_words
        if len(selected) >= max_sentences or total_words >= target_words:
            break
    if not selected and items:
        selected = [items[0]]
    selected.sort(key=lambda item: item.source_order)
    return clean(" ".join(item.text for item in selected))


def qwen_summary(
    client: OpenAI,
    model: str,
    entity_id: str,
    aspect: str,
    sentiment: str,
    items: list[EvidenceItem],
    target_words: int,
    summary_style: str = "standard",
) -> str:
    evidence_lines = "\n".join(f"- {item.text}" for item in items)
    sentiment_clause = "" if sentiment == "all" else f" Focus on {sentiment} evidence only."
    if summary_style == "copy_anchored":
        clusters = build_space_clusters(aspect, sentiment, items)
        cluster_payload = json.dumps(clusters, ensure_ascii=False, indent=2)
        prompt = f"""/no_think
Write a SPACE benchmark-style hotel review summary for aspect: {aspect}.{sentiment_clause}

You are doing Qwen-assisted evidence-grounded compressive summarization, not free abstractive rewriting.

Evidence bullets:
{evidence_lines}

Cluster/evidence map:
{cluster_payload}

Rules adapted from the original hotel pipeline:
- Use ONLY the evidence bullets and cluster/samples above.
- Evidence lock: every target, problem, praise item, or concrete detail must appear in the evidence.
- Copy important phrases exactly whenever possible; preserve nouns, adjectives, complaints, places, and hotel-specific details.
- Do not paraphrase concrete wording if the original wording is already concise.
- You may remove repetition and add short connectors only when needed.
- Preserve distinct target-level signals; do not collapse everything into generic labels such as room, staff, food, or service.
- Preserve sentiment contrast inside this {sentiment} bucket when the evidence itself is mixed.
- Do not mention sentiment labels, counts, clusters, methodology, or evidence.
- Keep about {target_words} words.
- Write in English.
- If evidence is sparse, return one concise sentence using exact evidence wording.

Entity: {entity_id}

Return only the summary text."""
    else:
        prompt = f"""/no_think
Write a short SPACE benchmark-style hotel summary for aspect: {aspect}.{sentiment_clause}

Rules:
- Use ONLY the evidence bullets.
- Preserve concrete review wording when possible.
- Do not mention sentiment labels, counts, clusters, or methodology.
- Keep it concise, about {target_words} words.
- Write in English.
- If evidence is sparse, return one concise sentence.

Entity: {entity_id}
Evidence:
{evidence_lines}

Return only the summary text."""
    return chat_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise evidence-grounded hotel review summarizer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=360,
    )


def summarize_sentiment_group(
    client: OpenAI | None,
    args: argparse.Namespace,
    entity_id: str,
    aspect: str,
    sentiment: str,
    items: list[EvidenceItem],
) -> str:
    if not items:
        return ""
    if args.mode == "extractive":
        return build_extractive_summary(items, args.target_words, args.max_sentences)
    assert client is not None
    try:
        summary = qwen_summary(
            client,
            args.qwen_model,
            entity_id,
            aspect,
            sentiment,
            items,
            args.target_words,
            args.summary_style,
        )
        if summary:
            return summary
        logging.warning(
            "Qwen returned empty summary for %s/%s/%s; falling back to extractive",
            entity_id,
            aspect,
            sentiment,
        )
    except Exception as exc:
        logging.warning(
            "Qwen failed for %s/%s/%s: %s; falling back to extractive",
            entity_id,
            aspect,
            sentiment,
            exc,
        )
    return build_extractive_summary(items, args.target_words, args.max_sentences)


def truncate_words(text: str, max_words: int) -> str:
    tokens = str(text).split()
    if max_words <= 0 or len(tokens) <= max_words:
        return clean(text)
    return clean(" ".join(tokens[:max_words]))


def build_global_summary_from_aspects(aspect_summaries: dict[str, str], target_words: int) -> str:
    parts = [clean(aspect_summaries.get(aspect, "")) for aspect in SPACE_ASPECTS]
    return truncate_words(" ".join(part for part in parts if part), target_words)


def qwen_global_summary(
    client: OpenAI,
    model: str,
    entity_id: str,
    aspect_summaries: dict[str, str],
    target_words: int,
    summary_style: str = "standard",
) -> str:
    aspect_lines = "\n".join(
        f"- {aspect}: {clean(aspect_summaries.get(aspect, ''))}"
        for aspect in SPACE_ASPECTS
        if clean(aspect_summaries.get(aspect, ""))
    )
    if not aspect_lines:
        return ""
    if summary_style == "copy_anchored":
        prompt = f"""/no_think
Write one overall SPACE hotel summary from the aspect summaries below.

You are doing Qwen-assisted evidence-grounded compressive summarization, not free abstractive rewriting.

Rules:
- Use ONLY the aspect summaries below.
- Copy important phrases exactly whenever possible.
- Preserve concrete details and do not add new claims.
- Remove repetition and stitch the strongest claims into one readable paragraph.
- Do not use headings or bullets.
- Write in English.
- Keep about {target_words} words.

Entity: {entity_id}
Aspect summaries:
{aspect_lines}

Return only the overall summary text."""
    else:
        prompt = f"""/no_think
Write one overall hotel summary from these SPACE aspect summaries.

Rules:
- Use ONLY the aspect summaries below.
- Preserve concrete details and avoid adding new claims.
- Do not use headings or bullets.
- Write in English.
- Keep it concise, about {target_words} words.

Entity: {entity_id}
Aspect summaries:
{aspect_lines}

Return only the overall summary text."""
    return chat_text(
        client,
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise evidence-grounded hotel review summarizer."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=480,
    )


def existing_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        return {}
    rows = {}
    for _, row in df.iterrows():
        rows[(clean(row.get("hotel_id")), clean(row.get("aspect")).lower())] = row.to_dict()
    return rows


def final_summary_row_complete(row: dict[str, Any]) -> bool:
    for sentiment in SENTIMENTS:
        count = int(row.get(f"{sentiment}_count", 0) or 0)
        if count > 0 and not clean(row.get(f"{sentiment}_summary", "")):
            return False
    return any(clean(row.get(f"{sentiment}_summary", "")) for sentiment in SENTIMENTS)


def existing_global_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path).fillna("")
    except Exception:
        return {}
    return {clean(row.get("hotel_id")): row.to_dict() for _, row in df.iterrows()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    preseg_csv = Path(args.preseg_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{args.run_name}_final_summary.csv"
    global_csv = out_dir / f"{args.run_name}_global_summary.csv"
    evidence_csv = out_dir / f"{args.run_name}_space_evidence.csv"
    cluster_evidence_csv = out_dir / f"{args.run_name}_space_cluster_evidence.csv"
    reroute_cache_jsonl = (
        Path(args.reroute_cache_jsonl)
        if args.reroute_cache_jsonl
        else out_dir / f"{args.run_name}_qwen_reroute_cache.jsonl"
    )

    client = None
    if args.mode == "qwen" or args.reroute_mode == "qwen":
        api_key = args.qwen_api_key or os.getenv("QWEN_API_KEY", DEFAULT_QWEN_API_KEY)
        client = OpenAI(base_url=args.qwen_base_url, api_key=api_key)
        client.models.list()

    logging.info("Loading SPACE-rerouted evidence from %s", preseg_csv)
    evidence = load_evidence(
        preseg_csv,
        min_score=args.min_score,
        reroute_mode=args.reroute_mode,
        client=client,
        model=args.qwen_model,
        qwen_batch_size=max(1, int(args.qwen_reroute_batch_size)),
        qwen_workers=max(1, int(args.qwen_reroute_workers)),
        qwen_min_confidence=max(0.0, min(1.0, float(args.qwen_reroute_min_confidence))),
        qwen_missing_policy=args.qwen_reroute_missing_policy,
        limit_entities=max(0, int(args.limit_entities)),
        reroute_cache_jsonl=reroute_cache_jsonl if args.reroute_mode == "qwen" else None,
    )
    entity_ids = sorted({entity_id for entity_id, _aspect in evidence})
    logging.info("Entities with evidence: %s", len(entity_ids))

    completed = existing_rows(out_csv) if args.resume else {}
    completed_global = existing_global_rows(global_csv) if args.resume else {}
    summary_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    aspect_summaries_by_entity: dict[str, dict[str, str]] = defaultdict(dict)
    aspect_evidence_counts_by_entity: dict[str, dict[str, int]] = defaultdict(dict)
    evidence_rows: list[dict[str, Any]] = []

    summary_tasks: list[tuple[str, str, str, list[EvidenceItem]]] = []
    cluster_evidence_rows: list[dict[str, Any]] = []

    def refresh_overall_summaries() -> None:
        for key, row in summary_rows_by_key.items():
            parts = [clean(row[f"{sentiment}_summary"]) for sentiment in SENTIMENTS]
            row["overall_aspect_summary"] = clean(" ".join(part for part in parts if part))
            if row["overall_aspect_summary"]:
                aspect_summaries_by_entity[key[0]][key[1]] = row["overall_aspect_summary"]

    def write_partial_summary() -> None:
        refresh_overall_summaries()
        partial_rows = [
            summary_rows_by_key[(entity_id, aspect)]
            for entity_id in entity_ids
            for aspect in SPACE_ASPECTS
            if (entity_id, aspect) in summary_rows_by_key
        ]
        write_csv(out_csv, partial_rows)

    for entity_id in entity_ids:
        for aspect in SPACE_ASPECTS:
            key = (entity_id, aspect)
            items = evidence.get(key, [])[: args.max_evidence]
            counts = Counter(item.sentiment for item in items)
            by_sentiment: dict[str, list[EvidenceItem]] = {
                sentiment: [item for item in items if item.sentiment == sentiment][: args.max_evidence]
                for sentiment in SENTIMENTS
            }
            cluster_fields: dict[str, Any] = {}
            for sentiment in SENTIMENTS:
                sentiment_items = by_sentiment[sentiment]
                clusters = build_space_clusters(aspect, sentiment, sentiment_items)
                cluster_fields[f"{sentiment}_avg_confidence"] = avg(
                    [min(max(item.score / 10.0, 0.0), 1.0) for item in sentiment_items]
                )
                cluster_fields[f"{sentiment}_cluster_count"] = len(clusters)
                cluster_fields[f"{sentiment}_clusters"] = json.dumps(clusters, ensure_ascii=False)
                for cluster in clusters:
                    cluster_evidence_rows.append(
                        {
                            "entity_id": f"space_{entity_id}",
                            "data_source": "space",
                            "hotel_id": entity_id,
                            "aspect": aspect,
                            "sentiment": sentiment,
                            "cluster_code": cluster["code"],
                            "measurement_scale": cluster["measurement_scale"],
                            "cluster_label": cluster["label"],
                            "count": cluster["count"],
                            "avg_confidence": cluster["avg_confidence"],
                            "descriptor_count": len(cluster["descriptors"]),
                            "descriptors_json": json.dumps(cluster["descriptors"], ensure_ascii=False),
                            "evidence_count": len(sentiment_items),
                            "evidence_texts_json": json.dumps([item.text for item in sentiment_items], ensure_ascii=False),
                            "segment_texts_json": json.dumps([item.text for item in sentiment_items], ensure_ascii=False),
                            "source_texts_json": json.dumps([item.text for item in sentiment_items], ensure_ascii=False),
                            "review_indexes_json": json.dumps([item.review_index for item in sentiment_items], ensure_ascii=False),
                            "source_files_json": json.dumps([preseg_csv.name], ensure_ascii=False),
                        }
                    )
            summary_rows_by_key[key] = {
                "hotel_id": entity_id,
                "aspect": aspect,
                "overall_aspect_summary": "",
                "positive_count": counts.get("positive", 0),
                "positive_avg_confidence": cluster_fields["positive_avg_confidence"],
                "positive_cluster_count": cluster_fields["positive_cluster_count"],
                "positive_clusters": cluster_fields["positive_clusters"],
                "positive_summary": "",
                "negative_count": counts.get("negative", 0),
                "negative_avg_confidence": cluster_fields["negative_avg_confidence"],
                "negative_cluster_count": cluster_fields["negative_cluster_count"],
                "negative_clusters": cluster_fields["negative_clusters"],
                "negative_summary": "",
                "neutral_count": counts.get("neutral", 0),
                "neutral_avg_confidence": cluster_fields["neutral_avg_confidence"],
                "neutral_cluster_count": cluster_fields["neutral_cluster_count"],
                "neutral_clusters": cluster_fields["neutral_clusters"],
                "neutral_summary": "",
                "evidence_count": len(items),
            }
            aspect_evidence_counts_by_entity[entity_id][aspect] = len(items)
            if key in completed:
                for field in ("overall_aspect_summary", "positive_summary", "negative_summary", "neutral_summary"):
                    summary_rows_by_key[key][field] = clean(completed[key].get(field, ""))
            if key in completed and final_summary_row_complete(summary_rows_by_key[key]):
                aspect_summaries_by_entity[entity_id][aspect] = clean(completed[key].get("overall_aspect_summary", ""))
            else:
                for sentiment, sentiment_items in by_sentiment.items():
                    if sentiment_items and not clean(summary_rows_by_key[key].get(f"{sentiment}_summary", "")):
                        summary_tasks.append((entity_id, aspect, sentiment, sentiment_items))
            for rank, item in enumerate(items, start=1):
                item_cluster_code = cluster_code(aspect, item.sentiment)
                evidence_rows.append(
                    {
                        "hotel_id": entity_id,
                        "aspect": aspect,
                        "cluster_code": item_cluster_code,
                        "cluster_label": f"{aspect.title()} {item.sentiment.title()} Evidence",
                        "rank": rank,
                        "score": item.score,
                        "sentiment": item.sentiment,
                        "sentence_id": item.sentence_id,
                        "review_index": item.review_index,
                        "text": item.text,
                    }
                )

    logging.info("Summary tasks: %s sentiment/aspect groups", len(summary_tasks))
    if summary_tasks:
        if args.mode == "qwen":
            assert client is not None
            max_workers = max(1, int(args.qwen_summary_workers))
        else:
            max_workers = 1
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task: dict[Any, tuple[str, str, str]] = {}
            next_idx = 0

            def submit_next() -> None:
                nonlocal next_idx
                if next_idx >= len(summary_tasks):
                    return
                entity_id, aspect, sentiment, items = summary_tasks[next_idx]
                next_idx += 1
                task_client = client
                if args.mode == "qwen":
                    api_key = args.qwen_api_key or os.getenv("QWEN_API_KEY", DEFAULT_QWEN_API_KEY)
                    task_client = OpenAI(base_url=args.qwen_base_url, api_key=api_key)
                future_to_task[
                    executor.submit(summarize_sentiment_group, task_client, args, entity_id, aspect, sentiment, items)
                ] = (entity_id, aspect, sentiment)

            for _ in range(min(len(summary_tasks), max_workers * 2)):
                submit_next()

            while future_to_task:
                for future in as_completed(future_to_task):
                    entity_id, aspect, sentiment = future_to_task.pop(future)
                    break
                summary = clean(future.result())
                key = (entity_id, aspect)
                summary_rows_by_key[key][f"{sentiment}_summary"] = summary
                completed_tasks += 1
                if completed_tasks % 25 == 0:
                    write_partial_summary()
                if completed_tasks % 50 == 0 or completed_tasks == len(summary_tasks):
                    logging.info("Completed %s/%s sentiment summary tasks", completed_tasks, len(summary_tasks))
                submit_next()

    refresh_overall_summaries()

    global_rows_by_entity: dict[str, dict[str, Any]] = {}
    global_tasks: list[tuple[str, dict[str, str]]] = []
    for entity_id in entity_ids:
        aspect_summaries = aspect_summaries_by_entity.get(entity_id, {})
        if entity_id in completed_global and clean(completed_global[entity_id].get("global_summary")):
            global_summary = clean(completed_global[entity_id].get("global_summary"))
            global_rows_by_entity[entity_id] = {
                "hotel_id": entity_id,
                "global_summary": global_summary,
                "building_summary": aspect_summaries.get("building", ""),
                "cleanliness_summary": aspect_summaries.get("cleanliness", ""),
                "food_summary": aspect_summaries.get("food", ""),
                "location_summary": aspect_summaries.get("location", ""),
                "rooms_summary": aspect_summaries.get("rooms", ""),
                "service_summary": aspect_summaries.get("service", ""),
                "total_evidence_count": sum(aspect_evidence_counts_by_entity.get(entity_id, {}).values()),
            }
        elif args.mode == "extractive":
            global_rows_by_entity[entity_id] = {
                "hotel_id": entity_id,
                "global_summary": build_global_summary_from_aspects(aspect_summaries, args.global_target_words),
                "building_summary": aspect_summaries.get("building", ""),
                "cleanliness_summary": aspect_summaries.get("cleanliness", ""),
                "food_summary": aspect_summaries.get("food", ""),
                "location_summary": aspect_summaries.get("location", ""),
                "rooms_summary": aspect_summaries.get("rooms", ""),
                "service_summary": aspect_summaries.get("service", ""),
                "total_evidence_count": sum(aspect_evidence_counts_by_entity.get(entity_id, {}).values()),
            }
        else:
            global_tasks.append((entity_id, dict(aspect_summaries)))

    if global_tasks:
        assert client is not None
        logging.info("Global summary tasks: %s entities", len(global_tasks))
        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=max(1, int(args.qwen_global_workers))) as executor:
            max_workers = max(1, int(args.qwen_global_workers))
            future_to_task: dict[Any, tuple[str, dict[str, str]]] = {}
            next_idx = 0

            def submit_next_global() -> None:
                nonlocal next_idx
                if next_idx >= len(global_tasks):
                    return
                entity_id, aspect_summaries = global_tasks[next_idx]
                next_idx += 1
                api_key = args.qwen_api_key or os.getenv("QWEN_API_KEY", DEFAULT_QWEN_API_KEY)
                task_client = OpenAI(base_url=args.qwen_base_url, api_key=api_key)
                future_to_task[
                    executor.submit(
                        qwen_global_summary,
                        task_client,
                        args.qwen_model,
                        entity_id,
                        aspect_summaries,
                        args.global_target_words,
                        args.summary_style,
                    )
                ] = (entity_id, aspect_summaries)

            for _ in range(min(len(global_tasks), max_workers * 2)):
                submit_next_global()

            while future_to_task:
                for future in as_completed(future_to_task):
                    entity_id, aspect_summaries = future_to_task.pop(future)
                    break
                try:
                    global_summary = clean(future.result())
                    if not global_summary:
                        logging.warning(
                            "Qwen returned empty global summary for %s; using concatenated aspect summary",
                            entity_id,
                        )
                        global_summary = build_global_summary_from_aspects(aspect_summaries, args.global_target_words)
                except Exception as exc:
                    logging.warning("Qwen global summary failed for %s: %s; using concatenated aspect summary", entity_id, exc)
                    global_summary = build_global_summary_from_aspects(aspect_summaries, args.global_target_words)
                global_rows_by_entity[entity_id] = {
                    "hotel_id": entity_id,
                    "global_summary": global_summary,
                    "building_summary": aspect_summaries.get("building", ""),
                    "cleanliness_summary": aspect_summaries.get("cleanliness", ""),
                    "food_summary": aspect_summaries.get("food", ""),
                    "location_summary": aspect_summaries.get("location", ""),
                    "rooms_summary": aspect_summaries.get("rooms", ""),
                    "service_summary": aspect_summaries.get("service", ""),
                    "total_evidence_count": sum(aspect_evidence_counts_by_entity.get(entity_id, {}).values()),
                }
                completed_tasks += 1
                if completed_tasks % 10 == 0 or completed_tasks == len(global_tasks):
                    logging.info("Completed %s/%s global summary tasks", completed_tasks, len(global_tasks))
                submit_next_global()

    rows = [summary_rows_by_key[(entity_id, aspect)] for entity_id in entity_ids for aspect in SPACE_ASPECTS]
    global_rows = [global_rows_by_entity[entity_id] for entity_id in entity_ids if entity_id in global_rows_by_entity]
    write_csv(out_csv, rows)
    write_csv(global_csv, global_rows)
    write_csv(evidence_csv, evidence_rows)
    write_csv(cluster_evidence_csv, cluster_evidence_rows)
    metadata = {
        "run_name": args.run_name,
        "mode": args.mode,
        "preseg_csv": str(preseg_csv.resolve()),
        "out_csv": str(out_csv.resolve()),
        "global_csv": str(global_csv.resolve()),
        "evidence_csv": str(evidence_csv.resolve()),
        "cluster_evidence_csv": str(cluster_evidence_csv.resolve()),
        "reroute_cache_jsonl": str(reroute_cache_jsonl.resolve()) if args.reroute_mode == "qwen" else "",
        "space_aspects": SPACE_ASPECTS,
        "entities": len(entity_ids),
        "rows": len(rows),
        "settings": {
            "summary_style": args.summary_style,
            "max_evidence": args.max_evidence,
            "target_words": args.target_words,
            "global_target_words": args.global_target_words,
            "max_sentences": args.max_sentences,
            "min_score": args.min_score,
        },
    }
    (out_dir / f"{args.run_name}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
