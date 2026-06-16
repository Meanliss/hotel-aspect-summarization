#!/usr/bin/env python3
"""Export SemAE/HASOS pipeline outputs into a single static JSON tree for the web app.

The web frontend (web/) reads the resulting file from web/public/data/<run_id>.json
and renders it 100% client-side (no backend needed for the Explore tab).

Inputs (all under outputs/ for a given --run_id):
  - <run_id>_hierarchical_summary.json   : per-entity child/parent/overall summaries
  - <run_id>_ranked_evidence.jsonl       : ranked evidence sentences (with sentiment)
  - <run_id>_threshold_evidence.jsonl    : fallback if ranked_evidence is absent
  - data/hasos/aspect_taxonomy.json      : aspect taxonomy (group -> codes, names)
  - data/hasos/hasos_summ.json           : entity_id -> entity_name (optional)

Output tree (web/public/data/<run_id>.json):
{
  "run_id": "...",
  "generated_at": "...",
  "aspect_taxonomy": [ {code, group, scale, description}, ... ],
  "entities": [
    {
      "entity_id": "100597",
      "entity_name": "Doubletree by Hilton Seattle Airport",
      "split": "dev",
      "overall_summary": "...",
      "parents": [
        {
          "code": "FACILITY",
          "summary": "...",
          "children": [
            {
              "code": "FAC_ROOM",
              "scale": "Room, Bed & Sleep Quality",
              "description": "...",
              "summaries": {"positive": "...", "negative": "..."},
              "evidence": {
                "positive": [{"sentence","score","review_id","rank"}, ...],
                "negative": [...],
                "neutral":  [...]
              }
            }, ...
          ]
        }, ...
      ]
    }, ...
  ]
}
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
DATA_DIR = REPO_ROOT / "data"
TAXONOMY_JSON = DATA_DIR / "hasos" / "aspect_taxonomy.json"
HASOS_SUMM_JSON = DATA_DIR / "hasos" / "hasos_summ.json"
WEB_DATA_DIR = REPO_ROOT / "web" / "public" / "data"

# Canonical sentiment label normalisation.
_SENTIMENT_CANON = {
    "pos": "positive", "positive": "positive",
    "neg": "negative", "negative": "negative",
    "neu": "neutral", "neutral": "neutral",
}


def canon_sentiment(label) -> str:
    return _SENTIMENT_CANON.get(str(label).strip().lower(), "neutral")


def load_taxonomy(path: Path) -> tuple[list[dict], dict[str, dict], list[str]]:
    """Return (flat_list, by_code, parent_order)."""
    with path.open("r", encoding="utf-8") as fin:
        data = json.load(fin)
    entries = data["aspects"] if isinstance(data, dict) else data
    flat = []
    by_code = {}
    parent_order: list[str] = []
    for entry in entries:
        code = entry.get("code") or entry.get("CODE")
        group = entry.get("group") or entry.get("GROUP") or ""
        scale = (entry.get("measurement_scale")
                 or entry.get("MEASUREMENT_SCALE") or code)
        description = entry.get("description") or entry.get("DESCRIPTION") or ""
        info = {
            "code": code,
            "group": group,
            "scale": scale,
            "description": description,
        }
        flat.append(info)
        by_code[code] = info
        if group and group not in parent_order:
            parent_order.append(group)
    return flat, by_code, parent_order


def load_entity_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    if not path.exists():
        return names
    try:
        with path.open("r", encoding="utf-8") as fin:
            data = json.load(fin)
    except (json.JSONDecodeError, OSError):
        return names
    items = data if isinstance(data, list) else data.values()
    for entity in items:
        if not isinstance(entity, dict):
            continue
        eid = str(entity.get("entity_id", ""))
        if eid:
            names[eid] = entity.get("entity_name", "") or eid
    return names


def normalize_child_summaries(value) -> dict[str, str]:
    """Accept either a plain string (legacy) or {positive, negative, ...}."""
    if isinstance(value, dict):
        return {
            "positive": str(value.get("positive", "") or ""),
            "negative": str(value.get("negative", "") or ""),
            "neutral": str(value.get("neutral", "") or ""),
        }
    text = str(value or "")
    # Legacy single summary: keep it under a neutral "summary" slot so the UI
    # can still show something even when sentiment was not split.
    return {"positive": "", "negative": "", "neutral": text}


def load_evidence(run_id: str) -> dict[tuple[str, str, str], dict[str, list[dict]]]:
    """Return evidence grouped by (split, entity_id, aspect) -> {sentiment: [rows]}."""
    path = OUTPUTS_DIR / f"{run_id}_ranked_evidence.jsonl"
    if not path.exists():
        path = OUTPUTS_DIR / f"{run_id}_threshold_evidence.jsonl"
    grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = defaultdict(
        lambda: {"positive": [], "negative": [], "neutral": []})
    if not path.exists():
        return grouped
    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            split = row.get("split", "")
            entity_id = str(row.get("entity_id", ""))
            aspect = row.get("aspect", "")
            if not (split and entity_id and aspect):
                continue
            sentiment = canon_sentiment(row.get("sentiment_label", "neutral"))
            grouped[(split, entity_id, aspect)][sentiment].append({
                "sentence": row.get("sentence", ""),
                "score": row.get("score"),
                "rank": row.get("rank"),
                "review_id": row.get("source_review_id") or row.get("review_id"),
            })
    # Sort each bucket by score (ascending = best SemAE rank first).
    for buckets in grouped.values():
        for rows in buckets.values():
            rows.sort(key=lambda r: (
                r.get("score") if r.get("score") is not None else 10**9,
                r.get("rank") if r.get("rank") is not None else 10**9,
            ))
    return grouped


def build_export(run_id: str, evidence_limit: int) -> dict:
    hier_path = OUTPUTS_DIR / f"{run_id}_hierarchical_summary.json"
    if not hier_path.exists():
        raise SystemExit(
            f"Missing hierarchical summary: {hier_path}\n"
            "Run synthesize_aspect_summaries.py --hierarchical first.")
    with hier_path.open("r", encoding="utf-8") as fin:
        hier = json.load(fin)

    flat_taxonomy, by_code, parent_order = load_taxonomy(TAXONOMY_JSON)
    entity_names = load_entity_names(HASOS_SUMM_JSON)
    evidence = load_evidence(run_id)

    entities_payload = hier.get("entities", {})
    entities_out = []
    for key, entity in sorted(entities_payload.items()):
        split = entity.get("split", "")
        entity_id = str(entity.get("entity_id", ""))
        child_aspects = entity.get("child_aspects", {})
        parent_aspects = entity.get("parent_aspects", {})

        # Group children under their taxonomy parent (group).
        children_by_parent: dict[str, list[dict]] = defaultdict(list)
        for code, summary_value in child_aspects.items():
            info = by_code.get(code, {})
            group = info.get("group", "") or "OTHER"
            buckets = evidence.get((split, entity_id, code),
                                   {"positive": [], "negative": [], "neutral": []})
            children_by_parent[group].append({
                "code": code,
                "scale": info.get("scale", code),
                "description": info.get("description", ""),
                "summaries": normalize_child_summaries(summary_value),
                "evidence": {
                    "positive": buckets["positive"][:evidence_limit],
                    "negative": buckets["negative"][:evidence_limit],
                    "neutral": buckets["neutral"][:evidence_limit],
                },
            })

        parents_out = []
        seen_parents = set()
        for group in parent_order + sorted(children_by_parent.keys()):
            if group in seen_parents or group not in children_by_parent:
                continue
            seen_parents.add(group)
            children = sorted(children_by_parent[group],
                              key=lambda c: c["code"])
            parents_out.append({
                "code": group,
                "summary": parent_aspects.get(group, ""),
                "children": children,
            })

        entities_out.append({
            "entity_id": entity_id,
            "entity_name": entity_names.get(entity_id, entity_id),
            "split": split,
            "overall_summary": entity.get("overall_summary", ""),
            "parents": parents_out,
        })

    return {
        "run_id": run_id,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "aspect_taxonomy": flat_taxonomy,
        "parent_order": parent_order,
        "entities": entities_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export pipeline outputs into a static JSON tree for the web app.")
    parser.add_argument("--run_id", default="space_hasos_threshold_full",
                        help="source run id under outputs/")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: web/public/data/<run_id>.json)")
    parser.add_argument("--evidence_limit", type=int, default=8,
                        help="max evidence sentences per sentiment bucket per aspect")
    parser.add_argument("--index", action="store_true",
                        help="also (re)write web/public/data/index.json listing all runs")
    args = parser.parse_args()

    payload = build_export(args.run_id, args.evidence_limit)

    out_path = Path(args.out) if args.out else WEB_DATA_DIR / f"{args.run_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes, "
          f"{len(payload['entities'])} entities)")

    if args.index:
        runs = sorted(p.stem for p in WEB_DATA_DIR.glob("*.json")
                      if p.stem != "index")
        index_path = WEB_DATA_DIR / "index.json"
        index_path.write_text(
            json.dumps({"runs": runs}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"wrote {index_path} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
