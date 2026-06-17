#!/usr/bin/env python3
"""Export SPACE 4-method summaries + gold into one JSON for the web app.

The four methods all derive from the SAME SemAE sentence selection; they differ
only in how the selected evidence is rendered:

  m1 : raw SemAE-selected sentences (outputs/<m1_run>/<aspect>/<split>_<eid>)
  m2 : FLAN-T5 rewrite, no sentiment split (flat <aspect>/<split>_<eid>)
  m3 : sentiment-split abstractive, keyword backend (<aspect>/{pos,neg}/<split>_<eid>)
  m4 : sentiment-split abstractive, BERT-ABSA backend (same layout as m3)

Each method also has an entity-level overall summary written by the hierarchical
synthesis step under <run>_entity/<split>_<eid>. SPACE gold (data/space/json/
space_summ.json) carries a non-aspectual `general` summary plus 6 per-aspect
references (3 each). There is no sentiment-level gold, so the web shows the
positive/negative split for m3/m4 but ROUGE only scores aspects + general.

Output: web/public/data/space_4method.json (shape consumed by lib/space.ts).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASPECTS = ["building", "cleanliness", "food", "location", "rooms", "service"]

# (method_id, aspect_tree_dir, entity_overall_dir, kind)
#   kind "extractive": tab-separated raw sentences, no sentiment, no overall gen
#   kind "flat":       one abstractive file per aspect, overall dir present
#   kind "sentiment":  <aspect>/{pos,neg}/<file>, overall dir present
METHOD_SOURCES = {
    "m1": dict(tree="outputs/space_eval_4method",
               entity=None, kind="extractive"),
    "m2": dict(tree="outputs/space_eval_4method_m2",
               entity="outputs/space_eval_4method_m2_entity", kind="flat"),
    "m3": dict(tree="outputs/space_eval_4method_m3_kw",
               entity="outputs/space_eval_4method_m3_kw_entity", kind="sentiment"),
    "m4": dict(tree="outputs/space_eval_4method_m4_bert",
               entity="outputs/space_eval_4method_m4_bert_entity", kind="sentiment"),
}

METHOD_META = {
    "m1": {"label": "M1 — Extractive (SemAE)", "short": "M1",
           "desc": "Raw top-ranked SemAE sentences, joined as-is.",
           "color": "slate"},
    "m2": {"label": "M2 — Abstractive (no sentiment)", "short": "M2",
           "desc": "FLAN-T5 rewrites the selected evidence; no sentiment split.",
           "color": "sky"},
    "m3": {"label": "M3 — Sentiment split · Keyword", "short": "M3",
           "desc": "Keyword sentiment split, then FLAN-T5 rewrites each polarity.",
           "color": "emerald"},
    "m4": {"label": "M4 — Sentiment split · BERT-ABSA", "short": "M4",
           "desc": "BERT-ABSA sentiment split, then FLAN-T5 rewrites each polarity.",
           "color": "violet"},
}


def clean_text(s: str) -> str:
    s = s.replace("\t", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def read_file(path: str) -> str:
    if os.path.isfile(path):
        with io.open(path, encoding="utf-8", errors="replace") as f:
            return clean_text(f.read())
    return ""


def load_gold(path: str):
    """Return {eid: {"general": [...], "aspects": {aspect: [...]}}, name}."""
    data = json.load(io.open(path, encoding="utf-8"))
    gold = {}
    names = {}
    for ent in data:
        eid = str(ent["entity_id"])
        names[eid] = ent.get("entity_name", eid)
        summaries = ent.get("summaries", {})
        aspects = {a: list(summaries.get(a, [])) for a in ASPECTS
                   if summaries.get(a)}
        gold[eid] = {
            "overall": list(summaries.get("general", [])),
            "aspects": aspects,
        }
    return gold, names


def load_splits(path: str):
    splits = {}
    if os.path.isfile(path):
        for line in io.open(path, encoding="utf-8"):
            parts = line.split()
            if len(parts) == 2:
                splits[parts[0]] = parts[1]
    return splits


def method_aspect_text(method: str, eid: str, split: str):
    """Return per-aspect dict for one method/entity. Shape depends on kind."""
    src = METHOD_SOURCES[method]
    tree = os.path.join(REPO, src["tree"])
    file_name = f"{split}_{eid}"
    out = {}
    for aspect in ASPECTS:
        adir = os.path.join(tree, aspect)
        if src["kind"] == "sentiment":
            pos = read_file(os.path.join(adir, "pos", file_name))
            neg = read_file(os.path.join(adir, "neg", file_name))
            if pos or neg:
                out[aspect] = {"positive": pos, "negative": neg}
        else:
            text = read_file(os.path.join(adir, file_name))
            if text:
                out[aspect] = {"overall": text}
    return out


def method_overall_text(method: str, eid: str, split: str, aspects: dict) -> str:
    """Entity overall: prefer generated <run>_entity file, else concatenate."""
    src = METHOD_SOURCES[method]
    file_name = f"{split}_{eid}"
    if src["entity"]:
        text = read_file(os.path.join(REPO, src["entity"], file_name))
        if text:
            return text
    # Fallback: concatenate available aspect text (M1 has no generated overall).
    parts = []
    for aspect in ASPECTS:
        cell = aspects.get(aspect)
        if not cell:
            continue
        if "overall" in cell:
            parts.append(cell["overall"])
        else:
            parts.append(" ".join(p for p in (cell.get("positive"),
                                               cell.get("negative")) if p))
    return clean_text(" ".join(parts))


def discover_entities():
    """Sorted [(split, eid)] from the M1 tree (all methods share entities)."""
    tree = os.path.join(REPO, METHOD_SOURCES["m1"]["tree"])
    found = set()
    for aspect in ASPECTS:
        adir = os.path.join(tree, aspect)
        if not os.path.isdir(adir):
            continue
        for name in os.listdir(adir):
            m = re.match(r"(dev|test|train)_(.+)$", name)
            if m:
                found.add((m.group(1), m.group(2)))
    return sorted(found)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/space/json/space_summ.json")
    ap.add_argument("--splits", default="data/space/json/space_summ_splits.txt")
    ap.add_argument("--rouge", default="reports/rouge_comparison_space.json",
                    help="optional ROUGE comparison JSON to embed")
    ap.add_argument("--out", default="web/public/data/space_4method.json")
    args = ap.parse_args()

    gold, names = load_gold(os.path.join(REPO, args.gold))
    splits = load_splits(os.path.join(REPO, args.splits))
    entities = discover_entities()

    rouge = None
    rouge_path = os.path.join(REPO, args.rouge)
    if os.path.isfile(rouge_path):
        rouge = json.load(io.open(rouge_path, encoding="utf-8"))

    out_entities = []
    for split, eid in entities:
        # prefer the authoritative split map; fall back to filename split
        ent_split = splits.get(eid, split)
        methods = {}
        for mid in ("m1", "m2", "m3", "m4"):
            aspects = method_aspect_text(mid, eid, split)
            overall = method_overall_text(mid, eid, split, aspects)
            methods[mid] = {"overall": overall, "aspects": aspects}
        out_entities.append({
            "entity_id": eid,
            "entity_name": names.get(eid, eid),
            "split": ent_split,
            "gold": gold.get(eid, {"overall": [], "aspects": {}}),
            "methods": methods,
        })

    payload = {
        "dataset": "space",
        "title": "Aspect-Based Sentiment Summarization for the Hotel Domain",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aspects": ASPECTS,
        "methods": ["m1", "m2", "m3", "m4"],
        "method_meta": METHOD_META,
        "rouge": rouge,
        "entities": out_entities,
    }

    out_path = os.path.join(REPO, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_overall = sum(1 for e in out_entities
                    if e["methods"]["m2"]["overall"])
    print(f"entities={len(out_entities)} with_m2_overall={n_overall}")
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
