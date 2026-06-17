"""Consolidate per-method ROUGE JSONs into one comparison report (md + json).

Reads reports/rouge_<m>_<dataset>.json for each method and emits a side-by-side
table of macro ROUGE-1/2/L F1 per split, plus per-aspect detail, and declares
the winner per metric.

Usage:
    python scripts/build_rouge_comparison.py --dataset hasos \
        --out reports/rouge_comparison_hasos
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = [
    ("m1", "M1 SemAE gốc (extractive)"),
    ("m2", "M2 Trước sentiment (abstractive)"),
    ("m3", "M3 Sau sentiment — Keyword"),
    ("m4", "M4 Sau sentiment — BERT-ABSA"),
]

PARENTS_BY_DATASET = {
    "hasos": ["facility", "amenity", "service", "experience"],
    "space": ["building", "cleanliness", "food", "location", "rooms", "service"],
}

GENERAL_RESULT_KEY = "GENERAL"


def load(dataset):
    data = {}
    for mid, label in METHODS:
        path = os.path.join(REPO, "reports", f"rouge_{mid}_{dataset}.json")
        if os.path.isfile(path):
            data[mid] = (label, json.load(io.open(path, encoding="utf-8")))
    return data


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hasos")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load(args.dataset)
    if not data:
        raise SystemExit(f"No rouge_*_{args.dataset}.json files found in reports/")

    parents = PARENTS_BY_DATASET[args.dataset]
    n_aspects = len(parents)
    label_by_mid = dict(METHODS)

    lines = []
    lines.append(f"# ROUGE Comparison — {args.dataset.upper()} ({len(data)} methods)\n")
    if args.dataset == "hasos":
        lines.append("Official pyrouge (ROUGE-1.5.5) F1 against human gold summaries, "
                     "aggregated to the 4 parent aspects (facility/amenity/service/experience). "
                     "All methods share identical SemAE sentence selection; they differ only "
                     "in how selected evidence is rendered.\n")
    else:
        lines.append("Official pyrouge (ROUGE-1.5.5) F1 against the human SPACE gold "
                     "summaries (6 flat aspects: building/cleanliness/food/location/rooms/service, "
                     "3 references each). All methods share identical SemAE sentence selection; "
                     "they differ only in how selected evidence is rendered.\n")

    # Macro summary table (split = all)
    lines.append(f"## Macro ROUGE F1 (mean over {n_aspects} aspects, split = all)\n")
    lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    lines.append("| --- | ---: | ---: | ---: |")
    macro_all = {}
    for mid, label in METHODS:
        if mid not in data:
            continue
        _, res = data[mid]
        m = res.get("by_split", {}).get("all", {}).get("MACRO", {})
        macro_all[mid] = m
        lines.append(f"| {label} | {fmt(m.get('rouge1'))} | "
                     f"{fmt(m.get('rouge2'))} | {fmt(m.get('rougeL'))} |")

    # SPACE entity-level / overall summary table (split = all)
    general_all = {}
    if args.dataset == "space":
        for mid, _label in METHODS:
            if mid not in data:
                continue
            _, res = data[mid]
            g = res.get("by_split", {}).get("all", {}).get(GENERAL_RESULT_KEY, {})
            if g:
                general_all[mid] = g
        if general_all:
            lines.append("\n## Overall / general ROUGE F1 (SPACE `general`, split = all)\n")
            lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | N |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for mid, label in METHODS:
                if mid not in general_all:
                    continue
                g = general_all[mid]
                lines.append(f"| {label} | {fmt(g.get('rouge1'))} | "
                             f"{fmt(g.get('rouge2'))} | {fmt(g.get('rougeL'))} | "
                             f"{g.get('n', '—')} |")

            lines.append("\n## Winner per overall/general metric (SPACE, split = all)\n")
            lines.append("| Metric | Best method | Score |")
            lines.append("| --- | --- | ---: |")
            for k, name in (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")):
                best_mid = max(general_all, key=lambda m: general_all[m].get(k, 0) or 0)
                lines.append(f"| {name} | {label_by_mid[best_mid]} | "
                             f"{fmt(general_all[best_mid].get(k))} |")

    # Winner per metric
    lines.append("\n## Winner per metric (macro, split = all)\n")
    lines.append("| Metric | Best method | Score |")
    lines.append("| --- | --- | ---: |")
    for k, name in (("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")):
        best_mid = max(macro_all, key=lambda m: macro_all[m].get(k, 0) or 0)
        lines.append(f"| {name} | {label_by_mid[best_mid]} | "
                     f"{fmt(macro_all[best_mid].get(k))} |")

    # Per-split macro
    lines.append("\n## Macro ROUGE F1 by split\n")
    for split in ("dev", "test", "all"):
        lines.append(f"\n### {split}\n")
        lines.append("| Method | ROUGE-1 | ROUGE-2 | ROUGE-L |")
        lines.append("| --- | ---: | ---: | ---: |")
        for mid, label in METHODS:
            if mid not in data:
                continue
            _, res = data[mid]
            m = res.get("by_split", {}).get(split, {}).get("MACRO", {})
            lines.append(f"| {label} | {fmt(m.get('rouge1'))} | "
                         f"{fmt(m.get('rouge2'))} | {fmt(m.get('rougeL'))} |")

    # Per-aspect detail (split = all), ROUGE-1
    lines.append("\n## Per-aspect ROUGE-1 F1 (split = all)\n")
    lines.append("| Method | " + " | ".join(p.capitalize() for p in parents) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in parents) + " |")
    for mid, label in METHODS:
        if mid not in data:
            continue
        _, res = data[mid]
        allsplit = res.get("by_split", {}).get("all", {})
        cells = []
        for p in parents:
            cells.append(fmt(allsplit.get(p, {}).get("rouge1")))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("\n## Notes\n")
    lines.append("- ROUGE-1.5.5 via pyrouge + Strawberry Perl (Windows).")
    if args.dataset == "hasos":
        lines.append("- Gold: `data/hasos/hasos_summ.json`, multi-reference where available.")
        lines.append("- 29 sub-aspects aggregated to 4 gold parents; Branding/Loyalty omitted (no gold).")
    else:
        lines.append("- Gold: `data/space/json/space_summ.json`, 3 references per aspect and 3 references for `general`.")
        lines.append("- 6 flat generic aspects are averaged into MACRO; `GENERAL` scores the overall entity summary separately when present.")
        lines.append("- SPACE has no sentiment-level gold references, so sentiment split is visualized but not scored independently.")
        lines.append("- Evidence selection uses a score threshold of 0.0082; since the SPACE evidence scores top out at ~0.0081, this effectively keeps all threshold-eligible evidence (no additional filtering).")
    lines.append("- M3/M4 concatenate positive + negative generated summaries per aspect.")
    lines.append("- M1 = raw SemAE sentences; M2 = FLAN-T5 rewrite (no split); "
                 "M3 = keyword-sentiment split; M4 = BERT-ABSA-sentiment split.")

    out_md = os.path.join(REPO, args.out + ".md")
    out_json = os.path.join(REPO, args.out + ".json")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with io.open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with io.open(out_json, "w", encoding="utf-8") as f:
        json.dump({mid: data[mid][1] for mid in data}, f, indent=2)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))
    print(f"\nwritten -> {args.out}.md / .json")


if __name__ == "__main__":
    main()
