"""Combine the SPACE and HASOS 4-method ROUGE comparisons into one report.

Reads reports/rouge_<m>_<dataset>.json and emits a side-by-side markdown table
with macro ROUGE-1/2/L F1 for split=all.

Usage:
    python scripts/build_combined_report.py --out reports/rouge_comparison_combined
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = [
    ("m1", "M1 SemAE extractive"),
    ("m2", "M2 abstractive before sentiment"),
    ("m3", "M3 keyword sentiment-split"),
    ("m4", "M4 BERT-ABSA sentiment-split"),
]
DATASETS = ["space", "hasos"]


def load(dataset: str) -> dict[str, dict]:
    out = {}
    for mid, _label in METHODS:
        path = os.path.join(REPO, "reports", f"rouge_{mid}_{dataset}.json")
        if os.path.isfile(path):
            out[mid] = json.load(io.open(path, encoding="utf-8"))
    return out


def fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "-"


def macro_all(result: dict) -> dict:
    return result.get("by_split", {}).get("all", {}).get("MACRO", {})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="reports/rouge_comparison_combined")
    args = parser.parse_args()

    data = {dataset: load(dataset) for dataset in DATASETS}
    label_by_mid = dict(METHODS)

    lines = [
        "# ROUGE Comparison - 4 methods x 2 datasets (SPACE + HASOS)",
        "",
        "Official pyrouge (ROUGE-1.5.5) macro F1 against human gold summaries, "
        "split = all. SPACE uses 6 flat generic aspects with 3 references each. "
        "HASOS uses 4 parent aspects aggregated from 29 sub-aspects.",
        "",
        "- **M1** - raw SemAE extractive sentences",
        "- **M2** - FLAN-T5 abstractive rewrite, no sentiment split",
        "- **M3** - sentiment-split abstractive, keyword backend",
        "- **M4** - sentiment-split abstractive, BERT-ABSA backend",
        "",
        "## Macro ROUGE F1 (split = all)",
        "",
        "| Method | SPACE R1 | SPACE R2 | SPACE RL | HASOS R1 | HASOS R2 | HASOS RL |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for mid, label in METHODS:
        space = macro_all(data["space"].get(mid, {})) if mid in data["space"] else {}
        hasos = macro_all(data["hasos"].get(mid, {})) if mid in data["hasos"] else {}
        lines.append(
            f"| {label} | {fmt(space.get('rouge1'))} | {fmt(space.get('rouge2'))} | "
            f"{fmt(space.get('rougeL'))} | {fmt(hasos.get('rouge1'))} | "
            f"{fmt(hasos.get('rouge2'))} | {fmt(hasos.get('rougeL'))} |"
        )

    lines += [
        "",
        "## Best method per dataset (macro ROUGE-1, split = all)",
        "",
        "| Dataset | Best method | ROUGE-1 | ROUGE-2 | ROUGE-L |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for dataset in DATASETS:
        scores = {mid: macro_all(result) for mid, result in data[dataset].items()}
        if not scores:
            continue
        best = max(scores, key=lambda mid: scores[mid].get("rouge1", 0) or 0)
        cell = scores[best]
        lines.append(
            f"| {dataset.upper()} | {label_by_mid[best]} | {fmt(cell.get('rouge1'))} | "
            f"{fmt(cell.get('rouge2'))} | {fmt(cell.get('rougeL'))} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        "- SPACE is very close across the four methods, with M3 slightly ahead on "
        "macro ROUGE-1/L in the baseline comparison.",
        "- In the non-optimized HASOS baseline table, no single method wins all "
        "ROUGE metrics: M2 leads ROUGE-1, M1 leads ROUGE-2, and M3 leads ROUGE-L. "
        "Use `reports/metrics/paper_tables.md` for the optimized HASOS paper table.",
        "- The optimized HASOS sweep is reported separately because M2/M3/M4 can "
        "be re-filtered and re-synthesized without changing the M1 extractive baseline.",
        "",
        "## Notes",
        "",
        "- ROUGE-1.5.5 via pyrouge + Strawberry Perl on Windows.",
        "- SPACE M1 macro ROUGE-1 0.3041 reproduces the official SemAE baseline "
        "(0.3033) computed earlier on the GPU box.",
        "- HASOS M1 is read from `reports/rouge_m1_hasos.json`; "
        "`reports/sweep/_sanity_m1_hasos.json` is a fixed-denominator sanity "
        "artifact and is not used for the paper table.",
        "- Detailed per-aspect / per-split tables: `reports/rouge_comparison_space.md`, "
        "`reports/rouge_comparison_hasos.md`.",
    ]

    out_md = os.path.join(REPO, args.out + ".md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    text = "\n".join(lines) + "\n"
    with io.open(out_md, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    sys.stdout.buffer.write(text.encode("utf-8"))
    print(f"\nwritten -> {args.out}.md")


if __name__ == "__main__":
    main()
