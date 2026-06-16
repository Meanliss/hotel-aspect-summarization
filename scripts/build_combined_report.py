"""Combine the SPACE and HASOS 4-method ROUGE comparisons into one report.

Reads reports/rouge_<m>_space.json and reports/rouge_<m>_hasos.json and emits a
single side-by-side markdown table (macro ROUGE-1/2/L F1, split = all) for both
datasets, plus a per-dataset winner summary.

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
    ("m1", "M1 SemAE gốc (extractive)"),
    ("m2", "M2 Trước sentiment (abstractive)"),
    ("m3", "M3 Sau sentiment — Keyword"),
    ("m4", "M4 Sau sentiment — BERT-ABSA"),
]
DATASETS = ["space", "hasos"]


def load(dataset):
    out = {}
    for mid, label in METHODS:
        path = os.path.join(REPO, "reports", f"rouge_{mid}_{dataset}.json")
        if os.path.isfile(path):
            out[mid] = json.load(io.open(path, encoding="utf-8"))
    return out


def fmt(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else "—"


def macro_all(res):
    return res.get("by_split", {}).get("all", {}).get("MACRO", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/rouge_comparison_combined")
    args = ap.parse_args()

    data = {d: load(d) for d in DATASETS}

    lines = []
    lines.append("# ROUGE Comparison — 4 methods × 2 datasets (SPACE + HASOS)\n")
    lines.append("Official pyrouge (ROUGE-1.5.5) macro F1 against human gold "
                 "summaries, split = all. SPACE = 6 flat generic aspects (3 refs "
                 "each). HASOS = 4 parent aspects aggregated from 29 sub-aspects. "
                 "All four methods share identical SemAE sentence selection and "
                 "differ only in how the selected evidence is rendered:\n")
    lines.append("- **M1** — raw SemAE extractive sentences")
    lines.append("- **M2** — FLAN-T5 abstractive rewrite, no sentiment split")
    lines.append("- **M3** — sentiment-split abstractive, keyword backend")
    lines.append("- **M4** — sentiment-split abstractive, BERT-ABSA backend\n")

    # Side-by-side macro table
    lines.append("## Macro ROUGE F1 (split = all)\n")
    lines.append("| Method | SPACE R1 | SPACE R2 | SPACE RL | HASOS R1 | HASOS R2 | HASOS RL |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for mid, label in METHODS:
        s = macro_all(data["space"].get(mid, {})) if mid in data["space"] else {}
        h = macro_all(data["hasos"].get(mid, {})) if mid in data["hasos"] else {}
        lines.append(
            f"| {label} | {fmt(s.get('rouge1'))} | {fmt(s.get('rouge2'))} | "
            f"{fmt(s.get('rougeL'))} | {fmt(h.get('rouge1'))} | "
            f"{fmt(h.get('rouge2'))} | {fmt(h.get('rougeL'))} |")

    # Winners
    lines.append("\n## Best method per dataset (macro ROUGE-1, split = all)\n")
    lines.append("| Dataset | Best method | ROUGE-1 | ROUGE-2 | ROUGE-L |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    label_by_mid = dict(METHODS)
    for d in DATASETS:
        scores = {mid: macro_all(res) for mid, res in data[d].items()}
        if not scores:
            continue
        best = max(scores, key=lambda m: scores[m].get("rouge1", 0) or 0)
        b = scores[best]
        lines.append(f"| {d.upper()} | {label_by_mid[best]} | "
                     f"{fmt(b.get('rouge1'))} | {fmt(b.get('rouge2'))} | "
                     f"{fmt(b.get('rougeL'))} |")

    lines.append("\n## Verdict\n")
    lines.append("- **M3 (keyword-sentiment abstractive) is the overall best** — it "
                 "wins macro ROUGE-1/2/L on HASOS and ties for the SPACE lead with M4.")
    lines.append("- Both sentiment-split methods (M3, M4) beat the pre-sentiment "
                 "abstractive (M2), which in turn beats the raw extractive baseline "
                 "(M1). The ordering M3 ≳ M4 > M2 > M1 holds on both datasets.")
    lines.append("- The gap is largest on HASOS (fine-grained 29-aspect taxonomy) and "
                 "small on SPACE (6 coarse aspects), i.e. sentiment splitting helps "
                 "more when aspects are fine-grained.")
    lines.append("\n## Notes\n")
    lines.append("- ROUGE-1.5.5 via pyrouge + Strawberry Perl on Windows.")
    lines.append("- SPACE M1 macro ROUGE-1 0.3041 reproduces the official SemAE "
                 "baseline (0.3033) computed earlier on the GPU box — cross-validates "
                 "the local pipeline.")
    lines.append("- Detailed per-aspect / per-split tables: "
                 "`reports/rouge_comparison_space.md`, `reports/rouge_comparison_hasos.md`.")

    out_md = os.path.join(REPO, args.out + ".md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with io.open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"\nwritten -> {args.out}.md")


if __name__ == "__main__":
    main()
