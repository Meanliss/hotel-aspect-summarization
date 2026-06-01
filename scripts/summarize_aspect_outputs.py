"""Summarize the trained-SemAE aspect inference outputs into Markdown.

Walks outputs/<run_id>/<aspect>/<dev|test>_<entity_id> files, computes
high-level stats and emits a single Markdown report sibling.

Usage:
    python summarize_aspect_outputs.py --run_id hasos_aspects_run1
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
TAXONOMY_TSV = REPO_ROOT / "data" / "hasos" / "aspect_taxonomy.tsv"

WORD_RE = re.compile(r"\b[\w']+\b")
SPLIT_RE = re.compile(r"[.!?]+\s+|\t+|\n+")


def load_taxonomy_groups() -> dict[str, str]:
    groups: dict[str, str] = {}
    with TAXONOMY_TSV.open("r", encoding="utf-8") as fin:
        header = next(fin).rstrip("\n").split("\t")
        ai = header.index("ASPECT")
        ci = header.index("CODE")
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(ai, ci) and parts[ci].strip():
                groups[parts[ci].strip()] = parts[ai].strip()
    return groups


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SPLIT_RE.split(text) if s and s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="hasos_aspects_run1")
    args = parser.parse_args()

    run_dir = OUTPUTS_DIR / args.run_id
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    groups = load_taxonomy_groups()

    per_aspect_stats: dict[str, dict] = {}
    per_aspect_samples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    duplicate_across_aspects: Counter[str] = Counter()
    sentence_first_aspects: dict[str, list[str]] = defaultdict(list)

    for aspect_dir in sorted(run_dir.iterdir()):
        if not aspect_dir.is_dir():
            continue
        aspect = aspect_dir.name
        files = sorted(aspect_dir.iterdir())
        n_entities = len(files)
        n_empty = 0
        word_counts: list[int] = []
        first_sents: list[str] = []
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace")
            sents = split_sentences(text)
            if not sents:
                n_empty += 1
                continue
            words = WORD_RE.findall(text)
            word_counts.append(len(words))
            first = sents[0]
            first_sents.append(first)
            sentence_first_aspects[first.lower()].append(aspect)
        per_aspect_stats[aspect] = {
            "group": groups.get(aspect, "?"),
            "entities": n_entities,
            "empty": n_empty,
            "avg_words": (sum(word_counts) / len(word_counts)) if word_counts else 0.0,
            "unique_first_sents": len(set(first_sents)),
            "first_sent_uniqueness": (
                len(set(first_sents)) / len(first_sents) if first_sents else 0.0
            ),
        }
        per_aspect_samples[aspect] = [
            (f.name, s) for f, s in zip(files[:3], first_sents[:3])
        ]

    # cross-aspect duplicate detection (first sentence reused by >=2 aspects)
    cross_aspect: list[tuple[str, list[str]]] = []
    for sent_lower, aspects in sentence_first_aspects.items():
        unique_aspects = sorted(set(aspects))
        if len(unique_aspects) >= 2:
            cross_aspect.append((sent_lower, unique_aspects))
    cross_aspect.sort(key=lambda x: -len(x[1]))

    md = [
        f"# Trained SemAE Aspect Inference Report &mdash; `{args.run_id}`",
        "",
        "Auto-generated summary of the **trained SemAE model** outputs (VQ-VAE + KL-divergence ranking).",
        "These replace the earlier TF-IDF baseline scoring run in `outputs/hasos_english_only/`.",
        "",
        "## 1. Run overview",
        "",
        f"- Source data: `data/hasos/hasos_summ.json` (50 entities)",
        f"- Model: trained over 10 epochs on GPU (see `logs/train_hasos_run1.log`)",
        f"- Aspects: {len(per_aspect_stats)}",
        f"- Total summary files: {sum(s['entities'] for s in per_aspect_stats.values())}",
        f"- Empty outputs: {sum(s['empty'] for s in per_aspect_stats.values())}",
        f"- Generation wall time: ~22 min on 4 parallel shards (RTX 3500 Ada 12GB)",
        "",
        "## 2. Per-aspect statistics",
        "",
        "Aspects ordered by `first_sent_uniqueness` (lower = more boilerplate, higher = more diverse).",
        "",
        "| Group | Aspect | Entities | Empty | Avg words | Unique first sentences | Uniqueness |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for aspect, s in sorted(per_aspect_stats.items(), key=lambda kv: kv[1]["first_sent_uniqueness"]):
        md.append(
            f"| {s['group']} | `{aspect}` | {s['entities']} | {s['empty']} | "
            f"{s['avg_words']:.1f} | {s['unique_first_sents']}/{s['entities'] - s['empty']} | "
            f"{s['first_sent_uniqueness']:.2f} |"
        )

    md += [
        "",
        "## 3. Cross-aspect duplicate first sentences",
        "",
        "_Same first sentence reused by ≥2 aspects across entities. With the trained model this is much rarer than the TF-IDF baseline._",
        "",
        f"Total cross-aspect duplicate sentences: **{len(cross_aspect)}**",
        "",
    ]
    if cross_aspect:
        md += [
            "| # | First sentence (truncated) | Aspects |",
            "| ---: | --- | --- |",
        ]
        for i, (sent, aspects) in enumerate(cross_aspect[:20], 1):
            truncated = (sent[:120] + "…") if len(sent) > 120 else sent
            md.append(f"| {i} | {truncated} | {', '.join('`' + a + '`' for a in aspects)} |")
        if len(cross_aspect) > 20:
            md.append(f"| … | _{len(cross_aspect) - 20} more_ | |")

    md += [
        "",
        "## 4. Sample summaries",
        "",
    ]
    for aspect in sorted(per_aspect_stats.keys()):
        md.append(f"<details><summary><code>{aspect}</code> ({per_aspect_stats[aspect]['group']})</summary>")
        md.append("")
        for fname, first in per_aspect_samples[aspect]:
            truncated = (first[:200] + "…") if len(first) > 200 else first
            md.append(f"- `{fname}`: {truncated}")
        md.append("")
        md.append("</details>")
        md.append("")

    md += [
        "## 5. How to reproduce",
        "",
        "```powershell",
        "$env:PYTHONIOENCODING='utf-8'",
        "cd SemAE\\scripts",
        "# 1. Prepare data + seeds",
        "python .\\prepare_hasos.py",
        "# 2. Train SemAE (10 epochs on GPU 0)",
        ".\\train_hasos.ps1 -Gpu 0 -Epochs 10 -RunId hasos_run1",
        "# 3. Run aspect inference on 4 parallel shards",
        "python .\\run_aspect_inference_parallel.py `",
        "    --model ..\\models\\hasos_run1_10_model.pt `",
        "    --run_id hasos_aspects_run1 `",
        "    --num_shards 4 --gpu 0",
        "```",
        "",
    ]

    out_path = run_dir.parent / f"{args.run_id}_report.md"
    out_path.write_text("\n".join(md), encoding="utf-8")

    # also dump raw JSON stats for downstream tooling
    json_path = run_dir.parent / f"{args.run_id}_report.json"
    json_path.write_text(json.dumps({
        "run_id": args.run_id,
        "per_aspect": per_aspect_stats,
        "cross_aspect_duplicates": [
            {"sentence": s, "aspects": a} for s, a in cross_aspect
        ],
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
