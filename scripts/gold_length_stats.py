#!/usr/bin/env python3
"""Reference-summary length statistics for SPACE and HASOS gold.

This anchors the token-budget sweep: instead of guessing values for
``--max_tokens`` (M1 extractive, words) and ``--max_new_tokens`` /
``--entity_max_new_tokens`` (M2/M3/M4 abstractive, sub-word tokens), we first
measure how long the human reference summaries actually are. The grid is then
chosen around the gold median / p90 rather than blindly.

Length is reported in *words* (whitespace split) because that is the unit the
extractive ``truncate_summary`` budget uses (``src/utils/summary.py``). A rough
sub-word multiplier (~1.3 tokens/word for FLAN-T5) is also printed so the
abstractive budgets can be calibrated.

Usage:
    python scripts/gold_length_stats.py
    python scripts/gold_length_stats.py --out reports/sweep/gold_length_stats.md
"""
from __future__ import annotations

import argparse
import io
import json
import os
import statistics
import sys

# Windows consoles default to cp1252 and choke on non-latin glyphs; force UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLD = {
    "hasos": os.path.join(REPO, "data", "hasos", "hasos_summ.json"),
    "space": os.path.join(REPO, "data", "space", "json", "space_summ.json"),
}

# Aspect keys per dataset (everything else is treated as entity-level overall).
ASPECT_KEYS = {
    "hasos": {"facility", "amenity", "service", "experience"},
    "space": {"building", "cleanliness", "food", "location", "rooms", "service"},
}
GENERAL_KEYS = {"general"}

# FLAN-T5 sentencepiece is roughly 1.3 sub-word tokens per whitespace word on
# English review prose. Used only to translate word stats into a token budget.
WORDS_TO_TOKENS = 1.3


def word_len(text: str) -> int:
    return len(str(text).split())


def percentile(values, q: float) -> float:
    """Linear-interpolation percentile (q in [0,1]); avoids numpy dependency."""
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    pos = q * (len(xs) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def summarize(values):
    if not values:
        return None
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": sum(values) / len(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def collect(dataset: str):
    """Return {'aspect': [...word lens...], 'general': [...]} for one dataset."""
    path = GOLD[dataset]
    data = json.load(io.open(path, encoding="utf-8"))
    aspect_keys = ASPECT_KEYS[dataset]
    buckets = {"aspect": [], "general": []}
    for ent in data:
        summaries = ent.get("summaries", {})
        for key, refs in summaries.items():
            if not refs:
                continue
            lvl = "aspect" if key in aspect_keys else (
                "general" if key in GENERAL_KEYS else "aspect")
            for ref in refs:
                buckets[lvl].append(word_len(ref))
    return buckets


def fmt_row(label: str, stats) -> str:
    if stats is None:
        return f"| {label} | — | — | — | — | — | — |"
    return (f"| {label} | {stats['n']} | {stats['min']} | "
            f"{stats['median']:.0f} | {stats['mean']:.1f} | "
            f"{stats['p90']:.0f} | {stats['max']} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="optional markdown output path (repo-relative)")
    args = ap.parse_args()

    lines = []
    lines.append("# Gold reference length statistics\n")
    lines.append("Word counts (whitespace split) of human reference summaries. "
                 "`tok≈` columns multiply words by "
                 f"{WORDS_TO_TOKENS} (FLAN-T5 sub-word estimate) to calibrate "
                 "abstractive `--max_new_tokens`.\n")

    suggestions = {}
    for dataset in ("space", "hasos"):
        buckets = collect(dataset)
        lines.append(f"\n## {dataset.upper()}\n")
        lines.append("| Level | n | min | median | mean | p90 | max |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for lvl in ("aspect", "general"):
            stats = summarize(buckets[lvl])
            lines.append(fmt_row(lvl, stats))
            if stats:
                suggestions[(dataset, lvl)] = stats

        # token-budget suggestion from p90 (covers 90% of refs without cutting)
        a = summarize(buckets["aspect"])
        g = summarize(buckets["general"])
        lines.append("")
        if a:
            tok_med = a["median"] * WORDS_TO_TOKENS
            tok_p90 = a["p90"] * WORDS_TO_TOKENS
            lines.append(
                f"- aspect refs: median {a['median']:.0f} w "
                f"(~{tok_med:.0f} tok), p90 {a['p90']:.0f} w "
                f"(~{tok_p90:.0f} tok)")
        if g:
            tok_med = g["median"] * WORDS_TO_TOKENS
            tok_p90 = g["p90"] * WORDS_TO_TOKENS
            lines.append(
                f"- general refs: median {g['median']:.0f} w "
                f"(~{tok_med:.0f} tok), p90 {g['p90']:.0f} w "
                f"(~{tok_p90:.0f} tok)")

    report = "\n".join(lines) + "\n"
    print(report)

    if args.out:
        out_path = os.path.join(REPO, args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with io.open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
