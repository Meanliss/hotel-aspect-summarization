#!/usr/bin/env python3
r"""Aggregate the per-cell sweep JSONs written by sweep_params.py into markdown
tables, mark the best value per metric, show the delta vs the current default,
and emit an optimality conclusion for each dataset.

Crucially, every row shows COVERAGE next to ROUGE. A threshold/token value that
scores higher only because it answered fewer (aspect, entity) instances is NOT
better — the fixed-denominator scorer already penalises empties with 0, and the
coverage column makes any remaining sparsity visible.

Reads:  reports/sweep/rouge_<method>_<dataset>_<phase>_<valuetag>.json
Writes: reports/sweep/threshold_sweep.md
        reports/sweep/token_sweep.md
        reports/sweep/optimality_summary.md

Run with the project venv python (no heavy deps, but keep it consistent):
    .venv/Scripts/python.exe scripts/build_sweep_tables.py
"""
from __future__ import annotations

import glob
import io
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_DIR = os.path.join(REPO, "reports", "sweep")

# Current production defaults, per the code (src/aspect_inference.py:157 and
# synthesize_aspect_summaries.py:1120,1123). Marked with * in the tables.
DEFAULTS = {
    # phase -> default value
    "threshold": {"space": 0.0082, "hasos": 0.005},
    "tokabs": {"space": 192, "hasos": 192},
}

METHOD_LABEL = {
    "m1": "M1 extractive",
    "m2": "M2 abstractive",
    "m3": "M3 kw-sentiment",
    "m4": "M4 bert-sentiment",
}

PHASE_LABEL = {
    "threshold": "evidence_score_threshold",
    "tokabs": "max_new_tokens (abstractive)",
}

CELL_RE = re.compile(
    r"rouge_(m[1-4])_(space|hasos)_(threshold|tokabs)_(.+)\.json$")


def untag(vt: str):
    """Inverse of sweep_params.tag(): 0p0067 -> 0.0067, neg0p02 -> -0.02."""
    s = vt.replace("neg", "-").replace("p", ".")
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return s


def load_cells():
    """Return {(dataset, phase, method): {value: cell_dict}}."""
    cells: dict = {}
    for path in glob.glob(os.path.join(SWEEP_DIR, "rouge_*.json")):
        m = CELL_RE.search(os.path.basename(path))
        if not m:
            continue
        method, dataset, phase, vt = m.groups()
        with io.open(path, encoding="utf-8") as f:
            data = json.load(f)
        allm = data.get("by_split", {}).get("all", {})
        macro = allm.get("MACRO", {})
        if not macro:
            continue
        # mean coverage across aspect keys (exclude MACRO/GENERAL)
        covs = [v.get("coverage") for k, v in allm.items()
                if k not in ("MACRO", "GENERAL") and isinstance(v, dict)
                and v.get("coverage") is not None]
        mean_cov = sum(covs) / len(covs) if covs else None
        n_aspects = len(covs)
        cells.setdefault((dataset, phase, method), {})[untag(vt)] = {
            "rouge1": macro.get("rouge1"),
            "rouge2": macro.get("rouge2"),
            "rougeL": macro.get("rougeL"),
            "coverage": mean_cov,
            "n_aspects": n_aspects,
            "fixed_denominator": data.get("fixed_denominator", False),
        }
    return cells


def fmt(x, nd=5):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "—"


def best_value(series, metric="rouge1"):
    """Return the value (key) with the max metric in this series."""
    best, best_v = None, -1.0
    for val, cell in series.items():
        r = cell.get(metric)
        if r is not None and r > best_v:
            best, best_v = val, r
    return best


def build_phase_table(cells, dataset, phase):
    """Markdown for one (dataset, phase): one block per method."""
    lines = []
    default_val = DEFAULTS.get(phase, {}).get(dataset)
    any_method = False
    for method in ("m1", "m2", "m3", "m4"):
        series = cells.get((dataset, phase, method))
        if not series:
            continue
        any_method = True
        best = best_value(series, "rouge1")
        lines.append(f"\n**{METHOD_LABEL[method]}** "
                     f"(default {PHASE_LABEL[phase]} = {default_val})\n")
        lines.append("| value | R1 | R2 | RL | coverage | n_asp | ΔR1 vs default |")
        lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        default_r1 = None
        if default_val in series:
            default_r1 = series[default_val].get("rouge1")
        for val in sorted(series.keys(), key=lambda v: (isinstance(v, str), v)):
            cell = series[val]
            marks = []
            if default_val is not None and val == default_val:
                marks.append("*")  # current default
            if val == best:
                marks.append("**best**")
            label = f"{val}{''.join(m for m in marks if m == '*')}"
            r1 = cell.get("rouge1")
            delta = ""
            if default_r1 is not None and r1 is not None:
                d = r1 - default_r1
                delta = f"{d:+.5f}" + (" ✓" if val == best and d > 0 else "")
            cov = cell.get("coverage")
            cov_s = f"{cov:.2f}" if cov is not None else "—"
            best_tag = " **(best)**" if val == best else ""
            lines.append(
                f"| {label}{best_tag} | {fmt(r1)} | {fmt(cell.get('rouge2'))} | "
                f"{fmt(cell.get('rougeL'))} | {cov_s} | {cell.get('n_aspects')} | "
                f"{delta} |")
    return lines, any_method


def write_threshold_table(cells):
    out = [
        "# Threshold sweep — macro ROUGE F1 (split=all), fixed denominator\n",
        "Every value scores the SAME (split, entity) universe; instances with no "
        "generated summary count as ROUGE 0. So a stricter threshold that drops "
        "evidence is penalised through both ROUGE and the coverage column. "
        "`*` = current code default. ROUGE-1 is the decision metric.\n",
    ]
    for dataset in ("space", "hasos"):
        out.append(f"\n## {dataset.upper()}\n")
        block, any_m = build_phase_table(cells, dataset, "threshold")
        if any_m:
            out += block
        else:
            out.append("_(no cells yet)_")
    path = os.path.join(SWEEP_DIR, "threshold_sweep.md")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path


def write_token_table(cells):
    out = [
        "# Token-budget sweep — macro ROUGE F1 (split=all), fixed denominator\n",
        "Abstractive `--max_new_tokens` (M2/M3/M4). Threshold held at the default. "
        "Token budget does not drop entities, so coverage stays flat here and the "
        "comparison is clean. `*` = current default (192).\n",
    ]
    for dataset in ("space", "hasos"):
        out.append(f"\n## {dataset.upper()}\n")
        block, any_m = build_phase_table(cells, dataset, "tokabs")
        if any_m:
            out += block
        else:
            out.append("_(no cells yet)_")
    path = os.path.join(SWEEP_DIR, "token_sweep.md")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path


def conclude(cells):
    """Per (dataset, phase, method): is the default optimal in the grid?"""
    out = ["# Optimality summary\n",
           "For each swept parameter, the value with the highest macro ROUGE-1 "
           "(fixed-denominator, split=all) is compared against the current code "
           "default. A non-default winner is only reported as an improvement when "
           "its coverage is within 0.02 of the default's — otherwise the gain is a "
           "coverage artifact and the default is kept.\n"]
    for dataset in ("space", "hasos"):
        out.append(f"\n## {dataset.upper()}\n")
        for phase in ("threshold", "tokabs"):
            default_val = DEFAULTS.get(phase, {}).get(dataset)
            for method in ("m1", "m2", "m3", "m4"):
                series = cells.get((dataset, phase, method))
                if not series:
                    continue
                best = best_value(series, "rouge1")
                best_r1 = series[best].get("rouge1")
                best_cov = series[best].get("coverage")
                def_cell = series.get(default_val)
                def_r1 = def_cell.get("rouge1") if def_cell else None
                def_cov = def_cell.get("coverage") if def_cell else None
                verdict = ""
                if def_r1 is None:
                    verdict = (f"default {default_val} not in grid; best={best} "
                               f"R1={fmt(best_r1)}")
                elif best == default_val:
                    verdict = f"**default {default_val} is optimal** (R1={fmt(def_r1)})"
                else:
                    cov_ok = (best_cov is not None and def_cov is not None
                              and abs(best_cov - def_cov) <= 0.02)
                    gain = best_r1 - def_r1 if (best_r1 and def_r1) else 0
                    if cov_ok and gain > 0:
                        verdict = (f"**{best} beats default {default_val}** "
                                   f"by ΔR1={gain:+.5f} at equal coverage "
                                   f"(cov {best_cov:.2f} vs {def_cov:.2f}) "
                                   f"→ recommend switching")
                    else:
                        verdict = (f"best={best} R1={fmt(best_r1)} but cov "
                                   f"{best_cov if best_cov is None else round(best_cov,2)} "
                                   f"vs default cov "
                                   f"{def_cov if def_cov is None else round(def_cov,2)} "
                                   f"→ gain is a coverage artifact, **keep default "
                                   f"{default_val}** (R1={fmt(def_r1)})")
                out.append(f"- **{PHASE_LABEL[phase]} / {METHOD_LABEL[method]}**: "
                           f"{verdict}")
    path = os.path.join(SWEEP_DIR, "optimality_summary.md")
    with io.open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    return path


def main():
    cells = load_cells()
    if not cells:
        print("no sweep cells found in", SWEEP_DIR)
        sys.exit(0)
    p1 = write_threshold_table(cells)
    p2 = write_token_table(cells)
    p3 = conclude(cells)
    n = sum(len(v) for v in cells.values())
    print(f"aggregated {n} cells across {len(cells)} (dataset,phase,method) series")
    for p in (p1, p2, p3):
        print("written ->", os.path.relpath(p, REPO))


if __name__ == "__main__":
    main()
