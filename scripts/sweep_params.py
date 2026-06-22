#!/usr/bin/env python3
r"""Grid-search harness for the two un-justified hyperparameters of the 4-method
SemAE opinion-summarization pipeline:

  1. ``--evidence_score_threshold`` — SemAE KL cutoff deciding how much evidence
     feeds the abstractive stage (affects M2/M3/M4; M1 is independent of it).
  2. token output budget — ``--max_tokens`` (M1 extractive, words) and
     ``--max_new_tokens`` (M2/M3/M4 abstractive, sub-word tokens).

The harness runs one factor at a time (Phase A = threshold, Phase B = token),
generates the needed outputs, scores them with the SAME official pyrouge driver
(``score_rouge_compare.py``) used for the committed baseline, and writes one
JSON per (dataset, method, value) cell under ``reports/sweep/``.

Cost model (verified 2026-06-22 on RTX 3050):
  * SPACE evidence pool was generated at threshold 0.0082 and contains every
    eligible sentence (max score 0.00811). Any threshold <= 0.0082 is therefore
    a pure RE-FILTER of the existing JSONL — NO SemAE rerun, just re-synthesis.
  * HASOS evidence pool is capped at 0.005 (max score 0.00500). Thresholds
    <= 0.005 re-filter; thresholds > 0.005 need a SemAE rerun (``--rerun_semae``).
  * Abstractive re-synthesis is ~1.3 s/task on GPU (flan-t5-base, 300 tasks/run).
  * M1 token sweep needs a SemAE rerun per value (extractive text comes from the
    model, not the JSONL).

MUST be run with the project venv python:
    D:\KHDL\LuanAn\tesing\.venv\Scripts\python.exe scripts/sweep_params.py ...

Examples:
    # Phase A — threshold sweep, abstractive methods, SPACE (no rerun needed)
    .venv/Scripts/python scripts/sweep_params.py \
        --dataset space --phase threshold \
        --grid 0.003,0.005,0.0067,0.0075,0.0082 --methods m2,m3,m4

    # Phase B — abstractive token sweep, HASOS
    .venv/Scripts/python scripts/sweep_params.py \
        --dataset hasos --phase token_abstractive \
        --grid 96,128,192,256 --methods m2,m3,m4
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # must be the project venv python
OUTPUTS = os.path.join(REPO, "outputs")
SWEEP_DIR = os.path.join(REPO, "reports", "sweep")

# ----------------------------------------------------------------------------
# Per-dataset, confirmed-baseline configuration (see memory: hasos-dir-mapping
# -confirmed, rouge-comparison-task). All run_ids / evidence files / model
# choices below reproduce the committed baseline numbers exactly.
# ----------------------------------------------------------------------------
CONFIG = {
    "space": {
        "base_run_id": "space_eval_4method",
        "gold": "data/space/json/space_summ.json",
        "dataset_flag": "space",
        # M1 extractive output dir (already 40-word baseline)
        "m1_dir": "space_eval_4method",
        # stable full-coverage dir defining the (split, entity) universe so a
        # sparse swept output cannot shrink the macro denominator.
        "universe_dir": "outputs/space_eval_4method",
        # evidence JSONL per abstractive method
        "evidence": {
            "m2": "outputs/space_eval_4method_threshold_evidence.jsonl",
            "m3": "outputs/space_eval_4method_kw_threshold_evidence.jsonl",
            "m4": "outputs/space_eval_4method_bert_threshold_evidence.jsonl",
        },
        # pool was generated at this cutoff; thresholds <= this just re-filter
        "pool_threshold": 0.0082,
        "child_model": "google/flan-t5-base",
        "include_general": True,
        # SPACE subdir == gold key (building/...), so M2 scores directly off the
        # flat child dir via --parent_dir. No hierarchical parent pass needed.
        "m2_hierarchical": False,
    },
    "hasos": {
        "base_run_id": "space_hasos_threshold_full",
        "gold": "data/hasos/hasos_summ.json",
        "dataset_flag": "hasos",
        "m1_dir": "space_hasos_threshold_full",
        "universe_dir": "outputs/space_hasos_threshold_full",
        "evidence": {
            "m2": "outputs/space_hasos_threshold_full_threshold_evidence.jsonl",
            "m3": "outputs/space_hasos_threshold_full_threshold_evidence.jsonl",
            "m4": "outputs/space_hasos_threshold_full_bert_threshold_evidence.jsonl",
        },
        "pool_threshold": 0.005,
        "child_model": "google/flan-t5-base",
        "include_general": False,
        # HASOS subdir == sub-aspect code (FAC_BATH/...), NOT the gold parent key.
        # M2 has no sentiment resolver to aggregate codes->parents, so it must run
        # the --hierarchical parent pass that writes a {run}_parent dir holding
        # FACILITY/AMENITY/SERVICE/EXPERIENCE files, scored via --parent_dir.
        # (M3/M4 are fine: their senti_dir resolver aggregates codes via code2group.)
        "m2_hierarchical": True,
    },
}

METHOD_SCORE = {
    "m1": "m1_extractive",
    "m2": "m2_abstractive",
    "m3": "m3_kw",
    "m4": "m4_bert",
}


def tag(val) -> str:
    """Filesystem-safe tag for a parameter value (0.0067 -> 0p0067)."""
    return str(val).replace(".", "p").replace("-", "neg")


def run(cmd, log_path=None):
    """Run a subprocess with the venv env; stream to log if given."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    if log_path:
        with io.open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call(cmd, cwd=REPO, env=env)
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(map(str, cmd))}")


def synthesize(dataset, method, out_run_id, threshold, max_new_tokens,
               entity_max_new_tokens, log_path):
    """Run child-level abstractive synthesis for one (method, value) cell.

    Child level is all the macro-over-aspects metric needs for SPACE (subdir ==
    gold key) and for M3/M4 (the senti_dir resolver aggregates sub-aspect codes
    -> parents via code2group). For HASOS M2 there is no such resolver, so we add
    --hierarchical: that runs the parent pass writing a {out_run_id}_parent dir
    with FACILITY/AMENITY/SERVICE/EXPERIENCE files, which --parent_dir can score.
    """
    cfg = CONFIG[dataset]
    evidence = cfg["evidence"][method]
    cmd = [
        PY, "-u", os.path.join("scripts", "synthesize_aspect_summaries.py"),
        "--run_id", cfg["base_run_id"],
        "--output_run_id", out_run_id,
        "--source_mode", "threshold_evidence",
        "--evidence_jsonl", evidence,
        "--evidence_score_threshold", str(threshold),
        "--model_name", cfg["child_model"],
        "--device", "auto",
        "--max_new_tokens", str(max_new_tokens),
        "--entity_max_new_tokens", str(entity_max_new_tokens),
        "--overwrite",
    ]
    if method in ("m3", "m4"):
        cmd.append("--split_sentiment")
    if method == "m2" and cfg.get("m2_hierarchical"):
        cmd.append("--hierarchical")
    run(cmd, log_path)


def score(dataset, method, value_tag, phase, run_dir=None, parent_dir=None,
          senti_dir=None):
    """Score one cell with the official pyrouge driver; return result dict."""
    cfg = CONFIG[dataset]
    out_json = f"reports/sweep/rouge_{method}_{dataset}_{phase}_{value_tag}.json"
    cmd = [
        PY, "-u", os.path.join("scripts", "score_rouge_compare.py"),
        "--method", METHOD_SCORE[method],
        "--dataset", cfg["dataset_flag"],
        "--gold", cfg["gold"],
        "--out", out_json,
    ]
    if run_dir:
        cmd += ["--run_dir", run_dir]
    if parent_dir:
        cmd += ["--parent_dir", parent_dir]
    if senti_dir:
        cmd += ["--senti_dir", senti_dir]
    # Fixed denominator + stable universe: every (split, entity) that the
    # full-coverage baseline M1 dir answers stays in the denominator, so a
    # sparse swept output is penalised (empties score 0) instead of silently
    # shrinking the macro. discover_entities walks any depth, so the M1 dir
    # works as the universe for all method layouts.
    cmd += ["--fixed_denominator",
            "--universe_dir", cfg["universe_dir"]]
    if dataset == "space" and cfg["include_general"]:
        cmd.append("--include_general")
    run(cmd)
    with io.open(os.path.join(REPO, out_json), encoding="utf-8") as f:
        return json.load(f)


def macro_all(result):
    m = result.get("by_split", {}).get("all", {}).get("MACRO", {})
    return {k: m.get(k) for k in ("rouge1", "rouge2", "rougeL")}


def out_dirs(out_run_id, method, dataset=None):
    """Map a synthesis output_run_id to the dir layout each method scores from."""
    base = os.path.join("outputs", out_run_id)
    if method == "m2":
        # HASOS M2 ran --hierarchical, so the parent-key files (FACILITY/...)
        # live in the sibling {out_run_id}_parent dir, not the flat child dir.
        if dataset and CONFIG[dataset].get("m2_hierarchical"):
            return {"parent_dir": base + "_parent"}
        return {"parent_dir": base}  # SPACE: <aspect>/<split>_<eid>
    # m3/m4 nested <aspect>/<sentiment>/<split>_<eid>
    return {"senti_dir": base}


def phase_threshold(dataset, grid, methods):
    """Phase A: vary evidence_score_threshold (M2/M3/M4 only)."""
    cfg = CONFIG[dataset]
    rows = []
    for method in methods:
        if method == "m1":
            print("[skip] m1 is independent of evidence threshold")
            continue
        for T in grid:
            if T > cfg["pool_threshold"]:
                print(f"[WARN] threshold {T} > pool {cfg['pool_threshold']} "
                      f"for {dataset}; needs SemAE rerun (not automated here). "
                      f"Skipping.")
                continue
            vt = tag(T)
            out_run_id = f"sweep_{dataset}_{method}_thr_{vt}"
            log_path = os.path.join(SWEEP_DIR,
                                    f"_log_{method}_{dataset}_thr_{vt}.log")
            print(f"\n=== [threshold] {dataset} {method} T={T} ===")
            synthesize(dataset, method, out_run_id, threshold=T,
                       max_new_tokens=192, entity_max_new_tokens=128,
                       log_path=log_path)
            dirs = out_dirs(out_run_id, method, dataset)
            res = score(dataset, method, vt, "threshold", **dirs)
            ma = macro_all(res)
            rows.append({"method": method, "value": T, **ma})
            print(f"    -> R1={ma['rouge1']:.5f} R2={ma['rouge2']:.5f} "
                  f"RL={ma['rougeL']:.5f}")
    return rows


def phase_token_abstractive(dataset, grid, methods):
    """Phase B: vary max_new_tokens for M2/M3/M4 (threshold fixed at default)."""
    rows = []
    for method in methods:
        if method == "m1":
            print("[skip] m1 uses --max_tokens; use phase token_extractive")
            continue
        for B in grid:
            vt = tag(B)
            out_run_id = f"sweep_{dataset}_{method}_tok_{vt}"
            log_path = os.path.join(SWEEP_DIR,
                                    f"_log_{method}_{dataset}_tok_{vt}.log")
            print(f"\n=== [token_abstractive] {dataset} {method} B={B} ===")
            synthesize(dataset, method, out_run_id, threshold=0.005,
                       max_new_tokens=B, entity_max_new_tokens=B,
                       log_path=log_path)
            dirs = out_dirs(out_run_id, method, dataset)
            res = score(dataset, method, vt, "tokabs", **dirs)
            ma = macro_all(res)
            rows.append({"method": method, "value": B, **ma})
            print(f"    -> R1={ma['rouge1']:.5f} R2={ma['rouge2']:.5f} "
                  f"RL={ma['rougeL']:.5f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["space", "hasos"], required=True)
    ap.add_argument("--phase", required=True,
                    choices=["threshold", "token_abstractive"])
    ap.add_argument("--grid", required=True,
                    help="comma-separated values for the swept parameter")
    ap.add_argument("--methods", default="m2,m3,m4",
                    help="comma-separated subset of m1,m2,m3,m4")
    ap.add_argument("--out", default=None,
                    help="summary JSON (default reports/sweep/<phase>_<ds>.json)")
    args = ap.parse_args()

    os.makedirs(SWEEP_DIR, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    if args.phase == "threshold":
        grid = [float(x) for x in args.grid.split(",")]
        rows = phase_threshold(args.dataset, grid, methods)
    else:
        grid = [int(x) for x in args.grid.split(",")]
        rows = phase_token_abstractive(args.dataset, grid, methods)

    out = args.out or os.path.join(
        "reports", "sweep", f"{args.phase}_{args.dataset}_summary.json")
    out_path = os.path.join(REPO, out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dataset": args.dataset, "phase": args.phase,
                   "grid": args.grid, "rows": rows}, f, indent=2)
    print(f"\nwritten summary -> {out}")


if __name__ == "__main__":
    main()
