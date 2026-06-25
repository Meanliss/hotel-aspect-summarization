#!/usr/bin/env python3
"""Run HASOS threshold cells that require a fresh SemAE pass per T.

The normal sweep harness re-filters an existing evidence pool for values that
are below the pool cutoff. For HASOS thresholds above 0.005, this script runs
the SemAE evidence generation separately for each requested threshold, then
synthesizes and scores M2/M3/M4 from that exact-T evidence.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SWEEP_DIR = REPO / "reports" / "sweep"
OUTPUTS = REPO / "outputs"

MODEL = (
    "checkpoints/space_full_11402x20_stable_resume_e7/"
    "space_full_11402x20_stable_resume_e7_20_model.pt"
)
UNIVERSE_DIR = "outputs/space_hasos_threshold_full"
GOLD = "data/hasos/hasos_summ.json"

METHOD_SCORE = {
    "m2": "m2_abstractive",
    "m3": "m3_kw",
    "m4": "m4_bert",
}


def tag(value: float | int) -> str:
    return str(value).replace(".", "p").replace("-", "neg")


def run(cmd: list[str], log_path: Path | None = None) -> None:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    print("$", " ".join(cmd), flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, cwd=REPO, env=env, stdout=lf,
                                 stderr=subprocess.STDOUT)
    else:
        rc = subprocess.call(cmd, cwd=REPO, env=env)
    if rc != 0:
        raise RuntimeError(f"command failed rc={rc}: {' '.join(cmd)}")


def macro_all(result: dict) -> dict:
    return result.get("by_split", {}).get("all", {}).get("MACRO", {})


def ensure_semae_run(threshold: float, force: bool) -> str:
    vt = tag(threshold)
    run_id = f"space_hasos_threshold_exact_{vt}"
    evidence = OUTPUTS / f"{run_id}_threshold_evidence.jsonl"
    if evidence.exists() and evidence.stat().st_size > 0 and not force:
        print(f"[skip] existing exact evidence: {evidence}", flush=True)
        return run_id

    # Interrupted runs can leave shard folders/logs behind. Clean only this
    # exact run_id namespace before starting a fresh SemAE pass.
    for path in OUTPUTS.glob(f"{run_id}*"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in (REPO / "logs").glob(f"{run_id}*"):
        path.unlink()

    run([
        PY, "scripts/run_space_hasos_aspect_parallel.py",
        "--model", MODEL,
        "--run_id", run_id,
        "--num_shards", "4",
        "--gpu", "0",
        "--max_tokens", "120",
        "--evidence_score_threshold", str(threshold),
        "--sentiment_split",
        "--no_cut_sents",
    ], SWEEP_DIR / f"_log_hasos_exact_semae_{vt}.log")
    return run_id


def ensure_bert_evidence(run_id: str, force: bool) -> Path:
    src = OUTPUTS / f"{run_id}_threshold_evidence.jsonl"
    out = OUTPUTS / f"{run_id}_bert_threshold_evidence.jsonl"
    if out.exists() and out.stat().st_size > 0 and not force:
        print(f"[skip] existing BERT evidence: {out}", flush=True)
        return out
    run([
        PY, "scripts/relabel_evidence_bert.py",
        "--in", str(src.relative_to(REPO)),
        "--out", str(out.relative_to(REPO)),
        "--taxonomy", "data/hasos/aspect_taxonomy.json",
    ], SWEEP_DIR / f"_log_hasos_exact_bert_{run_id}.log")
    return out


def synthesize(method: str, threshold: float, run_id: str,
               evidence_path: Path, force: bool) -> str:
    vt = tag(threshold)
    out_run_id = f"sweep_hasos_{method}_thr_{vt}"
    expected = OUTPUTS / (out_run_id + ("_parent" if method == "m2" else ""))
    if expected.exists() and any(expected.rglob("*")) and not force:
        print(f"[skip] existing synthesis: {expected}", flush=True)
        return out_run_id

    cmd = [
        PY, "-u", "scripts/synthesize_aspect_summaries.py",
        "--run_id", run_id,
        "--output_run_id", out_run_id,
        "--source_mode", "threshold_evidence",
        "--evidence_jsonl", str(evidence_path.relative_to(REPO)),
        "--evidence_score_threshold", str(threshold),
        "--model_name", "google/flan-t5-base",
        "--device", "auto",
        "--max_new_tokens", "192",
        "--entity_max_new_tokens", "128",
        "--overwrite",
    ]
    if method in {"m3", "m4"}:
        cmd.append("--split_sentiment")
    if method == "m2":
        cmd.append("--hierarchical")
        cmd.append("--skip_entity_summary")
    run(cmd, SWEEP_DIR / f"_log_{method}_hasos_exact_thr_{vt}.log")
    return out_run_id


def score(method: str, threshold: float, out_run_id: str) -> dict:
    vt = tag(threshold)
    out_json = SWEEP_DIR / f"rouge_{method}_hasos_threshold_{vt}.json"
    cmd = [
        PY, "-u", "scripts/score_rouge_compare.py",
        "--method", METHOD_SCORE[method],
        "--dataset", "hasos",
        "--gold", GOLD,
        "--out", str(out_json.relative_to(REPO)),
        "--fixed_denominator",
        "--universe_dir", UNIVERSE_DIR,
    ]
    if method == "m2":
        cmd += ["--parent_dir", f"outputs/{out_run_id}_parent"]
    else:
        cmd += ["--senti_dir", f"outputs/{out_run_id}"]
    run(cmd)
    with io.open(out_json, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="0.0067,0.0075,0.0082")
    ap.add_argument("--methods", default="m2,m3,m4")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.grid.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for threshold in thresholds:
        vt = tag(threshold)
        print(f"\n=== HASOS exact threshold T={threshold} ===", flush=True)
        run_id = ensure_semae_run(threshold, args.force)
        base_evidence = OUTPUTS / f"{run_id}_threshold_evidence.jsonl"
        bert_evidence = ensure_bert_evidence(run_id, args.force) if "m4" in methods else None

        for method in methods:
            evidence = bert_evidence if method == "m4" else base_evidence
            assert evidence is not None
            out_run_id = synthesize(method, threshold, run_id, evidence, args.force)
            result = score(method, threshold, out_run_id)
            ma = macro_all(result)
            rows.append({"method": method, "value": threshold, **ma})
            print(
                f"[score] {method} T={threshold} "
                f"R1={ma.get('rouge1'):.5f} R2={ma.get('rouge2'):.5f} "
                f"RL={ma.get('rougeL'):.5f}",
                flush=True,
            )

        summary_path = SWEEP_DIR / f"threshold_hasos_exact_high_{vt}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"dataset": "hasos", "phase": "threshold",
                       "threshold": threshold, "rows": rows}, f, indent=2)

    final_path = SWEEP_DIR / "threshold_hasos_exact_high_summary.json"
    with final_path.open("w", encoding="utf-8") as f:
        json.dump({"dataset": "hasos", "phase": "threshold",
                   "grid": args.grid, "rows": rows}, f, indent=2)
    print(f"\nwritten summary -> {final_path.relative_to(REPO)}", flush=True)


if __name__ == "__main__":
    main()
