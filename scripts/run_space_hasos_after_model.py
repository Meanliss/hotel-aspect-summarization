#!/usr/bin/env python3
"""Run the post-training SPACE->HASOS pipeline with hard artifact gates."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "data"
LOGS_DIR = REPO_ROOT / "logs"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Trace:
    def __init__(self, run_id: str):
        LOGS_DIR.mkdir(exist_ok=True)
        self.run_id = run_id
        self.jsonl_path = LOGS_DIR / f"{run_id}_pipeline_trace.jsonl"
        self.md_path = LOGS_DIR / f"{run_id}_pipeline_trace.md"
        self.rows: list[dict] = []
        self.jsonl_path.write_text("", encoding="utf-8")
        self.md_path.write_text("", encoding="utf-8")

    def emit(self, stage: str, status: str = "ok", **payload) -> None:
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "run_id": self.run_id,
            "stage": stage,
            "status": status,
        }
        row.update(payload)
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as fout:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.write_md()
        msg = f"[{status}] {stage}"
        if payload:
            msg += " " + " ".join(f"{k}={v}" for k, v in payload.items() if k != "stdout")
        print(msg, flush=True)

    def write_md(self) -> None:
        lines = [
            f"# SPACE SemAE -> HASOS pipeline trace: `{self.run_id}`",
            "",
            "| Time | Stage | Status | Details |",
            "| --- | --- | --- | --- |",
        ]
        for row in self.rows:
            details = ", ".join(
                f"`{k}`={v}" for k, v in row.items()
                if k not in {"time", "run_id", "stage", "status", "stdout"}
            )
            lines.append(
                f"| {row['time']} | `{row['stage']}` | **{row['status']}** | {details} |"
            )
        self.md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_cmd(trace: Trace, stage: str, cmd: list[str], *, allow_fail: bool = False) -> int:
    start = time.time()
    trace.emit(stage, "start", command=" ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - start
    log_path = LOGS_DIR / f"{trace.run_id}_{stage}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    status = "ok" if proc.returncode == 0 else "fail"
    trace.emit(stage, status, returncode=proc.returncode, seconds=round(elapsed, 2), log=str(log_path))
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(proc.returncode)
    return proc.returncode


def check_file(trace: Trace, label: str, path: Path, *, required: bool = True) -> bool:
    exists = path.exists()
    payload = {"path": str(path), "exists": exists}
    if exists and path.is_file():
        payload.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    status = "ok" if exists else ("fail" if required else "warn")
    trace.emit(f"artifact:{label}", status, **payload)
    return exists


def missing_abstractive_deps() -> list[str]:
    required = ["transformers", "sentencepiece", "accelerate"]
    return [
        name for name in required
        if importlib.util.find_spec(name) is None
    ]


def uv_command(trace: Trace) -> list[str]:
    uv_path = shutil.which("uv")
    if uv_path:
        return [uv_path]
    run_cmd(trace, "install_uv", [
        sys.executable, "-m", "pip", "install", "uv",
    ])
    return [sys.executable, "-m", "uv"]


def ensure_abstractive_requirements(trace: Trace) -> None:
    missing = missing_abstractive_deps()
    if not missing:
        trace.emit("abstractive_dependencies", "ok", missing="")
        return
    trace.emit("abstractive_dependencies", "warn",
               missing=",".join(missing))
    run_cmd(trace, "install_abstractive_requirements", [
        *uv_command(trace),
        "pip",
        "install",
        "-r",
        str(REPO_ROOT / "requirements_abstractive.txt"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="space_hasos_2k_e10")
    parser.add_argument(
        "--source_json",
        default=r"C:\Users\Windows\Downloads\space_summ_hasos.json",
    )
    parser.add_argument(
        "--model",
        default=str(REPO_ROOT / "models" / "space_2k_C_8" / "space_2k_C_8_10_model.pt"),
    )
    parser.add_argument(
        "--sentencepiece",
        default=str(DATA_DIR / "sentencepiece" / "space_unigram_32k.model"),
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--no_cut_sents", action="store_true",
                        default=True,
                        help="do not hard-cut the final extractive sentence")
    parser.add_argument("--cut_sents", dest="no_cut_sents",
                        action="store_false",
                        help="allow legacy hard-cut summary truncation")
    parser.add_argument("--no_early_stop", action="store_true",
                        help="continue selecting after an over-budget sentence")
    parser.add_argument("--evidence_top_k", type=int, default=5,
                        help="top full ranked evidence rows to emit per entity/aspect")
    parser.add_argument("--skip_bert_score", action="store_true")
    parser.add_argument("--abstractive", action="store_true",
                        help="run abstractive synthesis after extractive summarization")
    parser.add_argument("--abstractive_model", default="google/flan-t5-base")
    parser.add_argument("--abstractive_limit", type=int, default=None,
                        help="optional smoke-test limit for abstractive synthesis")
    parser.add_argument("--abstractive_max_input_sentences", type=int, default=5)
    parser.add_argument("--abstractive_max_new_tokens", type=int, default=192)
    parser.add_argument("--abstractive_backend",
                        choices=["vllm", "transformers"],
                        default="transformers")
    parser.add_argument("--abstractive_concurrency", type=int, default=16)
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    args = parser.parse_args()

    trace = Trace(args.run_id)
    trace.emit(
        "pipeline_start",
        python=sys.executable,
        cwd=str(REPO_ROOT),
        model=args.model,
        sentencepiece=args.sentencepiece,
        num_shards=args.num_shards,
    )

    source_json = Path(args.source_json)
    model = Path(args.model)
    sentencepiece = Path(args.sentencepiece)
    hasos_json = DATA_DIR / "hasos" / "hasos_summ.json"
    taxonomy_tsv = DATA_DIR / "hasos" / "aspect_taxonomy.tsv"
    taxonomy_json = DATA_DIR / "hasos" / "aspect_taxonomy.json"
    seeds_dir = DATA_DIR / "seeds_hasos"

    check_file(trace, "source_hasos_json", source_json)
    check_file(trace, "model_checkpoint", model)
    check_file(trace, "taxonomy_tsv", taxonomy_tsv)

    run_cmd(trace, "prepare_hasos", [
        sys.executable, str(SCRIPT_DIR / "prepare_hasos.py"),
        "--input-json", str(source_json),
        "--output-json", str(hasos_json),
        "--taxonomy-json", str(taxonomy_json),
        "--seeds-dir", str(seeds_dir),
    ])
    run_cmd(trace, "validate_hasos", [
        sys.executable, str(SCRIPT_DIR / "validate_hasos.py"),
        "--data", str(hasos_json),
        "--taxonomy", str(taxonomy_json),
        "--seeds-dir", str(seeds_dir),
    ])

    tokenizer_ok = check_file(trace, "space_sentencepiece_exact_gate", sentencepiece)
    if not tokenizer_ok:
        trace.emit(
            "blocked",
            "fail",
            reason=(
                "Missing data/sentencepiece/space_unigram_32k.model. "
                "Checkpoint cannot be used safely without the matching tokenizer."
            ),
        )
        run_cmd(trace, "build_pptx", [
            sys.executable, str(SCRIPT_DIR / "build_space_hasos_report_pptx.py"),
            "--run_id", args.run_id,
        ], allow_fail=True)
        raise SystemExit(3)

    inference_cmd = [
        sys.executable, str(SCRIPT_DIR / "run_space_hasos_aspect_parallel.py"),
        "--model", str(model),
        "--run_id", args.run_id,
        "--num_shards", str(args.num_shards),
        "--gpu", str(args.gpu),
        "--max_tokens", str(args.max_tokens),
        "--evidence_top_k", str(args.evidence_top_k),
        "--sentiment_split",
    ]
    if args.no_cut_sents:
        inference_cmd.append("--no_cut_sents")
    if args.no_early_stop:
        inference_cmd.append("--no_early_stop")
    run_cmd(trace, "inference", inference_cmd)
    run_cmd(trace, "export_lines", [
        sys.executable, str(SCRIPT_DIR / "export_space_hasos_lines.py"),
        "--run_id", args.run_id,
    ])
    run_cmd(trace, "summarize_outputs", [
        sys.executable, str(SCRIPT_DIR / "summarize_aspect_outputs.py"),
        "--run_id", args.run_id,
    ])

    abstractive_run_id = f"{args.run_id}_abstractive_ranked"
    if args.abstractive:
        ensure_abstractive_requirements(trace)
        synthesis_cmd = [
            sys.executable, str(SCRIPT_DIR / "synthesize_aspect_summaries.py"),
            "--run_id", args.run_id,
            "--output_run_id", abstractive_run_id,
            "--source_mode", "ranked_evidence",
            "--evidence_jsonl",
            str(OUTPUTS_DIR / f"{args.run_id}_ranked_evidence.jsonl"),
            "--evidence_top_k", str(args.evidence_top_k),
            "--backend", args.abstractive_backend,
            "--model_name", args.abstractive_model,
            "--max_input_sentences",
            str(args.abstractive_max_input_sentences),
            "--max_new_tokens", str(args.abstractive_max_new_tokens),
            "--concurrency", str(args.abstractive_concurrency),
        ]
        if args.abstractive_backend == "vllm":
            synthesis_cmd.extend([
                "--vllm_base_url", args.vllm_base_url,
                "--vllm_api_key", args.vllm_api_key,
            ])
        if args.abstractive_limit is not None:
            synthesis_cmd.extend(["--limit", str(args.abstractive_limit)])
        run_cmd(trace, "synthesize_abstractive", synthesis_cmd)
        run_cmd(trace, "export_abstractive_lines", [
            sys.executable, str(SCRIPT_DIR / "export_space_hasos_lines.py"),
            "--run_id", abstractive_run_id,
        ])
        run_cmd(trace, "summarize_abstractive_outputs", [
            sys.executable, str(SCRIPT_DIR / "summarize_aspect_outputs.py"),
            "--run_id", abstractive_run_id,
        ])

    score_cmd = [
        sys.executable, str(SCRIPT_DIR / "score_semae_run.py"),
        "--run_id", args.run_id,
    ]
    if not args.skip_bert_score:
        score_cmd.append("--bert_score")
    run_cmd(trace, "score_outputs", score_cmd)

    run_cmd(trace, "build_pptx", [
        sys.executable, str(SCRIPT_DIR / "build_space_hasos_report_pptx.py"),
        "--run_id", args.run_id,
    ])
    trace.emit("pipeline_complete", "ok", outputs=str(OUTPUTS_DIR / args.run_id))


if __name__ == "__main__":
    main()
