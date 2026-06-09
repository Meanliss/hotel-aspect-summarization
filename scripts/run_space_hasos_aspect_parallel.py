"""Run aspect_inference.py in parallel across entity shards for SPACE-trained
SemAE on HASOS hotel data, with optional sentiment split.

Based on run_aspect_inference_parallel.py with these changes:
  - SentencePiece: space_unigram_32k.model (SPACE vocabulary)
  - Seeds: seeds_hasos (29 HASOS aspects)
  - --sample_sentences enabled (paper recipe)
  - --sentiment_split enabled (keyword-based from taxonomy)
  - --taxonomy path for sentiment keywords

Usage:
    python run_space_hasos_aspect_parallel.py `
        --model ..\models\space_run1_20_model.pt `
        --run_id space_hasos_run1 `
        --num_shards 4 `
        --gpu 0 `
        --sentiment_split
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
LOGS_DIR = REPO_ROOT / "logs"


def build_cmd(args, shard_idx, shard_run_id):
    cmd = [
        sys.executable, "-u", str(SRC_DIR / "aspect_inference.py"),
        "--summary_data", str(DATA_DIR / "hasos" / "hasos_summ.json"),
        "--sentencepiece", str(DATA_DIR / "sentencepiece" / "space_unigram_32k.model"),
        "--seedsdir", str(DATA_DIR / "seeds_hasos"),
        "--gold_aspects", args.gold_aspects,
        "--model", args.model,
        "--run_id", shard_run_id,
        "--gpu", str(args.gpu),
        "--max_tokens", str(args.max_tokens),
        "--shard_idx", str(shard_idx),
        "--num_shards", str(args.num_shards),
        "--sample_sentences",
        "--no_eval",
        "--trace_jsonl", str(LOGS_DIR / f"{shard_run_id}.trace.jsonl"),
        "--trace_sample_limit", str(args.trace_sample_limit),
    ]
    if args.sentiment_split:
        cmd.append("--sentiment_split")
        cmd.extend(["--taxonomy", str(DATA_DIR / "hasos" / "aspect_taxonomy.json")])
    return cmd


def aspect_codes_from_taxonomy(taxonomy_tsv: Path) -> str:
    with taxonomy_tsv.open("r", encoding="utf-8") as fin:
        next(fin)  # header
        codes = []
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > 1 and parts[1].strip():
                codes.append(parts[1].strip())
    return ",".join(codes)


def merge_shards(shard_run_ids, final_run_id, sentiment_split=False):
    """Merge per-shard outputs into a single output folder.

    Handles both the main output tree and the optional sentiment tree.
    """
    for suffix in [None, '_sentiment'] if sentiment_split else [None]:
        if suffix:
            final_dir = OUTPUTS_DIR / (final_run_id + suffix)
        else:
            final_dir = OUTPUTS_DIR / final_run_id

        if final_dir.exists():
            shutil.rmtree(final_dir)
        final_dir.mkdir(parents=True)
        merged = 0

        for shard_run_id in shard_run_ids:
            if suffix:
                shard_dir = OUTPUTS_DIR / (shard_run_id + suffix)
            else:
                shard_dir = OUTPUTS_DIR / shard_run_id

            if not shard_dir.exists():
                continue
            for aspect_dir in shard_dir.iterdir():
                if not aspect_dir.is_dir():
                    continue
                dst_aspect = final_dir / aspect_dir.name
                dst_aspect.mkdir(exist_ok=True)
                for entity_file in aspect_dir.iterdir():
                    shutil.copy2(entity_file, dst_aspect / entity_file.name)
                    merged += 1

        print(f"[merge] copied {merged} files into {final_dir}")
    return OUTPUTS_DIR / final_run_id


def merge_provenance(shard_run_ids, final_run_id):
    """Merge per-shard sentence provenance into a final run-level JSONL."""
    final_path = OUTPUTS_DIR / f"{final_run_id}_provenance.jsonl"
    final_path.parent.mkdir(exist_ok=True)
    merged = 0
    with final_path.open("w", encoding="utf-8") as fout:
        for shard_run_id in shard_run_ids:
            shard_path = LOGS_DIR / f"{shard_run_id}.provenance.jsonl"
            if not shard_path.exists():
                continue
            for line in shard_path.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row["run_id"] = final_run_id
                row["shard_run_id"] = shard_run_id
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                merged += 1
    print(f"[merge] copied {merged} provenance rows into {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(
        description="Parallel aspect inference for SPACE-trained SemAE on HASOS data")
    parser.add_argument("--model", required=True,
                        help="path to trained SemAE model (.pt)")
    parser.add_argument("--run_id", default="space_hasos_run1")
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=40)
    parser.add_argument("--gold_aspects", default=None,
                        help="comma-separated aspect codes; default: taxonomy")
    parser.add_argument("--sentiment_split", action="store_true",
                        help="enable keyword-based sentiment splitting")
    parser.add_argument("--trace_sample_limit", type=int, default=40,
                        help="max trace sample rows emitted per shard")
    parser.add_argument("--keep_shards", action="store_true",
                        help="don't delete per-shard output folders")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = REPO_ROOT / model_path
    args.model = str(model_path.resolve())

    if not args.gold_aspects:
        args.gold_aspects = aspect_codes_from_taxonomy(
            DATA_DIR / "hasos" / "aspect_taxonomy.tsv"
        )

    LOGS_DIR.mkdir(exist_ok=True)
    shard_run_ids = [f"{args.run_id}__shard{i}" for i in range(args.num_shards)]

    processes = []
    log_files = []
    print(f"[main] launching {args.num_shards} shards on GPU {args.gpu}")
    print(f"[main] trace_sample_limit={args.trace_sample_limit}")
    if args.sentiment_split:
        print("[main] sentiment_split ENABLED — will write <run_id>_sentiment/ tree")
    start = time.time()

    for shard_idx, shard_run_id in enumerate(shard_run_ids):
        log_path = LOGS_DIR / f"{shard_run_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        cmd = build_cmd(args, shard_idx, shard_run_id)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        print(f"[main] shard {shard_idx} -> {log_path}")
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
            cwd=str(SRC_DIR), env=env,
        )
        processes.append(proc)
        log_files.append(log_file)

    failed = []
    for shard_idx, proc in enumerate(processes):
        rc = proc.wait()
        log_files[shard_idx].close()
        elapsed = time.time() - start
        status = "OK" if rc == 0 else f"FAIL rc={rc}"
        print(f"[main] shard {shard_idx} done in {elapsed:.0f}s ({status})")
        if rc != 0:
            failed.append(shard_idx)

    if failed:
        print(f"[main] FAILED shards: {failed}; aborting merge")
        sys.exit(2)

    final_dir = merge_shards(shard_run_ids, args.run_id,
                             sentiment_split=args.sentiment_split)
    merge_provenance(shard_run_ids, args.run_id)

    if not args.keep_shards:
        for shard_run_id in shard_run_ids:
            shard_dir = OUTPUTS_DIR / shard_run_id
            if shard_dir.exists():
                shutil.rmtree(shard_dir)
            # Also clean sentiment shard dirs
            sent_shard_dir = OUTPUTS_DIR / (shard_run_id + "_sentiment")
            if sent_shard_dir.exists():
                shutil.rmtree(sent_shard_dir)
        print("[main] removed per-shard folders")

    total = time.time() - start
    print(f"[main] total wall time: {total:.0f}s; final outputs: {final_dir}")


if __name__ == "__main__":
    main()
