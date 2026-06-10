#!/usr/bin/env python3
"""Generate abstractive HASOS aspect summaries from ranked SemAE evidence."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUTPUTS_DIR = REPO_ROOT / "outputs"
TAXONOMY_TSV = REPO_ROOT / "data" / "hasos" / "aspect_taxonomy.tsv"
SPLIT_RE = re.compile(r"\t+|\n+")
PROMPT_ECHO_MARKERS = (
    "Aspect meaning:",
    "Aspect means:",
    "Task:",
    "Evidence:",
    "Use only the evidence",
    "Write only the final summary",
    "Do not add facts",
    "Do not invent facts",
)


@dataclass(frozen=True)
class SummaryTask:
    source_run_id: str
    aspect: str
    split: str
    entity_id: str
    output_path: Path
    evidence: list[str]
    evidence_rows: list[dict]
    source_path: Path | None = None


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def split_summary(text: str) -> list[str]:
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]


def parse_entity_file(path: Path) -> tuple[str, str]:
    if "_" not in path.name:
        return "unknown", path.name
    return path.name.split("_", 1)


def evidence_hash(evidence: list[str]) -> str:
    payload = "\n".join(norm_text(item) for item in evidence)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_taxonomy(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin, delimiter="\t")
        return {
            row["CODE"].strip(): {
                "group": row.get("ASPECT", "").strip(),
                "scale": row.get("MEASUREMENT_SCALE", "").strip(),
                "description": row.get("DESCRIPTION", "").strip(),
            }
            for row in reader
            if row.get("CODE", "").strip()
        }


def load_provenance(run_id: str) -> dict[tuple[str, str, int], dict]:
    path = OUTPUTS_DIR / f"{run_id}_provenance.jsonl"
    rows: dict[tuple[str, str, int], dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        aspect = row.get("aspect")
        entity_id = str(row.get("entity_id", ""))
        try:
            sent_idx = int(row.get("summary_sentence_index"))
        except (TypeError, ValueError):
            continue
        if aspect and entity_id and sent_idx:
            rows[(aspect, entity_id, sent_idx)] = row
    return rows


def iter_extractive_tasks(run_id: str, output_run_id: str) -> Iterable[SummaryTask]:
    run_dir = OUTPUTS_DIR / run_id
    if not run_dir.exists():
        raise SystemExit(f"Missing extractive output dir: {run_dir}")
    output_dir = OUTPUTS_DIR / output_run_id
    provenance = load_provenance(run_id)
    for aspect_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        aspect = aspect_dir.name
        for source_path in sorted(p for p in aspect_dir.iterdir() if p.is_file()):
            split, entity_id = parse_entity_file(source_path)
            evidence = split_summary(
                source_path.read_text(encoding="utf-8", errors="replace"))
            evidence_rows = []
            for idx, sentence in enumerate(evidence, 1):
                prov = provenance.get((aspect, entity_id, idx), {})
                evidence_rows.append({
                    "sentence_index": idx,
                    "sentence": sentence,
                    "rank": prov.get("rank"),
                    "score": prov.get("score"),
                    "sentiment_label": prov.get("sentiment_label"),
                    "source_review_id": prov.get("source_review_id"),
                    "source_sentence_index": prov.get("source_sentence_index"),
                    "matched_aspect_seed": prov.get("matched_aspect_seed", []),
                    "was_truncated": prov.get("was_truncated", False),
                })
            yield SummaryTask(
                source_run_id=run_id,
                aspect=aspect,
                split=split,
                entity_id=entity_id,
                source_path=source_path,
                output_path=output_dir / aspect / source_path.name,
                evidence=evidence,
                evidence_rows=evidence_rows,
            )


def iter_ranked_evidence_tasks(run_id: str, output_run_id: str,
                               evidence_jsonl: Path,
                               evidence_top_k: int) -> Iterable[SummaryTask]:
    if not evidence_jsonl.exists():
        raise SystemExit(f"Missing ranked evidence JSONL: {evidence_jsonl}")
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    with evidence_jsonl.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (row.get("aspect"), row.get("split"),
                   str(row.get("entity_id", "")))
            if all(key):
                grouped[key].append(row)

    output_dir = OUTPUTS_DIR / output_run_id
    for (aspect, split, entity_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda r: (
            r.get("rank") if r.get("rank") is not None else 10**9,
            r.get("score") if r.get("score") is not None else 10**9,
        ))
        selected = rows[:evidence_top_k]
        evidence = [norm_text(row.get("sentence", "")) for row in selected
                    if norm_text(row.get("sentence", ""))]
        file_name = f"{split}_{entity_id}"
        yield SummaryTask(
            source_run_id=run_id,
            aspect=aspect,
            split=split,
            entity_id=entity_id,
            output_path=output_dir / aspect / file_name,
            evidence=evidence,
            evidence_rows=selected,
        )


def build_prompt(task: SummaryTask, taxonomy: dict[str, dict[str, str]],
                 max_input_sentences: int) -> str:
    info = taxonomy.get(task.aspect, {})
    aspect_name = info.get("scale") or task.aspect
    description = info.get("description", "")
    evidence = task.evidence[:max_input_sentences]
    evidence_lines = "\n".join(f"{i}. {sent}" for i, sent in enumerate(evidence, 1))
    desc = f" Aspect definition: {description}" if description else ""
    return (
        f"You are summarizing hotel reviews for one aspect: {aspect_name} "
        f"({task.aspect}).{desc}\n"
        "Use only the evidence below. Write one concise English sentence, or "
        "two sentences only if the evidence contains both positive and negative "
        "opinions. Do not copy the evidence verbatim unless there is only one "
        "clear fact. Do not mention ranks, evidence, instructions, or aspect codes.\n"
        f"Evidence:\n{evidence_lines}\n"
        "Final summary:"
    )


def clean_generation(text: str) -> str:
    text = norm_text(text)
    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1].strip()
    prefixes = ("Final summary:", "final summary:", "Summary:", "summary:")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text.strip(" \"'")


def looks_like_prompt_echo(summary: str) -> bool:
    return any(marker in summary for marker in PROMPT_ECHO_MARKERS)


def output_is_copy(summary: str, evidence: list[str]) -> bool:
    if not summary.strip() or not evidence:
        return False
    normalized_summary = norm_text(summary).lower()
    joined = norm_text(" ".join(evidence)).lower()
    return normalized_summary == joined or normalized_summary in {
        norm_text(sent).lower() for sent in evidence
    }


def fallback_evidence_summary(evidence: list[str], max_sentences: int = 2) -> str:
    return " ".join(norm_text(sent) for sent in evidence[:max_sentences])


def load_cache(path: Path) -> dict[tuple[str, str, str, str, str], dict]:
    cache: dict[tuple[str, str, str, str, str], dict] = {}
    if not path.exists():
        return cache
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (
            row.get("model_name", ""),
            row.get("aspect", ""),
            row.get("split", ""),
            str(row.get("entity_id", "")),
            row.get("evidence_hash", ""),
        )
        if all(key):
            cache[key] = row
    return cache


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fout:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fout:
        fout.write("\t".join(fields) + "\n")
        for row in rows:
            values = []
            for field in fields:
                value = row.get(field, "")
                if value is None:
                    value = ""
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                values.append(str(value).replace("\t", " ").replace("\n", " "))
            fout.write("\t".join(values) + "\n")


def build_transformers_generator(model_name: str, device: str):
    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            pipeline,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers dependencies. Install them with:\n"
            "  uv pip install -r requirements_abstractive.txt"
        ) from exc

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        device_id = -1
        if device != "cpu" and torch.cuda.is_available():
            device_id = 0
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        return {
            "kind": "seq2seq",
            "pipeline": pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                device=device_id,
            ),
        }

    dtype = torch.float32
    target_device = "cpu"
    if device != "cpu" and torch.cuda.is_available():
        dtype = torch.bfloat16
        target_device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(target_device)
    model.eval()
    return {
        "kind": "causal",
        "model": model,
        "tokenizer": tokenizer,
        "device": target_device,
    }


def generate_transformers(generator, prompt: str, max_new_tokens: int) -> str:
    if generator["kind"] == "seq2seq":
        result = generator["pipeline"](
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=4,
            repetition_penalty=1.2,
            truncation=True,
        )
        return clean_generation(result[0]["generated_text"])

    import torch

    tokenizer = generator["tokenizer"]
    model = generator["model"]
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise hotel aspect summaries. Return only the "
                "final summary text."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    inputs = tokenizer(text, return_tensors="pt").to(generator["device"])
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    return clean_generation(tokenizer.decode(generated, skip_special_tokens=True))


async def generate_vllm_one(client, semaphore: asyncio.Semaphore,
                            task: SummaryTask, prompt: str, args) -> str:
    request = {
        "model": args.model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write concise hotel aspect summaries. "
                    "Return only the final summary text."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_new_tokens,
    }
    async with semaphore:
        response = await client.chat.completions.create(**request)
    return clean_generation(response.choices[0].message.content or "")


async def generate_vllm_rows(tasks: list[SummaryTask], taxonomy: dict[str, dict[str, str]],
                             args, rows_by_key: dict[tuple[str, str, str, str], dict],
                             cache_path: Path) -> None:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise SystemExit(
            "Missing openai package for --backend vllm. Install with:\n"
            "  uv pip install openai"
        ) from exc

    client = AsyncOpenAI(base_url=args.vllm_base_url, api_key=args.vllm_api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    pending = []
    pending_tasks = []
    for task in tasks:
        key = task_key(args.model_name, task)
        if key in rows_by_key:
            continue
        prompt = build_prompt(task, taxonomy, args.max_input_sentences)
        pending.append(generate_vllm_one(client, semaphore, task, prompt, args))
        pending_tasks.append(task)

    for offset in range(0, len(pending), args.concurrency):
        chunk = pending[offset:offset + args.concurrency]
        chunk_tasks = pending_tasks[offset:offset + args.concurrency]
        results = await asyncio.gather(*chunk, return_exceptions=True)
        for task, result in zip(chunk_tasks, results):
            if isinstance(result, Exception):
                summary = fallback_evidence_summary(
                    task.evidence[:args.max_input_sentences])
                status = f"fallback_error:{type(result).__name__}"
            else:
                summary = result
                status = "generated"
                if not summary.strip():
                    summary = fallback_evidence_summary(
                        task.evidence[:args.max_input_sentences])
                    status = "fallback_empty"
                elif looks_like_prompt_echo(summary):
                    summary = fallback_evidence_summary(
                        task.evidence[:args.max_input_sentences])
                    status = "fallback_prompt_echo"
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            task.output_path.write_text(summary, encoding="utf-8")
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, status)
            rows_by_key[task_key(args.model_name, task)] = row
            append_jsonl(cache_path, row)
        logging.info("Processed %s/%s generated vLLM tasks",
                     min(offset + args.concurrency, len(pending)),
                     len(pending))


def task_key(model_name: str, task: SummaryTask) -> tuple[str, str, str, str, str]:
    return (
        model_name,
        task.aspect,
        task.split,
        task.entity_id,
        evidence_hash(task.evidence),
    )


def row_for_task(task: SummaryTask, output_run_id: str, model_name: str,
                 max_input_sentences: int, max_new_tokens: int, summary: str,
                 status: str) -> dict:
    used_evidence = task.evidence[:max_input_sentences]
    return {
        "run_id": output_run_id,
        "source_run_id": task.source_run_id,
        "model_name": model_name,
        "aspect": task.aspect,
        "split": task.split,
        "entity_id": task.entity_id,
        "summary": summary,
        "evidence_count": len(task.evidence),
        "evidence_used": len(used_evidence),
        "evidence_hash": evidence_hash(task.evidence),
        "max_input_sentences": max_input_sentences,
        "max_new_tokens": max_new_tokens,
        "copied_from_evidence": output_is_copy(summary, used_evidence),
        "status": status,
        "output_path": str(task.output_path),
        "source_path": str(task.source_path or ""),
        "evidence": task.evidence_rows[:max_input_sentences],
    }


def write_report(path: Path, rows: list[dict], output_run_id: str,
                 source_run_id: str, model_name: str,
                 source_mode: str) -> None:
    total = len(rows)
    empty = sum(1 for row in rows if row["status"] == "empty_evidence")
    generated = sum(1 for row in rows if row["status"] == "generated")
    cached = sum(1 for row in rows if row["status"].startswith("cached"))
    fallback = sum(1 for row in rows if row["status"].startswith("fallback"))
    copied = sum(1 for row in rows if row.get("copied_from_evidence"))
    aspects = sorted({row["aspect"] for row in rows})
    lines = [
        f"# Abstractive Aspect Synthesis Report - `{output_run_id}`",
        "",
        f"- Source run: `{source_run_id}`",
        f"- Source mode: `{source_mode}`",
        f"- Model: `{model_name}`",
        f"- Aspect/entity files processed: {total}",
        f"- Generated this run: {generated}",
        f"- Loaded from cache/existing output: {cached}",
        f"- Empty evidence files: {empty}",
        f"- Fallback rows: {fallback}",
        f"- Exact copy detections: {copied}",
        f"- Aspects: {len(aspects)}",
        "",
        "## Notes",
        "",
        "This stage rewrites top-ranked SemAE evidence into short abstractive "
        "summaries. In `ranked_evidence` mode, the evidence is captured before "
        "extractive summary truncation.",
        "",
        "Each row keeps the evidence records used for synthesis so the generated "
        "summary can be audited back to ranked SemAE sentences.",
        "",
        "## Samples",
        "",
    ]
    for row in rows[:20]:
        summary = row.get("summary", "")
        if len(summary) > 220:
            summary = summary[:217] + "..."
        lines.append(
            f"- `{row['aspect']}` `{row['split']}_{row['entity_id']}`: {summary}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_rows_from_cache(tasks: list[SummaryTask], cache: dict,
                            args) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    rows_by_key = {}
    for task in tasks:
        task.output_path.parent.mkdir(parents=True, exist_ok=True)
        key = task_key(args.model_name, task)
        if not task.evidence:
            summary = ""
            task.output_path.write_text("", encoding="utf-8")
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, "empty_evidence")
            rows.append(row)
            rows_by_key[key] = row
            continue
        if not args.overwrite and task.output_path.exists():
            summary = task.output_path.read_text(
                encoding="utf-8", errors="replace").strip()
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, "cached_existing_output")
            rows.append(row)
            rows_by_key[key] = row
            continue
        if key in cache:
            summary = cache[key].get("summary", "")
            task.output_path.write_text(summary, encoding="utf-8")
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, "cached_jsonl")
            rows.append(row)
            rows_by_key[key] = row
    return rows, rows_by_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abstractive synthesis for SemAE HASOS aspect evidence.")
    parser.add_argument("--run_id", default="space_hasos_full_e20")
    parser.add_argument("--output_run_id", default=None)
    parser.add_argument("--source_mode",
                        choices=["ranked_evidence", "extractive_tree"],
                        default="ranked_evidence")
    parser.add_argument("--evidence_jsonl", default=None)
    parser.add_argument("--evidence_top_k", type=int, default=5)
    parser.add_argument("--backend", choices=["vllm", "transformers"],
                        default="transformers")
    parser.add_argument("--model_name", default="google/flan-t5-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_input_sentences", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s",
    )

    if not args.output_run_id:
        if args.source_mode == "ranked_evidence":
            args.output_run_id = f"{args.run_id}_abstractive_ranked"
        else:
            args.output_run_id = f"{args.run_id}_abstractive"
    taxonomy = load_taxonomy(TAXONOMY_TSV)
    if args.source_mode == "ranked_evidence":
        evidence_jsonl = Path(
            args.evidence_jsonl
            or OUTPUTS_DIR / f"{args.run_id}_ranked_evidence.jsonl")
        tasks = list(iter_ranked_evidence_tasks(
            args.run_id, args.output_run_id, evidence_jsonl,
            args.evidence_top_k))
    else:
        tasks = list(iter_extractive_tasks(args.run_id, args.output_run_id))
    if args.limit is not None:
        tasks = tasks[:args.limit]

    cache_path = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_cache.jsonl"
    lines_jsonl = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_lines.jsonl"
    lines_tsv = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_lines.tsv"
    report_path = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_report.md"
    cache = {} if args.overwrite else load_cache(cache_path)
    if args.overwrite:
        cache_path.write_text("", encoding="utf-8")

    rows, rows_by_key = prepare_rows_from_cache(tasks, cache, args)
    for row in rows:
        if row["status"] == "empty_evidence":
            append_jsonl(cache_path, row)

    pending_tasks = [
        task for task in tasks
        if task.evidence and task_key(args.model_name, task) not in rows_by_key
    ]
    if pending_tasks and args.backend == "vllm":
        asyncio.run(generate_vllm_rows(
            pending_tasks, taxonomy, args, rows_by_key, cache_path))
    elif pending_tasks:
        logging.info("Loading generation model: %s", args.model_name)
        generator = build_transformers_generator(args.model_name, args.device)
        for index, task in enumerate(pending_tasks, 1):
            prompt = build_prompt(task, taxonomy, args.max_input_sentences)
            summary = generate_transformers(generator, prompt,
                                            args.max_new_tokens)
            status = "generated"
            if not summary.strip():
                summary = fallback_evidence_summary(
                    task.evidence[:args.max_input_sentences])
                status = "fallback_empty"
            elif looks_like_prompt_echo(summary):
                summary = fallback_evidence_summary(
                    task.evidence[:args.max_input_sentences])
                status = "fallback_prompt_echo"
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            task.output_path.write_text(summary, encoding="utf-8")
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, status)
            rows_by_key[task_key(args.model_name, task)] = row
            append_jsonl(cache_path, row)
            if index % 25 == 0:
                logging.info("Processed %s/%s files", index,
                             len(pending_tasks))

    rows = [rows_by_key[task_key(args.model_name, task)] for task in tasks
            if task_key(args.model_name, task) in rows_by_key]
    write_jsonl(lines_jsonl, rows)
    write_tsv(lines_tsv, rows, [
        "run_id", "source_run_id", "model_name", "aspect", "split",
        "entity_id", "summary", "evidence_count", "evidence_used",
        "copied_from_evidence", "status", "output_path", "source_path",
    ])
    write_report(report_path, rows, args.output_run_id, args.run_id,
                 args.model_name, args.source_mode)

    print(f"processed={len(rows)}")
    print(f"output_dir={OUTPUTS_DIR / args.output_run_id}")
    print(f"cache={cache_path}")
    print(f"jsonl={lines_jsonl}")
    print(f"tsv={lines_tsv}")
    print(f"report={report_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
