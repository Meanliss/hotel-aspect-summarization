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
from difflib import SequenceMatcher
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:  # pragma: no cover - fallback for minimal smoke environments
    TfidfVectorizer = None
    cosine_similarity = None


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
    level: str = "child"
    parent_aspect: str = ""
    sentiment: str = ""


def norm_text(text: str) -> str:
    return " ".join(str(text).split())


def norm_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", norm_text(text).lower()).strip()


def split_summary(text: str) -> list[str]:
    return [s.strip() for s in SPLIT_RE.split(text) if s.strip()]


def parse_entity_file(path: Path) -> tuple[str, str]:
    if "_" not in path.name:
        return "unknown", path.name
    return path.name.split("_", 1)


def evidence_hash(evidence: list[str]) -> str:
    payload = "\n".join(norm_text(item) for item in evidence)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def lexical_similarity(left: str, right: str) -> float:
    left_key = norm_key(left)
    right_key = norm_key(right)
    string_ratio = SequenceMatcher(None, left_key, right_key).ratio()
    left_tokens = set(left_key.split())
    right_tokens = set(right_key.split())
    overlap = (
        len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    )
    return max(string_ratio, overlap)


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


def taxonomy_parent_order(taxonomy: dict[str, dict[str, str]]) -> list[str]:
    seen = []
    for info in taxonomy.values():
        parent = info.get("group", "").strip()
        if parent and parent not in seen:
            seen.append(parent)
    return seen


def dedupe_evidence_rows(rows: list[dict], cosine_threshold: float) -> list[dict]:
    """Keep the best-scored representative for exact and near-duplicate evidence."""
    sorted_rows = sorted(rows, key=lambda r: (
        r.get("score") if r.get("score") is not None else 10**9,
        r.get("rank") if r.get("rank") is not None else 10**9,
    ))
    exact: dict[str, dict] = {}
    for row in sorted_rows:
        sentence = norm_text(row.get("sentence", ""))
        if not sentence:
            continue
        key = norm_key(sentence)
        if key and key not in exact:
            normalized = dict(row)
            normalized["sentence"] = sentence
            exact[key] = normalized
    representatives = list(exact.values())
    if len(representatives) < 2:
        return representatives

    sim = None
    if TfidfVectorizer is not None and cosine_similarity is not None:
        try:
            matrix = TfidfVectorizer(stop_words="english").fit_transform(
                [row["sentence"] for row in representatives])
            sim = cosine_similarity(matrix)
        except Exception:
            sim = None

    kept = []
    assigned = set()
    for idx, row in enumerate(representatives):
        if idx in assigned:
            continue
        kept.append(row)
        assigned.add(idx)
        for other_idx in range(idx + 1, len(representatives)):
            if other_idx in assigned:
                continue
            cosine_score = sim[idx, other_idx] if sim is not None else 0.0
            if (cosine_score >= cosine_threshold
                    or lexical_similarity(row["sentence"], representatives[other_idx]["sentence"]) >= cosine_threshold):
                assigned.add(other_idx)
    return kept


def evidence_limit(evidence: list[str], max_input_sentences: int) -> list[str]:
    if max_input_sentences and max_input_sentences > 0:
        return evidence[:max_input_sentences]
    return evidence


def estimate_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text.split()))


def chunk_evidence(evidence: list[str], max_source_tokens: int, tokenizer=None) -> list[list[str]]:
    if max_source_tokens <= 0:
        return [evidence]
    chunks: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for item in evidence:
        item_tokens = estimate_tokens(item, tokenizer)
        if current and current_tokens + item_tokens > max_source_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append(item)
        current_tokens += item_tokens
    if current:
        chunks.append(current)
    return chunks or [[]]


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
                               evidence_score_threshold: float | None,
                               dedupe_cosine_threshold: float,
                               split_sentiment: bool = False) -> Iterable[SummaryTask]:
    if not evidence_jsonl.exists():
        raise SystemExit(f"Missing ranked evidence JSONL: {evidence_jsonl}")
    # When split_sentiment is on we group by (aspect, sentiment, split, entity)
    # so each polarity (positive / negative) gets its own summary task. The
    # neutral bucket is skipped for highlighted summaries — it rarely carries an
    # opinion worth surfacing and would dilute the pos/neg contrast.
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    with evidence_jsonl.open("r", encoding="utf-8", errors="replace") as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if split_sentiment:
                sentiment = row.get("sentiment_label") or "neu"
                if sentiment not in ("pos", "neg"):
                    continue
                key = (row.get("aspect"), sentiment, row.get("split"),
                       str(row.get("entity_id", "")))
                if row.get("aspect") and row.get("split") and key[3]:
                    grouped[key].append(row)
            else:
                key = (row.get("aspect"), row.get("split"),
                       str(row.get("entity_id", "")))
                if all(key):
                    grouped[key].append(row)

    output_dir = OUTPUTS_DIR / output_run_id
    for group_key, rows in sorted(grouped.items()):
        if split_sentiment:
            aspect, sentiment, split, entity_id = group_key
        else:
            aspect, split, entity_id = group_key
            sentiment = ""
        selected = []
        for row in rows:
            score = row.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = None
            if evidence_score_threshold is None or (
                    score_value is not None
                    and score_value <= evidence_score_threshold):
                normalized = dict(row)
                normalized["score"] = score_value
                selected.append(normalized)
        selected.sort(key=lambda r: (
            r.get("score") if r.get("score") is not None else 10**9,
            r.get("rank") if r.get("rank") is not None else 10**9,
        ))
        selected = dedupe_evidence_rows(selected, dedupe_cosine_threshold)
        evidence = [norm_text(row.get("sentence", "")) for row in selected
                    if norm_text(row.get("sentence", ""))]
        file_name = f"{split}_{entity_id}"
        if split_sentiment:
            output_path = output_dir / aspect / sentiment / file_name
        else:
            output_path = output_dir / aspect / file_name
        yield SummaryTask(
            source_run_id=run_id,
            aspect=aspect,
            split=split,
            entity_id=entity_id,
            output_path=output_path,
            evidence=evidence,
            evidence_rows=selected,
            sentiment=sentiment,
        )


def build_prompt(task: SummaryTask, taxonomy: dict[str, dict[str, str]],
                 max_input_sentences: int) -> str:
    evidence = evidence_limit(task.evidence, max_input_sentences)
    evidence_lines = "\n".join(f"{i}. {sent}" for i, sent in enumerate(evidence, 1))
    if task.level == "parent":
        return (
            f"You are summarizing hotel review aspect summaries for parent aspect "
            f"{task.aspect}.\n"
            "Use only the child aspect summaries below. Preserve concrete details, "
            "merge repeated ideas, and write a detailed 3-5 sentence English "
            "paragraph. Mention the main positive and negative signals when both "
            "appear. Do not use generic filler such as 'good hotel' unless the "
            "child summaries say only that. Do not mention evidence numbers or "
            "aspect codes.\n"
            f"Child aspect summaries:\n{evidence_lines}\n"
            "Final summary:"
        )
    if task.level == "entity":
        return (
            "You are writing the overall SPACE hotel review summary for one entity.\n"
            "Use only the parent/aspect summaries below. Write a compact 5-6 sentence "
            "English paragraph that covers the six hotel dimensions when evidence "
            "exists: building/facilities, cleanliness, food, location, rooms, and "
            "service. Preserve concrete details and tradeoffs, including both praise "
            "and complaints. Avoid bland output like 'it is a good hotel' unless no "
            "better detail exists. Do not mention evidence numbers or aspect codes.\n"
            f"Parent/aspect summaries:\n{evidence_lines}\n"
            "Final summary:"
        )

    info = taxonomy.get(task.aspect, {})
    aspect_name = info.get("scale") or task.aspect
    description = info.get("description", "")
    desc = f" Aspect definition: {description}" if description else ""
    if task.sentiment == "pos":
        polarity = (
            " Every sentence below is a POSITIVE opinion about this aspect. "
            "Write only the positive summary (what guests liked and praised)."
        )
    elif task.sentiment == "neg":
        polarity = (
            " Every sentence below is a NEGATIVE opinion about this aspect. "
            "Write only the negative summary (complaints and problems guests "
            "reported)."
        )
    else:
        polarity = ""
    return (
        f"You are summarizing hotel reviews for one aspect: {aspect_name} "
        f"({task.aspect}).{desc}{polarity}\n"
        "Use only the evidence below. Write a detailed 2-4 sentence English "
        "summary that preserves concrete facts, amenities, complaints, and "
        "tradeoffs from the reviews. Merge repeated ideas instead of listing "
        "duplicates. Do not collapse the answer into generic wording such as "
        "'good hotel' or 'bad hotel'. Do not mention ranks, evidence, instructions, "
        "or aspect codes.\n"
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


def looks_too_generic(summary: str) -> bool:
    normalized = norm_text(summary).lower()
    if len(normalized.split()) < 10:
        return True
    generic_phrases = (
        "good hotel",
        "bad hotel",
        "not a bad hotel",
        "great place to stay",
        "good place to stay",
        "nice hotel",
    )
    return any(phrase in normalized for phrase in generic_phrases)


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


# Parent/entity evidence is built as "<aspect> (<polarity>): <summary>" strings
# so the generator can attribute each sentence. When we fall back to the raw
# evidence we strip that bookkeeping prefix so the surfaced summary reads like
# prose instead of leaking aspect codes. The labels can appear anywhere once
# several evidence strings are joined ("... friendly, cleanliness: The room"),
# so the pattern matches at the start of the text or after whitespace/comma/
# sentence punctuation, not only the leading position. Mirrors the standalone
# scripts/clean_overall_summaries.py cleaner.
_SPACE_ASPECT_ALT = "building|cleanliness|food|location|rooms|service"
_LEVEL_PREFIX_RE = re.compile(
    r"(?:^|(?<=[\s,.!?]))"
    r"(?:\d+\.\s*)?"
    r"(?:" + _SPACE_ASPECT_ALT + r")"
    r"\s*(?:\((?:positive|negative)\))?\s*:\s*",
    re.IGNORECASE,
)
# Orphan enumeration left behind after a label is removed ("... city. 2. The").
_ORPHAN_NUM_RE = re.compile(r"(?:^|(?<=\s))\d+\.\s+")


def strip_level_prefix(sentence: str) -> str:
    out = _LEVEL_PREFIX_RE.sub(" ", norm_text(sentence))
    out = _ORPHAN_NUM_RE.sub(" ", out)
    return norm_text(out)


def fallback_detailed_summary(task: SummaryTask, max_input_sentences: int) -> str:
    evidence = evidence_limit(task.evidence, max_input_sentences)
    if task.level in ("entity", "parent"):
        cleaned = [strip_level_prefix(sent) for sent in evidence]
        max_sentences = 6 if task.level == "entity" else 5
        return fallback_evidence_summary(cleaned, max_sentences=max_sentences)
    return fallback_evidence_summary(evidence, max_sentences=4)


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
            row.get("sentiment", "") or "all",
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
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing transformers dependencies. Install them with:\n"
            "  uv pip install -r requirements_abstractive.txt"
        ) from exc

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(config, "is_encoder_decoder", False):
        target_device = "cpu"
        if device != "cpu" and torch.cuda.is_available():
            target_device = "cuda"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        model.to(target_device)
        model.eval()
        return {
            "kind": "seq2seq",
            "model": model,
            "tokenizer": tokenizer,
            "device": target_device,
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
        import torch

        tokenizer = generator["tokenizer"]
        model = generator["model"]
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(generator["device"])
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                no_repeat_ngram_size=3,
                repetition_penalty=1.08,
                length_penalty=1.15,
            )
        return clean_generation(tokenizer.decode(output[0], skip_special_tokens=True))

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


def generator_tokenizer(generator):
    return generator.get("tokenizer")


def task_with_evidence(task: SummaryTask, evidence: list[str]) -> SummaryTask:
    return SummaryTask(
        source_run_id=task.source_run_id,
        aspect=task.aspect,
        split=task.split,
        entity_id=task.entity_id,
        output_path=task.output_path,
        evidence=evidence,
        evidence_rows=task.evidence_rows,
        source_path=task.source_path,
        level=task.level,
        parent_aspect=task.parent_aspect,
        sentiment=task.sentiment,
    )


def generate_transformers_task(generator, task: SummaryTask,
                               taxonomy: dict[str, dict[str, str]], args) -> str:
    tokenizer = generator_tokenizer(generator)
    evidence = evidence_limit(task.evidence, args.max_input_sentences)
    chunks = chunk_evidence(evidence, args.max_source_tokens, tokenizer)
    if len(chunks) <= 1:
        prompt = build_prompt(task_with_evidence(task, evidence), taxonomy, 0)
        return generate_transformers(generator, prompt, args.max_new_tokens)

    partials = []
    for chunk in chunks:
        prompt = build_prompt(task_with_evidence(task, chunk), taxonomy, 0)
        partial = generate_transformers(generator, prompt, args.max_new_tokens)
        if partial.strip():
            partials.append(partial)
    merge_task = task_with_evidence(task, partials)
    prompt = build_prompt(merge_task, taxonomy, 0)
    return generate_transformers(generator, prompt, args.max_new_tokens)


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
                    evidence_limit(task.evidence, args.max_input_sentences))
                status = f"fallback_error:{type(result).__name__}"
            else:
                summary = result
                status = "generated"
                if not summary.strip():
                    summary = fallback_evidence_summary(
                        evidence_limit(task.evidence, args.max_input_sentences))
                    status = "fallback_empty"
                elif looks_like_prompt_echo(summary):
                    summary = fallback_evidence_summary(
                        evidence_limit(task.evidence, args.max_input_sentences))
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


def task_key(model_name: str, task: SummaryTask) -> tuple[str, str, str, str, str, str]:
    return (
        model_name,
        task.aspect,
        task.sentiment,
        task.split,
        task.entity_id,
        evidence_hash(task.evidence),
    )


def row_for_task(task: SummaryTask, output_run_id: str, model_name: str,
                 max_input_sentences: int, max_new_tokens: int, summary: str,
                 status: str) -> dict:
    used_evidence = evidence_limit(task.evidence, max_input_sentences)
    return {
        "run_id": output_run_id,
        "source_run_id": task.source_run_id,
        "model_name": model_name,
        "level": task.level,
        "aspect": task.aspect,
        "sentiment": task.sentiment,
        "parent_aspect": task.parent_aspect,
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
        "selection_mode": "score_threshold" if task.level == "child" else "hierarchical_summary",
        "evidence": task.evidence_rows[:len(used_evidence)],
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
            status = "cached_existing_output"
            if looks_too_generic(summary):
                summary = fallback_detailed_summary(task, args.max_input_sentences)
                task.output_path.write_text(summary, encoding="utf-8")
                status = "fallback_generic_cached_output"
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, status)
            rows.append(row)
            rows_by_key[key] = row
            continue
        if key in cache:
            summary = cache[key].get("summary", "")
            status = "cached_jsonl"
            if looks_too_generic(summary):
                summary = fallback_detailed_summary(task, args.max_input_sentences)
                status = "fallback_generic_cached_jsonl"
            task.output_path.write_text(summary, encoding="utf-8")
            row = row_for_task(task, args.output_run_id, args.model_name,
                               args.max_input_sentences, args.max_new_tokens,
                               summary, status)
            rows.append(row)
            rows_by_key[key] = row
    return rows, rows_by_key


def validate_generated_summary(summary: str, task: SummaryTask,
                               max_input_sentences: int) -> tuple[str, str]:
    status = "generated"
    evidence = evidence_limit(task.evidence, max_input_sentences)
    if not summary.strip():
        return fallback_detailed_summary(task, max_input_sentences), "fallback_empty"
    if looks_like_prompt_echo(summary):
        return fallback_detailed_summary(task, max_input_sentences), "fallback_prompt_echo"
    if looks_too_generic(summary):
        return fallback_detailed_summary(task, max_input_sentences), "fallback_generic"
    return summary, status


def generate_rows_for_tasks(tasks: list[SummaryTask],
                            taxonomy: dict[str, dict[str, str]],
                            args,
                            model_name: str,
                            output_run_id: str,
                            cache_path: Path) -> list[dict]:
    original_model = args.model_name
    original_output = args.output_run_id
    args.model_name = model_name
    args.output_run_id = output_run_id
    try:
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
                summary = generate_transformers_task(generator, task, taxonomy, args)
                summary, status = validate_generated_summary(
                    summary, task, args.max_input_sentences)
                task.output_path.parent.mkdir(parents=True, exist_ok=True)
                task.output_path.write_text(summary, encoding="utf-8")
                row = row_for_task(task, args.output_run_id, args.model_name,
                                   args.max_input_sentences, args.max_new_tokens,
                                   summary, status)
                rows_by_key[task_key(args.model_name, task)] = row
                append_jsonl(cache_path, row)
                if index % 25 == 0:
                    logging.info("Processed %s/%s %s files", index,
                                 len(pending_tasks), task.level)

        return [
            rows_by_key[task_key(args.model_name, task)] for task in tasks
            if task_key(args.model_name, task) in rows_by_key
        ]
    finally:
        args.model_name = original_model
        args.output_run_id = original_output


def build_parent_tasks(child_rows: list[dict], output_run_id: str,
                       source_run_id: str,
                       taxonomy: dict[str, dict[str, str]]) -> list[SummaryTask]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in child_rows:
        aspect = row.get("aspect", "")
        parent = taxonomy.get(aspect, {}).get("group", "")
        summary = norm_text(row.get("summary", ""))
        if parent and summary:
            grouped[(row.get("split", ""), str(row.get("entity_id", "")), parent)].append(row)

    output_dir = OUTPUTS_DIR / f"{output_run_id}_parent"
    tasks = []
    for (split, entity_id, parent), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: (row.get("aspect", ""),
                                   row.get("sentiment", "")))
        evidence = []
        for row in rows:
            summary = norm_text(row.get("summary", ""))
            if not summary:
                continue
            scale = (taxonomy.get(row.get('aspect', ''), {}).get('scale')
                     or row.get('aspect'))
            polarity = row.get("sentiment", "")
            label = {"pos": "positive", "neg": "negative"}.get(polarity, "")
            prefix = f"{scale} ({label})" if label else f"{scale}"
            evidence.append(f"{prefix}: {summary}")
        tasks.append(SummaryTask(
            source_run_id=source_run_id,
            aspect=parent,
            parent_aspect=parent,
            split=split,
            entity_id=entity_id,
            output_path=output_dir / parent / f"{split}_{entity_id}",
            evidence=evidence,
            evidence_rows=rows,
            level="parent",
        ))
    return tasks


def build_entity_tasks(parent_rows: list[dict], output_run_id: str,
                       source_run_id: str,
                       parent_order: list[str]) -> list[SummaryTask]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    order = {parent: idx for idx, parent in enumerate(parent_order)}
    for row in parent_rows:
        summary = norm_text(row.get("summary", ""))
        if summary:
            grouped[(row.get("split", ""), str(row.get("entity_id", "")))].append(row)

    output_dir = OUTPUTS_DIR / f"{output_run_id}_entity"
    tasks = []
    for (split, entity_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: order.get(row.get("aspect", ""), 10**9))
        evidence = []
        for row in rows:
            summary = norm_text(row.get("summary", ""))
            if not summary:
                continue
            aspect = row.get("aspect")
            polarity = row.get("sentiment", "")
            label = {"pos": "positive", "neg": "negative"}.get(polarity, "")
            prefix = f"{aspect} ({label})" if label else str(aspect)
            evidence.append(f"{prefix}: {summary}")
        tasks.append(SummaryTask(
            source_run_id=source_run_id,
            aspect="ENTITY_OVERALL",
            split=split,
            entity_id=entity_id,
            output_path=output_dir / f"{split}_{entity_id}",
            evidence=evidence,
            evidence_rows=rows,
            level="entity",
        ))
    return tasks


def write_hierarchical_summary(path: Path, child_rows: list[dict],
                               parent_rows: list[dict],
                               entity_rows: list[dict]) -> None:
    payload: dict[str, dict] = {}
    for row in child_rows:
        key = f"{row.get('split')}_{row.get('entity_id')}"
        payload.setdefault(key, {
            "split": row.get("split"),
            "entity_id": row.get("entity_id"),
            "child_aspects": {},
            "parent_aspects": {},
            "overall_summary": "",
        })
        aspect = row.get("aspect")
        sentiment = row.get("sentiment", "")
        if sentiment in ("pos", "neg"):
            # Split mode: store both polarities under the aspect code as
            # {"positive": "...", "negative": "..."}.
            bucket = payload[key]["child_aspects"].setdefault(aspect, {})
            if not isinstance(bucket, dict):
                bucket = {}
                payload[key]["child_aspects"][aspect] = bucket
            polarity = "positive" if sentiment == "pos" else "negative"
            bucket[polarity] = row.get("summary", "")
        else:
            payload[key]["child_aspects"][aspect] = row.get("summary", "")
    for row in parent_rows:
        key = f"{row.get('split')}_{row.get('entity_id')}"
        payload.setdefault(key, {
            "split": row.get("split"),
            "entity_id": row.get("entity_id"),
            "child_aspects": {},
            "parent_aspects": {},
            "overall_summary": "",
        })
        payload[key]["parent_aspects"][row.get("aspect")] = row.get("summary", "")
    for row in entity_rows:
        key = f"{row.get('split')}_{row.get('entity_id')}"
        payload.setdefault(key, {
            "split": row.get("split"),
            "entity_id": row.get("entity_id"),
            "child_aspects": {},
            "parent_aspects": {},
            "overall_summary": "",
        })
        payload[key]["overall_summary"] = row.get("summary", "")
    path.write_text(json.dumps({
        "entities": payload,
        "counts": {
            "child_rows": len(child_rows),
            "parent_rows": len(parent_rows),
            "entity_rows": len(entity_rows),
        },
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Abstractive synthesis for SemAE HASOS aspect evidence.")
    parser.add_argument("--run_id", default="space_hasos_full_e20")
    parser.add_argument("--output_run_id", default=None)
    parser.add_argument("--source_mode",
                        choices=["ranked_evidence", "threshold_evidence", "extractive_tree"],
                        default="ranked_evidence")
    parser.add_argument("--evidence_jsonl", default=None)
    parser.add_argument("--evidence_top_k", type=int, default=0,
                        help="legacy no-op; threshold mode uses all eligible evidence")
    parser.add_argument("--evidence_score_threshold", type=float, default=0.005)
    parser.add_argument("--dedupe_cosine_threshold", type=float, default=0.82)
    parser.add_argument("--backend", choices=["vllm", "transformers"],
                        default="transformers")
    parser.add_argument("--model_name", default="google/flan-t5-small")
    parser.add_argument("--parent_model_name", default="google/flan-t5-base")
    parser.add_argument("--entity_model_name", default="google/flan-t5-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--vllm_base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm_api_key", default="EMPTY")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_input_sentences", type=int, default=0,
                        help="0 means use all threshold-selected evidence")
    parser.add_argument("--max_source_tokens", type=int, default=768,
                        help="chunk evidence before generation when prompt source exceeds this token estimate")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--parent_max_new_tokens", type=int, default=None,
                        help="Override output tokens for hierarchical parent summaries; defaults to --max_new_tokens.")
    parser.add_argument("--entity_max_new_tokens", type=int, default=128,
                        help="Output tokens for entity-level / overall summaries. 128 matches SPACE general refs.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--hierarchical", action="store_true",
                        help="also generate 6 parent aspect summaries and entity overall summaries")
    parser.add_argument("--skip_entity_summary", action="store_true",
                        help="with --hierarchical, stop after parent summaries")
    parser.add_argument("--split_sentiment", action="store_true",
                        help="generate separate positive and negative summaries "
                             "per sub-aspect (uses sentiment_label in the evidence "
                             "JSONL); neutral evidence is skipped for highlights")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(message)s",
    )

    if not args.output_run_id:
        if args.source_mode in {"ranked_evidence", "threshold_evidence"}:
            args.output_run_id = f"{args.run_id}_abstractive_threshold"
        else:
            args.output_run_id = f"{args.run_id}_abstractive"
    taxonomy = load_taxonomy(TAXONOMY_TSV)
    if args.source_mode in {"ranked_evidence", "threshold_evidence"}:
        evidence_jsonl = Path(
            args.evidence_jsonl
            or OUTPUTS_DIR / f"{args.run_id}_threshold_evidence.jsonl")
        if not evidence_jsonl.exists():
            evidence_jsonl = OUTPUTS_DIR / f"{args.run_id}_ranked_evidence.jsonl"
        tasks = list(iter_ranked_evidence_tasks(
            args.run_id, args.output_run_id, evidence_jsonl,
            args.evidence_score_threshold, args.dedupe_cosine_threshold,
            split_sentiment=args.split_sentiment))
    else:
        tasks = list(iter_extractive_tasks(args.run_id, args.output_run_id))
    if args.limit is not None:
        tasks = tasks[:args.limit]

    cache_path = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_cache.jsonl"
    lines_jsonl = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_lines.jsonl"
    lines_tsv = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_lines.tsv"
    report_path = OUTPUTS_DIR / f"{args.output_run_id}_synthesis_report.md"

    rows = generate_rows_for_tasks(
        tasks, taxonomy, args, args.model_name, args.output_run_id, cache_path)
    write_jsonl(lines_jsonl, rows)
    write_tsv(lines_tsv, rows, [
        "run_id", "source_run_id", "model_name", "level", "aspect", "parent_aspect",
        "sentiment", "split",
        "entity_id", "summary", "evidence_count", "evidence_used",
        "copied_from_evidence", "status", "output_path", "source_path",
    ])
    write_report(report_path, rows, args.output_run_id, args.run_id,
                 args.model_name, args.source_mode)

    child_alias_jsonl = OUTPUTS_DIR / f"{args.run_id}_child_synthesis_lines.jsonl"
    child_alias_tsv = OUTPUTS_DIR / f"{args.run_id}_child_synthesis_lines.tsv"
    write_jsonl(child_alias_jsonl, rows)
    write_tsv(child_alias_tsv, rows, [
        "run_id", "source_run_id", "model_name", "level", "aspect", "parent_aspect", "split",
        "entity_id", "summary", "evidence_count", "evidence_used",
        "copied_from_evidence", "status", "output_path", "source_path",
    ])

    parent_rows: list[dict] = []
    entity_rows: list[dict] = []
    if args.hierarchical:
        parent_tasks = build_parent_tasks(
            rows, args.output_run_id, args.run_id, taxonomy)
        if args.limit is not None:
            parent_tasks = parent_tasks[:args.limit]
        entity_source_rows = rows
        entity_parent_order = sorted({row.get("aspect", "") for row in rows if row.get("aspect")})
        if parent_tasks:
            parent_token_limit = args.max_new_tokens
            if args.parent_max_new_tokens is not None:
                parent_token_limit = args.parent_max_new_tokens
            original_max_new_tokens = args.max_new_tokens
            args.max_new_tokens = parent_token_limit
            parent_rows = generate_rows_for_tasks(
                parent_tasks, taxonomy, args, args.parent_model_name,
                f"{args.run_id}_parent_synthesis",
                OUTPUTS_DIR / f"{args.run_id}_parent_synthesis_cache.jsonl")
            args.max_new_tokens = original_max_new_tokens
            entity_source_rows = parent_rows
            entity_parent_order = taxonomy_parent_order(taxonomy)
        parent_jsonl = OUTPUTS_DIR / f"{args.run_id}_parent_synthesis_lines.jsonl"
        parent_tsv = OUTPUTS_DIR / f"{args.run_id}_parent_synthesis_lines.tsv"
        write_jsonl(parent_jsonl, parent_rows)
        write_tsv(parent_tsv, parent_rows, [
            "run_id", "source_run_id", "model_name", "level", "aspect", "parent_aspect",
            "sentiment", "split",
            "entity_id", "summary", "evidence_count", "evidence_used",
            "copied_from_evidence", "status", "output_path", "source_path",
        ])

        if not args.skip_entity_summary:
            entity_tasks = build_entity_tasks(
                entity_source_rows, args.output_run_id, args.run_id,
                entity_parent_order)
            if args.limit is not None:
                entity_tasks = entity_tasks[:args.limit]
            original_max_new_tokens = args.max_new_tokens
            args.max_new_tokens = args.entity_max_new_tokens
            entity_rows = generate_rows_for_tasks(
                entity_tasks, taxonomy, args, args.entity_model_name,
                f"{args.run_id}_entity_synthesis",
                OUTPUTS_DIR / f"{args.run_id}_entity_synthesis_cache.jsonl")
            args.max_new_tokens = original_max_new_tokens
            entity_jsonl = OUTPUTS_DIR / f"{args.run_id}_entity_synthesis_lines.jsonl"
            entity_tsv = OUTPUTS_DIR / f"{args.run_id}_entity_synthesis_lines.tsv"
            write_jsonl(entity_jsonl, entity_rows)
            write_tsv(entity_tsv, entity_rows, [
                "run_id", "source_run_id", "model_name", "level", "aspect", "parent_aspect",
                "sentiment", "split",
                "entity_id", "summary", "evidence_count", "evidence_used",
                "copied_from_evidence", "status", "output_path", "source_path",
            ])
        write_hierarchical_summary(
            OUTPUTS_DIR / f"{args.run_id}_hierarchical_summary.json",
            rows, parent_rows, entity_rows)

    print(f"processed={len(rows)}")
    print(f"output_dir={OUTPUTS_DIR / args.output_run_id}")
    print(f"cache={cache_path}")
    print(f"jsonl={lines_jsonl}")
    print(f"tsv={lines_tsv}")
    print(f"report={report_path}")
    if args.hierarchical:
        print(f"parent_rows={len(parent_rows)}")
        print(f"entity_rows={len(entity_rows)}")
        print(f"hierarchical_json={OUTPUTS_DIR / f'{args.run_id}_hierarchical_summary.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
