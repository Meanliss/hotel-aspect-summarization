#!/usr/bin/env python3
"""Run the DeepSeek LLM judge for the concrete HASOS metric dataset.

The API key is read only from DEEPSEEK_API_KEY. Results are cached per item and
rubric/model hash so the run can resume without paying twice.

Usage:
    python scripts/run_concrete_metric_judge.py --per-method 10
    python scripts/run_concrete_metric_judge.py --concurrency 12
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
METRICS_DIR = REPO / "reports" / "metrics"
DEFAULT_INPUT = METRICS_DIR / "concrete_metric_dataset.jsonl"
DEFAULT_OUT = METRICS_DIR / "concrete_metric_judgments.jsonl"
DEFAULT_FAILURES = METRICS_DIR / "judge_failures.jsonl"
DEFAULT_CACHE_DIR = METRICS_DIR / "judge_cache"

RUBRIC_VERSION = "concrete-v1"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com"

SCORE_FIELDS = [
    "evidence_support",
    "aspect_correctness",
    "sentiment_alignment",
    "coverage",
    "specificity",
    "usefulness",
]
FLAG_FIELDS = [
    "unsupported_claim_present",
    "aspect_mismatch_present",
    "sentiment_flip_present",
    "major_theme_missing",
    "generic_summary",
]
EVIDENCE_FLAG_FIELDS = [
    "relevant",
    "supports_summary_claims",
    "cross_aspect_leakage",
    "sentiment_leakage",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with io.open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO))
    except ValueError:
        return str(path)


def stable_hash(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_key(row: dict[str, Any], model: str, thinking: str | None = None) -> str:
    payload = {
        "rubric_version": RUBRIC_VERSION,
        "model": model,
        "method": row["method"],
        "entity_id": row["entity_id"],
        "split": row["split"],
        "aspect": row["aspect"],
        "sentiment": row.get("sentiment", ""),
        "summary": row["summary"],
        "evidence": row["evidence"],
    }
    if thinking:
        payload["thinking"] = thinking
    return stable_hash(payload)


def select_rows(rows: list[dict[str, Any]], limit: int | None, per_method: int | None, seed: int) -> list[dict[str, Any]]:
    if per_method is None and limit is None:
        return rows
    rng = random.Random(seed)
    if per_method is not None:
        selected = []
        by_method: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_method.setdefault(row["method"], []).append(row)
        for method in sorted(by_method):
            bucket = list(by_method[method])
            rng.shuffle(bucket)
            selected.extend(sorted(bucket[:per_method], key=lambda r: r["item_id"]))
        return sorted(selected, key=lambda r: (r["method"], r["item_id"]))
    shuffled = list(rows)
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda r: (r["method"], r["item_id"]))


def filter_methods(rows: list[dict[str, Any]], methods: str | None) -> list[dict[str, Any]]:
    if not methods:
        return rows
    wanted = {m.strip().lower() for m in methods.split(",") if m.strip()}
    if not wanted:
        return rows
    return [row for row in rows if str(row.get("method", "")).lower() in wanted]


def build_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    sentiment = row.get("sentiment") or "not specified"
    evidence_lines = []
    for idx, ev in enumerate(row.get("evidence") or [], 1):
        evidence_lines.append(
            json.dumps(
                {
                    "evidence_index": idx,
                    "rank": ev.get("rank"),
                    "aspect_seed": ev.get("matched_aspect_seed") or [],
                    "sentiment_label": ev.get("sentiment_label") or "",
                    "sentence": ev.get("sentence") or "",
                },
                ensure_ascii=False,
            )
        )
    user_payload = {
        "task": "Evaluate whether an aspect-based hotel summary is faithful to its evidence.",
        "target": {
            "dataset": row.get("dataset"),
            "method": row.get("method"),
            "entity_id": row.get("entity_id"),
            "aspect": row.get("aspect"),
            "sentiment": sentiment,
        },
        "summary": row.get("summary") or "",
        "evidence_top_k": evidence_lines,
        "instructions": [
            "Judge only against the provided evidence, target aspect, and target sentiment.",
            "If target sentiment is 'not specified', score sentiment_alignment by consistency with evidence and absence of sentiment reversal.",
            "Mark unsupported_claim_present true if the summary states information not supported by evidence.",
            "Mark aspect_mismatch_present true if the summary mainly discusses a different aspect.",
            "Mark sentiment_flip_present true if the summary polarity contradicts target/evidence polarity.",
            "For each evidence item, relevant means it matches the target aspect; supports_summary_claims means it supports at least one summary claim.",
        ],
    }
    schema = {
        "scores": {field: "integer 1-5" for field in SCORE_FIELDS},
        "flags": {field: "boolean" for field in FLAG_FIELDS},
        "evidence_labels": [
            {
                "evidence_index": "integer, 1-based",
                "relevant": "boolean",
                "supports_summary_claims": "boolean",
                "cross_aspect_leakage": "boolean",
                "sentiment_leakage": "boolean",
            }
        ],
        "verdict": "pass|warn|fail",
        "rationale": "short explanation, <= 35 words",
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict evaluation judge for an academic paper. "
                "Return only valid JSON. Do not include markdown fences."
            ),
        },
        {
            "role": "user",
            "content": (
                "Rubric version: "
                + RUBRIC_VERSION
                + "\nExpected JSON schema:\n"
                + json.dumps(schema, ensure_ascii=False)
                + "\nEvaluation item:\n"
                + json.dumps(user_payload, ensure_ascii=False)
            ),
        },
    ]


def strip_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def validate_judgment(obj: dict[str, Any], evidence_len: int) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError("judgment is not an object")
    scores = obj.get("scores")
    flags = obj.get("flags")
    evidence_labels = obj.get("evidence_labels")
    if not isinstance(scores, dict):
        raise ValueError("missing scores object")
    if not isinstance(flags, dict):
        raise ValueError("missing flags object")
    if not isinstance(evidence_labels, list):
        raise ValueError("missing evidence_labels list")
    clean_scores = {}
    for field in SCORE_FIELDS:
        value = scores.get(field)
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or not 1 <= value <= 5:
            raise ValueError(f"invalid score {field}: {value!r}")
        clean_scores[field] = value
    clean_flags = {}
    for field in FLAG_FIELDS:
        value = flags.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"invalid flag {field}: {value!r}")
        clean_flags[field] = value
    if len(evidence_labels) != evidence_len:
        raise ValueError(f"evidence_labels length {len(evidence_labels)} != {evidence_len}")
    clean_labels = []
    seen = set()
    for label in evidence_labels:
        if not isinstance(label, dict):
            raise ValueError("evidence label is not an object")
        idx = label.get("evidence_index")
        if not isinstance(idx, int) or not 1 <= idx <= evidence_len or idx in seen:
            raise ValueError(f"invalid evidence_index: {idx!r}")
        seen.add(idx)
        clean_label = {"evidence_index": idx}
        for field in EVIDENCE_FLAG_FIELDS:
            value = label.get(field)
            if not isinstance(value, bool):
                raise ValueError(f"invalid evidence label {field}: {value!r}")
            clean_label[field] = value
        clean_labels.append(clean_label)
    verdict = obj.get("verdict")
    if verdict not in {"pass", "warn", "fail"}:
        raise ValueError(f"invalid verdict: {verdict!r}")
    rationale = str(obj.get("rationale") or "").strip()
    return {
        "scores": clean_scores,
        "flags": clean_flags,
        "evidence_labels": sorted(clean_labels, key=lambda x: x["evidence_index"]),
        "verdict": verdict,
        "rationale": rationale[:500],
    }


def call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    timeout: int,
    thinking: str | None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if thinking:
        payload["thinking"] = {"type": thinking}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        return json.loads(resp.read().decode("utf-8"))


def judge_one(
    row: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    cache_dir: Path,
    max_retries: int,
    max_tokens: int,
    timeout: int,
    thinking: str | None,
) -> dict[str, Any]:
    key = cache_key(row, model, thinking)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        cached = json.load(io.open(cache_path, encoding="utf-8"))
        cached["cache_hit"] = True
        return cached

    evidence_len = len(row.get("evidence") or [])
    messages = build_prompt(row)
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = call_deepseek(
                api_key=api_key,
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                thinking=thinking,
            )
            content = response["choices"][0]["message"]["content"]
            json_text = strip_json_text(content)
            try:
                parsed = json.loads(json_text)
            except json.JSONDecodeError as exc:
                snippet = json_text[:180].replace("\n", "\\n")
                raise ValueError(f"invalid judge JSON: {exc}; prefix={snippet!r}") from exc
            judgment = validate_judgment(parsed, evidence_len)
            record = {
                "item_id": row["item_id"],
                "cache_key": key,
                "rubric_version": RUBRIC_VERSION,
                "model": model,
                "thinking": thinking or "",
                "dataset": row["dataset"],
                "method": row["method"],
                "method_label": row["method_label"],
                "split": row["split"],
                "entity_id": row["entity_id"],
                "aspect": row["aspect"],
                "sentiment": row.get("sentiment", ""),
                "status": row.get("status", ""),
                "judgment": judgment,
                "usage": response.get("usage") or {},
                "created": response.get("created"),
                "cache_hit": False,
            }
            tmp = cache_path.with_suffix(".json.tmp")
            with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
                json.dump(record, f, ensure_ascii=False, indent=2, sort_keys=True)
                f.write("\n")
            tmp.replace(cache_path)
            return record
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
            last_error = str(exc)
            sleep_s = min(2 ** attempt, 20) + random.random()
            time.sleep(sleep_s)
    raise RuntimeError(last_error or "unknown judge failure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--failures", default=str(DEFAULT_FAILURES))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--thinking", choices=["enabled", "disabled"],
                        help="DeepSeek V4 thinking mode toggle.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-method", type=int, help="Sample N rows per method for smoke runs.")
    parser.add_argument("--methods", help="Comma-separated method ids to judge, e.g. m1 or m1,m3.")
    parser.add_argument("--preserve-existing", action="store_true",
                        help="Keep existing judgments for unselected items in --out.")
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set; refusing to read keys from files or arguments.")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    rows = filter_methods(read_jsonl(Path(args.input)), args.methods)
    selected = select_rows(rows, args.limit, args.per_method, args.seed)
    selected_ids = {row["item_id"] for row in selected}
    preserved_results: list[dict[str, Any]] = []
    out_path = Path(args.out)
    if args.preserve_existing and out_path.exists():
        preserved_results = [
            row for row in read_jsonl(out_path)
            if row.get("item_id") not in selected_ids
        ]
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures_path = Path(args.failures)
    if failures_path.exists():
        failures_path.unlink()

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_map = {
            pool.submit(
                judge_one,
                row,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                cache_dir=cache_dir,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                thinking=args.thinking,
            ): row
            for row in selected
        }
        done_n = 0
        for future in concurrent.futures.as_completed(future_map):
            row = future_map[future]
            done_n += 1
            try:
                results.append(future.result())
            except Exception as exc:  # Keep the batch alive and report failures.
                failure = {
                    "item_id": row["item_id"],
                    "method": row["method"],
                    "entity_id": row["entity_id"],
                    "aspect": row["aspect"],
                    "sentiment": row.get("sentiment", ""),
                    "error": str(exc),
                }
                failures.append(failure)
                append_jsonl(failures_path, failure)
            if done_n % 25 == 0 or done_n == len(selected):
                elapsed = time.time() - started
                print(f"judged {done_n}/{len(selected)} rows in {elapsed:.1f}s")

    results.sort(key=lambda r: (r["method"], r["item_id"]))
    output_rows = preserved_results + results
    output_rows.sort(key=lambda r: (r["method"], r["item_id"]))
    write_jsonl(out_path, output_rows)
    if failures:
        print(f"completed with {len(failures)} failures -> {display_path(failures_path)}")
    if preserved_results:
        print(f"preserved {len(preserved_results)} existing judgments")
    print(f"written {len(output_rows)} judgments -> {display_path(out_path)}")


if __name__ == "__main__":
    main()
