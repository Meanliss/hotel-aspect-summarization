"""Compute non-ROUGE quality metrics on outputs/<run_id>/<aspect>/<split>_<entity>.

Metrics:
  1. source_fidelity          – % summary sentences appearing verbatim in source reviews
  2. aspect_keyword_coverage  – % summary sentences containing >=1 seed/taxonomy keyword for own aspect
  3. aspect_purity            – fraction of summary sentences whose best-matching aspect == target aspect
  4. distinct_1 / distinct_2  – type-token ratio for unigrams/bigrams across all summaries of an aspect
  5. self_bleu                – avg pairwise BLEU-4 between summaries of same aspect (lower = more diverse)
  6. cross_aspect_jaccard     – avg token-Jaccard between any two aspect summaries of the same entity
  7. compression_ratio        – summary tokens / source tokens
  8. avg_sentence_len         – tokens per summary sentence

Usage:
    python scripts/score_semae_run.py --run_id hasos_aspects_run1
Writes:
    outputs/<run_id>_metrics.json
    outputs/<run_id>_metrics.md
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def tok(s: str):
    return re.findall(r"[a-zA-Z']+", s.lower())


def sents(s: str):
    # split on sentence punctuation, keep non-empty trimmed pieces
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+", s.strip()) if p.strip()]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def bleu4(ref_tokens, hyp_tokens):
    """Mini BLEU-4 with brevity penalty, no smoothing (returns 0 if any n-gram precision is 0)."""
    if not hyp_tokens or not ref_tokens:
        return 0.0
    import math
    precisions = []
    for n in (1, 2, 3, 4):
        if len(hyp_tokens) < n:
            return 0.0
        hyp_ng = Counter(tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1))
        ref_ng = Counter(tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1))
        overlap = sum((hyp_ng & ref_ng).values())
        total = max(1, sum(hyp_ng.values()))
        if overlap == 0:
            return 0.0
        precisions.append(overlap / total)
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1 - len(ref_tokens) / max(1, len(hyp_tokens)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / 4)


def load_seeds(seeds_dir: Path):
    seeds = {}
    for f in seeds_dir.glob("*.txt"):
        words = set()
        for line in f.read_text(encoding="utf-8").splitlines():
            for w in tok(line):
                words.add(w)
        seeds[f.stem] = words
    return seeds


def load_taxonomy_keywords(tsv: Path):
    """Aspect-code -> set of aspect+sentiment keywords from taxonomy TSV."""
    out = {}
    lines = tsv.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        code = parts[1].strip()
        kws = set()
        for col in parts[4:8]:
            for w in tok(col):
                kws.add(w)
        out[code] = kws
    return out


def load_aspect_descriptions(tsv: Path):
    """Aspect-code -> a natural-language reference string (name + scale + description + keywords)."""
    out = {}
    lines = tsv.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) < 8:
            continue
        code = parts[1].strip()
        scale = parts[2].strip()
        desc = parts[3].strip()
        kws = parts[4].strip().replace(",", " ")
        out[code] = f"{scale}. {desc}. Keywords: {kws}."
    return out


def load_source_index(summary_json: Path):
    """entity_id -> (set of source sentences, list of source tokens)."""
    data = json.loads(summary_json.read_text(encoding="utf-8"))
    idx = {}
    for ent in data:
        ent_id = str(ent["entity_id"])
        all_sents = set()
        all_tokens = []
        for rev in ent.get("reviews", []):
            for s in rev.get("sentences", []):
                s_norm = s.strip()
                if s_norm:
                    all_sents.add(s_norm)
                    all_tokens.extend(tok(s_norm))
        idx[ent_id] = (all_sents, all_tokens)
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--outputs_dir", default=str(REPO_ROOT / "outputs"))
    ap.add_argument("--seeds_dir", default=str(REPO_ROOT / "data" / "seeds_hasos"))
    ap.add_argument("--taxonomy_tsv", default=str(REPO_ROOT / "data" / "hasos" / "aspect_taxonomy.tsv"))
    ap.add_argument("--summary_json", default=str(REPO_ROOT / "data" / "hasos" / "hasos_summ.json"))
    ap.add_argument("--self_bleu_max_pairs", type=int, default=200)
    ap.add_argument("--bert_score", action="store_true",
                    help="also compute BERTScore F1 vs aspect description and vs source pool")
    ap.add_argument("--bert_model", default="roberta-large")
    ap.add_argument("--bert_batch_size", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(args.outputs_dir) / args.run_id
    if not run_dir.is_dir():
        raise SystemExit(f"Run dir not found: {run_dir}")

    seeds = load_seeds(Path(args.seeds_dir))
    tax = load_taxonomy_keywords(Path(args.taxonomy_tsv))
    aspect_desc = load_aspect_descriptions(Path(args.taxonomy_tsv))
    # combined keyword set per aspect = seeds ∪ taxonomy
    aspect_kw = {a: seeds.get(a, set()) | tax.get(a, set()) for a in set(seeds) | set(tax)}

    source = load_source_index(Path(args.summary_json))

    per_aspect = {}
    all_aspects = sorted([d.name for d in run_dir.iterdir() if d.is_dir()])

    # collect per-aspect summary tokens + per-entity per-aspect tokens for cross-aspect jaccard
    entity_aspect_tokens = defaultdict(dict)
    # records for optional BERTScore pass: list of (aspect, ent_id, summary_text)
    summary_records = []

    for aspect in all_aspects:
        adir = run_dir / aspect
        files = sorted(adir.iterdir())
        kw = aspect_kw.get(aspect, set())

        total_sents = 0
        verbatim_sents = 0
        kw_hit_sents = 0
        pure_sents = 0
        sum_tokens_total = 0
        src_tokens_total = 0
        sent_lens = []

        unigrams = Counter()
        bigrams = Counter()
        n_uni = 0
        n_bi = 0

        summary_token_lists = []

        for f in files:
            stem = f.name  # e.g. dev_100597
            try:
                ent_id = stem.split("_", 1)[1]
            except IndexError:
                continue
            src_sents, src_tokens = source.get(ent_id, (set(), []))
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                continue
            s_list = sents(text)
            s_tokens_all = []
            for s in s_list:
                total_sents += 1
                if s in src_sents:
                    verbatim_sents += 1
                stoks = tok(s)
                sent_lens.append(len(stoks))
                s_tokens_all.extend(stoks)
                stoks_set = set(stoks)
                # keyword hit for own aspect
                if kw & stoks_set:
                    kw_hit_sents += 1
                # purity: which aspect has most kw hits in this sentence?
                best_aspect, best_hits = aspect, len(kw & stoks_set)
                for other, okw in aspect_kw.items():
                    h = len(okw & stoks_set)
                    if h > best_hits:
                        best_hits = h
                        best_aspect = other
                if best_aspect == aspect and best_hits > 0:
                    pure_sents += 1
                # distinct-n
                for w in stoks:
                    unigrams[w] += 1
                    n_uni += 1
                for i in range(len(stoks) - 1):
                    bigrams[(stoks[i], stoks[i + 1])] += 1
                    n_bi += 1
            sum_tokens_total += len(s_tokens_all)
            src_tokens_total += len(src_tokens)
            summary_token_lists.append(s_tokens_all)
            entity_aspect_tokens[ent_id][aspect] = set(s_tokens_all)
            summary_records.append((aspect, ent_id, text))

        # self-BLEU: avg pairwise BLEU-4 across up to N pairs
        pairs = list(combinations(range(len(summary_token_lists)), 2))
        if len(pairs) > args.self_bleu_max_pairs:
            step = len(pairs) // args.self_bleu_max_pairs
            pairs = pairs[::step][:args.self_bleu_max_pairs]
        if pairs:
            sb = sum(bleu4(summary_token_lists[i], summary_token_lists[j]) for i, j in pairs) / len(pairs)
        else:
            sb = 0.0

        per_aspect[aspect] = {
            "n_files": len(files),
            "n_sentences": total_sents,
            "source_fidelity": verbatim_sents / total_sents if total_sents else 0.0,
            "aspect_keyword_coverage": kw_hit_sents / total_sents if total_sents else 0.0,
            "aspect_purity": pure_sents / total_sents if total_sents else 0.0,
            "distinct_1": (len(unigrams) / n_uni) if n_uni else 0.0,
            "distinct_2": (len(bigrams) / n_bi) if n_bi else 0.0,
            "self_bleu4": sb,
            "compression_ratio": (sum_tokens_total / src_tokens_total) if src_tokens_total else 0.0,
            "avg_sentence_len": (sum(sent_lens) / len(sent_lens)) if sent_lens else 0.0,
        }

    # cross-aspect jaccard per entity (avg over all aspect pairs that both exist)
    jaccards = []
    for ent_id, by_aspect in entity_aspect_tokens.items():
        toks_lists = [(a, t) for a, t in by_aspect.items() if t]
        for (a1, t1), (a2, t2) in combinations(toks_lists, 2):
            jaccards.append(jaccard(t1, t2))
    cross_jacc = sum(jaccards) / len(jaccards) if jaccards else 0.0

    # optional BERTScore F1 (reference-free: vs aspect description and vs entity's source pool)
    if args.bert_score and summary_records:
        from bert_score import score as bert_score_fn
        print(f"[bert] scoring {len(summary_records)} summaries with {args.bert_model} ...")
        cands = [r[2] for r in summary_records]
        refs_aspect = [aspect_desc.get(r[0], r[0]) for r in summary_records]
        # source pool reference: first ~400 source tokens concatenated
        def source_ref(ent_id):
            _, src_toks = source.get(ent_id, (set(), []))
            return " ".join(src_toks[:400]) if src_toks else ""
        refs_source = [source_ref(r[1]) for r in summary_records]

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        _, _, f1_aspect = bert_score_fn(
            cands, refs_aspect, model_type=args.bert_model, lang="en",
            batch_size=args.bert_batch_size, device=device, verbose=False,
        )
        _, _, f1_source = bert_score_fn(
            cands, refs_source, model_type=args.bert_model, lang="en",
            batch_size=args.bert_batch_size, device=device, verbose=False,
        )
        f1_a = f1_aspect.tolist()
        f1_s = f1_source.tolist()
        # aggregate per aspect
        by_aspect_a = defaultdict(list)
        by_aspect_s = defaultdict(list)
        for (aspect, _ent, _txt), fa, fs in zip(summary_records, f1_a, f1_s):
            by_aspect_a[aspect].append(fa)
            by_aspect_s[aspect].append(fs)
        for aspect, vals in by_aspect_a.items():
            per_aspect[aspect]["bert_f1_aspect"] = sum(vals) / len(vals)
        for aspect, vals in by_aspect_s.items():
            per_aspect[aspect]["bert_f1_source"] = sum(vals) / len(vals)
        macro_bert_a = sum(f1_a) / len(f1_a)
        macro_bert_s = sum(f1_s) / len(f1_s)
    else:
        macro_bert_a = None
        macro_bert_s = None

    # macro aggregates
    def mean(key):
        vals = [v[key] for v in per_aspect.values() if v["n_sentences"] > 0]
        return sum(vals) / len(vals) if vals else 0.0

    macro = {
        "source_fidelity": mean("source_fidelity"),
        "aspect_keyword_coverage": mean("aspect_keyword_coverage"),
        "aspect_purity": mean("aspect_purity"),
        "distinct_1": mean("distinct_1"),
        "distinct_2": mean("distinct_2"),
        "self_bleu4": mean("self_bleu4"),
        "compression_ratio": mean("compression_ratio"),
        "avg_sentence_len": mean("avg_sentence_len"),
        "cross_aspect_jaccard": cross_jacc,
        "bert_f1_aspect": macro_bert_a,
        "bert_f1_source": macro_bert_s,
    }

    out_json = Path(args.outputs_dir) / f"{args.run_id}_metrics.json"
    out_md = Path(args.outputs_dir) / f"{args.run_id}_metrics.md"
    out_json.write_text(json.dumps({"macro": macro, "per_aspect": per_aspect}, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"# Metrics report — `{args.run_id}` (no ROUGE; gold summaries unavailable)\n")
    lines.append("## Macro averages (mean over aspects)\n")
    lines.append("| Metric | Value | What it measures |")
    lines.append("| --- | ---: | --- |")
    lines.append(f"| source_fidelity        | {macro['source_fidelity']:.4f} | fraction of summary sentences found verbatim in source reviews (extractive check, ideal=1.0) |")
    lines.append(f"| aspect_keyword_coverage| {macro['aspect_keyword_coverage']:.4f} | fraction of summary sentences containing ≥1 aspect/sentiment keyword for own aspect (higher=better) |")
    lines.append(f"| aspect_purity          | {macro['aspect_purity']:.4f} | fraction of summary sentences whose top-matching aspect == target aspect (higher=better) |")
    lines.append(f"| distinct_1             | {macro['distinct_1']:.4f} | unique unigrams / total unigrams across aspect's summaries (lexical diversity) |")
    lines.append(f"| distinct_2             | {macro['distinct_2']:.4f} | unique bigrams  / total bigrams  across aspect's summaries |")
    lines.append(f"| self_bleu4             | {macro['self_bleu4']:.4f} | avg pairwise BLEU-4 between summaries within same aspect (lower=more diverse) |")
    lines.append(f"| compression_ratio      | {macro['compression_ratio']:.4f} | summary tokens / source tokens (extractive compression) |")
    lines.append(f"| avg_sentence_len       | {macro['avg_sentence_len']:.2f} | mean tokens per summary sentence |")
    lines.append(f"| cross_aspect_jaccard   | {macro['cross_aspect_jaccard']:.4f} | avg token-Jaccard between any two aspect summaries of same entity (lower=better separation) |")
    if macro.get('bert_f1_aspect') is not None:
        lines.append(f"| bert_f1_aspect         | {macro['bert_f1_aspect']:.4f} | BERTScore-F1 (raw) between summary and aspect description text (higher=better aspect alignment) |")
        lines.append(f"| bert_f1_source         | {macro['bert_f1_source']:.4f} | BERTScore-F1 (raw) between summary and entity source-review pool (higher=better semantic fidelity) |")
    lines.append("")
    lines.append("## Per-aspect breakdown\n")
    header = "| Aspect | n_files | n_sents | src_fid | kw_cov | purity | distinct1 | distinct2 | self_bleu4 | compr | avg_len |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines.append(header)
    lines.append(sep)
    for a in sorted(per_aspect):
        v = per_aspect[a]
        lines.append(
            f"| {a} | {v['n_files']} | {v['n_sentences']} | "
            f"{v['source_fidelity']:.3f} | {v['aspect_keyword_coverage']:.3f} | {v['aspect_purity']:.3f} | "
            f"{v['distinct_1']:.3f} | {v['distinct_2']:.3f} | {v['self_bleu4']:.3f} | "
            f"{v['compression_ratio']:.4f} | {v['avg_sentence_len']:.1f} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print()
    print("Macro:")
    for k, v in macro.items():
        if v is None:
            print(f"  {k:25s} (skipped)")
        else:
            print(f"  {k:25s} {v:.4f}")


if __name__ == "__main__":
    main()
