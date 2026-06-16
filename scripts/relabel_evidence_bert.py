"""Re-label an evidence JSONL's sentiment_label column with the BERT-ABSA backend.

Sentence SELECTION is identical across sentiment backends (it depends only on
the SemAE aspect assignment + KL ranking), so M4 (BERT-ABSA) reuses the exact
same evidence rows as M3 (keyword) and only overwrites `sentiment_label`. This
lets us generate the M4 sentiment-split abstractive summaries without re-running
the GPU SemAE pipeline.

Usage:
    python scripts/relabel_evidence_bert.py \
        --in outputs/space_hasos_threshold_full_threshold_evidence.jsonl \
        --out outputs/space_hasos_threshold_full_bert_threshold_evidence.jsonl \
        --taxonomy data/hasos/aspect_taxonomy.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))


def load_display_names(taxonomy_path):
    tax = json.load(io.open(taxonomy_path, encoding="utf-8"))
    return {a["code"]: a.get("measurement_scale", a["code"])
            for a in tax["aspects"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--taxonomy", default="data/hasos/aspect_taxonomy.json",
                    help="HASOS taxonomy for aspect display names. Pass 'none' "
                         "for SPACE, whose aspect codes are already readable.")
    ap.add_argument("--model", default="yangheng/deberta-v3-base-absa-v1.1")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    from sentiment_classifier import get_classifier

    if args.taxonomy.lower() == "none":
        disp = {}
    else:
        disp = load_display_names(os.path.join(REPO, args.taxonomy))
    clf = get_classifier(model_name=args.model, aspect_aware=True,
                         batch_size=args.batch_size)

    rows = []
    with io.open(os.path.join(REPO, args.inp), encoding="utf-8",
                 errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    sentences = [r.get("sentence", "") for r in rows]
    aspect_names = [disp.get(r.get("aspect", ""), r.get("aspect", ""))
                    for r in rows]

    print(f"[relabel] {len(rows)} rows; classifying with {args.model} ...")
    labels = clf.classify_batch(sentences, aspect_names)

    dist = {}
    for r, (label, conf) in zip(rows, labels):
        r["sentiment_label"] = label
        r["matched_sentiment_keywords"] = [f"bert:{conf:.3f}"]
        dist[label] = dist.get(label, 0) + 1

    out_path = os.path.join(REPO, args.out)
    with io.open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[relabel] new distribution: {dist}")
    print(f"[relabel] written -> {args.out}")


if __name__ == "__main__":
    main()
