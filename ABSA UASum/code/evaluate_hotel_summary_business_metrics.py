"""Business-oriented metrics for hotel aspect summaries.

The existing pipeline metrics use ROUGE recall against every source segment in a
hotel/aspect bucket. That is useful as a rough lexical coverage signal, but it
penalizes concise abstractive summaries. This script adds metrics that match the
reader-facing output contract more directly:

- ROUGE precision/F1 and length-normalized recall.
- Sentiment count consistency against cluster evidence.
- Pos/Neg/Neu structured format compliance.
- Top-cluster target coverage in each sentiment summary.
- Optional multilingual embedding similarity against cluster-summary pseudo
  references.
- Optional BERTScore against the same compact pseudo references.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_hotel_review1_v3_full_narrative_summaries import ASPECT_CODE, SCALE_VI


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_RESULTS = WORKSPACE / "results"
DEFAULT_RUN_NAME = "hotel_review1_vi_100plus_llm_v3_full"
SENTIMENTS = ("positive", "negative", "neutral")
ASPECTS = ("facility", "amenity", "service", "experience", "branding", "loyalty")
WORD_RE = re.compile(r"\w+", re.UNICODE)

MANUAL_ALIASES = {
    "Breakfast Quality": ["bữa sáng", "ăn sáng", "buffet", "thực đơn"],
    "Wifi Quality": ["wifi", "wi fi", "internet", "mạng", "kết nối"],
    "Pool": ["hồ bơi", "bể bơi", "pool"],
    "Food & Beverage Quality": ["đồ ăn", "thức ăn", "món ăn", "đồ uống", "ẩm thực"],
    "Restaurant Availability": ["nhà hàng", "quầy bar", "bar"],
    "Gym": ["phòng gym", "phòng tập", "gym"],
    "Spa & Sauna": ["spa", "sauna", "xông hơi"],
    "Parking": ["bãi đậu xe", "chỗ đậu xe", "đỗ xe"],
    "Shuttle Service": ["đưa đón", "di chuyển", "taxi", "xe"],
    "Laundry Service": ["giặt ủi", "giặt là"],
    "In-room Amenities": ["tiện nghi trong phòng", "minibar", "tủ lạnh"],
    "Entertainment Facilities": ["tv", "tivi", "giải trí"],
    "Staff Friendliness": ["nhân viên", "thân thiện", "nhiệt tình", "hỗ trợ"],
    "Staff Professionalism": ["chuyên nghiệp", "quản lý", "xử lý"],
    "Check-in/Check-out": ["check in", "check out", "nhận phòng", "trả phòng"],
    "Responsiveness": ["phản hồi", "hỗ trợ", "nhanh chóng"],
    "Problem Solving": ["xử lý vấn đề", "giải quyết", "khắc phục"],
    "Restaurant Service": ["phục vụ nhà hàng", "nhà hàng"],
    "Room Service": ["dịch vụ phòng"],
    "Hospitality": ["hiếu khách", "chăm sóc"],
    "Communication Ability": ["giao tiếp", "tiếng anh"],
    "Room Cleanliness": ["độ sạch phòng", "phòng sạch", "vệ sinh phòng", "sạch sẽ"],
    "Room Comfort": ["thoải mái phòng", "phòng thoải mái"],
    "Bathroom Condition": ["phòng tắm", "nhà tắm", "vòi sen", "thoát nước"],
    "Bed Quality": ["giường", "nệm", "chăn ga"],
    "Interior Design": ["thiết kế", "nội thất", "trang trí"],
    "Furniture Quality": ["nội thất", "đồ đạc", "trang thiết bị"],
    "Reception Area": ["lễ tân", "sảnh"],
    "Building Condition": ["tòa nhà", "cơ sở", "bảo trì"],
    "Public Area Cleanliness": ["khu vực chung", "sảnh", "hành lang"],
    "Spaciousness": ["rộng", "rộng rãi", "diện tích"],
    "Noise Condition": ["tiếng ồn", "cách âm", "ồn"],
    "Air Conditioning": ["điều hòa", "máy lạnh"],
    "Physical Security Infrastructure": ["an ninh", "khóa", "két"],
    "View & Surrounding Scenery": ["tầm nhìn", "view", "cảnh quan"],
    "Location & Surroundings": ["vị trí", "gần", "khu vực xung quanh"],
    "Overall Satisfaction": ["hài lòng", "trải nghiệm tổng thể", "thất vọng"],
    "Comfort & Relaxation": ["thư giãn", "thoải mái", "yên tĩnh"],
    "Value for Money": ["giá trị", "chi phí", "đáng tiền", "giá"],
    "Convenience": ["tiện lợi", "thuận tiện"],
    "Enjoyment": ["thích", "thú vị", "tận hưởng"],
    "Safety Perception": ["an toàn", "lo ngại"],
    "Brand Reputation": ["danh tiếng", "nổi tiếng", "uy tín", "hình ảnh", "nhận diện"],
    "Brand Trust": ["tin tưởng", "niềm tin"],
    "Luxury Perception": ["sang trọng", "cao cấp"],
    "Brand Consistency": ["nhất quán", "đúng như", "tiêu chuẩn", "hình ảnh", "phù hợp", "thực tế"],
    "Expectation Fulfillment": ["kỳ vọng", "mong đợi", "đáp ứng", "vượt kỳ vọng"],
    "Revisit Intention": ["quay lại", "trở lại", "lần sau"],
    "Recommendation Intention": ["giới thiệu", "recommend", "khuyên"],
    "Customer Preference": ["lựa chọn", "ưa thích"],
    "Loyalty Behavior": ["trung thành", "ủng hộ"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute business-oriented hotel summary metrics.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--top-k-clusters", type=int, default=5)
    parser.add_argument("--enable-semantic", action="store_true")
    parser.add_argument("--semantic-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--semantic-batch-size", type=int, default=64)
    parser.add_argument("--enable-bertscore", action="store_true")
    parser.add_argument("--bertscore-language", choices=["vi", "en"], default="vi")
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--out-prefix", default="")
    return parser.parse_args()


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize(value: Any) -> str:
    text = clean(value).lower()
    text = "".join(ch for ch in unicodedata.normalize("NFD", text) if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def tokenize(value: Any) -> list[str]:
    return WORD_RE.findall(normalize(value))


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def prf(overlap: int, pred_total: int, ref_total: int) -> tuple[float, float, float]:
    precision = overlap / pred_total if pred_total > 0 else 0.0
    recall = overlap / ref_total if ref_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0]
        for idx, ref in enumerate(b, start=1):
            cur.append(prev[idx - 1] + 1 if token == ref else max(prev[idx], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_prf(summary: str, reference: str) -> dict[str, float]:
    summary_tokens = tokenize(summary)
    reference_tokens = tokenize(reference)
    out = {
        "summary_token_count": len(summary_tokens),
        "reference_token_count": len(reference_tokens),
    }
    for n in (1, 2):
        pred = ngrams(summary_tokens, n)
        ref = ngrams(reference_tokens, n)
        overlap = sum((pred & ref).values())
        precision, recall, f1 = prf(overlap, sum(pred.values()), sum(ref.values()))
        out[f"rouge{n}_precision"] = precision
        out[f"rouge{n}_recall"] = recall
        out[f"rouge{n}_f1"] = f1
    lcs = lcs_len(summary_tokens, reference_tokens)
    precision, recall, f1 = prf(lcs, len(summary_tokens), len(reference_tokens))
    out["rouge_l_precision"] = precision
    out["rouge_l_recall"] = recall
    out["rouge_l_f1"] = f1
    ref_len = max(len(reference_tokens), 1)
    length_bound = min(len(summary_tokens), len(reference_tokens)) / ref_len
    out["length_bound_recall"] = length_bound
    out["rouge1_recall_efficiency"] = recall_safe(out["rouge1_recall"], length_bound)
    out["rouge2_recall_efficiency"] = recall_safe(out["rouge2_recall"], length_bound)
    out["rouge_l_recall_efficiency"] = recall_safe(out["rouge_l_recall"], length_bound)
    return out


def recall_safe(value: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return min(value / denominator, 1.0)


def parse_json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def aliases_for_scale(scale: str) -> list[str]:
    aliases = [scale, SCALE_VI.get(scale, "")]
    aliases.extend(MANUAL_ALIASES.get(scale, []))
    out = []
    for alias in aliases:
        key = normalize(alias)
        if key and key not in out:
            out.append(key)
    return out


def cluster_is_mentioned(summary_text: str, cluster: dict[str, Any]) -> bool:
    summary_norm = normalize(summary_text)
    scale = clean(cluster.get("measurement_scale")) or clean(cluster.get("cluster_label"))
    aliases = aliases_for_scale(scale)
    descriptors = [normalize(value) for value in cluster.get("descriptors", []) if clean(value)]
    candidates = aliases + [value for value in descriptors[:4] if len(value) >= 3]
    return any(candidate and candidate in summary_norm for candidate in candidates)


def combined_summary(row: pd.Series) -> str:
    return " ".join(
        clean(row.get(col, ""))
        for col in ("overall_aspect_summary", "positive_summary", "negative_summary", "neutral_summary")
        if clean(row.get(col, ""))
    )


def format_metrics(row: pd.Series) -> dict[str, Any]:
    aspect = clean(row.get("aspect"))
    expected_code = ASPECT_CODE.get(aspect, aspect[:3].upper())
    text = str(row.get("overall_aspect_summary", ""))
    stripped = text.strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    first_ok = bool(lines and lines[0].startswith(f"{expected_code}:"))
    labels = {"Pos:": False, "Neg:": False, "Neu:": False}
    for line in lines[1:]:
        for label in labels:
            if line.startswith(label):
                labels[label] = True
    count_mentions_after_first = len(re.findall(r"\b\d+\s+câu\b", "\n".join(lines[1:]), flags=re.IGNORECASE))
    return {
        "format_first_line_ok": first_ok,
        "format_pos_line_ok": labels["Pos:"],
        "format_neg_line_ok": labels["Neg:"],
        "format_neu_line_ok": labels["Neu:"],
        "format_counts_only_first_line_ok": count_mentions_after_first == 0,
        "format_all_ok": first_ok and all(labels.values()) and count_mentions_after_first == 0,
    }


def count_consistency(row: pd.Series, evidence_counts: dict[tuple[str, str, str], int]) -> dict[str, Any]:
    hotel_id = str(row.get("hotel_id", ""))
    aspect = str(row.get("aspect", ""))
    out = {}
    mismatches = 0
    for sentiment in SENTIMENTS:
        actual = int(row.get(f"{sentiment}_count", 0) or 0)
        expected = int(evidence_counts.get((hotel_id, aspect, sentiment), 0))
        ok = actual == expected
        out[f"{sentiment}_count_expected"] = expected
        out[f"{sentiment}_count_match"] = ok
        mismatches += 0 if ok else 1
    out["all_sentiment_counts_match"] = mismatches == 0
    return out


def top_cluster_coverage(row: pd.Series, top_k: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    all_weighted_numer = 0
    all_weighted_denom = 0
    all_matched = 0
    all_total = 0
    for sentiment in SENTIMENTS:
        clusters = parse_json_list(row.get(f"{sentiment}_clusters"))
        clusters = sorted(
            (item for item in clusters if isinstance(item, dict)),
            key=lambda item: -int(item.get("count", 0) or 0),
        )[:top_k]
        summary = clean(row.get(f"{sentiment}_summary", ""))
        matched = 0
        weighted_numer = 0
        weighted_denom = 0
        for cluster in clusters:
            count = int(cluster.get("count", 0) or 0)
            hit = cluster_is_mentioned(summary, cluster)
            matched += int(hit)
            weighted_numer += count if hit else 0
            weighted_denom += count
        total = len(clusters)
        out[f"{sentiment}_top_cluster_count"] = total
        out[f"{sentiment}_top_cluster_mention_rate"] = matched / total if total else 1.0
        out[f"{sentiment}_top_cluster_weighted_coverage"] = weighted_numer / weighted_denom if weighted_denom else 1.0
        all_weighted_numer += weighted_numer
        all_weighted_denom += weighted_denom
        all_matched += matched
        all_total += total
    out["top_cluster_mention_rate"] = all_matched / all_total if all_total else 1.0
    out["top_cluster_weighted_coverage"] = all_weighted_numer / all_weighted_denom if all_weighted_denom else 1.0
    return out


def build_pseudo_reference(row: pd.Series, top_k: int) -> str:
    parts = []
    for sentiment in SENTIMENTS:
        clusters = parse_json_list(row.get(f"{sentiment}_clusters"))
        clusters = sorted(
            (item for item in clusters if isinstance(item, dict)),
            key=lambda item: -int(item.get("count", 0) or 0),
        )[:top_k]
        for cluster in clusters:
            scale = clean(cluster.get("measurement_scale")) or clean(cluster.get("cluster_label"))
            label = SCALE_VI.get(scale, scale)
            descriptors = [clean(value) for value in cluster.get("descriptors", []) if clean(value)]
            parts.append(" ".join([label, *descriptors[:6]]))
    return "; ".join(part for part in parts if part)


def aggregate(values: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(float(numeric.mean()), 6),
        "median": round(float(numeric.median()), 6),
        "min": round(float(numeric.min()), 6),
        "max": round(float(numeric.max()), 6),
    }


def add_semantic_similarity(rows: pd.DataFrame, model_name: str, batch_size: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # noqa: BLE001
        rows["semantic_similarity"] = 0.0
        return rows, {"available": False, "reason": f"sentence_transformers import failed: {exc}"}
    try:
        model = SentenceTransformer(model_name)
        summaries = rows["summary_text_for_metric"].fillna("").astype(str).tolist()
        refs = rows["pseudo_reference"].fillna("").astype(str).tolist()
        summary_emb = model.encode(summaries, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        ref_emb = model.encode(refs, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)
        sims = np.sum(np.asarray(summary_emb) * np.asarray(ref_emb), axis=1)
        rows["semantic_similarity"] = sims
        return rows, {"available": True, "model": model_name}
    except Exception as exc:  # noqa: BLE001
        rows["semantic_similarity"] = 0.0
        return rows, {"available": False, "reason": str(exc), "model": model_name}



def add_bertscore(
    rows: pd.DataFrame,
    language: str,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        from bert_score import BERTScorer  # type: ignore
    except Exception as exc:  # noqa: BLE001
        rows["bertscore_precision"] = 0.0
        rows["bertscore_recall"] = 0.0
        rows["bertscore_f1"] = 0.0
        return rows, {"available": False, "reason": f"bert_score import failed: {exc}"}
    try:
        scorer = BERTScorer(lang=language, rescale_with_baseline=False)
        summaries = rows["summary_text_for_metric"].fillna("").astype(str).tolist()
        refs = rows["pseudo_reference"].fillna("").astype(str).tolist()
        precision, recall, f1 = scorer.score(summaries, refs, batch_size=max(1, int(batch_size)))
        rows["bertscore_precision"] = precision.detach().cpu().numpy()
        rows["bertscore_recall"] = recall.detach().cpu().numpy()
        rows["bertscore_f1"] = f1.detach().cpu().numpy()
        return rows, {"available": True, "language": language}
    except Exception as exc:  # noqa: BLE001
        rows["bertscore_precision"] = 0.0
        rows["bertscore_recall"] = 0.0
        rows["bertscore_f1"] = 0.0
        return rows, {"available": False, "reason": str(exc), "language": language}

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    results = Path(args.results_dir)
    run = args.run_name
    out_prefix = args.out_prefix or f"{run}_business_summary_metrics"

    final_path = results / f"{run}_final_summary_narrative.csv"
    evidence_path = results / f"{run}_cluster_evidence.csv"
    if not final_path.exists():
        raise FileNotFoundError(final_path)
    if not evidence_path.exists():
        raise FileNotFoundError(evidence_path)

    summary = pd.read_csv(final_path).fillna("")
    summary = summary[summary["aspect"].isin(ASPECTS)].copy()
    evidence = pd.read_csv(evidence_path, usecols=["hotel_id", "aspect", "sentiment", "count"]).fillna("")

    evidence_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for _, row in evidence.iterrows():
        key = (str(row["hotel_id"]), str(row["aspect"]), str(row["sentiment"]))
        evidence_counts[key] += int(row["count"] or 0)

    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        summary_text = combined_summary(row)
        pseudo_ref = build_pseudo_reference(row, args.top_k_clusters)
        metric_row = {
            "hotel_id": str(row["hotel_id"]),
            "aspect": str(row["aspect"]),
            "summary_text_for_metric": summary_text,
            "pseudo_reference": pseudo_ref,
            **format_metrics(row),
            **count_consistency(row, evidence_counts),
            **top_cluster_coverage(row, args.top_k_clusters),
            **rouge_prf(summary_text, pseudo_ref),
        }
        rows.append(metric_row)

    metrics = pd.DataFrame(rows)
    semantic_info = {"available": False, "reason": "not requested"}
    if args.enable_semantic:
        metrics, semantic_info = add_semantic_similarity(metrics, args.semantic_model, args.semantic_batch_size)
    bertscore_info = {"available": False, "reason": "not requested"}
    if args.enable_bertscore:
        metrics, bertscore_info = add_bertscore(metrics, args.bertscore_language, args.bertscore_batch_size)

    metric_cols = [
        "format_all_ok",
        "all_sentiment_counts_match",
        "top_cluster_mention_rate",
        "top_cluster_weighted_coverage",
        "rouge1_precision",
        "rouge1_recall",
        "rouge1_f1",
        "rouge2_precision",
        "rouge2_recall",
        "rouge2_f1",
        "rouge_l_precision",
        "rouge_l_recall",
        "rouge_l_f1",
        "length_bound_recall",
        "rouge1_recall_efficiency",
        "rouge2_recall_efficiency",
        "rouge_l_recall_efficiency",
    ]
    if "semantic_similarity" in metrics.columns:
        metric_cols.append("semantic_similarity")
    if "bertscore_f1" in metrics.columns:
        metric_cols.extend(["bertscore_precision", "bertscore_recall", "bertscore_f1"])

    by_aspect = {}
    for aspect, group in metrics.groupby("aspect"):
        by_aspect[aspect] = {"rows": int(len(group)), **{col: aggregate(group[col]) for col in metric_cols}}

    payload = {
        "metric_type": "business_oriented_hotel_summary_metrics",
        "inputs": {
            "final_summary_narrative": str(final_path.resolve()),
            "cluster_evidence": str(evidence_path.resolve()),
        },
        "notes": {
            "why": "Adds metrics that fit concise, reader-facing multi-review summaries better than raw ROUGE recall.",
            "pseudo_reference": f"Top {args.top_k_clusters} clusters per sentiment from the final-summary cluster payload.",
            "top_cluster_weighted_coverage": "Share of top-cluster evidence volume whose target/descriptors are mentioned in the matching sentiment summary.",
            "rouge_precision_f1": "ROUGE against a compact cluster pseudo-reference, not the full source corpus.",
            "rouge_recall_efficiency": "ROUGE recall divided by the maximum recall allowed by summary/reference length.",
            "semantic_similarity": "Cosine similarity between summary and pseudo-reference embeddings; optional.",
            "bertscore": "BERTScore between summary and compact cluster pseudo-reference; optional.",
        },
        "rows": int(len(metrics)),
        "semantic_similarity": semantic_info,
        "bertscore": bertscore_info,
        "overall": {col: aggregate(metrics[col]) for col in metric_cols},
        "by_aspect": by_aspect,
    }

    row_path = results / f"{out_prefix}_rows.csv"
    json_path = results / f"{out_prefix}.json"
    compact = metrics.drop(columns=["summary_text_for_metric", "pseudo_reference"])
    compact.to_csv(row_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows_csv": str(row_path), "json": str(json_path), "overall": payload["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
