#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import unicodedata
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)


VIETNAMESE_ALIASES = {
    "FAC_ROOM": [
        "phong", "phong oc", "giuong", "nem", "dem", "chan", "goi", "ngu",
        "ban cong", "rong", "sach", "thoai mai", "am cung", "private",
    ],
    "FAC_BATH": [
        "phong tam", "nha tam", "toilet", "bon tam", "voi sen", "nuoc nong",
        "ap luc nuoc", "ve sinh", "bon cau",
    ],
    "FAC_INTERIOR": [
        "noi that", "trang tri", "thiet ke", "decor", "khong gian",
    ],
    "FAC_BUILDING": [
        "co so ha tang", "toa nha", "thang may", "hanh lang", "san",
    ],
    "FAC_ENV": [
        "yen tinh", "on", "on ao", "cach am", "moi truong", "khong khi",
    ],
    "FAC_CLIMATE": [
        "dieu hoa", "may lanh", "nong", "lanh", "quat", "thong gio",
    ],
    "FAC_SECURITY": [
        "an toan", "bao ve", "khoa", "camera", "an ninh",
    ],
    "FAC_VIEW_LOCATION": [
        "vi tri", "gan", "xa", "trung tam", "view", "tam nhin",
        "dia diem", "di lai", "moc chau", "bien", "nui", "ho",
    ],
    "AM_WIFI": [
        "wifi", "wi fi", "internet", "mang", "ket noi", "song",
    ],
    "AM_FOOD": [
        "do an", "mon an", "bua sang", "an sang", "nha hang", "cafe",
        "ca phe", "buffet", "ngon", "do uong",
    ],
    "AM_POOL": [
        "ho boi", "be boi", "pool", "jacuzzi",
    ],
    "AM_WELLNESS": [
        "spa", "massage", "gym", "phong tap", "xong hoi",
    ],
    "AM_ENT": [
        "giai tri", "karaoke", "tivi", "tv", "netflix", "choi",
    ],
    "AM_TRANSPORT": [
        "dua don", "xe", "taxi", "do xe", "bai xe", "parking", "san bay",
    ],
    "AM_ROOM_UTIL": [
        "tu lanh", "am dun", "may say", "mini bar", "bep", "tivi", "tv",
    ],
    "AM_UTILITY": [
        "giat ui", "giat la", "laundry",
    ],
    "SER_ATTITUDE": [
        "nhan vien", "le tan", "chu", "chi chu", "anh chu", "phuc vu",
        "than thien", "nhiet tinh", "chu dao", "thai do", "lich su",
    ],
    "SER_OPERATION": [
        "check in", "check out", "dat phong", "thu tuc", "nhan phong",
    ],
    "SER_SUPPORT": [
        "ho tro", "giup", "giai quyet", "yeu cau", "phan nan",
    ],
    "SER_COMM": [
        "giao tiep", "tieng anh", "noi chuyen", "tra loi", "nhan tin",
        "goi dien",
    ],
    "EXP_OVERALL": [
        "trai nghiem", "ky nghi", "chuyen di", "hai long", "tuyet voi",
        "rat thich", "that vong",
    ],
    "EXP_EMOTION": [
        "thu gian", "de chiu", "thoai mai", "am cung", "binh yen",
    ],
    "EXP_VALUE": [
        "gia", "dang tien", "gia ca", "re", "dat", "chi phi", "value",
    ],
    "EXP_SAFETY": [
        "an toan", "yen tam", "nguy hiem",
    ],
    "BRA_REPUTE": [
        "thuong hieu", "uy tin", "tieu chuan",
    ],
    "BRA_LUXURY": [
        "sang trong", "cao cap", "luxury", "5 sao", "nam sao",
    ],
    "LOY_RETURN": [
        "quay lai", "tro lai", "lan sau", "se den",
    ],
    "LOY_RECOMMEND": [
        "gioi thieu", "recommend", "de cu", "khuyen",
    ],
    "LOY_PREFERENCE": [
        "yeu thich", "ua thich", "favorite", "quen thuoc",
    ],
}

ENGLISH_MARKERS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "had", "has", "have", "he", "her", "his", "hotel", "i", "in", "is",
    "it", "its", "me", "my", "not", "of", "on", "or", "our", "she",
    "staff", "stay", "room", "rooms", "the", "their", "there", "they",
    "this", "to", "was", "we", "were", "with", "you",
}

ENGLISH_HOTEL_MARKERS = {
    "amazing", "bathroom", "bed", "breakfast", "central", "clean",
    "comfortable", "convenient", "delicious", "excellent", "friendly",
    "good", "great", "helpful", "location", "nice", "pool", "restaurant",
    "service", "shower", "view", "wifi",
}

ENGLISH_DETECTOR_RULE = (
    "English-like if repaired text has at least 8 alphabetic chars, "
    "ASCII-letter ratio >= 0.86, at least 3 normalized tokens, and either "
    "2 common English marker hits, or 1 marker plus 1 hotel-domain hit, "
    "or 2 hotel-domain hits with ASCII ratio >= 0.94."
)


def strip_accents(value):
    normalized = unicodedata.normalize("NFD", value)
    without_marks = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return without_marks.replace("đ", "d").replace("Đ", "D")


def repair_mojibake(value):
    markers = ("Ã", "Ä", "Â", "â€", "áº", "á»", "Æ")
    if not value or not any(marker in value for marker in markers):
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value
    original_markers = sum(value.count(marker) for marker in markers)
    repaired_markers = sum(repaired.count(marker) for marker in markers)
    if repaired_markers < original_markers:
        return repaired
    return value


def normalize(value):
    value = repair_mojibake(value)
    value = strip_accents(value.lower())
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def english_detection_details(value):
    value = repair_mojibake(value)
    letters = [ch for ch in value if ch.isalpha()]
    ascii_letters = [ch for ch in letters if "a" <= ch.lower() <= "z"]
    ascii_ratio = len(ascii_letters) / float(len(letters)) if letters else 0.0
    tokens = normalize(value).split()
    marker_hits = sum(1 for token in tokens if token in ENGLISH_MARKERS)
    hotel_hits = sum(1 for token in tokens if token in ENGLISH_HOTEL_MARKERS)

    is_english = False
    reason = "rejected"
    if len(letters) < 8:
        reason = "too_few_letters"
    elif ascii_ratio < 0.86:
        reason = "low_ascii_letter_ratio"
    elif len(tokens) < 3:
        reason = "too_few_tokens"
    elif marker_hits >= 2:
        is_english = True
        reason = "common_english_markers"
    elif marker_hits >= 1 and hotel_hits >= 1:
        is_english = True
        reason = "english_marker_plus_hotel_term"
    elif hotel_hits >= 2 and ascii_ratio >= 0.94:
        is_english = True
        reason = "hotel_terms_high_ascii"
    else:
        reason = "not_enough_english_markers"

    return {
        "is_english": is_english,
        "reason": reason,
        "ascii_letter_ratio": ascii_ratio,
        "letter_count": len(letters),
        "token_count": len(tokens),
        "marker_hits": marker_hits,
        "hotel_hits": hotel_hits,
        "repaired_text": value,
    }


def is_probably_english(value):
    return english_detection_details(value)["is_english"]


def split_sentences(review):
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", review.strip())
    return [part.strip() for part in parts if part.strip()]


def entity_from_ref(ref_id):
    match = re.match(r"(.+)_\d+$", ref_id or "")
    return match.group(1) if match else (ref_id or "unknown_entity")


def load_taxonomy(path):
    with open(path, "r", encoding="utf-8", newline="") as fin:
        return [row for row in csv.DictReader(fin, delimiter="\t") if row.get("CODE")]


def split_terms(value):
    return [term.strip() for term in (value or "").split(",") if term.strip()]


def build_aspect_terms(taxonomy, include_vietnamese_aliases=True):
    aspect_terms = {}
    for row in taxonomy:
        code = row["CODE"]
        terms = []
        terms.extend(split_terms(row.get("MEASUREMENT_SCALE")))
        terms.extend(split_terms(row.get("ASPECT_KEYWORDS")))
        terms.extend(split_terms(row.get("POSITIVE_SENTIMENT_KEYWORDS")))
        terms.extend(split_terms(row.get("NEGATIVE_SENTIMENT_KEYWORDS")))
        terms.extend(split_terms(row.get("NEUTRAL_SENTIMENT_KEYWORDS")))
        if include_vietnamese_aliases:
            terms.extend(VIETNAMESE_ALIASES.get(code, []))
        normalized_terms = []
        for term in terms:
            term = normalize(term)
            if term and term not in normalized_terms:
                normalized_terms.append(term)
        aspect_terms[code] = normalized_terms
    return aspect_terms


def build_term_matcher(aspect_terms):
    term_to_codes = defaultdict(list)
    for code, terms in aspect_terms.items():
        for term in terms:
            term_to_codes[term].append(code)
    terms = sorted(term_to_codes.keys(), key=len, reverse=True)
    pattern = re.compile(r"(?<!\S)({0})(?!\S)".format("|".join(re.escape(term) for term in terms)))
    return pattern, term_to_codes


def read_reviews(path, max_rows=None, language="all"):
    reviews = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        for row_idx, row in enumerate(reader):
            if max_rows is not None and row_idx >= max_rows:
                break
            text = repair_mojibake((row.get("review") or "").strip())
            if not text:
                continue
            sentences = split_sentences(text)
            if language == "english":
                if not is_probably_english(text):
                    continue
                sentences = [
                    sentence for sentence in sentences
                    if is_probably_english(sentence)
                ]
                if not sentences:
                    continue
            ref_id = row.get("ref_id") or "review_{0}".format(row_idx)
            reviews.append(
                {
                    "review_id": ref_id,
                    "entity_id": entity_from_ref(ref_id),
                    "sentences": sentences,
                    "review": text,
                }
            )
    return reviews


def match_aspects(sentence, matcher):
    normalized_sentence = normalize(sentence)
    pattern, term_to_codes = matcher
    matches = set()
    for match in pattern.finditer(normalized_sentence):
        for code in term_to_codes[match.group(1)]:
            matches.add(code)
    return list(matches)


def dedupe_sentences(sentences):
    seen = set()
    deduped = []
    for sentence in sentences:
        key = normalize(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return deduped


def top_centroid_sentences(sentences, max_sentences=3):
    if not sentences:
        return [], 0.0
    if len(sentences) == 1:
        return sentences, 1.0
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    matrix = vectorizer.fit_transform(sentences)
    centroid = np.asarray(matrix.mean(axis=0))
    scores = cosine_similarity(matrix, centroid).ravel()
    top_indices = np.argsort(-scores)[:max_sentences]
    return [sentences[idx] for idx in top_indices], float(np.mean(scores[top_indices]))


def cosine_between_texts(left, right):
    if not left or not right:
        return 0.0
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    matrix = vectorizer.fit_transform([left, right])
    return float(cosine_similarity(matrix[0], matrix[1])[0, 0])


def score_reviews(reviews,
                  taxonomy,
                  max_summary_sentences=3,
                  max_evidence_per_aspect=20000,
                  include_vietnamese_aliases=True,
                  language="all"):
    aspect_terms = build_aspect_terms(
        taxonomy,
        include_vietnamese_aliases=include_vietnamese_aliases)
    matcher = build_term_matcher(aspect_terms)
    aspect_sentences = defaultdict(list)
    unique_seen = defaultdict(set)
    matched_counts = defaultdict(int)
    total_sentences = 0
    for review in reviews:
        for sentence in review["sentences"]:
            total_sentences += 1
            matches = match_aspects(sentence, matcher)
            sentence_key = normalize(sentence)
            for code in matches:
                matched_counts[code] += 1
                if sentence_key not in unique_seen[code]:
                    unique_seen[code].add(sentence_key)
                    if len(aspect_sentences[code]) < max_evidence_per_aspect:
                        aspect_sentences[code].append(sentence)

    rows = []
    summaries = {}
    denominator = 0.0
    unique_by_code = {}
    for row in taxonomy:
        code = row["CODE"]
        unique_count = len(unique_seen.get(code, set()))
        unique_by_code[code] = unique_count
        denominator += math.log1p(unique_count)

    for row in taxonomy:
        code = row["CODE"]
        unique_sents = aspect_sentences.get(code, [])
        summary_sentences, cec = top_centroid_sentences(
            unique_sents, max_sentences=max_summary_sentences
        )
        summary_text = " ".join(summary_sentences)
        summaries[code] = summary_sentences
        unique_count = unique_by_code[code]
        weight = math.log1p(unique_count) / denominator if denominator else 0.0
        asc = weight * cosine_between_texts(summary_text, " ".join(unique_sents))
        rows.append(
            {
                "aspect": code,
                "group": row["ASPECT"],
                "measurement_scale": row["MEASUREMENT_SCALE"],
                "matched_sentences": matched_counts.get(code, 0),
                "unique_opinions": unique_count,
                "evidence_used_for_summary": len(unique_sents),
                "cluster_weight": weight,
                "cec": cec,
                "asc_contribution": asc,
                "summary": summary_sentences,
            }
        )

    overall = {
        "review_count": len(reviews),
        "sentence_count": total_sentences,
        "aspect_summary_cover": sum(row["asc_contribution"] for row in rows),
        "macro_cec": float(np.mean([row["cec"] for row in rows])) if rows else 0.0,
        "weighted_cec": sum(row["cluster_weight"] * row["cec"] for row in rows),
        "matched_aspects": sum(1 for row in rows if row["unique_opinions"] > 0),
        "aspect_count": len(rows),
        "rouge_note": (
            "ROUGE-1/2/L from the SemAE paper requires human gold summaries; "
            "this CSV has reviews only, so this trial reports image-formula CEC/ASC instead."
        ),
        "scoring_note": (
            "unique_opinions and cluster_weight are counted on the whole input; "
            "summary/CEC/ASC use a capped representative evidence set per aspect."
        ),
        "language_filter": language,
        "include_vietnamese_aliases": include_vietnamese_aliases,
    }
    return overall, rows


def write_outputs(outdir, input_csv, overall, rows, output_prefix=None):
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_csv))[0]
    prefix = output_prefix or (base + "_trial")
    json_path = os.path.join(outdir, prefix + "_scores.json")
    csv_path = os.path.join(outdir, prefix + "_scores.csv")
    summary_path = os.path.join(outdir, prefix + "_summaries.txt")

    payload = {"input_csv": os.path.abspath(input_csv), "overall": overall, "aspects": rows}
    with open(json_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=False, indent=2)
        fout.write("\n")

    fieldnames = [
        "aspect",
        "group",
        "measurement_scale",
        "matched_sentences",
        "unique_opinions",
        "evidence_used_for_summary",
        "cluster_weight",
        "cec",
        "asc_contribution",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})

    with open(summary_path, "w", encoding="utf-8") as fout:
        for row in sorted(rows, key=lambda item: item["asc_contribution"], reverse=True):
            if not row["summary"]:
                continue
            fout.write("[{0}] {1}\n".format(row["aspect"], row["measurement_scale"]))
            for sent in row["summary"]:
                fout.write("- {0}\n".format(sent))
            fout.write("\n")
    return json_path, csv_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Trial-score a hotel review CSV using HASOS aspects and the CEC/ASC formulas from the provided notes."
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.join(WORKSPACE_ROOT, "hotel_review1.csv"),
        help="CSV containing ref_id and review columns.",
    )
    parser.add_argument(
        "--taxonomy",
        default=os.path.join(REPO_ROOT, "data", "hasos", "aspect_taxonomy.tsv"),
        help="HASOS taxonomy TSV.",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(REPO_ROOT, "outputs", "hasos_file_trials"),
        help="Output directory.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for quick smoke runs.",
    )
    parser.add_argument(
        "--max-summary-sentences",
        type=int,
        default=3,
        help="Representative summary sentences per aspect.",
    )
    parser.add_argument(
        "--max-evidence-per-aspect",
        type=int,
        default=20000,
        help="Maximum unique sentences retained per aspect for summary and cosine scoring.",
    )
    parser.add_argument(
        "--language",
        choices=["all", "english"],
        default="all",
        help="Optional language filter applied before scoring.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Prefix for output files, e.g. 'file' creates file_scores.json.",
    )
    parser.add_argument(
        "--include-vietnamese-aliases",
        choices=["true", "false"],
        default="true",
        help="Whether to include Vietnamese alias keywords in aspect matching.",
    )
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    include_vietnamese_aliases = args.include_vietnamese_aliases == "true"
    reviews = read_reviews(args.input_csv,
                           max_rows=args.max_rows,
                           language=args.language)
    overall, rows = score_reviews(
        reviews,
        taxonomy,
        max_summary_sentences=args.max_summary_sentences,
        max_evidence_per_aspect=args.max_evidence_per_aspect,
        include_vietnamese_aliases=include_vietnamese_aliases,
        language=args.language,
    )
    json_path, csv_path, summary_path = write_outputs(
        args.outdir,
        args.input_csv,
        overall,
        rows,
        output_prefix=args.output_prefix,
    )

    print(json.dumps(overall, ensure_ascii=False, indent=2))
    print("json={0}".format(json_path))
    print("csv={0}".format(csv_path))
    print("summaries={0}".format(summary_path))
    top_rows = sorted(rows, key=lambda row: row["asc_contribution"], reverse=True)[:10]
    print("top_aspects=")
    for row in top_rows:
        print(
            "{0}\tunique={1}\tweight={2:.4f}\tcec={3:.4f}\tasc={4:.4f}".format(
                row["aspect"],
                row["unique_opinions"],
                row["cluster_weight"],
                row["cec"],
                row["asc_contribution"],
            )
        )


if __name__ == "__main__":
    main()
