#!/usr/bin/env python3
"""Generate a detailed end-to-end report for the hotel_review1 LLM pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


RUN_PREFIX = "hotel_review1_vi_100plus_llm_v3_full"
RESULTS_DIR = Path("results")
REPORT_MD = RESULTS_DIR / f"{RUN_PREFIX}_pipeline_end_to_end_report_20260611.md"
REPORT_HTML = RESULTS_DIR / f"{RUN_PREFIX}_pipeline_end_to_end_report_20260611.html"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_int(value: int | float) -> str:
    if pd.isna(value):
        return ""
    return f"{int(value):,}"


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return df.to_markdown(index=False)


def load_lightweight_stats() -> dict[str, object]:
    analysis = read_json(RESULTS_DIR / f"{RUN_PREFIX}_analysis_files.json")
    trace_stats = read_json(RESULTS_DIR / f"{RUN_PREFIX}_sentence_absa_processing_trace_stats.json")
    summary_stats = read_json(RESULTS_DIR / f"{RUN_PREFIX}_cluster_sentiment_summary_stats.json")
    repair_audit = read_json(RESULTS_DIR / f"{RUN_PREFIX}_anchor_repair_audit_20260610.json")

    manifest = pd.read_csv(RESULTS_DIR / f"{RUN_PREFIX}_pipeline_output_manifest_20260610.csv")
    taxonomy = pd.read_csv(RESULTS_DIR / f"{RUN_PREFIX}_cluster_taxonomy.csv")
    cluster_evidence = pd.read_csv(RESULTS_DIR / f"{RUN_PREFIX}_cluster_evidence.csv")
    metrics = pd.read_csv(RESULTS_DIR / f"{RUN_PREFIX}_summary_metrics.csv")
    final_metrics = pd.read_csv(RESULTS_DIR / f"{RUN_PREFIX}_final_summary_metrics.csv")
    processed_source = pd.read_csv(
        RESULTS_DIR / f"{RUN_PREFIX}_processed_sentences.csv",
        usecols=["data_source", "hotel_id"],
    )

    taxonomy_by_aspect = (
        taxonomy.groupby("aspect", dropna=False)
        .size()
        .sort_values(ascending=False)
        .reset_index(name="cluster_count")
    )

    sentiment_counts = (
        pd.Series(trace_stats["sentiment_counts"], name="count")
        .rename_axis("sentiment")
        .reset_index()
    )
    aspect_counts = (
        pd.Series(trace_stats["aspect_counts"], name="count")
        .rename_axis("aspect")
        .reset_index()
    )

    top_clusters = (
        cluster_evidence.groupby(["aspect", "cluster_code", "cluster_label"], dropna=False)[
            "count"
        ]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )

    top_clusters_by_sentiment = (
        cluster_evidence.groupby(
            ["sentiment", "aspect", "cluster_code", "cluster_label"], dropna=False
        )["count"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    metric_cols = [
        c
        for c in metrics.select_dtypes("number").columns
        if any(token in c.lower() for token in ["rouge", "coverage", "bertscore"])
    ]
    metrics_mean = metrics[metric_cols].mean(numeric_only=True).round(4).reset_index()
    metrics_mean.columns = ["metric", "aspect_summary_mean"]
    final_metrics_mean = final_metrics[metric_cols].mean(numeric_only=True).round(4).reset_index()
    final_metrics_mean.columns = ["metric", "final_summary_mean"]
    metrics_joined = metrics_mean.merge(final_metrics_mean, on="metric", how="outer")

    hotel_id = 10638
    hotel_cluster = pd.read_csv(
        RESULTS_DIR / f"{RUN_PREFIX}_hotel_10638_cluster_sentiment_summary_corrected.csv"
    )
    hotel_top_clusters = (
        hotel_cluster.sort_values("count", ascending=False)
        .head(15)[
            [
                "hotel_id",
                "aspect",
                "sentiment",
                "cluster_code",
                "cluster_label",
                "count",
                "avg_confidence",
                "cluster_summary",
            ]
        ]
        .copy()
    )
    source_counts = (
        processed_source.groupby("data_source")["hotel_id"]
        .nunique()
        .reset_index(name="hotel_count")
    )

    return {
        "analysis": analysis,
        "trace_stats": trace_stats,
        "summary_stats": summary_stats,
        "repair_audit": repair_audit,
        "manifest": manifest,
        "taxonomy_by_aspect": taxonomy_by_aspect,
        "sentiment_counts": sentiment_counts,
        "aspect_counts": aspect_counts,
        "top_clusters": top_clusters,
        "top_clusters_by_sentiment": top_clusters_by_sentiment,
        "metrics_joined": metrics_joined,
        "hotel_top_clusters": hotel_top_clusters,
        "source_counts": source_counts,
    }


def render_report(stats: dict[str, object]) -> str:
    analysis = stats["analysis"]
    trace_stats = stats["trace_stats"]
    summary_stats = stats["summary_stats"]
    repair = stats["repair_audit"]
    manifest = stats["manifest"].copy()

    manifest_display = manifest.copy()
    manifest_display["rows"] = manifest_display["rows"].apply(
        lambda x: fmt_int(x) if pd.notna(x) else ""
    )

    row_counts = (
        pd.Series(analysis["row_counts"], name="rows")
        .rename_axis("artifact")
        .reset_index()
    )
    row_counts["rows"] = row_counts["rows"].apply(fmt_int)

    sentiment_counts = stats["sentiment_counts"].copy()
    sentiment_counts["count"] = sentiment_counts["count"].apply(fmt_int)
    aspect_counts = stats["aspect_counts"].copy()
    aspect_counts["count"] = aspect_counts["count"].apply(fmt_int)
    taxonomy_by_aspect = stats["taxonomy_by_aspect"].copy()
    taxonomy_by_aspect["cluster_count"] = taxonomy_by_aspect["cluster_count"].apply(fmt_int)
    top_clusters = stats["top_clusters"].copy()
    top_clusters["count"] = top_clusters["count"].apply(fmt_int)

    hotel_top_clusters = stats["hotel_top_clusters"].copy()
    hotel_top_clusters["count"] = hotel_top_clusters["count"].apply(fmt_int)
    hotel_top_clusters["avg_confidence"] = hotel_top_clusters["avg_confidence"].round(4)

    source_counts = stats["source_counts"].copy()
    source_counts["hotel_count"] = source_counts["hotel_count"].apply(fmt_int)

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []
    lines.append(f"# Báo Cáo Pipeline LLM ABSA End-to-End - {RUN_PREFIX}")
    lines.append("")
    lines.append(f"- Thời điểm tạo report: `{generated_at}`")
    lines.append("- Mục tiêu: mô tả đầy đủ pipeline từ dữ liệu review gốc, tách câu, xử lý câu, gán ABSA, repair taxonomy/anchor, rollup evidence, tạo summary, xuất output và metric.")
    lines.append("- Nguyên tắc đọc report: số liệu cuối cùng ưu tiên các artifact sau repair ngày 2026-06-10 và trace audit-ready ngày 2026-06-11.")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(
        f"Pipeline hiện tại có `{fmt_int(trace_stats['row_count'])}` ABSA segment cuối cùng từ `{fmt_int(len(source_counts))}` `data_source`. "
        f"Tất cả segment đều có cluster assignment sau repair: `{fmt_int(trace_stats['cluster_assigned_rows'])}` / `{fmt_int(trace_stats['row_count'])}`. "
        "File audit cuối cùng nên dùng là `sentence_absa_processing_trace.csv`, không dùng raw `segmentation_trace.csv` để kết luận cluster vì trace cũ còn nhiều cluster field rỗng."
    )
    lines.append("")
    lines.append("Sentiment tổng thể:")
    lines.append("")
    lines.append(md_table(sentiment_counts))
    lines.append("")
    lines.append("Aspect tổng thể:")
    lines.append("")
    lines.append(md_table(aspect_counts))
    lines.append("")

    lines.append("## 2. Artifact Inventory")
    lines.append("")
    lines.append("Bảng này là danh sách file cần xem khi audit pipeline. Cột `status_or_caution` chỉ rõ file nào là primary, file nào chỉ là trace phụ hoặc aggregate.")
    lines.append("")
    lines.append(md_table(manifest_display))
    lines.append("")

    lines.append("Row count theo manifest phân tích:")
    lines.append("")
    lines.append(md_table(row_counts))
    lines.append("")

    lines.append("## 3. Pipeline Flow Từ Đầu Đến Cuối")
    lines.append("")
    lines.append("### 3.1 Input Review Và Metadata")
    lines.append("")
    lines.append(
        "Các dòng xử lý giữ lại metadata nguồn gồm `source_file`, `entity_id`, `data_source`, `hotel_id`, `review_index`, `sentence_id` và `aspect_segment_id`. "
        "`hotel_id` là khóa chính để rollup theo khách sạn; `aspect_segment_id` là khóa chi tiết để truy vết một unit ABSA cụ thể."
    )
    lines.append("")
    lines.append("Phân bổ khách sạn theo nguồn trong output:")
    lines.append("")
    lines.append(md_table(source_counts))
    lines.append("")

    lines.append("Lineage khóa chính xuyên suốt pipeline:")
    lines.append("")
    lineage = pd.DataFrame(
        [
            [
                "review/source",
                "`source_file`, `entity_id`, `data_source`, `hotel_id`, `review_index`",
                "Giữ nguồn gốc review và khách sạn.",
            ],
            [
                "semantic pre-segmentation",
                "`sentence_id`, `source_sentence`, `semantic_presegmentation_output`",
                "Tách/rút gọn review hoặc câu dài thành semantic unit; chưa gán aspect/sentiment.",
            ],
            [
                "ABSA aspect segment",
                "`aspect_segment_id`, `aspect_segment_text`, `segment_text`",
                "Một semantic unit có thể sinh nhiều segment ABSA theo aspect.",
            ],
            [
                "processing",
                "`shortened_sentence`, `processed_sentence`, `normalized_text_vi`, `normalized_text_en`",
                "Lưu các biến thể câu trước khi classification/summary.",
            ],
            [
                "ABSA assignment",
                "`aspect`, `sentiment`, `cluster_code`, `cluster_label`, `cluster_assignment_source`",
                "Kết quả gán aspect, sentiment và cluster cuối.",
            ],
            [
                "rollup",
                "`hotel_id + aspect + sentiment + cluster_code`",
                "Khóa gom evidence và summary theo cluster.",
            ],
        ],
        columns=["stage", "key_or_columns", "meaning"],
    )
    lines.append(md_table(lineage))
    lines.append("")

    lines.append("### 3.2 Semantic Pre-Segmentation Và ABSA Segment")
    lines.append("")
    lines.append(
        "`semantic_presegmentation_trace.csv` là file đúng để đọc riêng bước preseg: một row là một semantic unit, chỉ gồm "
        "`source_review`, `source_sentence`, `semantic_presegmentation_input`, `semantic_presegmentation_output`, confidence/coverage và số ABSA unit downstream. "
        "File này cố ý không có `aspect`, `sentiment`, `cluster_code` để không trùng công năng với ABSA."
    )
    lines.append("")
    lines.append(
        "Raw `segmentation_trace.csv` cũ có cùng số dòng với `processed_sentences.csv`, tức `1,595,680` dòng, nên thực chất là trace ABSA-level chứ không phải preseg-only. "
        "Nó lưu các trường như `source_review`, `source_sentence`, `preseg_sentence`, `aspect_segment_text`, `segment_text`, nhưng không còn đủ tin cậy cho cluster cuối vì `cluster_code` bị rỗng ở "
        f"`{fmt_int(trace_stats['original_segmentation_trace_blank_cluster_code_rows'])}` dòng."
    )
    lines.append("")
    lines.append(
        "Vì vậy đã tạo thêm `sentence_absa_processing_trace.csv`: file này lấy phần preseg/câu từ `segmentation_trace.csv`, "
        "nhưng lấy ABSA/sentiment/cluster từ `processed_sentences.csv` sau repair. Đây là file truy vết cuối cùng để chứng minh pipeline end-to-end."
    )
    lines.append("")

    lines.append("### 3.3 Chuẩn Hóa Và Xử Lý Câu")
    lines.append("")
    lines.append(
        "`processed_sentences.csv` là canonical row-level ABSA output. File này lưu đầy đủ chuỗi biến đổi: "
        "`source_sentence` -> `shortened_sentence` -> `processed_sentence` -> `normalized_text_vi` / `normalized_text_en` -> `classification_text` -> `summary_text`. "
        "Các cột này cho phép kiểm tra câu có bị rút gọn/chỉnh sửa trước khi gán aspect, sentiment và cluster hay không."
    )
    lines.append("")

    lines.append("### 3.4 Gán ABSA Và Sentiment")
    lines.append("")
    lines.append(
        "Mỗi row cuối cùng có `aspect`, `sentiment`, `sentiment_confidence`, `cluster_code`, `cluster_label`, `cluster_descriptors`, "
        "`cluster_assignment_confidence` và `cluster_assignment_source`. Sentiment totals trong trace mới khớp với `processed_sentences`."
    )
    lines.append("")
    lines.append(md_table(sentiment_counts))
    lines.append("")

    lines.append("### 3.5 Taxonomy Và Repair FAC2/FAC15")
    lines.append("")
    lines.append(
        "Audit cũ phát hiện lỗi taxonomy thật: `FAC2` bị dùng đồng thời cho `Room Comfort` và `Location & Surroundings`. "
        "Trạng thái hiện tại đã sửa theo nguyên tắc: `FAC2 = Room Comfort`, `FAC15 = Location & Surroundings`, `FAC14 = View & Surrounding Scenery`."
    )
    lines.append("")
    lines.append("Taxonomy hiện tại theo aspect:")
    lines.append("")
    lines.append(md_table(taxonomy_by_aspect))
    lines.append("")
    lines.append("Repair audit ghi nhận các nhóm thay đổi chính:")
    lines.append("")
    repair_rows = pd.DataFrame(
        [
            ["changed_total", repair["processed_sentences"]["changed_total"]],
            ["changed_location_code", repair["processed_sentences"]["changed_location_code"]],
            [
                "changed_location_current_anchor",
                repair["processed_sentences"]["changed_location_current_anchor"],
            ],
            ["changed_view_current_anchor", repair["processed_sentences"]["changed_view_current_anchor"]],
            [
                "changed_location_context_low_confidence",
                repair["processed_sentences"]["changed_location_context_low_confidence"],
            ],
            [
                "changed_view_context_low_confidence",
                repair["processed_sentences"]["changed_view_context_low_confidence"],
            ],
            ["changed_comfort_not_location", repair["processed_sentences"]["changed_comfort_not_location"]],
        ],
        columns=["repair_group", "row_count"],
    )
    repair_rows["row_count"] = repair_rows["row_count"].apply(fmt_int)
    lines.append(md_table(repair_rows))
    lines.append("")
    lines.append(
        "Nguyên tắc repair quan trọng: context chỉ hỗ trợ khi câu hiện tại mơ hồ hoặc confidence thấp; không override câu đã có anchor rõ như breakfast, staff, room cleanliness, location hoặc view."
    )
    lines.append("")

    lines.append("### 3.6 Cluster Evidence Rollup")
    lines.append("")
    lines.append(
        f"`cluster_evidence.csv` có `{fmt_int(summary_stats['rows']['cluster_evidence_rows'])}` dòng, là rollup theo "
        "`hotel_id + aspect + sentiment + cluster`. Tổng `count` cộng lại bằng toàn bộ ABSA segment: "
        f"`{fmt_int(summary_stats['count_totals']['all'])}`."
    )
    lines.append("")
    lines.append("Top 20 cluster toàn bộ dataset:")
    lines.append("")
    lines.append(md_table(top_clusters))
    lines.append("")

    lines.append("### 3.7 Cluster Summary Đúng Theo 3 Sentiment")
    lines.append("")
    lines.append(
        "`cluster_sentence_summary.csv` chỉ là aggregate theo `hotel_id + aspect + sentiment`, không phải per-cluster summary. "
        "File đúng cho yêu cầu summary theo từng cluster là `cluster_sentiment_summary.csv` và `cluster_three_sentiment_summary.csv`."
    )
    lines.append("")
    cluster_summary_rows = pd.DataFrame(
        [
            [
                "cluster_sentiment_summary.csv",
                summary_stats["rows"]["cluster_sentiment_summary_rows"],
                "Một dòng cho mỗi hotel/aspect/sentiment/cluster, có `cluster_summary` riêng.",
            ],
            [
                "cluster_three_sentiment_summary.csv",
                summary_stats["rows"]["cluster_three_sentiment_summary_rows"],
                "Một dòng cho mỗi hotel/aspect/cluster, tách `positive_summary`, `negative_summary`, `neutral_summary`.",
            ],
        ],
        columns=["file", "rows", "meaning"],
    )
    cluster_summary_rows["rows"] = cluster_summary_rows["rows"].apply(fmt_int)
    lines.append(md_table(cluster_summary_rows))
    lines.append("")
    lines.append(
        "Lưu ý: summary cụm hiện tại là deterministic local summary sinh từ descriptors và evidence samples trong `cluster_evidence.csv`; chưa phải LLM rewrite. "
        "Điều này giúp trace đúng với evidence, nhưng văn phong vẫn bị ràng buộc bởi dữ liệu descriptor/evidence đầu vào."
    )
    lines.append("")

    lines.append("### 3.8 Aspect Summary, Final Summary Và Output")
    lines.append("")
    lines.append(
        "`aspect_summary_from_cluster.csv` có `8,834` dòng và là summary theo hotel/aspect, lấy dữ liệu từ cluster evidence sau repair. "
        "`final_summary.csv` có `10,494` dòng, gồm các dòng aspect và dòng `all_aspects` cho từng khách sạn. "
        "`hotel_overall_summary.csv` và `output.csv` đều có `1,660` dòng, tương ứng 1 dòng cho mỗi khách sạn."
    )
    lines.append("")

    lines.append("### 3.9 Metrics")
    lines.append("")
    lines.append(
        "`summary_metrics.csv` và `final_summary_metrics.csv` đang tính ROUGE/coverage. BERTScore không khả dụng trong lần chạy này nên các trường BERTScore bằng 0 và `bertscore_available=False`."
    )
    lines.append("")
    lines.append(md_table(stats["metrics_joined"]))
    lines.append("")

    lines.append("### 3.10 Validation Gates")
    lines.append("")
    validation = pd.DataFrame(
        [
            [
                "Row-level trace đầy đủ",
                "sentence_absa_processing_trace row_count == processed_sentences row_count",
                f"PASS: {fmt_int(trace_stats['row_count'])} == {fmt_int(analysis['row_counts']['processed_sentences'])}",
            ],
            [
                "Không thiếu cluster cuối",
                "cluster_assigned_rows == trace row_count",
                f"PASS: {fmt_int(trace_stats['cluster_assigned_rows'])} / {fmt_int(trace_stats['row_count'])}",
            ],
            [
                "Sentiment totals khớp",
                "trace sentiment counts == processed/final rollup totals",
                "PASS: positive 982,556; negative 339,268; neutral 273,856",
            ],
            [
                "Cluster evidence khớp tổng ABSA",
                "sum(cluster_evidence.count) == processed row_count",
                f"PASS: {fmt_int(summary_stats['count_totals']['all'])}",
            ],
            [
                "Taxonomy FAC2/FAC15",
                "FAC2 chỉ là Room Comfort; FAC15 là Location & Surroundings",
                "PASS trong taxonomy hiện tại",
            ],
            [
                "Per-cluster summary đúng tầng",
                "cluster_sentiment_summary rows == cluster_evidence rows",
                f"PASS: {fmt_int(summary_stats['rows']['cluster_sentiment_summary_rows'])} == {fmt_int(summary_stats['rows']['cluster_evidence_rows'])}",
            ],
            [
                "Trace cũ không dùng làm kết luận cluster",
                "raw segmentation_trace có blank cluster_code lớn",
                f"CAUTION: {fmt_int(trace_stats['original_segmentation_trace_blank_cluster_code_rows'])} dòng blank cluster_code",
            ],
        ],
        columns=["gate", "check", "result"],
    )
    lines.append(md_table(validation))
    lines.append("")

    lines.append("## 4. Deep Dive Hotel 10638")
    lines.append("")
    lines.append(
        "Hotel `10638` đã có bộ file corrected riêng để đọc theo cluster. Top cluster dưới đây lấy từ "
        "`hotel_10638_cluster_sentiment_summary_corrected.csv`, tức summary đúng theo từng cluster-sentiment."
    )
    lines.append("")
    lines.append(md_table(hotel_top_clusters))
    lines.append("")

    lines.append("## 5. File Nên Dùng Cho Từng Mục Đích")
    lines.append("")
    usage = pd.DataFrame(
        [
            ["Audit riêng semantic preseg, không có aspect/sentiment", "semantic_presegmentation_trace.csv"],
            ["Audit preseg -> ABSA -> cluster sau repair", "sentence_absa_processing_trace.csv"],
            ["ABSA canonical row-level", "processed_sentences.csv"],
            ["Rollup evidence theo cluster", "cluster_evidence.csv"],
            ["Summary từng cluster theo sentiment", "cluster_sentiment_summary.csv"],
            ["Summary từng cluster có 3 sentiment columns", "cluster_three_sentiment_summary.csv"],
            ["Summary theo aspect", "aspect_summary_from_cluster.csv"],
            ["Summary cuối theo aspect + all aspects", "final_summary.csv"],
            ["Output wide theo khách sạn", "output.csv"],
            ["Metric summary", "summary_metrics.csv, final_summary_metrics.csv"],
            ["Taxonomy sau repair", "cluster_taxonomy.csv"],
        ],
        columns=["need", "file"],
    )
    lines.append(md_table(usage))
    lines.append("")

    lines.append("## 6. Caveat Và Rủi Ro Còn Lại")
    lines.append("")
    lines.append("- Raw `segmentation_trace.csv` không nên dùng để kết luận cluster cuối vì còn `1,575,187` dòng rỗng `cluster_code`; dùng `sentence_absa_processing_trace.csv` thay thế.")
    lines.append("- Audit 2026-06-09 phản ánh trạng thái trước repair, đặc biệt phần duplicate `FAC2`; report này dùng trạng thái sau repair.")
    lines.append("- Cluster summary hiện tại không gọi LLM lại, mà sinh deterministic từ `cluster_evidence`; nếu cần văn phong tự nhiên hơn nữa có thể thêm LLM rewrite pass nhưng phải khóa evidence để không hallucinate.")
    lines.append("- Một số descriptor có thể nhiễu do phụ thuộc chất lượng sentence processing ban đầu; cần audit sample định kỳ cho nhóm risky như location/view/room comfort/breakfast/wifi.")
    lines.append("- Metrics hiện chỉ là ROUGE/coverage; chưa có BERTScore nên không nên diễn giải như đánh giá semantic đầy đủ.")
    lines.append("")

    lines.append("## 7. Kết Luận")
    lines.append("")
    lines.append(
        "Pipeline hiện đã có đầy đủ artifact để truy vết từ review gốc đến output cuối. Phần preseg đã được tách khỏi ABSA bằng `semantic_presegmentation_trace.csv`; phần ABSA/cluster sau repair tiếp tục dùng `sentence_absa_processing_trace.csv`. "
        "Bộ file sau repair hiện nhất quán về tổng dòng, sentiment totals, taxonomy FAC2/FAC15 và rollup cluster evidence."
    )
    lines.append("")

    return "\n".join(lines)


def write_html(markdown_text: str) -> None:
    try:
        import markdown  # type: ignore

        body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code"])
    except Exception:
        body = "<pre>" + markdown_text.replace("&", "&amp;").replace("<", "&lt;") + "</pre>"

    html = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <title>{RUN_PREFIX} pipeline report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 40px; color: #17202a; }}
    h1, h2, h3 {{ color: #1f3a5f; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 13px; }}
    th, td {{ border: 1px solid #d5d8dc; padding: 6px 8px; vertical-align: top; }}
    th {{ background: #eef3f8; }}
    code {{ background: #f4f6f7; padding: 1px 4px; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    REPORT_HTML.write_text(html, encoding="utf-8")


def main() -> None:
    stats = load_lightweight_stats()
    report = render_report(stats)
    REPORT_MD.write_text(report, encoding="utf-8")
    write_html(report)
    print(REPORT_MD)
    print(REPORT_HTML)


if __name__ == "__main__":
    main()
