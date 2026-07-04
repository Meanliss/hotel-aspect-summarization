# ABSA UASum

**K22-Khóa luận tốt nghiệp - 22280077 - Đỗ Trần Sáng / 22280102 - Trần Kiết Tường**

**Tên đề tài:** Aspect-Based Sentiment-Aware Opinion Summarization for Hotel Reviews

Curated submission package for the UASum practical pipeline.

This folder keeps the code, answer files, intermediate artifacts, and main
metrics needed to inspect Method 1 without dumping every historical run or raw
input datasets.

## Folder Layout

| Path | Purpose |
|---|---|
| `code/` | Main Python scripts for UASum, ABSA evaluation, summary evaluation, and SPACE evaluation. |
| `scripts/` | Reproducible run wrapper. |
| `answer_files/` | Canonical answer files for ABSA, summary, and SPACE outputs. |
| `results/intermediate/` | Required intermediate UASum artifacts. |
| `results/metrics/` | Main metrics and end-to-end report. |
| `results/space/` | SPACE ROUGE reports and comparison table. |
| `results/sample_hotel_10638/` | Compact trace sample for qualitative audit. |
| `docs/` | Short documentation for method, datasets, and SPACE results. |

## Required Intermediate UASum Artifacts

The following files are intentionally included because they represent the
traceable pipeline path from raw review to final summary:

- `hotel_review1_vi_100plus_llm_v3_full_processed_sentences.csv`
- `hotel_review1_vi_100plus_llm_v3_full_sentence_absa_processing_trace.csv`
- `hotel_review1_vi_100plus_llm_v3_full_cluster_taxonomy.csv`
- `hotel_review1_vi_100plus_llm_v3_full_cluster_evidence.csv`
- `hotel_review1_vi_100plus_llm_v3_full_cluster_sentiment_summary.csv`
- `hotel_review1_vi_100plus_llm_v3_full_cluster_three_sentiment_summary.csv`
- `hotel_review1_vi_100plus_llm_v3_full_aspect_summary_from_cluster_polished.csv`
- `hotel_review1_vi_100plus_llm_v3_full_final_summary_polished.csv`
- `hotel_review1_vi_100plus_llm_v3_full_hotel_overall_summary_narrative.csv`

## Main Results

Annotated ABSA evaluation:

| Method | Aspect accuracy | Sentiment accuracy | Joint accuracy |
|---|---:|---:|---:|
| Qwen framework-v3 | 0.8311 | 0.9724 | 0.8113 |

Full hotel-review run:

| Metric | Value |
|---|---:|
| Hotels with final summaries | 1,660 |
| Final ABSA segments | 1,595,680 |
| Cluster-assigned ABSA rows | 1,595,680 / 1,595,680 |
| Cluster evidence rows | 131,104 |
| Three-sentiment cluster rows | 61,799 |
| Final summary rows | 10,494 |

SPACE benchmark:

| Run | Type | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| `space_qwen_native_full_qwenroute` | UASum/Qwen SPACE-native | 0.2500 | 0.0572 | 0.1695 |
| `space_qwen_native_full_qwenroute_global` | Generated global summary | 0.3124 | 0.0532 | 0.1668 |

## Excluded

The package excludes cache databases, checkpoint pickle files, runtime logs,
temporary files, raw/input datasets, and old duplicated run variants. Large CSV
artifacts are kept because they are the required intermediate/output results.

## Manifest

See `MANIFEST.csv` for file sizes and SHA-256 checksums.
