#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

QWEN_BASE_URL="${QWEN_BASE_URL:-http://localhost:8000/v1}"
QWEN_API_KEY="${QWEN_API_KEY:-local-dev-key}"
INPUT_CSV="${INPUT_CSV:-external/hotel_review1.csv}"
RUN_NAME="${RUN_NAME:-hotel_review1_vi_100plus_llm_v3_full}"
RESULTS_DIR="${RESULTS_DIR:-results/full/${RUN_NAME}}"
CACHE_DB="${CACHE_DB:-results/cache/${RUN_NAME}_cache.sqlite}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-results/cache/${RUN_NAME}_checkpoint.pkl}"

mkdir -p "$RESULTS_DIR" "$(dirname "$CACHE_DB")"

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "[ERROR] Missing input CSV: $INPUT_CSV" >&2
  echo "Set INPUT_CSV=/path/to/hotel_review1.csv before running this script." >&2
  exit 1
fi

if ! curl -sf -H "Authorization: Bearer ${QWEN_API_KEY}" "${QWEN_BASE_URL}/models" >/dev/null; then
  echo "[ERROR] Qwen server is not reachable at ${QWEN_BASE_URL}" >&2
  exit 1
fi

BERTSCORE_ARGS=()
if [[ "${ENABLE_BERTSCORE:-0}" != "1" ]]; then
  BERTSCORE_ARGS+=(--disable-bertscore)
fi

python hotel_aspect_sentiment_pipeline.py \
  --inputs "$INPUT_CSV" \
  --strategy sentence-qwen \
  --pre-segmentation "${PRE_SEGMENTATION:-semantic-qwen}" \
  --cluster-assignment "${CLUSTER_ASSIGNMENT:-llm}" \
  --summary-language vi \
  --sentiment-language en \
  --bertscore-language vi \
  --min-reviews-per-hotel "${MIN_REVIEWS_PER_HOTEL:-100}" \
  --max-rows-per-source "${MAX_ROWS_PER_SOURCE:-0}" \
  --max-hotels-per-source "${MAX_HOTELS_PER_SOURCE:-0}" \
  --processed-sentences-csv "${RESULTS_DIR}/${RUN_NAME}_processed_sentences.csv" \
  --output-csv "${RESULTS_DIR}/${RUN_NAME}_output.csv" \
  --aspect-output-dir "${RESULTS_DIR}/${RUN_NAME}_aspect_outputs" \
  --preseg-metrics-csv "${RESULTS_DIR}/${RUN_NAME}_preseg_information_coverage.csv" \
  --preseg-metrics-json "${RESULTS_DIR}/${RUN_NAME}_preseg_information_coverage.json" \
  --cluster-evidence-csv "${RESULTS_DIR}/${RUN_NAME}_cluster_evidence.csv" \
  --cluster-evidence-json "${RESULTS_DIR}/${RUN_NAME}_cluster_evidence.json" \
  --final-summary-csv "${RESULTS_DIR}/${RUN_NAME}_final_summary.csv" \
  --final-summary-json "${RESULTS_DIR}/${RUN_NAME}_final_summary.json" \
  --summary-metrics-csv "${RESULTS_DIR}/${RUN_NAME}_summary_metrics.csv" \
  --summary-metrics-json "${RESULTS_DIR}/${RUN_NAME}_summary_metrics.json" \
  --final-summary-metrics-csv "${RESULTS_DIR}/${RUN_NAME}_final_summary_metrics.csv" \
  --final-summary-metrics-json "${RESULTS_DIR}/${RUN_NAME}_final_summary_metrics.json" \
  --cache-db "$CACHE_DB" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --checkpoint-every-chunks 1 \
  --chunk-size "${CHUNK_SIZE:-8000}" \
  --batch-size "${BATCH_SIZE:-24}" \
  --qwen-extract-workers "${QWEN_EXTRACT_WORKERS:-24}" \
  --qwen-sentiment-workers "${QWEN_SENTIMENT_WORKERS:-16}" \
  --qwen-cluster-workers "${QWEN_CLUSTER_WORKERS:-16}" \
  --cluster-assignment-max-output-tokens "${CLUSTER_ASSIGNMENT_MAX_OUTPUT_TOKENS:-9000}" \
  --cluster-assignment-min-confidence "${CLUSTER_ASSIGNMENT_MIN_CONFIDENCE:-0.55}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-5000}" \
  --semantic-max-output-tokens "${SEMANTIC_MAX_OUTPUT_TOKENS:-3000}" \
  --summary-max-output-tokens "${SUMMARY_MAX_OUTPUT_TOKENS:-3000}" \
  --final-summary-max-output-tokens "${FINAL_SUMMARY_MAX_OUTPUT_TOKENS:-3000}" \
  --max-sentence-chars "${MAX_SENTENCE_CHARS:-700}" \
  --semantic-max-review-chars "${SEMANTIC_MAX_REVIEW_CHARS:-1800}" \
  --semantic-min-source-precision "${SEMANTIC_MIN_SOURCE_PRECISION:-0.72}" \
  --semantic-max-units-per-review "${SEMANTIC_MAX_UNITS_PER_REVIEW:-24}" \
  --final-summary-samples-per-sentiment "${FINAL_SUMMARY_SAMPLES_PER_SENTIMENT:-12}" \
  --final-summary-sample-chars "${FINAL_SUMMARY_SAMPLE_CHARS:-220}" \
  --final-summary-max-clusters "${FINAL_SUMMARY_MAX_CLUSTERS:-8}" \
  --final-summary-cluster-samples "${FINAL_SUMMARY_CLUSTER_SAMPLES:-2}" \
  --final-summary-cluster-descriptors "${FINAL_SUMMARY_CLUSTER_DESCRIPTORS:-12}" \
  --summary-batch-size "${SUMMARY_BATCH_SIZE:-1}" \
  --summary-workers "${SUMMARY_WORKERS:-8}" \
  --final-summary-batch-size "${FINAL_SUMMARY_BATCH_SIZE:-1}" \
  --final-summary-workers "${FINAL_SUMMARY_WORKERS:-8}" \
  --timeout-sec "${TIMEOUT_SEC:-240}" \
  --qwen-base-url "$QWEN_BASE_URL" \
  --qwen-api-key "$QWEN_API_KEY" \
  "${BERTSCORE_ARGS[@]}"
