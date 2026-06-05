#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/workspace/SemAE}"
cd "$REMOTE_ROOT"

TRAIN_RUN_ID="${TRAIN_RUN_ID:-space_600x4_fast}"
INFER_RUN_ID="${INFER_RUN_ID:-space_hasos_600x4_e4}"
MAX_ENTITIES="${MAX_ENTITIES:-600}"
EPOCHS="${EPOCHS:-4}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-3}"
GPU="${GPU:-0}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MAX_TOKENS="${MAX_TOKENS:-40}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

LOG_DIR="$REMOTE_ROOT/logs"
CKPT_DIR="$REMOTE_ROOT/checkpoints/$TRAIN_RUN_ID"
TRACE_JSONL="$LOG_DIR/${INFER_RUN_ID}_pipeline_trace.jsonl"
TRACE_MD="$LOG_DIR/${INFER_RUN_ID}_pipeline_trace.md"
PIPELINE_LOG="$LOG_DIR/${INFER_RUN_ID}_remote_pipeline.log"
SPM="$REMOTE_ROOT/data/sentencepiece/space_unigram_32k.model"
MODEL="$CKPT_DIR/${TRAIN_RUN_ID}_${EPOCHS}_model.pt"

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$REMOTE_ROOT/data/space/json" "$REMOTE_ROOT/data/sentencepiece" "$REMOTE_ROOT/outputs"
: > "$TRACE_JSONL"

trace() {
  local stage="$1"
  local status="$2"
  local detail="${3:-}"
  "$PYTHON_BIN" - "$TRACE_JSONL" "$INFER_RUN_ID" "$stage" "$status" "$detail" <<'PY'
import json, sys, time
path, run_id, stage, status, detail = sys.argv[1:6]
row = {
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "run_id": run_id,
    "stage": stage,
    "status": status,
    "detail": detail,
}
with open(path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

render_trace_md() {
  "$PYTHON_BIN" - "$TRACE_JSONL" "$TRACE_MD" "$INFER_RUN_ID" <<'PY'
import json, sys
jsonl, md, run_id = sys.argv[1:4]
rows = []
with open(jsonl, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))
lines = [
    f"# SPACE SemAE -> HASOS pipeline trace: `{run_id}`",
    "",
    "| Time | Stage | Status | Detail |",
    "| --- | --- | --- | --- |",
]
for row in rows:
    lines.append("| {time} | `{stage}` | **{status}** | {detail} |".format(**row))
with open(md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
PY
}

exec > >(tee -a "$PIPELINE_LOG") 2>&1

trap 'trace failed fail "line=$LINENO"; render_trace_md' ERR

trace pipeline_start ok "train=$TRAIN_RUN_ID infer=$INFER_RUN_ID max_entities=$MAX_ENTITIES epochs=$EPOCHS"
echo "=== pipeline start $(date -Is) ==="
echo "REMOTE_ROOT=$REMOTE_ROOT"
echo "TRAIN_RUN_ID=$TRAIN_RUN_ID INFER_RUN_ID=$INFER_RUN_ID MAX_ENTITIES=$MAX_ENTITIES EPOCHS=$EPOCHS"

trace gpu_check start ""
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON_BIN" - <<'PY'
try:
    import torch
except Exception as exc:
    print("torch preinstall missing:", repr(exc))
    raise SystemExit(0)
import torch
print("python torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
trace gpu_check ok ""

trace install_deps start "pip install python deps"
PIP_BREAK_FLAG=""
if "$PYTHON_BIN" -m pip help install 2>/dev/null | grep -q -- "--break-system-packages"; then
  PIP_BREAK_FLAG="--break-system-packages"
fi
"$PYTHON_BIN" -m pip install $PIP_BREAK_FLAG --upgrade pip -q || true
"$PYTHON_BIN" -m pip install $PIP_BREAK_FLAG torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 -q
"$PYTHON_BIN" -m pip install $PIP_BREAK_FLAG sentencepiece tqdm numpy scipy scikit-learn nltk tensorboard gdown -q
"$PYTHON_BIN" - <<'PY'
import torch
print("python torch after install", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
trace install_deps ok ""

trace prepare_hasos start ""
"$PYTHON_BIN" scripts/prepare_hasos.py \
  --input-json data/hasos/hasos_summ.json \
  --output-json data/hasos/hasos_summ.prepared.json \
  --taxonomy-json data/hasos/aspect_taxonomy.json \
  --seeds-dir data/seeds_hasos
mv data/hasos/hasos_summ.prepared.json data/hasos/hasos_summ.json
"$PYTHON_BIN" scripts/validate_hasos.py \
  --data data/hasos/hasos_summ.json \
  --taxonomy data/hasos/aspect_taxonomy.json \
  --seeds-dir data/seeds_hasos
trace prepare_hasos ok ""

trace download_space start "gdown space_train"
if [ ! -f data/space/json/space_train.json ]; then
  "$PYTHON_BIN" -m gdown --no-check-certificate \
    "https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT" \
    -O data/space/json/space_train.tar
  "$PYTHON_BIN" - <<'PY'
import tarfile
with tarfile.open("data/space/json/space_train.tar", "r") as t:
    t.extractall("data/space/json/")
PY
  rm -f data/space/json/space_train.tar
fi
test -f data/space/json/space_train.json
trace download_space ok ""

trace train_sentencepiece start ""
if [ ! -f "$SPM" ]; then
  (cd src/utils && "$PYTHON_BIN" train-spm.py ../../data/space/json/space_train.json ../../data/sentencepiece/space_unigram_32k)
fi
test -f "$SPM"
test -f "${SPM%.model}.vocab"
sha256sum "$SPM" "${SPM%.model}.vocab"
trace train_sentencepiece ok "tokenizer=$SPM"

trace train_model start "max_entities=$MAX_ENTITIES epochs=$EPOCHS warmup=$WARMUP_EPOCHS"
export PYTHONPATH="$REMOTE_ROOT/src:${PYTHONPATH:-}"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" src/train.py \
  --data data/space/json/space_train.json \
  --sentencepiece data/sentencepiece/space_unigram_32k.model \
  --run_id "$TRAIN_RUN_ID" \
  --gpu "$GPU" \
  --epochs "$EPOCHS" \
  --batch_size 5 \
  --d_model 320 \
  --codebook_size 1024 \
  --nlayers 3 \
  --internal_nheads 4 \
  --output_nheads 8 \
  --d_ff 512 \
  --warmup_epochs "$WARMUP_EPOCHS" \
  --lr 0.0001 \
  --lr_decay 0.9 \
  --label_smoothing 0.1 \
  --commitment_cost 1.0 \
  --l1_cost 1000.0 \
  --entropy_cost 0.00005 \
  --grad_clip 1.0 \
  --diagnose_nonfinite \
  --nonfinite_policy fail \
  --safe_normalize \
  --kmeans_bad_vector_policy fail \
  --max_num_entities "$MAX_ENTITIES" \
  --no_eval \
  --savedir "$CKPT_DIR" \
  --save_every 1
test -f "$MODEL"
trace train_model ok "model=$MODEL"

trace inference start "shards=$NUM_SHARDS"
"$PYTHON_BIN" scripts/run_space_hasos_aspect_parallel.py \
  --model "$MODEL" \
  --run_id "$INFER_RUN_ID" \
  --num_shards "$NUM_SHARDS" \
  --gpu "$GPU" \
  --max_tokens "$MAX_TOKENS" \
  --sentiment_split
trace inference ok ""

trace export_lines start ""
"$PYTHON_BIN" scripts/export_space_hasos_lines.py --run_id "$INFER_RUN_ID"
trace export_lines ok ""

trace summarize_outputs start ""
"$PYTHON_BIN" scripts/summarize_aspect_outputs.py --run_id "$INFER_RUN_ID"
trace summarize_outputs ok ""

trace score_outputs start ""
"$PYTHON_BIN" scripts/score_semae_run.py --run_id "$INFER_RUN_ID"
trace score_outputs ok ""

trace metadata start ""
"$PYTHON_BIN" - <<PY
import json, time
payload = {
    "run_id": "$INFER_RUN_ID",
    "train_run_id": "$TRAIN_RUN_ID",
    "train_label": f"SPACE {int('$MAX_ENTITIES')} entities x {int('$EPOCHS')} epochs",
    "model_label": f"SPACE SemAE {int('$MAX_ENTITIES')} entities x {int('$EPOCHS')} epochs",
    "checkpoint": "$MODEL",
    "sentencepiece": "$SPM",
    "max_num_entities": int("$MAX_ENTITIES"),
    "epochs": int("$EPOCHS"),
    "warmup_epochs": int("$WARMUP_EPOCHS"),
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
}
with open("outputs/${INFER_RUN_ID}_metadata.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
trace metadata ok ""

trace build_pptx start ""
"$PYTHON_BIN" scripts/build_space_hasos_report_pptx.py --run_id "$INFER_RUN_ID"
trace build_pptx ok ""

render_trace_md
trace pipeline_complete ok "outputs=$REMOTE_ROOT/outputs/$INFER_RUN_ID"
render_trace_md
echo "=== pipeline done $(date -Is) ==="
