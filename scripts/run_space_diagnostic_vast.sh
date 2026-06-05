#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SemAE}"
RUN_KIND="${RUN_KIND:-A}"
RUN_ID="${RUN_ID:-space_full_diag_${RUN_KIND}}"
SESSION="${SESSION:-semae_${RUN_ID}}"
LOGFILE="$WORKDIR/logs/${RUN_ID}.log"
CHECKPOINT_DIR="$WORKDIR/checkpoints/$RUN_ID"
SPACE_JSON="$WORKDIR/data/space/json/space_train.json"
SPM_MODEL="$WORKDIR/data/sentencepiece/space_unigram_32k.model"

mkdir -p "$WORKDIR/logs" "$CHECKPOINT_DIR"

case "$RUN_KIND" in
  A)
    LR="${LR:-0.001}"
    GRAD_CLIP="${GRAD_CLIP:-0.0}"
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---diagnose_nonfinite --nonfinite_policy fail --max_train_batches 5000 --no_eval --kmeans_bad_vector_policy fail}"
    ;;
  B)
    LR="${LR:-0.0001}"
    GRAD_CLIP="${GRAD_CLIP:-0.0}"
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---diagnose_nonfinite --nonfinite_policy fail --safe_normalize --kmeans_bad_vector_policy fail}"
    ;;
  C)
    LR="${LR:-0.0001}"
    GRAD_CLIP="${GRAD_CLIP:-1.0}"
    EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---diagnose_nonfinite --nonfinite_policy fail --safe_normalize --kmeans_bad_vector_policy fail}"
    ;;
  *)
    echo "RUN_KIND must be A, B, or C" >&2
    exit 2
    ;;
esac

cat > "$WORKDIR/run_${RUN_ID}.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SemAE}"
RUN_ID="${RUN_ID:-space_full_diag}"
LR="${LR:-0.001}"
GRAD_CLIP="${GRAD_CLIP:-0.0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
LOGFILE="$WORKDIR/logs/${RUN_ID}.log"
CHECKPOINT_DIR="$WORKDIR/checkpoints/$RUN_ID"
SPACE_JSON="$WORKDIR/data/space/json/space_train.json"
SPM_MODEL="$WORKDIR/data/sentencepiece/space_unigram_32k.model"

mkdir -p "$WORKDIR/logs" "$CHECKPOINT_DIR"
exec > >(tee -a "$LOGFILE") 2>&1

START_TS=$(date +%s)
echo "=== SPACE diagnostic start: $(date -Iseconds) ==="
echo "RUN_ID=$RUN_ID"
echo "LR=$LR"
echo "GRAD_CLIP=$GRAD_CLIP"
echo "EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS"
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id,temperature.gpu,power.limit --format=csv,noheader
python - <<'PY'
import numpy, scipy, sys, torch
print("python", sys.version)
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("scipy", scipy.__version__)
print("numpy", numpy.__version__)
PY

test -f "$SPACE_JSON"
test -f "$SPM_MODEL"
cd "$WORKDIR/src"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

set +e
python train.py \
  --data "$SPACE_JSON" \
  --sentencepiece "$SPM_MODEL" \
  --run_id "$RUN_ID" \
  --gpu 0 \
  --epochs 20 \
  --batch_size 5 \
  --d_model 320 \
  --codebook_size 1024 \
  --nlayers 3 \
  --internal_nheads 4 \
  --output_nheads 8 \
  --d_ff 512 \
  --warmup_epochs 4 \
  --lr "$LR" \
  --lr_decay 0.9 \
  --label_smoothing 0.1 \
  --commitment_cost 1.0 \
  --l1_cost 1000.0 \
  --entropy_cost 0.00005 \
  --grad_clip "$GRAD_CLIP" \
  $EXTRA_TRAIN_ARGS \
  --savedir "$CHECKPOINT_DIR" \
  --save_every 1
EXIT_CODE=$?
set -e

END_TS=$(date +%s)
echo "=== SPACE diagnostic end: $(date -Iseconds) exit=$EXIT_CODE ==="
echo "WALL_SECONDS=$((END_TS - START_TS))"
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader || true
find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "${RUN_ID}_*_model.pt" -printf "%f %s bytes\n" | sort -V | tail -25 || true
exit "$EXIT_CODE"
RUNEOF

chmod +x "$WORKDIR/run_${RUN_ID}.sh"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" "WORKDIR='$WORKDIR' RUN_ID='$RUN_ID' LR='$LR' GRAD_CLIP='$GRAD_CLIP' EXTRA_TRAIN_ARGS='$EXTRA_TRAIN_ARGS' bash '$WORKDIR/run_${RUN_ID}.sh'"
  echo "started tmux session: $SESSION"
else
  nohup env WORKDIR="$WORKDIR" RUN_ID="$RUN_ID" LR="$LR" GRAD_CLIP="$GRAD_CLIP" EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS" bash "$WORKDIR/run_${RUN_ID}.sh" > "$WORKDIR/logs/${RUN_ID}.nohup" 2>&1 &
  echo "started nohup pid: $!"
fi
