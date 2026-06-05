#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SemAE}"
RUN_ID="${RUN_ID:-space_full_11402x20}"
SESSION="${SESSION:-semae_full}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---diagnose_nonfinite --nonfinite_policy fail --safe_normalize --kmeans_bad_vector_policy fail}"
LOGFILE="$WORKDIR/logs/${RUN_ID}.log"
CHECKPOINT_DIR="$WORKDIR/checkpoints/$RUN_ID"
SPACE_JSON="$WORKDIR/data/space/json/space_train.json"
SPM_MODEL="$WORKDIR/data/sentencepiece/space_unigram_32k.model"

mkdir -p "$WORKDIR/logs" "$CHECKPOINT_DIR"

cat > "$WORKDIR/run_${RUN_ID}.sh" <<'RUNEOF'
#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace/SemAE}"
RUN_ID="${RUN_ID:-space_full_11402x20}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-0.0001}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:---diagnose_nonfinite --nonfinite_policy fail --safe_normalize --kmeans_bad_vector_policy fail}"
LOGFILE="$WORKDIR/logs/${RUN_ID}.log"
CHECKPOINT_DIR="$WORKDIR/checkpoints/$RUN_ID"
SPACE_JSON="$WORKDIR/data/space/json/space_train.json"
SPM_MODEL="$WORKDIR/data/sentencepiece/space_unigram_32k.model"

mkdir -p "$WORKDIR/logs" "$CHECKPOINT_DIR"
exec > >(tee -a "$LOGFILE") 2>&1

START_TS=$(date +%s)
echo "=== SPACE full start: $(date -Iseconds) ==="
echo "RUN_ID=$RUN_ID"
echo "EPOCHS=$EPOCHS"
echo "LR=$LR"
echo "GRAD_CLIP=$GRAD_CLIP"
echo "EXTRA_TRAIN_ARGS=$EXTRA_TRAIN_ARGS"
nvidia-smi --query-gpu=name,memory.total,driver_version,pci.bus_id,temperature.gpu,power.limit --format=csv,noheader
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
if torch.cuda.is_available():
    print("capability", torch.cuda.get_device_capability(0))
PY

test -f "$SPACE_JSON"
test -f "$SPM_MODEL"
echo "Data: $(du -sh "$SPACE_JSON" | cut -f1) $SPACE_JSON"
echo "SPM:  $(ls -lh "$SPM_MODEL" | awk '{print $5}') $SPM_MODEL"
echo "Checkpoints: $CHECKPOINT_DIR"

cd "$WORKDIR/src"
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1

TRAIN_START=$(date +%s)
python train.py \
  --data "$SPACE_JSON" \
  --sentencepiece "$SPM_MODEL" \
  --run_id "$RUN_ID" \
  --gpu 0 \
  --epochs "$EPOCHS" \
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
TRAIN_END=$(date +%s)

END_TS=$(date +%s)
echo "=== done: $(date -Iseconds) ==="
echo "TRAIN_SECONDS=$((TRAIN_END - TRAIN_START))"
echo "WALL_SECONDS=$((END_TS - START_TS))"
find "$CHECKPOINT_DIR" -maxdepth 1 -type f -name "${RUN_ID}_*_model.pt" -printf "%f %s bytes\n" | sort -V | tail -25
RUNEOF

chmod +x "$WORKDIR/run_${RUN_ID}.sh"

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  tmux new-session -d -s "$SESSION" "WORKDIR='$WORKDIR' RUN_ID='$RUN_ID' EPOCHS='$EPOCHS' LR='$LR' GRAD_CLIP='$GRAD_CLIP' EXTRA_TRAIN_ARGS='$EXTRA_TRAIN_ARGS' bash '$WORKDIR/run_${RUN_ID}.sh'"
  echo "started tmux session: $SESSION"
else
  nohup env WORKDIR="$WORKDIR" RUN_ID="$RUN_ID" bash "$WORKDIR/run_${RUN_ID}.sh" > "$WORKDIR/logs/${RUN_ID}.nohup" 2>&1 &
  echo "started nohup pid: $!"
fi
