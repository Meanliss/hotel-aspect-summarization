#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$WORKDIR"

mkdir -p logs checkpoints/space_full_11402x20_stable_resume_e7

LOG_PATH="logs/space_full_11402x20_stable_resume_e10_$(date +%Y%m%d_%H%M%S).log"

python -m py_compile src/train.py

echo "Starting resume from epoch 10"
echo "Log: ${LOG_PATH}"
echo "Checkpoint source: checkpoints/space_full_11402x20_stable_resume_e7/space_full_11402x20_stable_resume_e7_10_model.pt"

nohup python3 src/train.py \
  --data data/space/json/space_train.json \
  --sentencepiece data/sentencepiece/space_unigram_32k.model \
  --run_id space_full_11402x20_stable_resume_e7 \
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
  --lr 0.00003 \
  --lr_decay 0.9 \
  --label_smoothing 0.1 \
  --commitment_cost 1.0 \
  --l1_cost 1000.0 \
  --entropy_cost 0.00005 \
  --grad_clip 1.0 \
  --safe_normalize \
  --diagnose_nonfinite \
  --nonfinite_policy fail \
  --kmeans_bad_vector_policy fail \
  --no_eval \
  --resume_model checkpoints/space_full_11402x20_stable_resume_e7/space_full_11402x20_stable_resume_e7_10_model.pt \
  --start_epoch 10 \
  --savedir checkpoints/space_full_11402x20_stable_resume_e7 \
  --save_every 1 \
  > "${LOG_PATH}" 2>&1 &

echo "$!" > logs/space_full_11402x20_stable_resume_e10.pid
echo "PID: $(cat logs/space_full_11402x20_stable_resume_e10.pid)"
echo "Monitor with:"
echo "  tail -f ${LOG_PATH}"
