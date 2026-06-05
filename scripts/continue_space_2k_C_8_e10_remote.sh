#!/usr/bin/env bash
set -euo pipefail

cd /workspace/SemAE
mkdir -p logs checkpoints/space_2k_C_8
export PYTHONPATH=/workspace/SemAE/src:${PYTHONPATH:-}

LOG=logs/space_2k_C_8_continue_e10.log
echo "=== CONTINUE e9-e10 start: $(date -Is) ===" | tee "$LOG"

/workspace/venv/bin/python train.py \
  --data /workspace/SemAE/data/space/json/space_train.json \
  --sentencepiece /workspace/SemAE/data/sentencepiece/space_unigram_32k.model \
  --run_id space_2k_C_8 \
  --gpu 0 \
  --epochs 10 \
  --start_epoch 8 \
  --resume_model /workspace/SemAE/checkpoints/space_2k_C_8/space_2k_C_8_8_model.pt \
  --batch_size 5 \
  --d_model 320 \
  --codebook_size 1024 \
  --nlayers 3 \
  --internal_nheads 4 \
  --output_nheads 8 \
  --d_ff 512 \
  --warmup_epochs 4 \
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
  --max_num_entities 2000 \
  --no_eval \
  --savedir /workspace/SemAE/checkpoints/space_2k_C_8 \
  --save_every 1 2>&1 | tee -a "$LOG"

echo "=== CONTINUE e9-e10 end: $(date -Is) ===" | tee -a "$LOG"
