#!/bin/bash
# ╔══════════════════════════════════════════════════════════════╗
# ║  SemAE SPACE Training — One-Click Autonomous Pipeline       ║
# ║  Paste this ENTIRE block into your vast.ai terminal.        ║
# ║  It will:                                                   ║
# ║    1. Install dependencies                                  ║
# ║    2. Download SPACE data from Google Drive                 ║
# ║    3. Train SentencePiece tokenizer                         ║
# ║    4. Train SemAE on SPACE (1000 entities, 20 epochs)       ║
# ║  Everything runs in tmux → survives disconnect.             ║
# ║  Reconnect with: tmux attach -t semae                       ║
# ╚══════════════════════════════════════════════════════════════╝

set -e

WORKDIR="/workspace/SemAE"

# ── Step 0: Clone/upload the repo ──
# If you already uploaded files, skip this. Otherwise:
# cd /workspace && git clone https://github.com/YOUR_USER/SemAE.git

# ── Step 1: Create the pipeline script ──
cat > "$WORKDIR/scripts/run_pipeline.sh" << 'PIPELINE_EOF'
#!/bin/bash
set -e
set -o pipefail

WORKDIR="/workspace/SemAE"
LOGFILE="$WORKDIR/logs/pipeline.log"
mkdir -p "$WORKDIR/logs" "$WORKDIR/models" "$WORKDIR/data/space/json" "$WORKDIR/data/sentencepiece"

exec > >(tee -a "$LOGFILE") 2>&1

echo "============================================"
echo "SemAE Pipeline Started: $(date)"
echo "============================================"

# ── Phase 1: Install dependencies ──
echo ""
echo ">>> Phase 1: Installing dependencies..."
pip install --upgrade pip setuptools 2>/dev/null
pip install sentencepiece tqdm scipy scikit-learn nltk tensorboard 2>/dev/null
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPUs:', torch.cuda.device_count())"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem/1024**3:.0f} GB)') for i in range(torch.cuda.device_count())]"

# ── Phase 2: Download SPACE training data ──
echo ""
echo ">>> Phase 2: Downloading SPACE data..."
SPACE_JSON="$WORKDIR/data/space/json/space_train.json"
if [ -f "$SPACE_JSON" ] && [ $(stat -c%s "$SPACE_JSON" 2>/dev/null || stat -f%z "$SPACE_JSON" 2>/dev/null) -gt 100000000 ]; then
    echo "  space_train.json already exists ($(du -sh "$SPACE_JSON" | cut -f1)), skipping"
else
    pip install gdown 2>/dev/null

    echo "  Downloading from Google Drive..."
    python -c "
import gdown
import os
# SPACE dataset Google Drive ID
url = 'https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT'
output = '$WORKDIR/data/space/space_download'
gdown.download(url, output, quiet=False, fuzzy=True)

# Check if tar/zip
import tarfile, zipfile
if tarfile.is_tarfile(output):
    print('  Extracting tar archive...')
    with tarfile.open(output) as t:
        t.extractall('$WORKDIR/data/space/json/')
    os.remove(output)
elif zipfile.is_zipfile(output):
    print('  Extracting zip archive...')
    with zipfile.ZipFile(output) as z:
        z.extractall('$WORKDIR/data/space/json/')
    os.remove(output)
else:
    # Assume raw json
    os.rename(output, '$SPACE_JSON')
"

    # Find the json if extracted
    if [ ! -f "$SPACE_JSON" ]; then
        FOUND=$(find "$WORKDIR/data/space" -name "*.json" -size +100M 2>/dev/null | head -1)
        if [ -n "$FOUND" ]; then
            mv "$FOUND" "$SPACE_JSON"
        fi
    fi
fi

if [ ! -f "$SPACE_JSON" ]; then
    echo "ERROR: space_train.json not found!"
    echo "You may need to upload it manually or fix the Google Drive link."
    echo "Expected path: $SPACE_JSON"
    find "$WORKDIR/data/space" -type f 2>/dev/null
    exit 1
fi
echo "  Data ready: $(du -sh "$SPACE_JSON" | cut -f1)"

# ── Phase 3: Train SentencePiece model ──
echo ""
echo ">>> Phase 3: Training SentencePiece (space_unigram_32k)..."
SPM_MODEL="$WORKDIR/data/sentencepiece/space_unigram_32k.model"
if [ -f "$SPM_MODEL" ]; then
    echo "  SPM model already exists, skipping"
else
    cd "$WORKDIR/src/utils"
    python train-spm.py "$SPACE_JSON" "$WORKDIR/data/sentencepiece/space_unigram_32k"
    cd "$WORKDIR"
    echo "  SPM model trained: $(ls -lh "$SPM_MODEL" | awk '{print $5}')"
fi

# ── Phase 4: Train SemAE on SPACE ──
echo ""
echo ">>> Phase 4: Training SemAE (1000 entities x 20 epochs)..."
FINAL_MODEL="$WORKDIR/models/space_run1_20_model.pt"
if [ -f "$FINAL_MODEL" ]; then
    echo "  Final model already exists, skipping training"
else
    cd "$WORKDIR/src"
    export PYTHONIOENCODING=utf-8
    export PYTHONUNBUFFERED=1

    python train.py \
        --data "$WORKDIR/data/space/json/space_train.json" \
        --sentencepiece "$WORKDIR/data/sentencepiece/space_unigram_32k.model" \
        --max_num_entities 1000 \
        --run_id space_run1 \
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
        --lr 0.001 \
        --lr_decay 0.9 \
        --label_smoothing 0.1 \
        --commitment_cost 1.0 \
        --l1_cost 1000.0 \
        --entropy_cost 0.00005

    cd "$WORKDIR"
fi

# Verify
if [ -f "$FINAL_MODEL" ]; then
    echo ""
    echo "============================================"
    echo "✅ TRAINING COMPLETE!"
    echo "============================================"
    echo "Model: $FINAL_MODEL ($(ls -lh "$FINAL_MODEL" | awk '{print $5}'))"
    echo "Log:   $LOGFILE"
    echo ""
    echo "Download model to local machine:"
    echo "  scp -P <port> root@<ip>:$FINAL_MODEL ."
else
    FOUND_MODEL=$(find "$WORKDIR/models" -name "space_run1_*_model.pt" | sort | tail -1)
    if [ -n "$FOUND_MODEL" ]; then
        cp "$FOUND_MODEL" "$FINAL_MODEL"
        echo "Model saved from checkpoint: $FOUND_MODEL"
    else
        echo "ERROR: No model checkpoint found!"
        exit 1
    fi
fi

echo ""
echo "Pipeline finished at $(date)"
PIPELINE_EOF

chmod +x "$WORKDIR/scripts/run_pipeline.sh"

# ── Step 2: Launch in tmux (survives disconnect) ──
tmux kill-session -t semae 2>/dev/null || true
tmux new-session -d -s semae "bash $WORKDIR/scripts/run_pipeline.sh"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  Pipeline launched in tmux session 'semae'       ║"
echo "║                                                  ║"
echo "║  Monitor:   tmux attach -t semae                 ║"
echo "║  Detach:    Ctrl+B then D                        ║"
echo "║  Log file:  tail -f logs/pipeline.log            ║"
echo "║                                                  ║"
echo "║  You can disconnect SSH now — training continues ║"
echo "╚══════════════════════════════════════════════════╝"
