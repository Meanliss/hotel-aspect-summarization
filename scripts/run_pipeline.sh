#!/bin/bash
# SemAE SPACE Training Pipeline — Autonomous Runner
# Runs in tmux so it survives SSH disconnect / Claude Code close.
#
# Usage (on vast.ai instance):
#   bash /workspace/SemAE/scripts/run_pipeline.sh
#
# Monitor:
#   tmux attach -t semae
#   tail -f /workspace/SemAE/logs/pipeline.log

set -e
set -o pipefail

WORKDIR="/workspace/SemAE"
LOGFILE="$WORKDIR/logs/pipeline.log"
mkdir -p "$WORKDIR/logs" "$WORKDIR/models"

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
if [ -f "$SPACE_JSON" ]; then
    echo "  space_train.json already exists, skipping download"
else
    pip install gdown 2>/dev/null
    mkdir -p "$WORKDIR/data/space/json"
    # Google Drive file ID for space_train.json
    # If this is a tar.gz, extract it; if json, just save
    echo "  Downloading from Google Drive..."
    python -m gdown --no-check-certificate \
        "https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT" \
        -O "$WORKDIR/data/space/space_download" || {
        echo "  gdown failed. Trying wget..."
        wget --no-check-certificate \
            "https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT" \
            -O "$WORKDIR/data/space/space_download"
    }

    # Check if it's a tar archive or raw json
    if file "$WORKDIR/data/space/space_download" | grep -q "tar\|gzip\|Zip"; then
        echo "  Extracting archive..."
        cd "$WORKDIR/data/space/json"
        tar xf "$WORKDIR/data/space/space_download" || python -c "
import tarfile, zipfile, sys
f = '$WORKDIR/data/space/space_download'
try:
    tarfile.open(f).extractall('.')
except:
    zipfile.ZipFile(f).extractall('.')
"
        rm -f "$WORKDIR/data/space/space_download"
    else
        mv "$WORKDIR/data/space/space_download" "$SPACE_JSON"
    fi
    cd "$WORKDIR"
fi

# Verify data exists
if [ ! -f "$SPACE_JSON" ]; then
    # Maybe extracted to a different name
    FOUND=$(find "$WORKDIR/data/space" -name "*.json" -size +100M 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        echo "  Found data at: $FOUND"
        if [ "$FOUND" != "$SPACE_JSON" ]; then
            mv "$FOUND" "$SPACE_JSON"
        fi
    else
        echo "ERROR: space_train.json not found! Check Google Drive link."
        echo "Available files:"
        find "$WORKDIR/data/space" -type f 2>/dev/null
        exit 1
    fi
fi
echo "  Data ready: $(du -sh "$SPACE_JSON" | cut -f1)"

# ── Phase 3: Train SentencePiece model ──
echo ""
echo ">>> Phase 3: Training SentencePiece (space_unigram_32k)..."
SPM_MODEL="$WORKDIR/data/sentencepiece/space_unigram_32k.model"
if [ -f "$SPM_MODEL" ]; then
    echo "  SPM model already exists, skipping"
else
    mkdir -p "$WORKDIR/data/sentencepiece"
    cd "$WORKDIR/src/utils"
    python train-spm.py \
        "$SPACE_JSON" \
        "$WORKDIR/data/sentencepiece/space_unigram_32k" \
        --vocab_size 32000
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
        --warmup_epochs 4

    cd "$WORKDIR"
fi

# Verify model exists
if [ -f "$FINAL_MODEL" ]; then
    echo "  Training complete! Model: $(ls -lh "$FINAL_MODEL" | awk '{print $5}')"
else
    # Try finding any model checkpoint
    FOUND_MODEL=$(find "$WORKDIR/models" -name "space_run1_*_model.pt" | sort | tail -1)
    if [ -n "$FOUND_MODEL" ]; then
        echo "  Using checkpoint: $FOUND_MODEL"
        cp "$FOUND_MODEL" "$FINAL_MODEL"
    else
        echo "ERROR: No model found after training!"
        exit 1
    fi
fi

# ── Phase 5: Summary ──
echo ""
echo "============================================"
echo "Pipeline Complete: $(date)"
echo "============================================"
echo "Model: $FINAL_MODEL"
echo "Log:   $LOGFILE"
echo ""
echo "Next steps (on LOCAL machine):"
echo "  scp -P <port> root@<ip>:$FINAL_MODEL SemAE/models/"
echo "  Then run aspect inference locally."
