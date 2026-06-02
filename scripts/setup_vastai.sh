#!/bin/bash
# SemAE vast.ai setup script
# Run on vast.ai instance after SSH: bash setup_vastai.sh
#
# This script:
#   1. Installs Python dependencies
#   2. Downloads SPACE training data (1.18 GB from Google Drive)
#   3. Trains SPACE SentencePiece model (~8 min)
#
# Prerequisites: project files already uploaded to /workspace/SemAE/

set -euo pipefail

WORKSPACE="/workspace/SemAE"
cd "$WORKSPACE"

echo "============================================"
echo "  SemAE vast.ai Setup"
echo "============================================"
echo ""

# ─── 1. System deps ────────────────────────────────────────────────
echo "[1/4] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git wget curl > /dev/null 2>&1

# ─── 2. Python deps ────────────────────────────────────────────────
echo "[2/4] Installing Python dependencies..."
pip install --upgrade pip setuptools -q

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# Install other deps (ignore pinned versions, use latest compatible)
pip install sentencepiece tqdm numpy scipy scikit-learn nltk tensorboard -q

echo "  Python: $(python --version)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "  GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

# ─── 3. Download SPACE data ────────────────────────────────────────
echo "[3/4] Downloading SPACE training data (1.18 GB)..."
pip install gdown -q

mkdir -p data/space/json

if [ -f "data/space/json/space_train.json" ]; then
    echo "  space_train.json already exists, skipping download."
else
    # Google Drive file ID from qt repo
    python -m gdown --no-check-certificate \
        "https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT" \
        -O data/space/json/space_train.tar

    echo "  Extracting..."
    python -c "import tarfile; t=tarfile.open('data/space/json/space_train.tar','r'); t.extractall('data/space/json/'); t.close()"
    rm -f data/space/json/space_train.tar
    echo "  Done. Files:"
    ls -lh data/space/json/
fi

# ─── 4. Train SPACE SentencePiece ──────────────────────────────────
echo "[4/4] Training SPACE SentencePiece model (~8 min)..."
cd src/utils
python train-spm.py ../../data/space/json/space_train.json ../../data/sentencepiece/space_unigram_32k
cd ../..

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To train SemAE (Option B: 1000 entities x 20 epochs):"
echo ""
echo "  cd $WORKSPACE/src"
echo "  export PYTHONIOENCODING=utf-8"
echo "  export PYTHONUNBUFFERED=1"
echo "  python train.py \\"
echo "      --data ../data/space/json/space_train.json \\"
echo "      --sentencepiece ../data/sentencepiece/space_unigram_32k.model \\"
echo "      --max_num_entities 1000 \\"
echo "      --run_id space_run1 \\"
echo "      --gpu 0 \\"
echo "      --epochs 20 \\"
echo "      2>&1 | tee ../logs/train_space.log"
echo ""
echo "Estimated time: 7-16 hours depending on GPU."
echo "Model saved to: models/space_run1_20_model.pt"
echo ""
echo "To download model back to local machine:"
echo "  scp -P <port> root@<vast_ip>:/workspace/SemAE/models/space_run1_20_model.pt ."
