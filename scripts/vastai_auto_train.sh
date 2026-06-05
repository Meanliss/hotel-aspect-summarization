#!/bin/bash
# SemAE vast.ai Auto-Training Script
# Run from Git Bash / WSL / Linux terminal (NOT from corporate network!)
#
# Usage: bash vastai_auto_train.sh
#
# This script:
#   1. Finds and rents cheapest GPU on vast.ai
#   2. Uploads code
#   3. Installs deps + downloads SPACE data + trains SPM
#   4. Trains SemAE (Option B: 1000 entities x 20 epochs)
#   5. Downloads trained model back to local

set -euo pipefail

API_KEY="${VAST_API_KEY:?Set VAST_API_KEY before running this script}"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$PROJECT_DIR/models"
LOG_FILE="$PROJECT_DIR/logs/vastai_train.log"

mkdir -p "$PROJECT_DIR/logs" "$MODEL_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ─── 1. Find and rent GPU ──────────────────────────────────────────
log "Searching for cheapest GPU on vast.ai..."

OFFERS=$(curl -s "https://console.vast.ai/api/v0/bundles/?api_key=$API_KEY" 2>/dev/null)

OFFER_ID=$(echo "$OFFERS" | python -c "
import sys, json
data = json.load(sys.stdin)
offers = data.get('offers', [])
good = [o for o in offers if o.get('gpu_ram', 0) >= 8000]
good.sort(key=lambda x: x.get('dph_total', 999))
if good:
    o = good[0]
    print(o['id'])
    import sys as s
    s.stderr.write(f\"Found: {o['gpu_name']} {o['gpu_ram']}MB at \${o['dph_total']:.3f}/h\n")
else:
    print('NONE')
" 2>&1 | head -1)

if [ "$OFFER_ID" = "NONE" ]; then
    log "ERROR: No GPU offers available. Try again later."
    exit 1
fi

log "Renting offer $OFFER_ID..."

RESULT=$(curl -s -X PUT "https://console.vast.ai/api/v0/asks/$OFFER_ID/?api_key=$API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "client_id": "me",
    "image": "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
    "disk": 30,
    "label": "semae-space-training",
    "runtype": "ssh",
    "python_utf8": true,
    "lang_utf8": true
  }' 2>/dev/null)

SUCCESS=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin).get('success',False))" 2>/dev/null)
if [ "$SUCCESS" != "True" ]; then
    log "ERROR: Failed to rent GPU."
    echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin).get('msg','unknown'))" 2>/dev/null
    exit 1
fi

CONTRACT_ID=$(echo "$RESULT" | python -c "import sys,json; print(json.load(sys.stdin)['new_contract'])" 2>/dev/null)
log "Instance rented! Contract ID: $CONTRACT_ID"

# ─── 2. Wait for instance to be ready ──────────────────────────────
log "Waiting for instance to boot..."

for i in $(seq 1 30); do
    INFO=$(curl -s "https://console.vast.ai/api/v0/instances/$CONTRACT_ID/?api_key=$API_KEY" 2>/dev/null)
    STATUS=$(echo "$INFO" | python -c "import sys,json; d=json.load(sys.stdin)['instances']; print(d.get('actual_status') or d.get('cur_state','?'))" 2>/dev/null)
    SSH_PORT=$(echo "$INFO" | python -c "import sys,json; print(json.load(sys.stdin)['instances'].get('ssh_port',''))" 2>/dev/null)
    SSH_HOST=$(echo "$INFO" | python -c "import sys,json; print(json.load(sys.stdin)['instances'].get('ssh_host',''))" 2>/dev/null)
    PUBLIC_IP=$(echo "$INFO" | python -c "import sys,json; print(json.load(sys.stdin)['instances'].get('public_ipaddr',''))" 2>/dev/null)
    GPU_NAME=$(echo "$INFO" | python -c "import sys,json; print(json.load(sys.stdin)['instances'].get('gpu_name','?'))" 2>/dev/null)
    PRICE=$(echo "$INFO" | python -c "import sys,json; print(json.load(sys.stdin)['instances'].get('dph_total',0))" 2>/dev/null)

    if [ "$STATUS" = "running" ] && [ -n "$SSH_PORT" ]; then
        # Try SSH connection
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -p "$SSH_PORT" root@"$PUBLIC_IP" "echo ok" 2>/dev/null; then
            log "Instance ready! $GPU_NAME at \$$PRICE/h"
            log "SSH: ssh -p $SSH_PORT root@$PUBLIC_IP"
            break
        fi
    fi

    if [ "$i" = "30" ]; then
        log "ERROR: Instance didn't become ready in 5 minutes."
        log "Try connecting manually: ssh -p $SSH_PORT root@$PUBLIC_IP"
        log "Or destroy: curl -X DELETE 'https://console.vast.ai/api/v0/instances/$CONTRACT_ID/?api_key=$API_KEY'"
        exit 1
    fi

    log "  [$i/30] Status: $STATUS, waiting..."
    sleep 10
done

SSH_CMD="ssh -o StrictHostKeyChecking=no -p $SSH_PORT root@$PUBLIC_IP"
SCP_CMD="scp -o StrictHostKeyChecking=no -P $SSH_PORT"

# ─── 3. Upload code ────────────────────────────────────────────────
log "Uploading code to instance..."

# Create tarball excluding large/unnecessary files
TAR_FILE="/tmp/semae_upload.tar.gz"
cd "$PROJECT_DIR"
tar czf "$TAR_FILE" \
    --exclude='models/' \
    --exclude='logs/' \
    --exclude='outputs/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    --exclude='data/space/' \
    --exclude='*.tar' \
    .

$SCP_CMD "$TAR_FILE" "root@$PUBLIC_IP:/tmp/semae.tar.gz"
$SSH_CMD "mkdir -p /workspace/SemAE && cd /workspace/SemAE && tar xzf /tmp/semae.tar.gz && rm /tmp/semae.tar.gz"

log "Code uploaded!"

# ─── 4. Setup + Train ──────────────────────────────────────────────
log "Starting setup and training on remote instance..."

# Run setup + training in a single SSH session
$SSH_CMD bash -s <<'REMOTE_SCRIPT'
set -euo pipefail
cd /workspace/SemAE

echo "=== Installing dependencies ==="
pip install --upgrade pip setuptools -q
pip install sentencepiece tqdm numpy scipy scikit-learn nltk tensorboard -q

echo "=== Downloading SPACE data (1.18 GB) ==="
pip install gdown -q
mkdir -p data/space/json
python -m gdown --no-check-certificate \
    "https://drive.google.com/uc?id=1C6SaRQkas2B-9MolbwZbl0fuLgqdSKDT" \
    -O data/space/json/space_train.tar
python -c "import tarfile; t=tarfile.open('data/space/json/space_train.tar','r'); t.extractall('data/space/json/'); t.close()"
rm -f data/space/json/space_train.tar

echo "=== Training SentencePiece (~8 min) ==="
cd src/utils
python train-spm.py ../../data/space/json/space_train.json ../../data/sentencepiece/space_unigram_32k
cd ../..

echo "=== Training SemAE (Option B: 1000 entities x 20 epochs) ==="
mkdir -p logs
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
python src/train.py \
    --data data/space/json/space_train.json \
    --sentencepiece data/sentencepiece/space_unigram_32k.model \
    --max_num_entities 1000 \
    --run_id space_run1 \
    --gpu 0 \
    --epochs 20 \
    2>&1 | tee logs/train_space.log

echo "=== Training complete! ==="
ls -lh models/space_run1_20_model.pt
REMOTE_SCRIPT

log "Training finished!"

# ─── 5. Download model ─────────────────────────────────────────────
log "Downloading trained model..."

$SCP_CMD "root@$PUBLIC_IP:/workspace/SemAE/models/space_run1_20_model.pt" "$MODEL_DIR/"

log "Model saved to: $MODEL_DIR/space_run1_20_model.pt"

# ─── 6. Destroy instance ───────────────────────────────────────────
log "Destroying instance to stop charges..."
curl -s -X DELETE "https://console.vast.ai/api/v0/instances/$CONTRACT_ID/?api_key=$API_KEY" > /dev/null 2>&1

log "=========================================="
log "  DONE! Model ready at:"
log "  $MODEL_DIR/space_run1_20_model.pt"
log ""
log "  Next step: run inference locally"
log "  python scripts/run_space_hasos_aspect_parallel.py \\"
log "      --model models/space_run1_20_model.pt \\"
log "      --run_id space_hasos_run1 --num_shards 4 --gpu 0 --sentiment_split"
log "=========================================="
