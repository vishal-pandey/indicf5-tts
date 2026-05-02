#!/bin/bash

# ==========================================
# LAUNCHD ENVIRONMENT FIXES
# ==========================================
# 1. Explicitly set PATH so the daemon can find conda
export PATH="/Users/vishal/miniconda3/bin:/Users/vishal/anaconda3/bin:/opt/homebrew/bin:/opt/homebrew/Caskroom/miniforge/base/bin:/opt/homebrew/Caskroom/miniconda/base/bin:/opt/homebrew/Caskroom/miniconda/base/condabin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

# 2. Explicitly set HOME so HuggingFace can find your login token
export HOME="/Users/vishal"

# 3. Change into the correct directory so Uvicorn can find 'app/main.py'
cd /Users/vishal/indicf5-tts || exit 1

# ==========================================

set -e

ENV_NAME="indicf5"
PYTHON_VERSION="3.10"
PORT=8000

echo "========================================="
echo "  Indic TTS API Setup & Start"
echo "  Engines: IndicF5 + Indic Parler TTS"
echo "========================================="

# ==========================================
# Wait for network (LaunchDaemon may start before DNS is ready)
# ==========================================
echo "Waiting for network..."
MAX_WAIT=60
WAITED=0
while ! host github.com > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo "ERROR: Network not available after ${MAX_WAIT}s. Exiting."
        exit 1
    fi
done
echo "Network ready (waited ${WAITED}s)."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed or not in PATH."
    echo "Current PATH is: $PATH"
    exit 1
fi

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

# Create conda env if it doesn't exist
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo ""
    echo "[1/4] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
else
    echo ""
    echo "[1/4] Conda environment '${ENV_NAME}' already exists. Skipping creation."
fi

# Activate the environment
conda activate "$ENV_NAME"
echo "       Activated: $(python --version)"

# Install dependencies (skip git installs if already present)
echo ""
echo "[2/4] Installing dependencies..."

python -c "import f5_tts" 2>/dev/null || pip install -q git+https://github.com/ai4bharat/IndicF5.git
python -c "import parler_tts" 2>/dev/null || pip install -q git+https://github.com/huggingface/parler-tts.git
pip install -q fastapi 'uvicorn[standard]' torchcodec 2>/dev/null || true
conda install -c conda-forge ffmpeg -y -q 2>/dev/null || true

# Check HuggingFace login
echo ""
echo "[3/4] Checking HuggingFace authentication..."
python -c "
from huggingface_hub import HfApi
try:
    user = HfApi().whoami()
    print(f'       Logged in as: {user[\"name\"]}')
except Exception:
    print('       ERROR: Not logged in to HuggingFace.')
    print('       Run: huggingface-cli login')
    print('       Then request access at:')
    print('         https://huggingface.co/ai4bharat/IndicF5')
    print('         https://huggingface.co/ai4bharat/indic-parler-tts')
    exit(1)
"

# Pre-download model weights
echo ""
echo "[4/4] Downloading model weights (first run only)..."
python -c "
from huggingface_hub import hf_hub_download
import sys

files = [
    ('ai4bharat/IndicF5', 'model.safetensors'),
    ('ai4bharat/IndicF5', 'checkpoints/vocab.txt'),
    ('ai4bharat/IndicF5', 'prompts/PAN_F_HAPPY_00001.wav'),
    ('charactr/vocos-mel-24khz', 'config.yaml'),
    ('charactr/vocos-mel-24khz', 'pytorch_model.bin'),
]
for repo, fname in files:
    try:
        hf_hub_download(repo, fname)
        print(f'       ✓ {repo}/{fname}')
    except Exception as e:
        print(f'       ✗ {repo}/{fname} — {e}')
        if 'gated' in str(e).lower() or '403' in str(e):
            print(f'       → Request access at: https://huggingface.co/{repo}')
            sys.exit(1)

# Verify Parler model access
try:
    hf_hub_download('ai4bharat/indic-parler-tts', 'config.json')
    print('       ✓ ai4bharat/indic-parler-tts (config verified)')
except Exception as e:
    print(f'       ✗ ai4bharat/indic-parler-tts — {e}')
    if 'gated' in str(e).lower() or '403' in str(e):
        print('       → Request access at: https://huggingface.co/ai4bharat/indic-parler-tts')
        sys.exit(1)
"

echo ""
echo "========================================="
echo "  Starting API server on port ${PORT}"
echo "  Docs: http://localhost:${PORT}/docs"
echo "========================================="
echo ""

uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
