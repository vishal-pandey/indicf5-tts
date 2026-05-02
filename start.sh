#!/bin/bash
set -e

ENV_NAME="indicf5"
PYTHON_VERSION="3.10"
PORT=8000

echo "========================================="
echo "  IndicF5 TTS API Setup & Start"
echo "========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda is not installed."
    echo "Install Miniconda from: https://docs.anaconda.com/miniconda/"
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

# Install dependencies
echo ""
echo "[2/4] Installing dependencies..."
pip install -q git+https://github.com/ai4bharat/IndicF5.git
pip install -q fastapi 'uvicorn[standard]' torchcodec
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
    print('       Then request access at: https://huggingface.co/ai4bharat/IndicF5')
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
            print('       → Request access at: https://huggingface.co/ai4bharat/IndicF5')
            sys.exit(1)
"

echo ""
echo "========================================="
echo "  Starting API server on port ${PORT}"
echo "  Docs: http://localhost:${PORT}/docs"
echo "========================================="
echo ""

uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
