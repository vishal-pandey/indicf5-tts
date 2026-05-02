FROM python:3.10-slim

# System deps for audio processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch (CPU — no CUDA needed in container)
RUN pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0

# Install remaining Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Pre-download model weights at build time so container startup is fast.
# Pass your HF token: docker build --build-arg HF_TOKEN=hf_xxx ...
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('ai4bharat/IndicF5', 'model.safetensors'); \
hf_hub_download('ai4bharat/IndicF5', 'checkpoints/vocab.txt'); \
hf_hub_download('ai4bharat/IndicF5', 'prompts/PAN_F_HAPPY_00001.wav'); \
hf_hub_download('ai4bharat/IndicF5', 'config.json'); \
hf_hub_download('charactr/vocos-mel-24khz', 'config.yaml'); \
hf_hub_download('charactr/vocos-mel-24khz', 'pytorch_model.bin'); \
"

# Clear the token from the image
ENV HF_TOKEN=""

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
