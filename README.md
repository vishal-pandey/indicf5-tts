# IndicF5 TTS API

Text-to-Speech API for 11 Indian languages powered by [IndicF5](https://huggingface.co/ai4bharat/IndicF5), running on Apple Silicon (MPS).

## Supported Languages

Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- [Miniconda](https://docs.anaconda.com/miniconda/) installed
- HuggingFace account with access to [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5)

## Quick Start

```bash
# 1. Login to HuggingFace (one-time)
pip install huggingface_hub
huggingface-cli login

# 2. Request model access (one-time)
# Visit https://huggingface.co/ai4bharat/IndicF5 and click "Access repository"

# 3. Start the API
./start.sh
```

## API Usage

Once running, the API is available at `http://localhost:8000`.

### Generate Speech

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते, आप कैसे हैं?"}' \
  -o output.wav
```

### Request Body

| Field | Type  | Default | Description                |
|-------|-------|---------|----------------------------|
| text  | str   | —       | Text to synthesize (required) |
| speed | float | 1.0     | Speed multiplier (0.5–2.0) |

### Interactive Docs

Open `http://localhost:8000/docs` in your browser for the Swagger UI.

## Performance (Mac Mini M4, 24GB)

| Input Length | Audio Duration | Generation Time |
|-------------|---------------|-----------------|
| 1 sentence  | ~1.5s         | ~19s            |
| 4 sentences | ~20s          | ~88s            |

## Docker (CPU, slower)

If you prefer Docker (runs on CPU, ~5x slower):

```bash
docker build --build-arg HF_TOKEN=$(cat ~/.cache/huggingface/token) -t indicf5-tts .
docker run --rm -p 8000:8000 indicf5-tts
```
