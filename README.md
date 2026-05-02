# IndicF5 TTS API

Text-to-Speech API for 11 Indian languages powered by [IndicF5](https://huggingface.co/ai4bharat/IndicF5), optimized for Apple Silicon (MPS).

## Supported Languages

Assamese · Bengali · Gujarati · Hindi · Kannada · Malayalam · Marathi · Odia · Punjabi · Tamil · Telugu

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
#    Visit https://huggingface.co/ai4bharat/IndicF5 and click "Access repository"

# 3. Start the API
./start.sh
```

The script handles everything — creates a conda environment, installs dependencies, downloads model weights, and starts the server. First run takes a few minutes; subsequent runs start in ~10 seconds.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/synthesize` | Generate speech from text, returns WAV audio |
| GET | `/health` | Check if model is loaded and ready |
| GET | `/languages` | List supported languages |
| GET | `/docs` | Interactive Swagger UI with full documentation |

## Usage

### Generate Speech

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "नमस्ते, आप कैसे हैं?"}' \
  -o output.wav
```

### With Custom Parameters

```bash
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "भारत एक विविधताओं से भरा देश है।",
    "speed": 1.0,
    "nfe_step": 16,
    "cfg_strength": 0.0
  }' \
  -o output.wav
```

### Other Languages

```bash
# Tamil
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"}' \
  -o tamil.wav

# Bengali
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "নমস্কার! আপনি কেমন আছেন?"}' \
  -o bengali.wav

# Telugu
curl -X POST http://localhost:8000/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text": "నమస్కారం! మీరు ఎలా ఉన్నారు?"}' \
  -o telugu.wav
```

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `text` | string | *(required)* | 1–2000 chars | Text to synthesize in any supported language |
| `speed` | float | 1.0 | 0.5–2.0 | Speech speed multiplier |
| `nfe_step` | int | 16 | 4–64 | ODE solver steps (fewer = faster) |
| `cfg_strength` | float | 0.0 | 0.0–5.0 | Classifier-Free Guidance strength |

### `nfe_step` — Diffusion Steps

Controls how many iterations the model takes to generate audio. More steps refine the output but take proportionally longer.

| Value | Speed | Quality | When to Use |
|-------|-------|---------|-------------|
| 8 | Fastest | Acceptable — may sound slightly rough | Quick previews, testing |
| **16** | **Fast** | **Good — natural sounding** | **Default, recommended for production** |
| 32 | Slow | Best — subtle improvements over 16 | High-quality final renders |
| 64 | Very slow | Diminishing returns | Not recommended |

### `cfg_strength` — Classifier-Free Guidance

Controls how strongly the model follows text conditioning. Any value above 0 **doubles compute** because the model runs twice per step (once with guidance, once without).

| Value | Speed | Quality | When to Use |
|-------|-------|---------|-------------|
| **0.0** | **Fastest** | **Good — natural prosody** | **Default, recommended for production** |
| 1.0 | 2x slower | More expressive | When 0.0 sounds flat |
| 2.0 | 2x slower | Most guided (original default) | Maximum text adherence |
| 3.0+ | 2x slower | Over-guided, can distort | Not recommended |

### `speed` — Speech Rate

Simple multiplier on output duration. Does not significantly affect generation time.

| Value | Effect |
|-------|--------|
| 0.5 | Half speed (slow, stretched) |
| **1.0** | **Normal speed** |
| 1.5 | Faster speech |
| 2.0 | Double speed (compressed) |

## Recommended Presets

| Preset | nfe_step | cfg_strength | RTF* | Best For |
|--------|----------|--------------|------|----------|
| ⚡ Fastest | 8 | 0.0 | ~0.7x | Drafts, testing |
| 🎯 **Balanced** | **16** | **0.0** | **~1.5x** | **Production (default)** |
| 🎵 High Quality | 32 | 1.0 | ~6x | Final renders |

*RTF = Real-Time Factor (seconds of compute per second of audio). Lower is faster.
Benchmarked on Mac Mini M4, 24GB unified memory, MPS backend.

## Performance

Measured on Mac Mini M4 with 24GB unified memory:

| Input | Audio Length | Default (16 steps, cfg=0) | Original (32 steps, cfg=2) |
|-------|-------------|---------------------------|----------------------------|
| 1 short sentence | ~1.5s | **4.3s** | 19.4s |
| 4 sentences | ~20s | **28.8s** | 88s |

The default settings are **3–4x faster** than the original model defaults with minimal quality loss.

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── model.py          # Model loader with MPS support & weight remapping
│   └── main.py           # FastAPI app with documented endpoints
├── start.sh              # One-command setup & run
├── Dockerfile            # Optional Docker build (CPU, slower)
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Docker (Optional, CPU only)

Docker runs on CPU inside the container (~5x slower than native MPS). Use only if you need containerization.

```bash
# Build (pass your HF token for the gated model download)
docker build --build-arg HF_TOKEN=$(cat ~/.cache/huggingface/token) -t indicf5-tts .

# Run
docker run --rm -p 8000:8000 indicf5-tts
```

## How It Works

IndicF5 is a flow-matching TTS model based on [F5-TTS](https://github.com/SWivid/F5-TTS). It uses a Diffusion Transformer (DiT) to generate mel spectrograms from text, then a Vocos vocoder to convert them to audio waveforms.

The model requires a reference audio clip to guide voice characteristics (prosody, speaker identity). This API uses a bundled Punjabi reference audio by default.

Key optimizations for Apple Silicon:
- **MPS backend** — runs on the Apple GPU instead of CPU
- **Reduced diffusion steps** (16 vs 32) — halves ODE solver iterations
- **Disabled CFG** (0 vs 2) — eliminates the second forward pass per step
- **Weight remapping** — handles `torch.compile` key prefixes from the original checkpoint

## Credits

- [AI4Bharat](https://github.com/AI4Bharat/IndicF5) for the IndicF5 model
- [F5-TTS](https://github.com/SWivid/F5-TTS) for the base architecture
- [Vocos](https://github.com/gemelo-ai/vocos) for the vocoder
