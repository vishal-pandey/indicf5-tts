"""
IndicF5 TTS API
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from app.model import IndicF5TTS, SUPPORTED_LANGUAGES

DESCRIPTION = """
## IndicF5 Text-to-Speech API

Generate natural-sounding speech in **11 Indian languages** using the IndicF5 model.

### Supported Languages

Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu

---

### Performance Tuning Guide

The two key parameters that control speed vs quality are **`nfe_step`** and **`cfg_strength`**.

#### `nfe_step` — Number of diffusion steps

Controls how many iterations the ODE solver takes to generate the mel spectrogram.
More steps = better quality but slower.

| Value | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| 8 | Fastest | Lower — may sound slightly robotic | Quick previews, testing |
| **16** | **Fast (recommended)** | **Good — natural sounding** | **Production default** |
| 32 | Slow | Best — subtle improvements over 16 | High-quality final renders |
| 64 | Very slow | Diminishing returns | Not recommended |

#### `cfg_strength` — Classifier-Free Guidance

Controls how strongly the model follows the text conditioning.
Higher values = more faithful to text but **doubles compute** (runs the model twice per step).

| Value | Speed | Quality | Use Case |
|-------|-------|---------|----------|
| **0.0** | **Fastest (recommended)** | **Good — natural prosody** | **Production default** |
| 1.0 | 2x slower | Slightly more expressive | When 0.0 sounds flat |
| 2.0 | 2x slower | Original default, most guided | Maximum text adherence |
| 3.0+ | 2x slower | Over-guided, can sound unnatural | Not recommended |

#### `speed` — Speech rate

Simple multiplier on the output duration. Does not affect generation time significantly.

| Value | Effect |
|-------|--------|
| 0.5 | Half speed (slow, stretched) |
| **1.0** | **Normal speed** |
| 1.5 | Faster speech |
| 2.0 | Double speed (compressed) |

---

### Recommended Presets

| Preset | nfe_step | cfg_strength | RTF* | Notes |
|--------|----------|--------------|------|-------|
| ⚡ Fastest | 8 | 0.0 | ~0.7x | Quick drafts |
| 🎯 Balanced (default) | 16 | 0.0 | ~1.5x | Best speed/quality tradeoff |
| 🎵 High Quality | 32 | 1.0 | ~6x | Final production audio |

*RTF = Real-Time Factor (seconds to generate 1 second of audio). Lower is faster.
Measured on Mac Mini M4 with 24GB unified memory using MPS.

---

### Example Requests

**Fast generation (default):**
```json
{"text": "नमस्ते, आप कैसे हैं?"}
```

**High quality:**
```json
{"text": "नमस्ते, आप कैसे हैं?", "nfe_step": 32, "cfg_strength": 1.0}
```

**Tamil:**
```json
{"text": "வணக்கம்! நீங்கள் எப்படி இருக்கிறீர்கள்?"}
```
"""

app = FastAPI(
    title="IndicF5 TTS API",
    description=DESCRIPTION,
    version="1.0.0",
)

# Load model at startup
tts: IndicF5TTS | None = None


@app.on_event("startup")
async def load_model():
    global tts
    tts = IndicF5TTS()


class TTSRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Text to synthesize. Supports Hindi, Tamil, Bengali, Telugu, Marathi, Gujarati, Kannada, Malayalam, Odia, Punjabi, and Assamese.",
    )
    speed: float = Field(
        1.0,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier. 1.0 = normal, 0.5 = slow, 2.0 = fast.",
    )
    nfe_step: int = Field(
        16,
        ge=4,
        le=64,
        description="Number of ODE solver steps. Fewer = faster. 16 is the sweet spot for speed+quality. 32 for best quality.",
    )
    cfg_strength: float = Field(
        0.0,
        ge=0.0,
        le=5.0,
        description="Classifier-Free Guidance strength. 0 = fastest (no guidance, still good quality). Values > 0 double the compute. Use 1.0-2.0 for more expressive output.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "नमस्ते! आप कैसे हैं?",
                    "speed": 1.0,
                    "nfe_step": 16,
                    "cfg_strength": 0.0,
                }
            ]
        }
    }


@app.post(
    "/synthesize",
    response_class=Response,
    summary="Generate speech from text",
    description="Converts text to speech and returns a WAV audio file (24kHz, mono, float32). "
    "Adjust nfe_step and cfg_strength to trade off speed vs quality.",
    responses={
        200: {"content": {"audio/wav": {}}, "description": "Generated WAV audio (24kHz mono)"},
        503: {"description": "Model is still loading"},
    },
)
async def synthesize(req: TTSRequest):
    """Generate speech from text. Returns a WAV audio file."""
    if tts is None:
        raise HTTPException(status_code=503, detail="Model is still loading")
    try:
        wav_bytes = tts.synthesize(
            req.text, speed=req.speed, nfe_step=req.nfe_step, cfg_strength=req.cfg_strength
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=wav_bytes, media_type="audio/wav")


@app.get("/health", summary="Health check")
async def health():
    """Check if the API and model are ready."""
    return {"status": "ready" if tts is not None else "loading"}


@app.get("/languages", summary="List supported languages")
async def languages():
    """Returns the list of supported languages."""
    return {"languages": SUPPORTED_LANGUAGES}
