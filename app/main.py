"""
IndicF5 + Indic Parler TTS API
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from enum import Enum
from app.model import (
    IndicF5TTS, ParlerTTS, select_device, wav_to_ogg_opus,
    INDICF5_LANGUAGES, PARLER_LANGUAGES,
)

DESCRIPTION = """
## Indic TTS API

Generate speech in Indian languages using two engines:

### Engines

| Engine | Languages | Speed (Mac MPS) | Features |
|--------|-----------|-----------------|----------|
| **indicf5** | 11 languages | ~1.5x RTF (fast) | Reference-audio cloning |
| **parler** | 21 languages | ~6-10x RTF (slower) | Named speakers, emotions, style control |

### Output Format

Returns **OGG/Opus** audio by default (WhatsApp compatible).
Set `format=wav` to get WAV instead.

---

### IndicF5 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nfe_step` | 16 | Diffusion steps (8=fastest, 16=balanced, 32=best) |
| `cfg_strength` | 0.0 | Guidance strength (0=fastest, >0 doubles compute) |
| `speed` | 1.0 | Speech rate (0.5–2.0) |

### Parler Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `speaker` | Divya | Named speaker (e.g. Rohit, Divya, Arjun, Aditi) |
| `description` | auto | Full voice description (overrides speaker) |

### Parler Speakers by Language

| Language | Recommended |
|----------|-------------|
| Hindi | Rohit, Divya, Aman, Rani |
| Bengali | Arjun, Aditi |
| Tamil | Jaya, Kavitha |
| Telugu | Prakash, Lalitha |
| Marathi | Sanjay, Sunita |
| Kannada | Suresh, Anu |
| Malayalam | Anjali, Harish |
| Gujarati | Yash, Neha |
| English | Thoma, Mary |

See full list at the model page.

---

### Example Requests

**IndicF5 (fast, Hindi):**
```json
{"text": "नमस्ते, आप कैसे हैं?", "engine": "indicf5"}
```

**Parler (with speaker):**
```json
{"text": "नमस्ते, आप कैसे हैं?", "engine": "parler", "speaker": "Rohit"}
```

**Parler (with full description):**
```json
{
  "text": "Hello, how are you?",
  "engine": "parler",
  "description": "Thoma speaks with a clear British accent at a moderate pace. Very high quality recording."
}
```
"""

app = FastAPI(
    title="Indic TTS API",
    description=DESCRIPTION,
    version="2.0.0",
)

indicf5: IndicF5TTS | None = None
parler: ParlerTTS | None = None


@app.on_event("startup")
async def load_models():
    global indicf5, parler
    device = select_device()
    print(f"Device: {device}")

    print("Loading IndicF5...")
    indicf5 = IndicF5TTS(device)

    print("Loading Indic Parler TTS...")
    parler = ParlerTTS(device)

    print("All models ready.")


class EngineEnum(str, Enum):
    indicf5 = "indicf5"
    parler = "parler"


class AudioFormat(str, Enum):
    ogg = "ogg"
    wav = "wav"


class TTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=2000,
        description="Text to synthesize in any supported Indian language.",
    )
    engine: EngineEnum = Field(
        EngineEnum.indicf5,
        description="TTS engine: 'indicf5' (fast) or 'parler' (more features).",
    )
    format: AudioFormat = Field(
        AudioFormat.ogg,
        description="Output format: 'ogg' (WhatsApp/Opus) or 'wav'.",
    )
    # IndicF5 params
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Speech speed (IndicF5 only).")
    nfe_step: int = Field(16, ge=4, le=64, description="Diffusion steps (IndicF5 only).")
    cfg_strength: float = Field(0.0, ge=0.0, le=5.0, description="CFG strength (IndicF5 only).")
    # Parler params
    speaker: str | None = Field(
        None,
        description="Named speaker for Parler (e.g. Rohit, Divya, Arjun). Ignored by IndicF5.",
    )
    description: str | None = Field(
        None,
        description="Full voice description for Parler. Overrides speaker. Ignored by IndicF5.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "नमस्ते! आप कैसे हैं?",
                    "engine": "indicf5",
                    "format": "ogg",
                },
                {
                    "text": "नमस्ते! आप कैसे हैं?",
                    "engine": "parler",
                    "speaker": "Divya",
                    "format": "ogg",
                },
            ]
        }
    }


@app.post(
    "/synthesize",
    response_class=Response,
    summary="Generate speech from text",
    description="Returns audio in OGG/Opus (WhatsApp compatible) or WAV format.",
    responses={
        200: {
            "content": {
                "audio/ogg": {},
                "audio/wav": {},
            },
            "description": "Generated audio",
        },
        503: {"description": "Models still loading"},
    },
)
async def synthesize(req: TTSRequest):
    if indicf5 is None or parler is None:
        raise HTTPException(status_code=503, detail="Models are still loading")

    try:
        if req.engine == EngineEnum.indicf5:
            wav_bytes = indicf5.synthesize(
                req.text, speed=req.speed,
                nfe_step=req.nfe_step, cfg_strength=req.cfg_strength,
            )
        else:
            wav_bytes = parler.synthesize(
                req.text, description=req.description, speaker=req.speaker,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if req.format == AudioFormat.ogg:
        audio_bytes = wav_to_ogg_opus(wav_bytes)
        media_type = "audio/ogg"
    else:
        audio_bytes = wav_bytes
        media_type = "audio/wav"

    return Response(content=audio_bytes, media_type=media_type)


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ready" if (indicf5 and parler) else "loading",
        "engines": {
            "indicf5": indicf5 is not None,
            "parler": parler is not None,
        },
    }


@app.get("/languages", summary="List supported languages per engine")
async def languages():
    return {
        "indicf5": INDICF5_LANGUAGES,
        "parler": PARLER_LANGUAGES,
    }
