"""
TTS engines — IndicF5 and Indic Parler TTS.
"""
import torch
import numpy as np
import io
import subprocess
import soundfile as sf
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from safetensors.torch import load_file
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT

INDICF5_LANGUAGES = [
    "assamese", "bengali", "gujarati", "hindi", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu",
]

PARLER_LANGUAGES = [
    "assamese", "bengali", "bodo", "dogri", "english", "gujarati",
    "hindi", "kannada", "konkani", "maithili", "malayalam", "manipuri",
    "marathi", "nepali", "odia", "sanskrit", "santali", "sindhi",
    "tamil", "telugu", "urdu",
]


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def wav_to_ogg_opus(wav_bytes: bytes) -> bytes:
    """Convert WAV bytes to OGG/Opus using ffmpeg (WhatsApp compatible)."""
    proc = subprocess.run(
        ["ffmpeg", "-i", "pipe:0", "-c:a", "libopus", "-b:a", "64k",
         "-f", "ogg", "pipe:1"],
        input=wav_bytes,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode()}")
    return proc.stdout


def _postprocess_audio(audio_np: np.ndarray, sample_rate: int) -> bytes:
    """Remove silence, normalize loudness, return WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, samplerate=sample_rate, format="WAV")
    buf.seek(0)
    seg = AudioSegment.from_file(buf, format="wav")

    non_silent = silence.split_on_silence(
        seg, min_silence_len=1000, silence_thresh=-50,
        keep_silence=500, seek_step=10,
    )
    if non_silent:
        seg = sum(non_silent, AudioSegment.silent(duration=0))

    seg = seg.apply_gain(-20.0 - seg.dBFS)

    final = np.array(seg.get_array_of_samples())
    if final.dtype == np.int16:
        final = final.astype(np.float32) / 32768.0

    out = io.BytesIO()
    sf.write(out, np.array(final, dtype=np.float32), samplerate=sample_rate, format="WAV")
    out.seek(0)
    return out.read()


# ---------------------------------------------------------------------------
# IndicF5 Engine
# ---------------------------------------------------------------------------

class IndicF5TTS:
    def __init__(self, device: str):
        self.device = device
        repo = "ai4bharat/IndicF5"

        self.ref_audio_path = hf_hub_download(repo, "prompts/PAN_F_HAPPY_00001.wav")
        self.ref_text = (
            "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ "
            "ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
        )
        vocab_path = hf_hub_download(repo, "checkpoints/vocab.txt")

        print("  Loading IndicF5 vocoder...")
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

        print("  Loading IndicF5 DiT model...")
        self.ema_model = load_model(
            DiT,
            dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            device=device,
        )

        safetensors_path = hf_hub_download(repo, "model.safetensors")
        raw_sd = load_file(safetensors_path, device="cpu")
        ema_sd, vocoder_sd = {}, {}
        for k, v in raw_sd.items():
            if k.startswith("ema_model._orig_mod."):
                ema_sd[k.replace("ema_model._orig_mod.", "")] = v
            elif k.startswith("vocoder._orig_mod."):
                vocoder_sd[k.replace("vocoder._orig_mod.", "")] = v

        self.ema_model.load_state_dict(ema_sd, strict=False)
        self.vocoder.load_state_dict(vocoder_sd, strict=False)
        self.ema_model.to(device)
        self.vocoder.to(device)
        print("  IndicF5 ready.")

    def synthesize(
        self, text: str, speed: float = 1.0, nfe_step: int = 16, cfg_strength: float = 0.0
    ) -> bytes:
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio_path, self.ref_text)
        audio, sr, _ = infer_process(
            ref_audio, ref_text, text,
            self.ema_model, self.vocoder,
            mel_spec_type="vocos", speed=speed,
            nfe_step=nfe_step, cfg_strength=cfg_strength,
            sway_sampling_coef=-1.0, device=self.device,
        )
        return _postprocess_audio(audio, 24000)


# ---------------------------------------------------------------------------
# Indic Parler TTS Engine
# ---------------------------------------------------------------------------

class ParlerTTS:
    def __init__(self, device: str):
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self.device = device
        repo = "ai4bharat/indic-parler-tts"

        print("  Loading Indic Parler TTS model...")
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(repo).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(repo)
        self.desc_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self.sample_rate = self.model.config.sampling_rate
        print("  Indic Parler TTS ready.")

    def synthesize(
        self, text: str, description: str | None = None, speaker: str | None = None,
    ) -> bytes:
        if description is None:
            name = speaker or "Divya"
            description = (
                f"{name} speaks with a clear, high-quality voice at a moderate pace. "
                "The recording is of very high quality, with the speaker's voice "
                "sounding clear and very close up."
            )

        desc_ids = self.desc_tokenizer(description, return_tensors="pt").to(self.device)
        prompt_ids = self.tokenizer(text, return_tensors="pt").to(self.device)

        generation = self.model.generate(
            input_ids=desc_ids.input_ids,
            attention_mask=desc_ids.attention_mask,
            prompt_input_ids=prompt_ids.input_ids,
        )
        audio = generation.cpu().numpy().squeeze()
        return _postprocess_audio(audio, self.sample_rate)
