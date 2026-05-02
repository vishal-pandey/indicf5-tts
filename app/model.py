"""
IndicF5 model loader — handles weight remapping and device selection.
"""
import torch
import numpy as np
import io
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

REPO_ID = "ai4bharat/IndicF5"

SUPPORTED_LANGUAGES = [
    "assamese", "bengali", "gujarati", "hindi", "kannada",
    "malayalam", "marathi", "odia", "punjabi", "tamil", "telugu",
]


class IndicF5TTS:
    def __init__(self):
        self.device = self._select_device()
        print(f"Using device: {self.device}")

        # Download assets
        self.ref_audio_path = hf_hub_download(REPO_ID, "prompts/PAN_F_HAPPY_00001.wav")
        self.ref_text = (
            "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ "
            "ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
        )
        vocab_path = hf_hub_download(REPO_ID, "checkpoints/vocab.txt")

        # Load vocoder
        print("Loading vocoder...")
        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=self.device)

        # Load DiT model
        print("Loading IndicF5 DiT model...")
        self.ema_model = load_model(
            DiT,
            dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            device=self.device,
        )

        # Load and remap weights from safetensors
        safetensors_path = hf_hub_download(REPO_ID, "model.safetensors")
        raw_sd = load_file(safetensors_path, device="cpu")

        ema_sd = {}
        vocoder_sd = {}
        for k, v in raw_sd.items():
            if k.startswith("ema_model._orig_mod."):
                ema_sd[k.replace("ema_model._orig_mod.", "")] = v
            elif k.startswith("vocoder._orig_mod."):
                vocoder_sd[k.replace("vocoder._orig_mod.", "")] = v

        self.ema_model.load_state_dict(ema_sd, strict=False)
        self.vocoder.load_state_dict(vocoder_sd, strict=False)
        self.ema_model.to(self.device)
        self.vocoder.to(self.device)
        print("Model ready.")

    @staticmethod
    def _select_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def synthesize(
        self, text: str, speed: float = 1.0, nfe_step: int = 16, cfg_strength: float = 0.0
    ) -> bytes:
        """
        Synthesize speech from text. Returns WAV bytes.

        Args:
            text: Text to synthesize.
            speed: Speech speed multiplier.
            nfe_step: Number of ODE solver steps (fewer = faster, 16 is good quality).
            cfg_strength: Classifier-free guidance strength (0 = no guidance, fastest).
        """
        ref_audio, ref_text = preprocess_ref_audio_text(
            self.ref_audio_path, self.ref_text
        )

        audio, sr, _ = infer_process(
            ref_audio,
            ref_text,
            text,
            self.ema_model,
            self.vocoder,
            mel_spec_type="vocos",
            speed=speed,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1.0,
            device=self.device,
        )

        # Post-process: remove silence, normalize loudness
        buf = io.BytesIO()
        sf.write(buf, audio, samplerate=24000, format="WAV")
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
        sf.write(out, np.array(final, dtype=np.float32), samplerate=24000, format="WAV")
        out.seek(0)
        return out.read()
