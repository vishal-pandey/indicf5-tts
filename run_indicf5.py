"""
IndicF5 TTS inference on Apple Silicon (MPS).
Manually loads components to avoid torch.compile/meta device issues.
"""
import torch
import numpy as np
import soundfile as sf
import time
import io
from huggingface_hub import hf_hub_download
from pydub import AudioSegment, silence
from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT
from safetensors.torch import load_file

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Download reference prompt audio
ref_audio_path = hf_hub_download("ai4bharat/IndicF5", "prompts/PAN_F_HAPPY_00001.wav")
print(f"Reference audio: {ref_audio_path}")

# Download vocab
vocab_path = hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")

# Load vocoder
print("Loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

# Load DiT model (empty weights)
print("Loading IndicF5 model...")
t0 = time.time()
ema_model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    device=device,
)

# Load safetensors and remap keys
safetensors_path = hf_hub_download("ai4bharat/IndicF5", "model.safetensors")
print(f"Loading weights from {safetensors_path}...")
raw_sd = load_file(safetensors_path, device="cpu")

# The saved keys look like:
#   ema_model._orig_mod.transformer.xxx  -> goes into ema_model (CFM wrapper with .transformer)
#   vocoder._orig_mod.xxx                -> goes into vocoder
ema_sd = {}
vocoder_sd = {}
for k, v in raw_sd.items():
    if k.startswith("ema_model._orig_mod."):
        new_key = k.replace("ema_model._orig_mod.", "")
        ema_sd[new_key] = v
    elif k.startswith("vocoder._orig_mod."):
        new_key = k.replace("vocoder._orig_mod.", "")
        vocoder_sd[new_key] = v

# Load into models
missing, unexpected = ema_model.load_state_dict(ema_sd, strict=False)
print(f"EMA model - missing: {len(missing)}, unexpected: {len(unexpected)}")
if missing:
    print(f"  Missing keys (first 5): {missing[:5]}")

missing2, unexpected2 = vocoder.load_state_dict(vocoder_sd, strict=False)
print(f"Vocoder - missing: {len(missing2)}, unexpected: {len(unexpected2)}")

ema_model.to(device)
vocoder.to(device)
print(f"Model loaded in {time.time() - t0:.1f}s")

# Prepare reference audio
ref_audio, ref_text = preprocess_ref_audio_text(
    ref_audio_path,
    "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# Generate speech
text = "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए."
print(f"Generating speech...")
t0 = time.time()

audio, final_sample_rate, _ = infer_process(
    ref_audio,
    ref_text,
    text,
    ema_model,
    vocoder,
    mel_spec_type="vocos",
    speed=1.0,
    device=device,
)

elapsed = time.time() - t0
print(f"Generation took {elapsed:.1f}s")

# Post-process: remove silence and normalize
buffer = io.BytesIO()
sf.write(buffer, audio, samplerate=24000, format="WAV")
buffer.seek(0)
audio_segment = AudioSegment.from_file(buffer, format="wav")

non_silent_segs = silence.split_on_silence(
    audio_segment,
    min_silence_len=1000,
    silence_thresh=-50,
    keep_silence=500,
    seek_step=10,
)
if non_silent_segs:
    non_silent_wave = sum(non_silent_segs, AudioSegment.silent(duration=0))
    audio_segment = non_silent_wave

# Normalize loudness
target_dBFS = -20.0
change_in_dBFS = target_dBFS - audio_segment.dBFS
audio_segment = audio_segment.apply_gain(change_in_dBFS)

# Save
final_audio = np.array(audio_segment.get_array_of_samples())
if final_audio.dtype == np.int16:
    final_audio = final_audio.astype(np.float32) / 32768.0

output_path = "namaste.wav"
sf.write(output_path, np.array(final_audio, dtype=np.float32), samplerate=24000)
duration = len(final_audio) / 24000
print(f"Audio saved to {output_path} ({duration:.1f}s of audio)")
print(f"Real-time factor: {elapsed/duration:.2f}x")
