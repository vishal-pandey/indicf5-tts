"""
IndicF5 TTS — Multi-language samples on Apple Silicon (MPS).
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

# Download reference prompt audios
ref_punjabi = hf_hub_download("ai4bharat/IndicF5", "prompts/PAN_F_HAPPY_00001.wav")
ref_marathi = hf_hub_download("ai4bharat/IndicF5", "prompts/MAR_F_HAPPY_00001.wav")
vocab_path = hf_hub_download("ai4bharat/IndicF5", "checkpoints/vocab.txt")

# Load vocoder
print("Loading vocoder...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)

# Load model
print("Loading IndicF5 model...")
ema_model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    mel_spec_type="vocos",
    vocab_file=vocab_path,
    device=device,
)

safetensors_path = hf_hub_download("ai4bharat/IndicF5", "model.safetensors")
raw_sd = load_file(safetensors_path, device="cpu")

ema_sd = {}
vocoder_sd = {}
for k, v in raw_sd.items():
    if k.startswith("ema_model._orig_mod."):
        ema_sd[k.replace("ema_model._orig_mod.", "")] = v
    elif k.startswith("vocoder._orig_mod."):
        vocoder_sd[k.replace("vocoder._orig_mod.", "")] = v

ema_model.load_state_dict(ema_sd, strict=False)
vocoder.load_state_dict(vocoder_sd, strict=False)
ema_model.to(device)
vocoder.to(device)
print("Model loaded.\n")

# Reference texts for the bundled audios
ref_punjabi_text = "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
ref_marathi_text = "हम्पी मधील स्मारकांच्या वास्तुकलेचे तपशील गुंतागुंतीचे आणि विस्मयकारक आहेत, ज्यामुळे मला आनंद होतो."

# Samples to generate: (language, text, ref_audio, ref_text, output_file)
samples = [
    (
        "Tamil",
        "வணக்கம்! இசையைப் போல வாழ்க்கையும் அழகானது, சரியான தாளத்தில் வாழ கற்றுக்கொள்ள வேண்டும்.",
        ref_punjabi, ref_punjabi_text,
        "tamil_sample.wav",
    ),
    (
        "Bengali",
        "নমস্কার! সংগীতের মতো জীবনও সুন্দর, শুধু সঠিক তালে বাঁচতে শিখতে হবে।",
        ref_punjabi, ref_punjabi_text,
        "bengali_sample.wav",
    ),
    (
        "Telugu",
        "నమస్కారం! సంగీతం లాగే జీవితం కూడా అందమైనది, సరైన లయలో జీవించడం నేర్చుకోవాలి.",
        ref_punjabi, ref_punjabi_text,
        "telugu_sample.wav",
    ),
    (
        "Marathi",
        "नमस्कार! संगीतासारखे आयुष्यही सुंदर असते, फक्त योग्य तालात जगायला शिकले पाहिजे.",
        ref_marathi, ref_marathi_text,
        "marathi_sample.wav",
    ),
    (
        "Gujarati",
        "નમસ્તે! સંગીતની જેમ જીવન પણ સુંદર છે, બસ યોગ્ય તાલમાં જીવતા શીખવું જોઈએ.",
        ref_punjabi, ref_punjabi_text,
        "gujarati_sample.wav",
    ),
]


def postprocess_and_save(audio_np, output_path):
    buffer = io.BytesIO()
    sf.write(buffer, audio_np, samplerate=24000, format="WAV")
    buffer.seek(0)
    audio_segment = AudioSegment.from_file(buffer, format="wav")

    non_silent_segs = silence.split_on_silence(
        audio_segment, min_silence_len=1000, silence_thresh=-50,
        keep_silence=500, seek_step=10,
    )
    if non_silent_segs:
        audio_segment = sum(non_silent_segs, AudioSegment.silent(duration=0))

    target_dBFS = -20.0
    audio_segment = audio_segment.apply_gain(target_dBFS - audio_segment.dBFS)

    final = np.array(audio_segment.get_array_of_samples())
    if final.dtype == np.int16:
        final = final.astype(np.float32) / 32768.0
    sf.write(output_path, np.array(final, dtype=np.float32), samplerate=24000)
    return len(final) / 24000


for lang, text, ref_path, ref_txt, out_file in samples:
    print(f"--- {lang} ---")
    ref_audio, ref_text = preprocess_ref_audio_text(ref_path, ref_txt)

    t0 = time.time()
    audio, sr, _ = infer_process(
        ref_audio, ref_text, text,
        ema_model, vocoder,
        mel_spec_type="vocos", speed=1.0, device=device,
    )
    elapsed = time.time() - t0

    duration = postprocess_and_save(audio, out_file)
    print(f"  Saved {out_file} ({duration:.1f}s audio, generated in {elapsed:.1f}s)\n")

print("All done!")
