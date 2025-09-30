#!/usr/bin/env python
# compare_musicgen_base_vs_lora_continuation_nocli.py
"""
Generate two continuations (Base vs LoRA) of a given seed audio piece.

Base model  : Audiocraft MusicGen native continuation (true continuation).
LoRA model  : HF Transformers + PEFT; approximates continuation via melody conditioning.

Outputs (./generated_comparison):
  <PREFIX>_seed_normalized.wav (if SAVE_SEED_REF)
  <PREFIX>_base_continuation.wav
  <PREFIX>_base_full.wav
  <PREFIX>_lora_continuation.wav
  <PREFIX>_lora_full.wav
"""

import gc
from pathlib import Path
import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from transformers import AutoProcessor, AutoModelForTextToWaveform
from peft import PeftModel

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
PROMPT             = ""
BASE_MODEL_ID      = "facebook/musicgen-melody"
LORA_BASE_MODEL_ID = "facebook/musicgen-melody"
LORA_PATH          = "./musicgen-melody-lora-punk"
SEED_WAV_PATH      = "/home/alels_star/Desktop/AI_composition_assistant/v0.01/musicgen_wav_generator/retrain_models/training_songs/train/song2_fixed_a.wav"
CONTINUATION_SEC   = 30
GUIDANCE_SCALE     = 3.0
TOP_K              = 250
TEMPERATURE        = 1.1
CFG_COEF           = 1.4
OUTPUT_PREFIX      = "comparison"
SAVE_SEED_REF      = True
USE_CROSSFADE      = False
CROSSFADE_MS       = 300
TOKENS_PER_SEC_EST = 40          # heuristic for HF LoRA length
OUTPUT_DIR         = Path("./generated_comparison")
# ───────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
print(f"▶ Device: {DEVICE} (dtype={DTYPE})")

def ensure_path(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    return p

def load_seed(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.size(0) > 1:
        wav = wav.mean(0, keepdim=True)
    else:
        wav = wav[:1]
    peak = wav.abs().max().clamp(min=1e-9)
    wav = wav / peak
    return wav.to(torch.float32)  # ensure f32

def crossfade_concat(a: torch.Tensor, b: torch.Tensor, sr: int, fade_ms: int) -> torch.Tensor:
    fade_samples = int(sr * fade_ms / 1000)
    fade_samples = min(
        fade_samples,
        a.shape[-1] // 4 if a.shape[-1] > 0 else 0,
        b.shape[-1] // 4 if b.shape[-1] > 0 else 0
    )
    if fade_samples <= 0:
        return torch.cat([a, b], dim=-1)
    fade_out = torch.linspace(1, 0, fade_samples, device=a.device)
    fade_in  = torch.linspace(0, 1, fade_samples, device=b.device)
    a_tail = a[..., -fade_samples:] * fade_out
    b_head = b[..., :fade_samples] * fade_in
    mixed  = a_tail + b_head
    return torch.cat([a[..., :-fade_samples], mixed, b[..., fade_samples:]], dim=-1)

def to_f32(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float32)

# ───────────────────────────────────────────────
# 1) Base model continuation
# ───────────────────────────────────────────────
print("▶ Loading base MusicGen for true continuation ...")
base_model: MusicGen = MusicGen.get_pretrained(BASE_MODEL_ID, device=DEVICE)
base_model.set_generation_params(
    duration=CONTINUATION_SEC,
    use_sampling=True,
    top_k=TOP_K,
    top_p=0,
    temperature=TEMPERATURE,
    cfg_coef=CFG_COEF,
)
sample_rate = base_model.sample_rate

seed_path = ensure_path(Path(SEED_WAV_PATH).expanduser().resolve())
print("▶ Loading and normalizing seed audio ...")
seed_wav = load_seed(seed_path, sample_rate)      # (1, n)
seed_batch = seed_wav.unsqueeze(0)                # (B=1, 1, n)
print(f"   Seed shape: {seed_wav.shape}, sr={sample_rate}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if SAVE_SEED_REF:
    audio_write(str(OUTPUT_DIR / f"{OUTPUT_PREFIX}_seed_normalized.wav"),
                seed_wav, sample_rate, strategy="loudness")

print("▶ Generating base continuation (audiocraft) ...")
with torch.no_grad():
    if DEVICE == "cuda":
        # Use new AMP API (avoids deprecation warning)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            base_cont = base_model.generate_continuation(
                seed_batch.to(DEVICE),
                sample_rate,
                descriptions=[PROMPT],
                progress=False
            )[0]
    else:
        base_cont = base_model.generate_continuation(
            seed_batch,
            sample_rate,
            descriptions=[PROMPT],
            progress=False
        )[0]

# Cast result to float32 before audio_write (critical fix)
base_cont = to_f32(base_cont).cpu()

audio_write(str(OUTPUT_DIR / f"{OUTPUT_PREFIX}_base_continuation.wav"),
            base_cont, sample_rate, strategy="loudness")

if USE_CROSSFADE:
    base_full = crossfade_concat(seed_wav, base_cont, sample_rate, CROSSFADE_MS)
else:
    base_full = torch.cat([seed_wav, base_cont], dim=-1)

audio_write(str(OUTPUT_DIR / f"{OUTPUT_PREFIX}_base_full.wav"),
            base_full, sample_rate, strategy="loudness")
print("✓ Base continuation done.")

# Free memory
del base_model
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# ───────────────────────────────────────────────
# 2) LoRA model approximate continuation
# ───────────────────────────────────────────────
print("▶ Loading HF processor & base model for LoRA ...")
processor = AutoProcessor.from_pretrained(LORA_BASE_MODEL_ID)

print("▶ Loading HF base model ...")
hf_model = AutoModelForTextToWaveform.from_pretrained(
    LORA_BASE_MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True
)

print("▶ Loading LoRA adapter ...")
hf_model = PeftModel.from_pretrained(hf_model, LORA_PATH)
hf_model.to(DEVICE).eval()

# Melody conditioning
seed_for_processor = seed_wav.squeeze(0)  # (samples,)
inputs = processor(
    text=[PROMPT],
    audio=seed_for_processor,
    sampling_rate=sample_rate,
    return_tensors="pt"
)

gen_tokens = int(TOKENS_PER_SEC_EST * CONTINUATION_SEC)
print(f"▶ Approx continuation tokens (LoRA): {gen_tokens}")

for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        inputs[k] = v.to(DEVICE)

print("▶ Generating LoRA conditioned segment ...")
with torch.no_grad():
    if DEVICE == "cuda":
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            lora_gen = hf_model.generate(
                **inputs,
                max_new_tokens=gen_tokens,
                guidance_scale=GUIDANCE_SCALE,
            )[0]
    else:
        lora_gen = hf_model.generate(
            **inputs,
            max_new_tokens=gen_tokens,
            guidance_scale=GUIDANCE_SCALE,
        )[0]

lora_cont = to_f32(lora_gen).cpu()

audio_write(str(OUTPUT_DIR / f"{OUTPUT_PREFIX}_lora_continuation.wav"),
            lora_cont, sample_rate, strategy="loudness")

if USE_CROSSFADE:
    lora_full = crossfade_concat(seed_wav, lora_cont, sample_rate, CROSSFADE_MS)
else:
    lora_full = torch.cat([seed_wav, lora_cont], dim=-1)

audio_write(str(OUTPUT_DIR / f"{OUTPUT_PREFIX}_lora_full.wav"),
            lora_full, sample_rate, strategy="loudness")

print("✓ LoRA pseudo-continuation done.")

print("\n✔ All files written to:", OUTPUT_DIR.resolve())
for p in sorted(OUTPUT_DIR.iterdir()):
    print("  •", p.name)
