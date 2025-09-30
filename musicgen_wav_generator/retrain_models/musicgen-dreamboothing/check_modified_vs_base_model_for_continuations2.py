#!/usr/bin/env python
# compare_hf_pseudo_continuation_base_vs_lora.py
"""
Fair (same-path) comparison of base vs LoRA 'continuations' using only the HF pipeline.
Continuation is *approximated* by conditioning on the seed (melody/audio), not true internal AR state carry-over.

Outputs (OUTPUT_DIR):
  seed_normalized.wav
  greedy_base_continuation.wav
  greedy_base_full.wav
  greedy_lora_continuation.wav
  greedy_lora_full.wav
  sampled_base_continuation.wav
  sampled_base_full.wav
  sampled_lora_continuation.wav
  sampled_lora_full.wav

Printed stats: duration, RMS, peak, and max abs diffs.
"""

import os, random, math
from pathlib import Path
import numpy as np
import torch, torchaudio
from transformers import AutoProcessor, AutoModelForTextToWaveform
from peft import PeftModel

# ───────── CONFIG ─────────
PROMPT               = "PIANO"
BASE_MODEL_ID        = "facebook/musicgen-melody"
LORA_PATH            = "./musicgen-melody-lora-punk"   # Set to None to compare base vs base
SEED_WAV_PATH        = "/home/alels_star/Desktop/AI_composition_assistant/v0.01/musicgen_wav_generator/retrain_models/training_songs/train/04 Dynamique De La Résonance_part004.wav"

CONTINUATION_SEC     = 10          # desired continuation length
TOKENS_PER_SEC_EST   = 40          # adjust if length off (≈40 tokens/sec)

GUIDANCE_SCALE       = 3.0         # keep same as training (sampled mode)
GREEDY_GUIDANCE      = 1000.0 # or set 1.0 for pure deterministic baseline

TOP_K                = 250
TEMPERATURE          = 20.0

GREEDY_SEED          = 1146
SAMPLED_SEED         = 12374

USE_CROSSFADE        = False
CROSSFADE_MS         = 300
ALIGN_LENGTH         = True
SAVE_SEED_REF        = True
# If set, only the first L samples of the (resampled, mono, normalized) seed are used.
# L is in *samples* at the internal SR (default 32 kHz). Example: for first 5 seconds at 32 kHz → 32_000*5 = 160_000.
SEED_FIRST_L_SAMPLES = 300000  # e.g., 160_000  (set to None to use the whole file)


OUTPUT_DIR           = Path("./hf_pseudo_continuation_outputs")
DEVICE               = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE                = torch.float16 if DEVICE == "cuda" else torch.float32
USE_AMP              = (DEVICE == "cuda")
# ──────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Device: {DEVICE} dtype={DTYPE}")

def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_and_norm_seed(path: Path, target_sr: int = 32_000):
    wav, sr = torchaudio.load(str(path))
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    else:
        wav = wav[:1]
    wav = wav / wav.abs().max().clamp(min=1e-9)
    return wav.to(torch.float32), target_sr

def crossfade_concat(a: torch.Tensor, b: torch.Tensor, sr: int, ms: int):
    fade_samples = int(sr * ms / 1000)
    fade_samples = min(
        fade_samples,
        a.shape[-1] // 4 if a.shape[-1] else 0,
        b.shape[-1] // 4 if b.shape[-1] else 0
    )
    if fade_samples <= 0:
        return torch.cat([a, b], -1)
    fade_out = torch.linspace(1, 0, fade_samples, device=a.device)
    fade_in  = torch.linspace(0, 1, fade_samples, device=b.device)
    mixed = a[..., -fade_samples:] * fade_out + b[..., :fade_samples] * fade_in
    return torch.cat([a[..., :-fade_samples], mixed, b[..., fade_samples:]], -1)

def describe(wav: torch.Tensor, sr: int, label: str):
    dur = wav.shape[-1] / sr
    rms = wav.pow(2).mean().sqrt().item()
    peak = wav.abs().max().item()
    print(f"[{label}] dur={dur:.2f}s rms={rms:.5f} peak={peak:.5f}")

def save_wav(path: Path, wav: torch.Tensor, sr: int):
    torchaudio.save(str(path), wav.to(torch.float32).cpu(), sr)
    print(" Saved:", path.name)

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

# Load processor & models
print("Loading processor & base model...")
processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)
base = AutoModelForTextToWaveform.from_pretrained(
    BASE_MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True
).to(DEVICE).eval()

lora = None
if LORA_PATH:
    print("Loading LoRA adapter...")
    lora = AutoModelForTextToWaveform.from_pretrained(
        BASE_MODEL_ID, torch_dtype=DTYPE, low_cpu_mem_usage=True
    )
    lora = PeftModel.from_pretrained(lora, LORA_PATH).to(DEVICE).eval()

# Seed audio
seed_wav, SR = load_and_norm_seed(Path(SEED_WAV_PATH))

if SEED_FIRST_L_SAMPLES is not None:
    L = int(SEED_FIRST_L_SAMPLES)
    if L <= 0:
        raise ValueError("SEED_FIRST_L_SAMPLES must be > 0")
    # Cap L to the available length
    L = min(L, seed_wav.shape[-1])
    seed_wav = seed_wav[..., :L]
    print(f"Using first {L} samples of seed (~{L / SR:.2f} s at {SR} Hz).")


if SAVE_SEED_REF:
    save_wav(OUTPUT_DIR / "seed_normalized.wav", seed_wav, SR)

def prepare_inputs():
    return processor(
        text=[PROMPT],
        audio=seed_wav.squeeze(0),
        sampling_rate=SR,
        return_tensors="pt"
    )

@torch.no_grad()
def generate(model, mode: str, seed: int):
    assert mode in ("greedy","sampled")
    set_all_seeds(seed)
    inputs = prepare_inputs()
    for k,v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(DEVICE)
    gen_tokens = int(TOKENS_PER_SEC_EST * CONTINUATION_SEC)

    kwargs = dict(
        max_new_tokens=gen_tokens,
        guidance_scale = GREEDY_GUIDANCE if mode == "greedy" else GUIDANCE_SCALE,
    )
    if mode == "greedy":
        kwargs.update(dict(do_sample=False, top_k=None, temperature=1.0))
    else:
        kwargs.update(dict(do_sample=True, top_k=TOP_K, temperature=TEMPERATURE))

    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if (USE_AMP and DEVICE=="cuda") else torch.no_grad()
    with amp_ctx:
        wav = model.generate(**inputs, **kwargs)[0]   # (1, n_samples)

    wav = wav.to(torch.float32).cpu()
    if ALIGN_LENGTH:
        target = int(CONTINUATION_SEC * SR)
        if wav.shape[-1] > target:
            wav = wav[..., :target]
        elif wav.shape[-1] < target:
            wav = torch.nn.functional.pad(wav, (0, target - wav.shape[-1]))
    return wav

def assemble_full(seed_audio, cont):
    return crossfade_concat(seed_audio, cont, SR, CROSSFADE_MS) if USE_CROSSFADE else torch.cat([seed_audio, cont], -1)

results = {}
for mode, seed in [("greedy", GREEDY_SEED), ("sampled", SAMPLED_SEED)]:
    print(f"\n=== {mode.upper()} mode (seed={seed}) ===")
    base_cont = generate(base, mode, seed)
    base_full = assemble_full(seed_wav, base_cont)
    save_wav(OUTPUT_DIR / f"{mode}_base_continuation.wav", base_cont, SR)
    save_wav(OUTPUT_DIR / f"{mode}_base_full.wav", base_full, SR)
    describe(base_cont, SR, f"{mode}_base_cont")

    if lora:
        lora_cont = generate(lora, mode, seed)
        lora_full = assemble_full(seed_wav, lora_cont)
        save_wav(OUTPUT_DIR / f"{mode}_lora_continuation.wav", lora_cont, SR)
        save_wav(OUTPUT_DIR / f"{mode}_lora_full.wav", lora_full, SR)
        describe(lora_cont, SR, f"{mode}_lora_cont")

        diff_cont = max_abs_diff(base_cont, lora_cont)
        diff_full = max_abs_diff(base_full, lora_full)
        print(f"max_abs_diff (continuation): {diff_cont:.6e}")
        print(f"max_abs_diff (full):         {diff_full:.6e}")
        results[mode] = dict(
            cont_diff=diff_cont,
            base_rms=base_cont.pow(2).mean().sqrt().item(),
            lora_rms=lora_cont.pow(2).mean().sqrt().item()
        )

print("\n=== Summary ===")
for k,v in results.items():
    print(f"{k}: cont_diff={v['cont_diff']:.6e} base_rms={v['base_rms']:.5f} lora_rms={v['lora_rms']:.5f}")

print("\nOutputs in:", OUTPUT_DIR.resolve())
