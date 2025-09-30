#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 16:07:27 2025

@author: alels_star
"""

#!/usr/bin/env python
# compare_musicgen_base_vs_lora_dual_mode.py
"""
Compare base MusicGen vs LoRA adapter:
 - Deterministic GREEDY mode (often near silent, for exact equality testing)
 - Controlled SAMPLING mode (audible music, reproducible)

Outputs:
  greedy_base.wav
  greedy_lora.wav
  sampled_base.wav
  sampled_lora.wav

Also prints:
  * Max abs diff (greedy)
  * Max abs diff (sampled)
  * Basic audio stats (duration, RMS, peak)
"""

import os, math, random
import numpy as np
import torch, torchaudio
from transformers import AutoProcessor, AutoModelForTextToWaveform
from peft import PeftModel

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
PROMPT        = "a classical mellody"
BASE_MODEL_ID = "facebook/musicgen-melody"
LORA_PATH     = "./musicgen-melody-lora-punk"   # Set to None if you only want base
OUT_SR        = 32_000
GEN_TOKENS    = 400          # ~10 seconds if ~40 tokens/sec
GUIDANCE      = 3.0
GREEDY_SEED   = 1111         # seed for greedy equivalence test
SAMPLED_SEED  = 1234         # seed for sampled musical generation
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.float16 if (DEVICE == "cuda") else torch.float32
USE_AMP       = (DEVICE == "cuda")  # disable if you want pure fp32
OUTPUT_DIR    = "./dual_mode_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ─────────────────────────────────────────

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def describe(wav: torch.Tensor, sr: int, label: str):
    rms  = wav.pow(2).mean().sqrt().item()
    peak = wav.abs().max().item()
    dur  = wav.shape[-1] / sr
    print(f"[{label}] duration={dur:.2f}s rms={rms:.5f} peak={peak:.5f}")

def save_wav(path: str, wav: torch.Tensor, sr: int):
    wav = wav.to(torch.float32).cpu()
    torchaudio.save(path, wav, sr)
    print("Saved:", path)

def load_models():
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    print("Loading base model...")
    base = AutoModelForTextToWaveform.from_pretrained(
        BASE_MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    lora = None
    if LORA_PATH is not None:
        print("Loading LoRA model...")
        lora = AutoModelForTextToWaveform.from_pretrained(
            BASE_MODEL_ID, torch_dtype=DTYPE
        )
        lora = PeftModel.from_pretrained(lora, LORA_PATH).to(DEVICE).eval()
    return processor, base, lora

@torch.no_grad()
def generate(model, inputs, deterministic: bool, seed: int):
    set_all_seeds(seed)
    gen_kwargs = dict(
        max_new_tokens=GEN_TOKENS,
        guidance_scale=GUIDANCE,
    )
    if deterministic:
        gen_kwargs.update(
            dict(do_sample=False, top_k=None, temperature=1.0)
        )
    else:
        gen_kwargs.update(
            dict(do_sample=True, top_k=250, temperature=1.1)
        )

    ctx = (
        torch.autocast("cuda", dtype=torch.float16)
        if (USE_AMP and DEVICE == "cuda")
        else torch.enable_grad()  # dummy context; no grad anyway
    )
    with ctx:
        wav = model.generate(**inputs, **gen_kwargs)[0]  # (1, n_samples)
    return wav

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()

def main():
    print(f"Device: {DEVICE} dtype={DTYPE}")
    processor, base, lora = load_models()

    # Prepare text inputs once
    text_inputs = processor.tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    # ========== GREEDY (Deterministic) MODE ==========
    print("\n=== GREEDY MODE (deterministic, likely near-silent) ===")
    wav_base_greedy = generate(base, text_inputs, deterministic=True, seed=GREEDY_SEED)
    save_wav(os.path.join(OUTPUT_DIR, "greedy_base.wav"), wav_base_greedy, OUT_SR)
    describe(wav_base_greedy, OUT_SR, "greedy_base")

    if lora is not None:
        wav_lora_greedy = generate(lora, text_inputs, deterministic=True, seed=GREEDY_SEED)
        save_wav(os.path.join(OUTPUT_DIR, "greedy_lora.wav"), wav_lora_greedy, OUT_SR)
        describe(wav_lora_greedy, OUT_SR, "greedy_lora")
        print("Greedy MAX ABS DIFF:", max_abs_diff(
            wav_base_greedy.to(torch.float32),
            wav_lora_greedy.to(torch.float32)
        ))

    # ========== SAMPLED (Musical) MODE ==========
    print("\n=== SAMPLED MODE (audible, reproducible seed) ===")
    wav_base_sampled = generate(base, text_inputs, deterministic=False, seed=SAMPLED_SEED)
    save_wav(os.path.join(OUTPUT_DIR, "sampled_base.wav"), wav_base_sampled, OUT_SR)
    describe(wav_base_sampled, OUT_SR, "sampled_base")

    if lora is not None:
        wav_lora_sampled = generate(lora, text_inputs, deterministic=False, seed=SAMPLED_SEED)
        save_wav(os.path.join(OUTPUT_DIR, "sampled_lora.wav"), wav_lora_sampled, OUT_SR)
        describe(wav_lora_sampled, OUT_SR, "sampled_lora")
        print("Sampled MAX ABS DIFF:", max_abs_diff(
            wav_base_sampled.to(torch.float32),
            wav_lora_sampled.to(torch.float32)
        ))

    # Optional: If you want to assert identity when expecting zero LoRA
    if lora is not None:
        expected_identical = False  # set True if you expect LoRA == zero
        if expected_identical:
            assert max_abs_diff(
                wav_base_sampled.to(torch.float32),
                wav_lora_sampled.to(torch.float32)
            ) < 1e-6, "LoRA outputs differ unexpectedly."

    print("\nDone.")

if __name__ == "__main__":
    main()
