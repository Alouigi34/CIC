#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 01:42:25 2025

@author: alels_star
"""

#!/usr/bin/env python
# compare_musicgen_base_vs_lora.py
import torch, torchaudio, gc
from transformers import AutoProcessor, AutoModelForTextToWaveform
from peft import PeftModel

# ───────────────────────────────────────────────
# SETTINGS ─ change only the bits below
# ───────────────────────────────────────────────
PROMPT      = "a classical mellody"
BASE_MODEL  = "facebook/musicgen-melody"
LORA_PATH   = "./musicgen-melody-lora-punk"   # your trained adapter
OUT_SR      = 32_000                          # MusicGen default
GEN_TOKENS  = 400                             # ≈ 10 s of audio
GUIDANCE    = 3.0                             # keep same as training
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE       = torch.float16 if DEVICE == "cuda" else torch.float32
# ───────────────────────────────────────────────

processor = AutoProcessor.from_pretrained(BASE_MODEL)

def generate_and_save(model, file_name: str):
    model.to(DEVICE).eval()
    inputs = processor.tokenizer(PROMPT, return_tensors="pt").to(DEVICE)

    with torch.cuda.amp.autocast(enabled=DTYPE == torch.float16):
        wav = model.generate(
            **inputs,
            max_new_tokens=GEN_TOKENS,
            guidance_scale=GUIDANCE,
        )[0]                       # shape: (1, n_samples)

    # 1. cast to float32              (torchaudio-friendly dtype)
    # 2. move to CPU
    wav = wav.to(dtype=torch.float32, device="cpu")

    # (1, n_samples) is already the correct 2-D shape → do **not** unsqueeze
    torchaudio.save(file_name, wav, OUT_SR)   # channels_first=True by default
    print(f"✔ Saved → {file_name}")




# 1) Vanilla MusicGen-Melody
base = AutoModelForTextToWaveform.from_pretrained(BASE_MODEL,
                                                  torch_dtype=DTYPE)
generate_and_save(base, "musicgen_base.wav")

# 2) MusicGen-Melody + LoRA adapter
lora = AutoModelForTextToWaveform.from_pretrained(BASE_MODEL,
                                                  torch_dtype=DTYPE)
lora = PeftModel.from_pretrained(lora, LORA_PATH).to(DEVICE)
generate_and_save(lora, "musicgen_lora.wav")
