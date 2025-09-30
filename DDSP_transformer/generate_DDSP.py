#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_DDSP.py — CLI for DDSP timbre-transfer
"""

import os
import time
import argparse
import pickle
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import soundfile as sf
import crepe
import ddsp
import ddsp.training
from ddsp.training.postprocessing import detect_notes, fit_quantile_transform
import ddsp.core
import gin
import librosa
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import colab_utils
from colab_utils import (
    auto_tune,
    get_tuning_factor,
    specplot,
    upload,
    audio_bytes_to_np,
    DEFAULT_SAMPLE_RATE
)

# ───────────────── Helpers ─────────────────
def shift_f0(af, octaves):
    af["f0_hz"] = af["f0_hz"] * (2.0 ** octaves)
    af["f0_hz"] = np.clip(
        af["f0_hz"],
        0.0,
        librosa.midi_to_hz(110.0)
    )
    return af

def shift_ld(af, db):
    af["loudness_db"] = af["loudness_db"] + db
    return af

# ───────────────── argparse ─────────────────
parser = argparse.ArgumentParser(description="Generate timbre-transfer with DDSP")
parser.add_argument("--organ",  required=True, help="Model: Violin, Flute, …")
parser.add_argument("--input",  required=True, help="Input WAV/MP3")
parser.add_argument("--output", default=None,  help="Out WAV (defaults to <in>_out.wav)")
args = parser.parse_args()

if args.output is None:
    args.output = args.input + "_out.wav"

# ───────────────── Paths ─────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "DDSP_models"

# ───────────────── Load audio & features ─────────────────
filenames, audios = upload(args.input)
audio = audios[0]
if audio.ndim == 1:
    audio = audio[np.newaxis, :]

print("Extracting audio features…")
ddsp.spectral_ops.reset_crepe()
t0 = time.time()
audio_features = ddsp.training.metrics.compute_audio_features(audio)
print(f"Features extracted in {time.time()-t0:.1f}s")

# ───────────────── Load & configure model ─────────────────
m = args.organ.lower()
ckpt_dir = MODELS_DIR / f"solo_{m}_ckpt"
gin_file = ckpt_dir / "operative_config-0.gin"
stats_f  = ckpt_dir / "dataset_statistics.pkl"

print(f"Loading stats from {stats_f}")
with tf.io.gfile.GFile(str(stats_f), "rb") as f:
    DATASET_STATS = pickle.load(f)

with gin.unlock_config():
    gin.parse_config_file(str(gin_file), skip_unknown=True)

# find a checkpoint
ckpts = [f for f in tf.io.gfile.listdir(str(ckpt_dir)) if f.startswith("ckpt")]
if not ckpts:
    raise RuntimeError(f"No checkpoint in {ckpt_dir}")
ckpt = ckpt_dir / ckpts[0].split(".")[0]

# match time / sample lengths
t_steps_train = gin.query_parameter("F0LoudnessPreprocessor.time_steps")
n_samp_train  = gin.query_parameter("Harmonic.n_samples")
hop = int(n_samp_train / t_steps_train)
t_steps = int(audio.shape[1] / hop)
n_samp   = t_steps * hop

with gin.unlock_config():
    gin.parse_config([
        f"Harmonic.n_samples = {n_samp}",
        f"FilteredNoise.n_samples = {n_samp}",
        f"F0LoudnessPreprocessor.time_steps = {t_steps}",
        "oscillator_bank.use_angular_cumsum = True"
    ])

# trim features
for k in ("f0_hz","f0_confidence","loudness_db"):
    audio_features[k] = audio_features[k][:t_steps]
audio_features["audio"] = audio_features["audio"][:,:n_samp]

# build & restore
model = ddsp.training.models.Autoencoder()
model.restore(str(ckpt))
_ = model(audio_features, training=False)

# ───────────────── Adjust conditioning ─────────────────
threshold      = 1.0
quiet_db       = 20
autotune_amt   = 0.2
pitch_shift    = 0     # manual pitch shift in octaves
loudness_shift = 0     # manual loudness shift in dB

# detect_notes returns Tensors—convert to NumPy
mask_on, note_on = detect_notes(
    audio_features["loudness_db"],
    audio_features["f0_confidence"],
    threshold
)
mask_on = np.array(mask_on)
note_on = np.array(note_on)

audio_features_mod = {k: tf.identity(v) for k,v in audio_features.items()}

if mask_on.any():
    # auto pitch-register shift
    midi = np.array(ddsp.core.hz_to_midi(audio_features["f0_hz"]))
    mean_pitch = midi[mask_on].mean()
    diff = (DATASET_STATS["mean_pitch"] - mean_pitch) / 12.0
    octs = np.floor(diff) if diff > 1.5 else np.ceil(diff)
    audio_features_mod = shift_f0(audio_features_mod, octs)

    # quantile loudness transform: convert to NumPy so .copy() works
    loud_db_np = np.array(audio_features["loudness_db"])
    _, loud_norm = fit_quantile_transform(
        loud_db_np,
        mask_on,
        inv_quantile=DATASET_STATS["quantile_transform"]
    )
    off = ~mask_on
    loud_norm[off] -= quiet_db * (1.0 - note_on[off][:,None])
    audio_features_mod["loudness_db"] = loud_norm

    if autotune_amt > 0:
        tf_midi = np.array(ddsp.core.hz_to_midi(audio_features_mod["f0_hz"]))
        tune   = get_tuning_factor(tf_midi, audio_features["f0_confidence"], mask_on)
        at     = auto_tune(tf_midi, tune, mask_on, amount=autotune_amt)
        audio_features_mod["f0_hz"] = ddsp.core.midi_to_hz(at)
else:
    print("No notes detected; skipping auto-adjust")

# manual shifts
audio_features_mod = shift_ld(audio_features_mod, loudness_shift)
audio_features_mod = shift_f0(audio_features_mod, pitch_shift)

# ───────────────── Resynthesize & write ─────────────────
t1 = time.time()
out = model(audio_features_mod, training=False)
y   = model.get_audio_from_outputs(out)
print(f"Resynthesis took {time.time()-t1:.1f}s")

wav = np.array(y).reshape(-1,1)
pcm = (wav * np.iinfo(np.int16).max).astype(np.int16)
sf.write(args.output, pcm, DEFAULT_SAMPLE_RATE)

print("Output written to", args.output)
