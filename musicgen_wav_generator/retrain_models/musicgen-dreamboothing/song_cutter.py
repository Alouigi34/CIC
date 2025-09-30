#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 01:05:54 2025

@author: alels_star
"""

#!/usr/bin/env python3
"""
split_wavs_10s.py – cut every .wav found under TRAINING_DIR into 10-second chunks
Requires: python -m pip install pydub soundfile
           (and ffmpeg/libav in your PATH; on Ubuntu: sudo apt install ffmpeg)
"""

from pathlib import Path
from pydub import AudioSegment

TRAINING_DIR = Path(
    "/home/alels_star/Desktop/AI_composition_assistant/"
    "v0.01/musicgen_wav_generator/retrain_models/training_songs"
)
CHUNK_SEC = 15            # target duration of each split
OUT_SUBDIR = "splits"     # created inside the directory that holds the source .wav


def split_wav(wav_path: Path, chunk_sec: int = CHUNK_SEC) -> None:
    """Cut wav_path into consecutive chunk_sec-second files."""
    audio = AudioSegment.from_file(wav_path)
    ms_per_chunk = chunk_sec * 1000

    out_dir = wav_path.parent / OUT_SUBDIR
    out_dir.mkdir(exist_ok=True)

    n_chunks = (len(audio) + ms_per_chunk - 1) // ms_per_chunk
    stem = wav_path.stem

    for i in range(n_chunks):
        start_ms = i * ms_per_chunk
        end_ms = min((i + 1) * ms_per_chunk, len(audio))
        chunk = audio[start_ms:end_ms]

        out_file = out_dir / f"{stem}_part{i+1:03d}.wav"
        chunk.export(out_file, format="wav")
        print(f"✔︎ wrote {out_file.relative_to(TRAINING_DIR)}")


def main():
    wav_files = list(TRAINING_DIR.rglob("*.wav"))
    if not wav_files:
        print(f"No wav files found under {TRAINING_DIR}")
        return

    print(f"Found {len(wav_files)} wav file(s). Splitting …\n")
    for wav in wav_files:
        split_wav(wav)

    print("\nAll done!")


if __name__ == "__main__":
    main()
