#!/usr/bin/env python3
"""
create_dataset_from_songs.py  –  v2
Turn a folder of WAV/FLAC/MP3/etc. into a MusicGen-ready HuggingFace
DatasetDict.  All chunked / resampled clips are *copied into the output
folder*, and the CSV stores absolute paths, so nothing goes missing later.

Usage examples
──────────────
# You already have train/  and validation/
python create_dataset_from_songs.py /path/to/training_songs --out punk_dataset

# Autosplit 90 % / 10 %, chunk to 30-s, force 32 kHz stereo
python create_dataset_from_songs.py /path/to/raw_audio \
       --auto_split 0.9 --chunk --fix_audio --out punk_dataset
"""
from __future__ import annotations
import argparse, csv, math, pathlib, random, shutil, sys
import soundfile as sf, torchaudio
from datasets import Dataset, DatasetDict, Audio

AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac"}


# ── helpers ────────────────────────────────────────────────────────────────────
def gather(folder: pathlib.Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def chunk_write(piece, sr, dst: pathlib.Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, piece.T.numpy(), sr)


def ensure_32k_stereo(sig, sr):
    if sr != 32_000:
        sig = torchaudio.functional.resample(sig, sr, 32_000)
        sr = 32_000
    if sig.shape[0] == 1:                      # mono → stereo
        sig = sig.repeat(2, 1)
    return sig, sr


# ── main ───────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create a MusicGen HF dataset with rock-solid audio paths.")
    ap.add_argument("root", type=pathlib.Path,
                    help="Folder containing audio; may already have train/ & validation/")
    ap.add_argument("--out", default="musicgen_dataset",
                    help="Where to save the HF Dataset (and the processed WAVs)")
    ap.add_argument("--auto_split", type=float,
                    help="e.g. 0.9 → auto 90 %% train / 10 %% val when no sub-dirs")
    ap.add_argument("--chunk", action="store_true",
                    help="Slice files longer than 30 s into 30 s chunks")
    ap.add_argument("--fix_audio", action="store_true",
                    help="Resample to 32 kHz stereo (recommended for MusicGen)")
    args = ap.parse_args(argv)

    root: pathlib.Path = args.root.expanduser().resolve()
    out_root: pathlib.Path = pathlib.Path(args.out).expanduser().resolve()
    audio_root = out_root / "audio"            # where all clips will live

    # fresh start
    if out_root.exists():
        shutil.rmtree(out_root)
    audio_root.mkdir(parents=True, exist_ok=True)

    # ── gather files ──
    if (root / "train").exists():
        train_files = gather(root / "train")
        val_files   = gather(root / "validation")
    else:
        all_files = gather(root)
        if not args.auto_split:
            sys.exit("No train/validation sub-dirs and --auto_split not set")
        random.shuffle(all_files)
        split_pt = int(len(all_files) * args.auto_split)
        train_files, val_files = all_files[:split_pt], all_files[split_pt:]

    print(f"📀  {len(train_files)} train  |  {len(val_files)} validation clips")

    # ── copy / chunk / fix ──
    rows: list[dict[str, str]] = []

    def process(files: list[pathlib.Path], split: str):
        for wav in files:
            sig, sr = torchaudio.load(wav)
            dur = sig.shape[-1] / sr
            parts = [(sig, sr)]

            if args.chunk and dur > 30:
                hop = 30 * sr
                parts = [(sig[..., i*hop:(i+1)*hop], sr)
                         for i in range(math.ceil(dur / 30))
                         if sig[..., i*hop:(i+1)*hop].numel()]

            for idx, (piece, piece_sr) in enumerate(parts):
                if args.fix_audio or args.chunk:
                    piece, piece_sr = ensure_32k_stereo(piece, piece_sr)

                dst = audio_root / split / f"{wav.stem}_{idx:03d}.wav"
                chunk_write(piece, piece_sr, dst)
                rows.append({"audio": str(dst), "split": split})

    process(train_files, "train")
    process(val_files,   "validation")
    print(f"📑  Wrote {len(rows)} processed clips into {audio_root}")

    # ── CSV → DatasetDict ──
    csv_path = out_root / "paths.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio", "split"])
        writer.writeheader()
        writer.writerows(rows)

    ds_full = Dataset.from_csv(str(csv_path)).cast_column("audio", Audio())
    dsdict  = DatasetDict({
        "train":      ds_full.filter(lambda r: r["split"] == "train").remove_columns("split"),
        "validation": ds_full.filter(lambda r: r["split"] == "validation").remove_columns("split"),
    })

    dsdict.save_to_disk(str(out_root))
    print("✅  Saved HuggingFace dataset to", out_root)


if __name__ == "__main__":
    main()
