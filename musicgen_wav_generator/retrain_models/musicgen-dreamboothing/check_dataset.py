#!/usr/bin/env python
# check_dataset.py
"""
Set DATASET_DIR below if you’d like to hard-code the dataset location and just hit
Run in your IDE.  Leave it as None to fall back to argv or an input() prompt.
"""

# ───  EDIT THIS IF YOU WANT TO “DEFINE IT BY HAND”  ────────────────────────────
DATASET_DIR: str | None = r"/home/alels_star/Desktop/AI_composition_assistant/v0.01/musicgen_wav_generator/retrain_models/musicgen-dreamboothing/punk_dataset"        # e.g.  "punk_dataset" or r"C:\data\punk"
#DATASET_DIR: str | None = ""

# ────────────────────────────────────────────────────────────────────────────────

#!/usr/bin/env python
# check_dataset.py
"""
Scan every audio file referenced in a MusicGen dataset on disk and report ones
that are missing, unreadable, or have 0 channels.

Hard-code DATASET_DIR below if you prefer launching straight from an IDE, or
leave it None and pass the folder on the CLI (or let the script ask).
"""

# ─── EDIT IF YOU WANT TO HARD-CODE YOUR DATASET FOLDER ─────────────────────────
# ────────────────────────────────────────────────────────────────────────────────

from datasets import load_from_disk, concatenate_datasets, Audio
from torchaudio import info as audio_info
from pathlib import Path
import sys


def _extract_path(audio_field, root: Path) -> Path:
    """
    Return a Path to the underlying WAV file, whatever shape the `audio` column
    is in. If the path is *relative*, resolve it against the dataset root.
    """
    # 1. figure out the raw path string
    if isinstance(audio_field, (str, Path)):
        raw = Path(audio_field)
    elif isinstance(audio_field, dict) and "path" in audio_field:
        raw = Path(audio_field["path"])
    else:
        raw = Path(getattr(audio_field, "path", ""))  # AudioDecoder etc.

    # 2. if it's already absolute or exists as-is → done
    if raw.is_absolute() or raw.exists():
        return raw

    # 3. otherwise treat it as relative to the dataset folder
    candidate = root / raw
    return candidate


def check(dataset_dir: str) -> None:
    ds_root = Path(dataset_dir).expanduser().resolve()

    ds_disk = load_from_disk(ds_root)
    ds_all = concatenate_datasets(
        [
            ds_disk["train"].cast_column("audio", Audio(decode=False)),
            ds_disk["validation"].cast_column("audio", Audio(decode=False)),
        ]
    )

    bad = 0
    for row in ds_all:
        wav_path = _extract_path(row["audio"], ds_root)
        try:
            meta = audio_info(str(wav_path))
            if meta.num_channels == 0:
                raise RuntimeError("0-channel audio")
        except Exception as err:
            print(f"❌  {wav_path.relative_to(ds_root) if wav_path.is_relative_to(ds_root) else wav_path}"
                  f"\n    ↳ {err}")
            bad += 1

    print(f"\nScanned {len(ds_all):,} files — {bad} problematic.")


if __name__ == "__main__":
    # decide which folder to use
    if DATASET_DIR:
        folder = DATASET_DIR
    else:
        cli = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
        folder = cli[0] if cli else input("Dataset folder: ").strip()

    if not folder:
        print("⚠️  No dataset folder provided — exiting.")
        sys.exit(1)

    check(folder)
