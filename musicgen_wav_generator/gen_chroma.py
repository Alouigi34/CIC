# gen_chroma.py
import argparse
from pathlib import Path
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# ------------------ Parse CLI arguments ------------------
parser = argparse.ArgumentParser(description="Generate audio with chroma from a seed WAV file.")
parser.add_argument("--model_id", default="melody", help="MusicGen model to use (e.g., melody, small, medium)")
parser.add_argument("--duration", type=int, default=16, help="Length of the generated audio in seconds")
parser.add_argument("--description", default="whispering", help="Text prompt/description for generation")
parser.add_argument("--output", default="output_chroma", help="Output filename (no extension)")
parser.add_argument("--seed", required=True, help="Path to seed WAV file")
args = parser.parse_args()

# ------------------ Load Model ------------------
print(f"▶ Loading model: {args.model_id}")
model = MusicGen.get_pretrained(args.model_id)
model.set_generation_params(duration=args.duration)

# ------------------ Load seed melody ------------------
seed_path = Path(args.seed).expanduser().resolve()
if not seed_path.exists():
    raise FileNotFoundError(f"Seed file not found: {seed_path}")

melody, sr = torchaudio.load(str(seed_path))
melody_batch = melody[None].expand(1, -1, -1)  # Add batch dimension

# ------------------ Generate ------------------
print(f"▶ Generating chroma with description: '{args.description}'")
wav_output = model.generate_with_chroma([args.description], melody_batch, sr)

# ------------------ Save Output ------------------
# Determine path to common_data_space relative to this script
script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "common_data_space/generated_data/musicgen"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / f"{args.output}.wav"
audio_write(str(output_path), wav_output[0].cpu(), model.sample_rate, strategy="loudness")
print(f"✓ Saved generated audio to: {output_path}")
