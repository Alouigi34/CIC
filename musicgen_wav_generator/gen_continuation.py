# gen_continuation.py
import argparse
import torch
import torchaudio
from pathlib import Path
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# ------------------ Parse CLI arguments ------------------
parser = argparse.ArgumentParser(description="Generate continuation from a seed WAV file.")
parser.add_argument("--model_id", default="melody", help="MusicGen model to use (e.g., melody, small, medium)")
parser.add_argument("--duration", type=int, default=16, help="Length of continuation in seconds")
parser.add_argument("--description", default="orchestra plays alongside", help="Text prompt/instruction")
parser.add_argument("--output", default="output_continuation", help="Output filename (no extension)")
parser.add_argument("--seed", required=True, help="Path to seed WAV file")
args = parser.parse_args()




# ------------------ Load Model ------------------
print(f"▶ Loading model: {args.model_id}")
model: MusicGen = MusicGen.get_pretrained(args.model_id)
model.set_generation_params(
    duration=args.duration,
    use_sampling=True,
    top_k=250,
    top_p=0,
    temperature=1.1,
    cfg_coef=1.4
)

# ------------------ Load Seed ------------------
def load_seed(path: str, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    wav = wav.mean(0, keepdim=True)  # Convert to mono
    wav /= wav.abs().max().clamp(min=1e-9)  # Normalize
    return wav,sr

seed_path = Path(args.seed).expanduser().resolve()
if not seed_path.exists():
    raise FileNotFoundError(f"Seed file not found: {seed_path}")

print("▶ Loading seed WAV...")
seed,sr = load_seed(str(seed_path), model.sample_rate)
print(len(seed) / sr)   # seconds
print(sr)
seed = seed.expand(1, -1, -1)

# ------------------ Generate ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"▶ Generating continuation on device: {DEVICE}")

with torch.no_grad():
    if DEVICE == "cuda":
        with torch.cuda.amp.autocast():
            output = model.generate_continuation(
                seed,
                model.sample_rate,
                descriptions=[args.description],
                progress=False
            )[0]
    else:
        output = model.generate_continuation(
            seed,
            model.sample_rate,
            descriptions=[args.description],
            progress=False
        )[0]

# ------------------ Save Output to common_data_space ------------------
script_dir = Path(__file__).resolve().parent
output_dir = script_dir.parent / "/home/alels_star/Desktop"#"common_data_space/generated_data/musicgen"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / f"{args.output}.wav"
#udio_write(str(output_path), output.cpu(), model.sample_rate, strategy="loudness")
audio_write(str(output_path), output.cpu().float(), model.sample_rate, strategy="loudness")
print(f"✓ Saved generated audio to: {output_path}")
