#!/usr/bin/env python3
import pygame, sys, subprocess, tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path
import json
import datetime
import os

# -- path to the environment that has torch / torchaudio / audiocraft
GEN_ENV_PYTHON = "/home/alels_star/anaconda3/envs/musicgen_/bin/python"

# ────────────────── Scaling ─────────────────────────────────────────────────────
SCALE = 1.0
def sc(v: float) -> int:
    return int(v * SCALE)

# ────────────────── Paths & Metadata Config ────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent / "musicgen_wav_generator"
COMMON_DIR    = Path(__file__).resolve().parent.parent / "common_data_space"
OUTPUT_DIR    = COMMON_DIR / "generated_data" / "musicgen"
METADATA_PATH = Path("/home/alels_star/Desktop/AI_composition_assistant/v0.01/metadata/metadata.json")

# ────────────────── Pygame Setup ────────────────────────────────────────────────
pygame.init()
WIDTH, HEIGHT = sc(700), sc(600)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MusicGen Music Generator - GUI")
font  = pygame.font.SysFont(None, sc(28))
clock = pygame.time.Clock()

WHITE, GREY, DARK = (255,255,255), (200,200,200), (40,40,40)
BLUE,  RED,  GREEN = (50,100,255), (255,80,80), (80,255,120)

# ────────────────── File‐dialog helper ───────────────────────────────────────────
def open_file_dialog():
    global screen
    pygame.event.set_grab(False)
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        parent=root,
        title="Select seed WAV",
        filetypes=[("WAV files", "*.wav")]
    )
    root.destroy()
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MusicGen Music Generator - GUI")
    return path

# ────────────────── UI helpers ───────────────────────────────────────────────────
def draw_text(txt, x, y, col=WHITE):
    screen.blit(font.render(txt, True, col), (x, y))

def make_button(label, x, y, w, h, callback, col=BLUE):
    r = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, col, r, border_radius=sc(8))
    draw_text(label, x+sc(10), y+sc(10))
    return r, callback

# ────────────────── State & Defaults ────────────────────────────────────────────
def default_seed() -> str | None:
    folder = COMMON_DIR / "input_data" / "musicgen"
    wavs = list(folder.glob("*.wav")) if folder.exists() else []
    return str(wavs[0]) if wavs else None

mode, model_id, duration        = "chroma", "melody", 16
description, output_name        = "orchestra plays alongside", "output"
seed_file                       = default_seed()
message                         = f"Using default seed: {Path(seed_file).name}" if seed_file else "No seed file!"
editing_field, text_input       = None, ""

# ────────────────── Callbacks ───────────────────────────────────────────────────
def toggle_mode():
    global mode
    mode = "continuation" if mode == "chroma" else "chroma"

def change_model():
    global model_id
    models = ["melody", "small", "medium"]
    model_id = models[(models.index(model_id)+1) % len(models)]

def change_duration():
    global duration
    durs = [8, 16, 32]
    duration = durs[(durs.index(duration)+1) % len(durs)]

def select_seed():
    global seed_file, message
    path = open_file_dialog()
    if path:
        seed_file = path
        message = f"Seed: {Path(seed_file).name}"

def start_edit(field):
    global editing_field, text_input
    editing_field, text_input = field, ""

def run_script():
    global message
    if not seed_file:
        message = "❌ No seed file!"
        return

    # record when we start
    start_time = datetime.datetime.now()

    script = "gen_chroma.py" if mode == "chroma" else "gen_continuation.py"
    args = [
        GEN_ENV_PYTHON,
        str(BASE_DIR / script),
        "--model_id", model_id,
        "--duration", str(duration),
        "--description", description,
        "--output", output_name,
        "--seed", seed_file
    ]

    message = "▶ Generating..."
    pygame.display.flip()

    try:
        subprocess.run(args, cwd=str(BASE_DIR), check=True)
    except subprocess.CalledProcessError as e:
        message = f"❌ Failed: {e}"
        return

    message = f"✅ Saved: {output_name}.wav"

    # now find every new .wav in OUTPUT_DIR modified since start_time
    new_files = []
    for f in OUTPUT_DIR.rglob("*.wav"):
        if datetime.datetime.fromtimestamp(f.stat().st_mtime) >= start_time:
            new_files.append(f)

    if not new_files:
        message = "⚠️ No new files detected"
        return

    # prompt once for notes & provenance
    root = tk.Tk(); root.withdraw()
    extra = simpledialog.askstring("Additional Notes",
                                   "Any extra notes? (optional)",
                                   parent=root)
    provenance = simpledialog.askstring("Provenance",
                                        "Enter provenance/source info (optional):",
                                        parent=root)
    root.destroy()

    # load or init metadata DB
    try:
        with open(METADATA_PATH, "r") as mf:
            meta_db = json.load(mf)
    except (FileNotFoundError, json.JSONDecodeError):
        meta_db = {}

    default_notes = (
        f"Mode: {mode}; Model: {model_id}; Duration: {duration}s; "
        f"Desc: {description}"
    )

    for f in new_files:
        rel_out = str(f.relative_to(COMMON_DIR))
        notes = default_notes
        if extra and extra.strip():
            notes += "; " + extra.strip()
        meta_db[rel_out] = {
            "generated_at": start_time.isoformat(),
            "mode":         mode,
            "model_id":     model_id,
            "duration_s":   duration,
            "description":  description,
            "output_file":  rel_out,
            "notes":        notes,
            "provenance":   provenance or ""
        }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as mf:
        json.dump(meta_db, mf, indent=2)

# ────────────────── Main Loop ───────────────────────────────────────────────────
while True:
    screen.fill(DARK)

    # Text (scaled positions)
    draw_text(f"Mode: {mode}",               sc(30),  sc(30))
    draw_text(f"Model: {model_id}",         sc(30),  sc(70))
    draw_text(f"Duration: {duration}s",     sc(30), sc(110))
    draw_text(f"Description: {description}",sc(30), sc(150))
    draw_text(f"Output: {output_name}.wav", sc(30), sc(190))
    draw_text(f"Seed File: {Path(seed_file).name if seed_file else 'None'}",
              sc(30), sc(230))

    if message:
        col = GREEN if message.startswith("✅") else RED
        draw_text(message, sc(30), sc(440), col)
    if editing_field:
        draw_text(f"Editing {editing_field}: {text_input}",
                  sc(30), sc(400), GREY)

    # Buttons
    buttons = [
        make_button("Toggle Mode",      sc(400), sc(30),  sc(240), sc(60),  toggle_mode),
        make_button("Change Model",     sc(400), sc(100), sc(240), sc(60),  change_model),
        make_button("Change Duration",  sc(400), sc(170), sc(240), sc(60),  change_duration),
        make_button("Edit Description", sc(400), sc(240), sc(240), sc(60),
                    lambda: start_edit("description")),
        make_button("Edit Output Name", sc(400), sc(310), sc(240), sc(60),
                    lambda: start_edit("output_name")),
        make_button("Select Seed File", sc(400), sc(380), sc(240), sc(60),
                    select_seed),
        make_button("Run Generation",   sc(250), sc(480), sc(300), sc(75),
                    run_script, GREEN),
    ]

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif e.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for rect, cb in buttons:
                if rect.collidepoint(pos):
                    cb()
        elif e.type == pygame.KEYDOWN and editing_field:
            if e.key == pygame.K_RETURN:
                globals()[editing_field] = text_input
                editing_field, text_input = None, ""
            elif e.key == pygame.K_BACKSPACE:
                text_input = text_input[:-1]
            else:
                text_input += e.unicode

    pygame.display.flip()
    clock.tick(30)
