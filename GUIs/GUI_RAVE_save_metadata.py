#!/usr/bin/env python3
import pygame, sys, subprocess, tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path
import json
import datetime
import os

# ─── Environment Python ────────────────────────────────────────────────────────
RAVE_ENV_PYTHON = "/home/alels_star/anaconda3/envs/transformers_/bin/python"

# ─── Paths & Metadata Config ───────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
COMMON_DIR     = BASE_DIR / "common_data_space"
TRANSFORMER    = BASE_DIR / "RAVE_transformer"
SCRIPT         = TRANSFORMER / "generate_RAVE.py"
OUTPUT_DIR     = COMMON_DIR / "generated_data" / "RAVE"
METADATA_PATH  = BASE_DIR / "metadata" / "metadata.json"
MODELS_DIR     = TRANSFORMER / "RAVE_models"

# ─── Scaling ───────────────────────────────────────────────────────────────────
SCALE = 1.0
def sc(v: float) -> int:
    return int(v * SCALE)

# ─── Pygame Setup ─────────────────────────────────────────────────────────────
pygame.init()
WIDTH, HEIGHT = sc(900), sc(420)
screen        = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("RAVE GUI")
font   = pygame.font.SysFont(None, sc(26))
clock  = pygame.time.Clock()

COL_BG, COL_TEXT = (30, 30, 30), (255, 255, 255)
COL_BTN, COL_HL   = (40, 120, 255), (80, 220, 120)
COL_ERR          = (255, 80, 80)

def draw(txt, x, y, col=COL_TEXT):
    screen.blit(font.render(txt, True, col), (sc(x), sc(y)))

def button(lbl, x, y, w, h, cb, col=COL_BTN):
    r = pygame.Rect(sc(x), sc(y), sc(w), sc(h))
    pygame.draw.rect(screen, col, r, border_radius=sc(5))
    txt_y = sc(y) + (sc(h) - font.get_height())//2
    screen.blit(font.render(lbl, True, COL_TEXT), (sc(x+10), txt_y))
    return r, cb

def tk_open(wav_only=False):
    global screen
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    ftypes = [("WAV","*.wav")] if wav_only else [("All files","*.*")]
    path = filedialog.askopenfilename(parent=root, title="Select file", filetypes=ftypes)
    root.destroy()
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RAVE GUI")
    return path

def tk_save_wav(default_name):
    global screen
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        parent=root,
        title="Save output as…",
        defaultextension=".wav",
        initialfile=default_name,
        filetypes=[("WAV","*.wav")]
    )
    root.destroy()
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("RAVE GUI")
    return path

# ─── State & Defaults ─────────────────────────────────────────────────────────
model_path   = str(MODELS_DIR / "nasa.ts")
input_wav    = str(COMMON_DIR / "input_data" / "RAVE" / "song2_cutted.wav")
default_out  = Path(input_wav).stem + "_out.wav"
output_wav   = str(OUTPUT_DIR / default_out)
duration     = 9000001.0

message       = "Ready"
editing_field = None
text_input    = ""

# ─── Callbacks ────────────────────────────────────────────────────────────────
def start_edit(field):
    global editing_field, text_input
    editing_field, text_input = field, ""

def set_model():
    global model_path, message
    p = tk_open(wav_only=False)
    if p:
        model_path = p
        message = "Model set"

def set_input():
    global input_wav, output_wav, message
    p = tk_open(wav_only=True)
    if p:
        input_wav = p
        stem = Path(p).stem + "_out.wav"
        output_wav = str(OUTPUT_DIR / stem)
        message = "Input set"

def set_output():
    global output_wav, message
    stem = Path(input_wav).stem + "_out.wav"
    p = tk_save_wav(stem)
    if p:
        output_wav = p
        message = "Output set"

def run_rave():
    global message
    if not (model_path and input_wav and output_wav):
        message = "❌ pick model, input & output"
        return

    # record start time
    start_time = datetime.datetime.now()

    args = [
        RAVE_ENV_PYTHON, str(SCRIPT),
        "--model",    model_path,
        "--input",    input_wav,
        "--output",   output_wav,
        "--duration", str(duration),
    ]
    message = "▶ Running..."
    pygame.display.flip()

    try:
        subprocess.run(args, cwd=TRANSFORMER, check=True)
        message = "✅ Done"
    except subprocess.CalledProcessError as e:
        message = f"❌ Error: {e}"
        return

    # gather new files modified since start_time
    new_files = []
    out_dir = Path(output_wav).parent
    for f in out_dir.rglob("*.wav"):
        if f.is_file() and datetime.datetime.fromtimestamp(f.stat().st_mtime) >= start_time:
            new_files.append(f)

    if not new_files:
        message = "⚠️ No new files detected"
        return

    # prompt once for notes & provenance
    root = tk.Tk(); root.withdraw()
    extra = simpledialog.askstring("Additional Notes",
                                   "Any extra notes? (optional)",
                                   parent=root)
    prov = simpledialog.askstring("Provenance",
                                  "Enter provenance/source info (optional):",
                                  parent=root)
    root.destroy()

    # load or init metadata db
    try:
        with open(METADATA_PATH, "r") as mf:
            meta_db = json.load(mf)
    except (FileNotFoundError, json.JSONDecodeError):
        meta_db = {}

    # write entry for each new file
    for f in new_files:
        rel_in  = str(Path(input_wav).relative_to(COMMON_DIR))
        rel_out = str(f.relative_to(COMMON_DIR))
        notes = f"Duration: {duration}; Model: {Path(model_path).name}"
        if extra and extra.strip():
            notes += "; " + extra.strip()
        meta_db[rel_out] = {
            "generated_at": start_time.isoformat(),
            "source_wav":   rel_in,
            "output_wav":   rel_out,
            "model":        Path(model_path).name,
            "duration":     duration,
            "notes":        notes,
            "provenance":   prov or ""
        }

    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w") as mf:
        json.dump(meta_db, mf, indent=2)

# ─── Main Loop ─────────────────────────────────────────────────────────────────
while True:
    screen.fill(COL_BG)
    draw("RAVE Audio GUI", 30, 20, COL_HL)
    draw(f"Model:    {Path(model_path).name}", 30, 70)
    draw(f"Input:    {Path(input_wav).name}",   30, 110)
    draw(f"Output:   {Path(output_wav).name}", 30, 150)
    draw(f"Duration: {duration}",              30, 190)
    col = COL_HL if message.startswith("✅") else (COL_ERR if message.startswith("❌") else COL_TEXT)
    draw(message, 30, HEIGHT/SCALE - 40, col)
    if editing_field:
        draw(f"Editing {editing_field}: {text_input}", 30, HEIGHT/SCALE - 80, GREY)

    btns = []
    x0 = 500
    btns.append(button("Select Model",     x0,   70, 250, 40, set_model))
    btns.append(button("Select Input WAV", x0,  120, 250, 40, set_input))
    btns.append(button("Select Output",    x0,  170, 250, 40, set_output))
    btns.append(button("Edit Duration",    x0,  220, 250, 40, lambda: start_edit("duration")))
    btns.append(button("RUN RAVE",         x0,  280, 250, 55, run_rave, COL_HL))

    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif e.type == pygame.MOUSEBUTTONDOWN:
            for rect, cb in btns:
                if rect.collidepoint(e.pos):
                    cb()
        elif e.type == pygame.KEYDOWN and editing_field:
            if e.key == pygame.K_RETURN:
                try:
                    duration = float(text_input)
                    message = "Duration set"
                except Exception as exc:
                    message = f"❌ {exc}"
                editing_field, text_input = None, ""
            elif e.key == pygame.K_BACKSPACE:
                text_input = text_input[:-1]
            else:
                text_input += e.unicode

    pygame.display.flip()
    clock.tick(30)
