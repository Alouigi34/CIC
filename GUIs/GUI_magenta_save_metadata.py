#!/usr/bin/env python3
import pygame
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import datetime
from pathlib import Path
import os

# ────────────────── UI Scale ──────────────────────────────────────────────────
UI_SCALE = 1.0  # ← increase for a larger UI, decrease for smaller

# ────────────────── Scaled Constants ──────────────────────────────────────────
BASE_WIDTH, BASE_HEIGHT = 1400, 700
WIDTH, HEIGHT          = int(BASE_WIDTH * UI_SCALE), int(BASE_HEIGHT * UI_SCALE)
FONT_SIZE              = int(26 * UI_SCALE)
PADDING_X              = int(30 * UI_SCALE)
LINE_THICKNESS         = max(1, int(2 * UI_SCALE))

# TRAIN panel
TRAIN_X      = PADDING_X
TRAIN_Y      = int(80  * UI_SCALE)
TRAIN_SPACING= int(40  * UI_SCALE)

# GENERATE panel
GEN_X        = WIDTH // 2 + int(50 * UI_SCALE)
GEN_Y        = TRAIN_Y
GEN_SPACING  = TRAIN_SPACING
BTN_X_OFFSET = int(400 * UI_SCALE)

# Status / Editing
STATUS_Y     = HEIGHT - int(50 * UI_SCALE)
EDIT_Y       = HEIGHT - int(90 * UI_SCALE)

# Buttons
BTN_W    = int(200 * UI_SCALE)
BTN_H    = int(40 * UI_SCALE)
RUN_W    = int(200 * UI_SCALE)
RUN_H    = int(60 * UI_SCALE)
TOG_W    = int(200 * UI_SCALE)
TOG_H    = int(40 * UI_SCALE)
BORDER_R = int(7   * UI_SCALE)
TEXT_PAD = int(10  * UI_SCALE)

# Label offsets
LABEL_PAD = int(10 * UI_SCALE)

# ────────────────── Paths & Metadata Config ───────────────────────────────────
MAG_ENV_PYTHON = "/home/alels_star/anaconda3/envs/magenta_/bin/python"
BASE_DIR       = Path(__file__).resolve().parent.parent / "magenta_midi_generator"
COMMON         = Path(__file__).resolve().parent.parent / "common_data_space"
METADATA_PATH  = Path(__file__).resolve().parent.parent / "metadata" / "metadata.json"

# Train defaults
mdi         = str(COMMON / "training_data" / "magenta" / "training_midis")
run_dir     = str(BASE_DIR)
bundle_file = str(BASE_DIR / "models" / "performance_with_dynamics.mag")
run_steps   = 200
batch_size  = 64

# Generate defaults
use_pre     = True
gen_bundle  = bundle_file
gen_run_dir = str(BASE_DIR)
primer_midi = str(COMMON / "input_data" / "magenta" / "input_cutted_.mid")
output_dir  = str(COMMON / "generated_data" / "magenta")
num_outputs = 3
gen_steps   = 3000
gen_temp    = 1.0

message       = "Ready"
editing_field = None
text_input    = ""

# ────────────────── Pygame Setup ─────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Magenta – Performance-RNN GUI")
font  = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()

WHITE, GREY, DARK = (255,255,255), (180,180,180), (30,30,30)
BLUE, RED, GREEN = (40,120,255), (255,80,80), (80,220,120)

def draw(txt, x, y, col=WHITE):
    screen.blit(font.render(txt, True, col), (x, y))

def button(lbl, x, y, w, h, cb, col=BLUE):
    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, col, rect, border_radius=BORDER_R)
    text_y = y + (h - font.get_height()) // 2
    screen.blit(font.render(lbl, True, WHITE), (x + TEXT_PAD, text_y))
    return rect, cb

def tk_dialog(select_dir=False, midi_only=False):
    global screen
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    if select_dir:
        path = filedialog.askdirectory(parent=root, title="Select folder")
    else:
        ftypes = [("MIDI","*.mid *.midi")] if midi_only else [("All","*.*")]
        path = filedialog.askopenfilename(parent=root, title="Select file", filetypes=ftypes)
    root.destroy()
    pygame.display.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Magenta – Performance-RNN GUI")
    return path

# ────────────────── Callbacks ─────────────────────────────────────────────────
def toggle_mode():
    global mode, message
    mode = "generate" if mode == "train" else "train"
    message = f"Mode → {mode}"

def start_edit(field):
    global editing_field, text_input
    editing_field, text_input = field, ""

def cycle_bool(name):
    globals()[name] = not globals()[name]

def set_path(var, folder=False, midi=False):
    global message
    p = tk_dialog(select_dir=folder, midi_only=midi)
    if p:
        globals()[var] = p
        message = f"Set {var}"

def run_magenta():
    global message
    # Record start time
    start_time = datetime.datetime.now()
    if mode == "train":
        args = [
            MAG_ENV_PYTHON, str(BASE_DIR/"train.py"),
            "--MIDI_DIR",    mdi,
            "--RUN_DIR",     run_dir,
            "--BUNDLE_FILE", bundle_file,
            "--TRAIN_STEPS", str(run_steps),
            "--BATCH_SIZE",  str(batch_size),
        ]
    else:
        args = [
            MAG_ENV_PYTHON, str(BASE_DIR/"generate.py"),
            "--use_pretrained_model", str(use_pre),
            "--BUNDLE_FILE",          gen_bundle,
            "--RUN_DIR",              gen_run_dir,
            "--PRIMER_MIDI",          primer_midi,
            "--OUTPUT_DIR",           output_dir,
            "--NUM_OUTPUTS",          str(num_outputs),
            "--NUM_STEPS",            str(gen_steps),
            "--TEMPERATURE",          str(gen_temp),
        ]

    message = "▶ Running..."
    pygame.display.flip()

    try:
        subprocess.run(args, cwd=BASE_DIR, check=True)
        message = "✅ Finished"
    except subprocess.CalledProcessError as e:
        message = f"❌ Error: {e}"
        return

    # If in generate mode, find new files
    if mode == "generate":
        new_files = []
        out_dir = Path(output_dir)
        for f in out_dir.rglob("*"):
            if f.is_file() and datetime.datetime.fromtimestamp(f.stat().st_mtime) >= start_time:
                new_files.append(f)

        if new_files:
            # Prompt once
            default_notes = (
                f"Pretrained: {use_pre}; Bundle: {Path(gen_bundle).name}; "
                f"Primer: {Path(primer_midi).name}; Outputs: {num_outputs}; "
                f"Steps: {gen_steps}; Temp: {gen_temp}"
            )
            root = tk.Tk(); root.withdraw()
            extra = simpledialog.askstring("Additional Notes",
                                           "Any extra notes? (optional)",
                                           parent=root)
            prov = simpledialog.askstring("Provenance",
                                          "Enter provenance/source info (optional):",
                                          parent=root)
            root.destroy()

            # Load or init metadata
            try:
                with open(METADATA_PATH, "r") as mf:
                    meta_db = json.load(mf)
            except (FileNotFoundError, json.JSONDecodeError):
                meta_db = {}

            for f in new_files:
                rel_out = str(f.relative_to(COMMON))
                notes = default_notes
                if extra and extra.strip():
                    notes += "; " + extra.strip()
                meta_db[rel_out] = {
                    "generated_at": start_time.isoformat(),
                    "mode":         mode,
                    "bundle":       Path(gen_bundle).name,
                    "primer":       Path(primer_midi).name,
                    "output":       rel_out,
                    "notes":        notes,
                    "provenance":   prov or ""
                }

            METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(METADATA_PATH, "w") as mf:
                json.dump(meta_db, mf, indent=2)

# ────────────────── Main Loop ─────────────────────────────────────────────────
mode = "train"
while True:
    screen.fill(DARK)

    # header and divider
    draw(f"Mode: {mode.upper()}", PADDING_X, int(20 * UI_SCALE),
         GREEN if mode=="train" else BLUE)
    pygame.draw.line(screen, GREY,
                     (WIDTH//2, 0), (WIDTH//2, HEIGHT),
                     LINE_THICKNESS)

    # TRAIN panel
    draw("TRAIN", TRAIN_X, TRAIN_Y, GREEN)
    train_labels = [
        f"MIDI dir    : {Path(mdi).name}",
        f"Run dir     : {Path(run_dir).name}",
        f"Bundle file : {Path(bundle_file).name}",
        f"Steps       : {run_steps}",
        f"Batch size  : {batch_size}",
    ]
    for i, lbl in enumerate(train_labels):
        draw(lbl, TRAIN_X, TRAIN_Y + (i+1)*TRAIN_SPACING)

    # GENERATE panel
    draw("GENERATE", GEN_X, GEN_Y, BLUE)
    gen_labels = [
        f"Use pretrained: {use_pre}",
        f"Bundle file   : {Path(gen_bundle).name}",
        f"Run dir       : {Path(gen_run_dir).name}",
        f"Primer MIDI   : {Path(primer_midi).name}",
        f"Output dir    : {Path(output_dir).name}",
        f"Outputs       : {num_outputs}",
        f"Length (steps): {gen_steps}",
        f"Temperature   : {gen_temp}",
    ]
    for i, lbl in enumerate(gen_labels):
        y = GEN_Y + (i+1)*GEN_SPACING
        draw(lbl, GEN_X, y)

    # Status / editing
    draw(message, PADDING_X, STATUS_Y,
         GREEN if message.startswith("✅") else RED)
    if editing_field:
        draw(f"Editing {editing_field}: {text_input}",
             PADDING_X, EDIT_Y, GREY)

    # Buttons
    btns = [
        button("Toggle Train/Gen",
               WIDTH//2 - TOG_W//2, int(20*UI_SCALE),
               TOG_W, TOG_H, toggle_mode)
    ]
    # train controls
    btns += [
        button("Select MIDI dir", TRAIN_X, TRAIN_Y + 5*TRAIN_SPACING, BTN_W, BTN_H,
               lambda: set_path("mdi", True)),
        button("Select Run dir",  TRAIN_X + BTN_W + LABEL_PAD,
               TRAIN_Y + 5*TRAIN_SPACING, BTN_W, BTN_H,
               lambda: set_path("run_dir", True)),
        button("Select Bundle",   TRAIN_X, TRAIN_Y + 6*TRAIN_SPACING, BTN_W, BTN_H,
               lambda: set_path("bundle_file")),
        button("Edit Steps",      TRAIN_X + BTN_W + LABEL_PAD,
               TRAIN_Y + 6*TRAIN_SPACING, BTN_W//2, BTN_H,
               lambda: start_edit("run_steps")),
        button("Edit Batch",      TRAIN_X + BTN_W + LABEL_PAD + BTN_W//2 + LABEL_PAD,
               TRAIN_Y + 6*TRAIN_SPACING, BTN_W//2, BTN_H,
               lambda: start_edit("batch_size")),
    ]
    # generate controls
    gen_btns = [
        ("Toggle Pretrained", lambda: cycle_bool("use_pre")),
        ("Select Bundle",     lambda: set_path("gen_bundle")),
        ("Select Run dir",    lambda: set_path("gen_run_dir", True)),
        ("Select Primer MIDI",lambda: set_path("primer_midi", midi=True)),
        ("Select Output dir", lambda: set_path("output_dir", True)),
        ("Edit Outputs",      lambda: start_edit("num_outputs")),
        ("Edit Steps",        lambda: start_edit("gen_steps")),
        ("Edit Temp",         lambda: start_edit("gen_temp")),
    ]
    for i, (lbl, cb) in enumerate(gen_btns):
        y = TRAIN_Y + (5 + i)*GEN_SPACING
        bw = BTN_W if i < 5 else (BTN_W * 2 // 3)
        btns.append(button(lbl, GEN_X + BTN_X_OFFSET, y, bw, BTN_H, cb))

    # run button
    btns.append(
        button("RUN",
               WIDTH - RUN_W - PADDING_X,
               HEIGHT - RUN_H - PADDING_X,
               RUN_W, RUN_H,
               run_magenta, GREEN)
    )

    # event handling
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
                    if editing_field in ("run_steps","batch_size","num_outputs","gen_steps"):
                        val = int(text_input)
                    elif editing_field == "gen_temp":
                        val = float(text_input)
                    else:
                        val = text_input
                    globals()[editing_field] = val
                    message = f"Set {editing_field}"
                except Exception as exc:
                    message = f"Bad value: {exc}"
                editing_field, text_input = None, ""
            elif e.key == pygame.K_BACKSPACE:
                text_input = text_input[:-1]
            else:
                text_input += e.unicode

    pygame.display.flip()
    clock.tick(30)
