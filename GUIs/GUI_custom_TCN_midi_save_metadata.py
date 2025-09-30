#!/usr/bin/env python3
import sys
import subprocess
import pygame
import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import datetime
from pathlib import Path

# ─── UI SCALE CONFIG ─────────────────────────────────────────────────────────
UI_SCALE   = 1.0    # ← change this to 1.2, 2.0, etc.
BASE_W, BASE_H = 1450, 260
WIDTH, HEIGHT = int(BASE_W * UI_SCALE), int(BASE_H * UI_SCALE)

# ─── Paths & Metadata Config ─────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent.parent
COMMON_DIR     = BASE_DIR / "common_data_space"
METADATA_PATH  = BASE_DIR / "metadata" / "metadata.json"

ENV_PYTHON     = "/home/alels_star/anaconda3/envs/custom_generators_/bin/python"
SCRIPT         = BASE_DIR / "custom_generators" / "TCN_midi_generator" / \
                 "train_test_on_piece_whole_process.py"
DEFAULT_INPUT  = COMMON_DIR / "input_data" / "custom_generators" / "TCN_midi_generator"
DEFAULT_OUTPUT = COMMON_DIR / "generated_data" / "custom_generators" / "TCN_midi_generator"

# ─── Helpers for scaled coords ─────────────────────────────────────────────────
def sx(x): return int(x * UI_SCALE)
def sy(y): return int(y * UI_SCALE)

# ─── Pygame Setup ─────────────────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MIDI → TCN Prediction")
font   = pygame.font.SysFont(None, int(28 * UI_SCALE))
clock  = pygame.time.Clock()

WHITE, GREY, DARK = (255,255,255), (200,200,200), (40,40,40)
BLUE, RED, GREEN = (50,100,255), (255,80,80), (80,255,120)

def draw_text(txt, x, y, col=WHITE):
    screen.blit(font.render(txt, True, col), (sx(x), sy(y)))

def button(lbl, x, y, w, h, cb, col=BLUE):
    r = pygame.Rect(sx(x), sy(y), sx(w), sy(h))
    pygame.draw.rect(screen, col, r, border_radius=int(6*UI_SCALE))
    txt_y = sy(y) + (sy(h) - font.get_height()) // 2
    screen.blit(font.render(lbl, True, WHITE), (sx(x) + sx(10), txt_y))
    return r, cb

def tk_dialog(select_dir=False):
    """Show a file/folder dialog, pausing pygame."""
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    if select_dir:
        p = filedialog.askdirectory(parent=root, title="Select output folder")
    else:
        p = filedialog.askopenfilename(
            parent=root,
            title="Select MIDI file",
            filetypes=[("MIDI","*.mid")]
        )
    root.destroy()
    pygame.display.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MIDI → TCN Prediction")
    return p

def ask_integer(prompt):
    """Pop up a simple integer input dialog."""
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    val = simpledialog.askinteger("Parameter", prompt, parent=root, minvalue=0)
    root.destroy()
    pygame.display.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("MIDI → TCN Prediction")
    return val

# ─── State & Defaults ─────────────────────────────────────────────────────────
found         = list(DEFAULT_INPUT.glob("*.mid"))
input_path    = str(found[0]) if found else ""
output_folder = str(DEFAULT_OUTPUT)
mode          = "option_1"
skip_thresh   = 10
retrain       = False
status_msg    = f"Input: {Path(input_path).name}" if input_path else "Ready"

# ─── Callbacks ─────────────────────────────────────────────────────────────────
def set_input():
    global input_path, status_msg
    p = tk_dialog(False)
    if p:
        input_path = p
        status_msg = f"Input: {Path(p).name}"

def set_output():
    global output_folder, status_msg
    p = tk_dialog(True)
    if p:
        output_folder = p
        status_msg = f"Output: {Path(p).name}"

def toggle_mode():
    global mode, status_msg
    mode = "option_2" if mode=="option_1" else "option_1"
    status_msg = f"Mode set to {mode}"

def set_skip():
    global skip_thresh, status_msg
    val = ask_integer("Skip notes lower than (MIDI pitch):")
    if val is not None:
        skip_thresh = val
        status_msg = f"Skip threshold ← {skip_thresh}"

def toggle_retrain():
    global retrain, status_msg
    retrain = not retrain
    status_msg = f"ReTrain = {retrain}"

def run_prediction():
    global status_msg
    if not input_path or not output_folder:
        status_msg = "❌ Select both input and output"
        return

    start_time = datetime.datetime.now()
    status_msg = "▶ Running…"
    pygame.display.flip()

    cmd = [
        ENV_PYTHON,
        str(SCRIPT),
        "--input",  input_path,
        "--output", output_folder,
        "--mode",   mode,
        "--skip_notes_lower_than", str(skip_thresh)
    ]
    if retrain:
        cmd.append("--retrain")

    try:
        subprocess.run(
            cmd,
            cwd=str(SCRIPT.parent),
            check=True,
            capture_output=True,
            text=True
        )
        status_msg = "✅ Done"

        # Find all files in output_folder with mtime >= start_time
        out_dir = Path(output_folder)
        new_files = []
        for f in out_dir.rglob("*"):
            if f.is_file() and datetime.datetime.fromtimestamp(f.stat().st_mtime) >= start_time:
                new_files.append(f)

        if not new_files:
            status_msg = "⚠️ No new files detected"
            return

        # Prompt once for user annotations
        default_notes = (
            f"Input: {Path(input_path).name}; "
            f"Output Dir: {out_dir.name}; "
            f"Mode: {mode}; "
            f"Skip< {skip_thresh}; "
            f"Retrain: {retrain}"
        )
        root = tk.Tk(); root.withdraw()
        extra_notes = simpledialog.askstring("Additional Notes",
                                             "Any extra notes? (optional)",
                                             parent=root)
        provenance = simpledialog.askstring("Provenance",
                                            "Enter provenance/source info (optional):",
                                            parent=root)
        root.destroy()

        # Load or init metadata DB
        try:
            with open(METADATA_PATH, "r") as mf:
                meta_db = json.load(mf)
        except (FileNotFoundError, json.JSONDecodeError):
            meta_db = {}

        # Write an entry for each new file
        for f in new_files:
            rel_in  = str(Path(input_path).relative_to(COMMON_DIR))
            rel_out = str(f.relative_to(COMMON_DIR))

            notes = default_notes
            if extra_notes and extra_notes.strip():
                notes += "; " + extra_notes.strip()

            meta_db[rel_out] = {
                "generated_at":     start_time.isoformat(),
                "source_midi":      rel_in,
                "output_predict":   rel_out,
                "mode":             mode,
                "skip_threshold":   skip_thresh,
                "retrain":          retrain,
                "notes":            notes,
                "provenance":       provenance or ""
            }

        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, "w") as mf:
            json.dump(meta_db, mf, indent=2)

    except subprocess.CalledProcessError as e:
        last = (e.stderr or "").strip().splitlines()[-1]
        status_msg = f"❌ {last}"

# ─── Main Loop ─────────────────────────────────────────────────────────────────
while True:
    screen.fill(DARK)

    # Display current settings
    draw_text(f"Input   : {Path(input_path).name or '—'}",      20,  20)
    draw_text(f"Output  : {Path(output_folder).name or '—'}",   20,  60)
    draw_text(f"Mode    : {mode}",                             480, 20)
    draw_text(f"Skip <  : {skip_thresh}",                      480, 60)
    draw_text(f"ReTrain : {retrain}",                         800, 20)

    # Status
    col = GREEN if status_msg.startswith("✅") else RED if status_msg.startswith("❌") else WHITE
    draw_text(status_msg, 20, 110, col)

    # Buttons
    btns = [
        button("Select MIDI",       20, 150, 200, 40, set_input),
        button("Select Output dir",240,150, 200, 40, set_output),
        button("Toggle Mode",       480,150, 200, 40, toggle_mode),
        button("Set Skip ≥",        730,150, 200, 40, set_skip),
        button("ReTrain",           980,150, 200, 40, toggle_retrain, GREY),
        button("Run Prediction",    1200,150,200, 50, run_prediction, GREEN),
    ]

    # Event loop
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        elif e.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            for r, cb in btns:
                if r.collidepoint(pos):
                    cb()

    pygame.display.flip()
    clock.tick(30)
