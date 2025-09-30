#!/usr/bin/env python3
import sys
import subprocess
import pygame
import tkinter as tk
import tkinter.simpledialog as simpledialog
from tkinter import filedialog
from pathlib import Path

# ─── UI SCALE CONFIG ─────────────────────────────────────────────────────────
UI_SCALE   = 1.0    # ← change this to 1.2, 2.0, etc.
BASE_W, BASE_H = 1450, 260
WIDTH, HEIGHT = int(BASE_W * UI_SCALE), int(BASE_H * UI_SCALE)

# ─── Configuration ────────────────────────────────────────────────────────────
ENV_PYTHON     = "/home/alels_star/anaconda3/envs/custom_generators_/bin/python"
BASE_DIR       = Path(__file__).resolve().parent.parent
SCRIPT         = BASE_DIR / "custom_generators" / "TCN_midi_generator" / \
                 "train_test_on_piece_whole_process.py"
DEFAULT_INPUT  = BASE_DIR / "common_data_space" / "input_data" / \
                 "custom_generators" / "TCN_midi_generator"
DEFAULT_OUTPUT = BASE_DIR / "common_data_space" / "generated_data" / \
                 "custom_generators" / "TCN_midi_generator"

# ─── Helper for scaled coords ─────────────────────────────────────────────────
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
found        = list(DEFAULT_INPUT.glob("*.mid"))
input_path   = str(found[0])     if found else ""
output_folder= str(DEFAULT_OUTPUT)
mode         = "option_1"
skip_thresh  = 10
retrain      = False
status_msg   = f"Input: {Path(input_path).name}" if input_path else "Ready"

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

    status_msg = "▶ Running…"
    pygame.display.flip()
    try:
        subprocess.run(
            cmd,
            cwd=str(SCRIPT.parent),
            check=True,
            capture_output=True,
            text=True
        )
        status_msg = "✅ Done"
    except subprocess.CalledProcessError as e:
        last = (e.stderr or "").strip().splitlines()[-1]
        status_msg = f"❌ {last}"

# ─── Main Loop ─────────────────────────────────────────────────────────────────
while True:
    screen.fill(DARK)

    # ── Current settings
    draw_text(f"Input   : {Path(input_path).name or '—'}",      20,  20)
    draw_text(f"Output  : {Path(output_folder).name or '—'}",  20,  60)
    draw_text(f"Mode    : {mode}",                             480, 20)
    draw_text(f"Skip <  : {skip_thresh}",                      480, 60)
    draw_text(f"ReTrain : {retrain}",                         800, 20)

    # ── Status line
    col = (GREEN if status_msg.startswith("✅")
           else RED   if status_msg.startswith("❌")
           else WHITE)
    draw_text(status_msg, 20, 110, col)

    # ── Buttons
    btns = [
        button("Select MIDI",       20, 150, 200, 40, set_input),
        button("Select Output dir",240,150, 200, 40, set_output),
        button("Toggle Mode",       480,150, 200, 40, toggle_mode),
        button("Set Skip ≥",        730,150, 200, 40, set_skip),
        button("ReTrain",           980,150, 200, 40, toggle_retrain, GREY),
        button("Run Prediction",    1200,150, 200, 50, run_prediction, GREEN),
    ]

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
