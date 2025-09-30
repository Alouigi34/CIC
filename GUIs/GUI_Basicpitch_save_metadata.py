#!/usr/bin/env python3
import sys
import pygame
import subprocess
import tkinter as tk
from tkinter import filedialog, simpledialog
import json
import datetime
from pathlib import Path

# ────────────────── UI Scale ──────────────────────────────────────────────────
UI_SCALE = 1.0   # ← change this to scale the entire UI

# ────────────────── Scaled Constants ──────────────────────────────────────────
WIDTH, HEIGHT     = int(800 * UI_SCALE), int(500 * UI_SCALE)
FONT_SIZE         = int(24  * UI_SCALE)
X_MARGIN          = int(20  * UI_SCALE)
Y_INPUT           = int(40  * UI_SCALE)
Y_OUTPUT          = int(80  * UI_SCALE)
Y_STATUS          = HEIGHT - int(40 * UI_SCALE)

BUTTON_X          = int(550 * UI_SCALE)
BTN1_Y            = int(60  * UI_SCALE)
BTN2_Y            = int(140 * UI_SCALE)
BTN3_Y            = int(300 * UI_SCALE)

BTN_WIDTH         = int(200 * UI_SCALE)
BTN_HEIGHT        = int(40  * UI_SCALE)
RUN_BTN_HEIGHT    = int(60  * UI_SCALE)

BORDER_RADIUS     = int(5 * UI_SCALE)
TEXT_LEFT_PAD     = int(10 * UI_SCALE)

# ────────────────── Paths ──────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent.parent
COMMON_DIR    = BASE_DIR / "common_data_space"
METADATA_PATH = BASE_DIR / "metadata" / "metadata.json"

# ────────────────── Pygame Setup ───────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("BasicPitch – WAV → MIDI GUI")
font  = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()

WHITE, GREY, DARK = (255,255,255), (180,180,180), (20,20,20)
BLUE, RED, GREEN = (40,120,255), (255,80,80), (80,220,120)

def draw_text(txt, x, y, col=WHITE):
    screen.blit(font.render(txt, True, col), (x, y))

def button(lbl, x, y, w, h, cb, col=BLUE):
    r = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, col, r, border_radius=BORDER_RADIUS)
    text_y = y + (h - font.get_height()) // 2
    screen.blit(font.render(lbl, True, WHITE), (x + TEXT_LEFT_PAD, text_y))
    return r, cb

def tk_dialog(select_dir=False, wav_only=False):
    """Open file/folder dialog, then re-init Pygame."""
    pygame.display.quit()
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    if select_dir:
        p = filedialog.askdirectory(parent=root, title="Select folder")
    else:
        ftypes = [("WAV","*.wav")] if wav_only else [("All","*.*")]
        p = filedialog.askopenfilename(parent=root, title="Select file", filetypes=ftypes)
    root.destroy()
    pygame.display.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BasicPitch – WAV → MIDI GUI")
    return p

# ────────────────── State & Defaults ───────────────────────────────────────────
BASICPITCH_PYTHON = "/home/alels_star/anaconda3/envs/transformers_/bin/python"
SCRIPT = BASE_DIR / "BasicPitch_transformer" / "generate_Basicpitch.py"

DEFAULT_INPUT  = COMMON_DIR / "input_data" / "BasicPitch" / "song2_cutted.wav"
DEFAULT_OUTPUT = COMMON_DIR / "generated_data" / "BasicPitch"

input_path  = str(DEFAULT_INPUT)
output_dir  = str(DEFAULT_OUTPUT)
status_msg  = "Ready"

# ────────────────── Callbacks ──────────────────────────────────────────────────
def set_input():
    global input_path, status_msg
    p = tk_dialog(wav_only=True)
    if p:
        input_path = p
        status_msg = f"Input → {Path(p).name}"

def set_output():
    global output_dir, status_msg
    p = tk_dialog(select_dir=True)
    if p:
        output_dir = p
        status_msg = f"Output Dir → {Path(p).name}"

def run_basicpitch():
    global status_msg
    if not input_path or not output_dir:
        status_msg = "❌ Select input & output"
        return

    inp = Path(input_path)
    out = Path(output_dir) / f"{inp.stem}_BasicPitch.mid"
    cmd = [
        BASICPITCH_PYTHON,
        str(SCRIPT),
        "-i", str(inp),
        "-o", str(out)
    ]
    status_msg = "▶ Running…"
    pygame.display.flip()

    try:
        subprocess.run(
            cmd,
            cwd=str(SCRIPT.parent),
            capture_output=True,
            text=True,
            check=True
        )
        # ─── SUCCESS ─────────────────────────────────────────────────────────
        status_msg = f"✅ {out.name}"

        # Build default notes from GUI selections
        default_notes = f"Input WAV: {inp.name}; Output Dir: {Path(output_dir).name}"

        # Prompt for additional notes
        root = tk.Tk(); root.withdraw()
        extra_notes = simpledialog.askstring(
            "Additional Notes",
            "Add any extra notes? (leave blank if none)",
            parent=root
        )

        # Prompt for provenance
        provenance = simpledialog.askstring(
            "Provenance",
            "Enter provenance/source info:",
            parent=root
        )
        root.destroy()

        # Load or initialize metadata
        try:
            with open(METADATA_PATH, "r") as mf:
                meta_db = json.load(mf)
        except (FileNotFoundError, json.JSONDecodeError):
            meta_db = {}

        # Compose final notes string
        notes_field = default_notes
        if extra_notes and extra_notes.strip():
            notes_field += "; " + extra_notes.strip()

        # Compute paths relative to COMMON_DIR
        rel_out = str(out.relative_to(COMMON_DIR))
        rel_inp = str(inp.relative_to(COMMON_DIR))

        # Add or update record
        meta_db[rel_out] = {
            "generated_at": datetime.datetime.now().isoformat(),
            "source_wav": rel_inp,
            "output_midi": rel_out,
            "notes": notes_field,
            "provenance": provenance or ""
        }

        # Ensure parent dir exists, then write
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(METADATA_PATH, "w") as mf:
            json.dump(meta_db, mf, indent=2)

    except subprocess.CalledProcessError as e:
        last = e.stderr.strip().splitlines()[-1]
        status_msg = f"❌ {last}"

# ────────────────── Main Loop ──────────────────────────────────────────────────
while True:
    screen.fill(DARK)

    # Status display
    draw_text(f"Input WAV  : {Path(input_path).name}", X_MARGIN, Y_INPUT)
    draw_text(f"Output Dir : {Path(output_dir).name}", X_MARGIN, Y_OUTPUT)
    col = GREEN if status_msg.startswith("✅") else RED if status_msg.startswith("❌") else WHITE
    draw_text(status_msg, X_MARGIN, Y_STATUS, col)

    # Buttons
    btns = [
        button("Select WAV",     BUTTON_X, BTN1_Y, BTN_WIDTH,    BTN_HEIGHT, set_input),
        button("Select OutDir",  BUTTON_X, BTN2_Y, BTN_WIDTH,    BTN_HEIGHT, set_output),
        button("RUN BasicPitch", BUTTON_X, BTN3_Y, BTN_WIDTH, RUN_BTN_HEIGHT, run_basicpitch, GREEN),
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
