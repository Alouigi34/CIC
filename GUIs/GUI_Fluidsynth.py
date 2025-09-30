#!/usr/bin/env python3
import sys, pygame, subprocess
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ────────────────── UI Scale ──────────────────────────────────────────────────
UI_SCALE = 1.0  # ← change this to scale the entire UI

# ────────────────── Scaled Constants ──────────────────────────────────────────
WIDTH, HEIGHT      = int(800 * UI_SCALE), int(500 * UI_SCALE)
FONT_SIZE          = int(24  * UI_SCALE)
X_MARGIN           = int(20  * UI_SCALE)
Y_INST             = int(20  * UI_SCALE)
Y_INPUT            = int(60  * UI_SCALE)
Y_OUTPUT           = int(100 * UI_SCALE)
Y_STATUS           = HEIGHT - int(40 * UI_SCALE)

BUTTON_X           = int(550 * UI_SCALE)
BTN1_Y             = int(20  * UI_SCALE)
BTN2_Y             = int(80  * UI_SCALE)
BTN3_Y             = int(140 * UI_SCALE)
BTN4_Y             = int(300 * UI_SCALE)

BTN_WIDTH          = int(200 * UI_SCALE)
BTN_HEIGHT         = int(40  * UI_SCALE)
RUN_BTN_HEIGHT     = int(60  * UI_SCALE)

BORDER_RADIUS      = int(5   * UI_SCALE)
TEXT_LEFT_PAD      = int(10  * UI_SCALE)

# ────────────────── Script Configuration ──────────────────────────────────────
FLUIDSYNTH_PYTHON = "/home/alels_star/anaconda3/envs/transformers_/bin/python"
SCRIPT = Path(__file__).resolve().parent.parent / "Fluidsynth_transformer" / "generate_fluidsynth.py"
SF2_PATH = Path(__file__).resolve().parent.parent / "Fluidsynth_transformer" / "Fluidsynth_models" / "ChateauGrand-Plus-Instruments-bs16i-v4.sf2"

INSTR_MAP = {
    "AcousticGrandPiano": 0,
    "ChurchOrgan": 19,
    "ReedOrgan": 20,
    "Accordion": 21,
    "Harmonica": 22,
    "Clavinet": 7,
    "Celesta": 8,
    "Harpsichord": 6
}
instruments = list(INSTR_MAP.keys())
idx = 0
selected_inst = instruments[idx]

BASE_DIR      = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = BASE_DIR/"common_data_space"/"input_data"/"fluidsynth"/"input_cutted_.mid"
DEFAULT_OUTPUT= BASE_DIR/"common_data_space"/"generated_data"/"fluidsynth"

input_path = str(DEFAULT_INPUT)
output_dir = str(DEFAULT_OUTPUT)
status_msg = "Ready"

# ────────────────── Pygame Setup ───────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fluidsynth – MIDI → WAV GUI")
font  = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()

WHITE, GREY, DARK = (255,255,255), (180,180,180), (20,20,20)
BLUE, RED, GREEN  = (40,120,255), (255,80,80), (80,220,120)

def draw_text(txt, x, y, col=WHITE):
    screen.blit(font.render(txt, True, col), (x, y))

def button(lbl, x, y, w, h, cb, col=BLUE):
    r = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, col, r, border_radius=BORDER_RADIUS)
    text_y = y + (h - font.get_height()) // 2
    screen.blit(font.render(lbl, True, WHITE), (x + TEXT_LEFT_PAD, text_y))
    return r, cb

def tk_dialog(select_dir=False, mid_only=False):
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    if select_dir:
        p = filedialog.askdirectory(parent=root, title="Select folder")
    else:
        ftypes = [("MIDI","*.mid")] if mid_only else [("All","*.*")]
        p = filedialog.askopenfilename(parent=root, title="Select file", filetypes=ftypes)
    root.destroy()
    pygame.display.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fluidsynth – MIDI → WAV GUI")
    return p

# ────────────────── Callbacks ──────────────────────────────────────────────────
def cycle_inst():
    global idx, selected_inst, status_msg
    idx = (idx + 1) % len(instruments)
    selected_inst = instruments[idx]
    status_msg = f"Instrument → {selected_inst}"

def set_input():
    global input_path, status_msg
    p = tk_dialog(mid_only=True)
    if p:
        input_path = p
        status_msg = f"Input → {Path(p).name}"

def set_output():
    global output_dir, status_msg
    p = tk_dialog(select_dir=True)
    if p:
        output_dir = p
        status_msg = f"Output Dir → {Path(p).name}"

def run_fluidsynth():
    global status_msg
    if not input_path or not selected_inst:
        status_msg = "❌ Select input & instrument"
        return

    inp = Path(input_path)
    out = Path(output_dir) / f"{inp.stem}_{selected_inst}.wav"
    cmd = [
        FLUIDSYNTH_PYTHON,
        str(SCRIPT),
        "-i", str(inp),
        "-o", str(out),
        "-s", str(SF2_PATH),
        "-r", "16000",
        "-p", str(INSTR_MAP[selected_inst])
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
        status_msg = f"✅ {out.name}"
    except subprocess.CalledProcessError as e:
        last = e.stderr.strip().splitlines()[-1]
        status_msg = f"❌ {last}"

# ────────────────── Main Loop ──────────────────────────────────────────────────
while True:
    screen.fill(DARK)

    # Display state
    draw_text(f"Inst       : {selected_inst}",     X_MARGIN, Y_INST,   GREEN)
    draw_text(f"Input File : {Path(input_path).name}", X_MARGIN, Y_INPUT)
    draw_text(f"Output Dir : {Path(output_dir).name}", X_MARGIN, Y_OUTPUT)
    col = GREEN if status_msg.startswith("✅") else RED if status_msg.startswith("❌") else WHITE
    draw_text(status_msg, X_MARGIN, Y_STATUS, col)

    # Buttons
    btns = [
        button("Cycle Inst",   BUTTON_X, BTN1_Y, BTN_WIDTH,    BTN_HEIGHT,    cycle_inst),
        button("Select MIDI",  BUTTON_X, BTN2_Y, BTN_WIDTH,    BTN_HEIGHT,    set_input),
        button("Select OutDir",BUTTON_X, BTN3_Y, BTN_WIDTH,    BTN_HEIGHT,    set_output),
        button("RUN Synth",    BUTTON_X, BTN4_Y, BTN_WIDTH, RUN_BTN_HEIGHT, run_fluidsynth, GREEN),
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
