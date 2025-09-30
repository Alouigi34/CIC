#!/usr/bin/env python3
import sys, os, pygame, subprocess, tkinter as tk
from tkinter import filedialog, simpledialog
from pathlib import Path

# ───────────────── Configuration ────────────────────────────────────────────────
UI_SCALE    = 1.0  # ← change this to make everything larger/smaller

TCN_PYTHON  = "/home/alels_star/anaconda3/envs/custom_generators_/bin/python"

BASE_DIR    = Path(__file__).resolve().parent.parent
TCN_DIR     = BASE_DIR / "custom_generators" / "TCN_wav_generator"
SCRIPT      = TCN_DIR / "initializations.py"

DEFAULT_INPUT  = BASE_DIR / "common_data_space/input_data/custom_generators/TCN_wav_generator" / "song1.wav"
DEFAULT_OUTPUT = BASE_DIR / "common_data_space/generated_data/custom_generators/TCN_wav_generator"

MODELS_PATH = str(TCN_DIR)
SONGS_PATH  = str(BASE_DIR / "common_data_space/training_data/custom_generators/TCN_wav_generator")

# ───────────────── Scaled UI Constants ─────────────────────────────────────────
WIDTH, HEIGHT   = int(800 * UI_SCALE), int(400 * UI_SCALE)
FONT_SIZE       = int(24  * UI_SCALE)
BTN_W           = int(150 * UI_SCALE)
BTN_H           = int(30  * UI_SCALE)
RUN_W           = int(500 * UI_SCALE)
RUN_H           = int(30  * UI_SCALE)
X1, X2, X3      = int(20  * UI_SCALE), int(200 * UI_SCALE), int(380 * UI_SCALE)
X4              = X3 + BTN_W + int(10 * UI_SCALE)   # ← new fourth column
Y_STATE_START   = int(20  * UI_SCALE)
Y_LINE_SPACING  = int(30  * UI_SCALE)
Y_BTN_ROW1      = int(280 * UI_SCALE)
Y_BTN_ROW2      = int(320 * UI_SCALE)
Y_RUN_BTN       = int(360 * UI_SCALE)

# ───────────────── Pygame Setup ───────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TCN – Waveform Predictor")
font  = pygame.font.SysFont(None, FONT_SIZE)
clock = pygame.time.Clock()

WHITE, GREY, DARK = (255,255,255), (180,180,180), (20,20,20)
BLUE, RED, GREEN = (40,120,255), (255,80,80), (80,220,120)

def draw_text(txt, x, y, col=WHITE):
    screen.blit(font.render(txt, True, col), (x, y))

def button(lbl, x, y, w, h, cb, col=BLUE):
    r = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, col, r, border_radius=5)
    text_y = y + (h - font.get_height()) // 2
    screen.blit(font.render(lbl, True, WHITE), (x + int(10*UI_SCALE), text_y))
    return r, cb

def tk_dialog(select_dir=False):
    pygame.display.quit()
    root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
    if select_dir:
        p = filedialog.askdirectory(parent=root, title="Select folder")
    else:
        p = filedialog.askopenfilename(parent=root,
            title="Select WAV file", filetypes=[("WAV","*.wav")])
    root.destroy()
    pygame.display.init()
    global screen
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TCN – Waveform Predictor")
    return p

# ───────────────── State & Defaults ───────────────────────────────────────────
models            = [f"model{i}" for i in range(1,10)]
idx               = 0
sel_model         = models[idx]

input_path        = str(DEFAULT_INPUT)
output_dir        = str(DEFAULT_OUTPUT)
status_msg        = "Ready"
user_idx_flag     = False
user_index_value  = 0
length_pct        = 1.0
retrain_flag      = False      # ← New flag

# ───────────────── Callbacks ───────────────────────────────────────────────────
def cycle_model():
    global idx, sel_model, status_msg
    idx = (idx + 1) % len(models)
    sel_model = models[idx]
    status_msg = f"Model → {sel_model}"

def set_input():
    global input_path, status_msg
    p = tk_dialog(False)
    if p:
        input_path = p
        status_msg = f"Input → {Path(p).name}"

def set_output():
    global output_dir, status_msg
    p = tk_dialog(True)
    if p:
        output_dir = p
        status_msg = f"Output → {Path(p).name}"

def toggle_index():
    global user_idx_flag, status_msg
    user_idx_flag = not user_idx_flag
    status_msg = f"UseIdx → {user_idx_flag}"

def set_index():
    global user_index_value, status_msg
    val = simpledialog.askinteger("Index", "Enter time-step index:", initialvalue=user_index_value)
    if val is not None:
        user_index_value = val
        status_msg = f"Idx → {user_index_value}"

def set_length():
    global length_pct, status_msg
    val = simpledialog.askfloat("Length %", "Enter length percentage [0.0–1.0]:", initialvalue=length_pct)
    if val is not None and 0.0 <= val <= 1.0:
        length_pct = val
        status_msg = f"Len% → {length_pct:.2f}"

def toggle_retrain():
    global retrain_flag, status_msg
    retrain_flag = not retrain_flag
    status_msg   = f"ReTrain → {retrain_flag}"

def run_tcn():
    global status_msg
    if not sel_model or not input_path or not output_dir:
        status_msg = "❌ Pick model, input & output"
        return

    out = Path(output_dir) / f"{Path(input_path).stem}_{sel_model}.wav"
    env = os.environ.copy()
    env.update({
        "CHOSEN_MODEL":                     sel_model,
        "_path_":                           str(output_dir),
        "audio_file_another":               input_path,
        "models_path":                      MODELS_PATH,
        "songs_path":                       SONGS_PATH,
        "user_defined_time_prediction_index": str(user_idx_flag),
        "index_check":                      str(user_index_value),
        "length_percentage":                str(length_pct),
        "RETRAIN_FLAG":                     str(retrain_flag),
    })
    cmd = [TCN_PYTHON, str(SCRIPT)]
    status_msg = "▶ Running…"
    pygame.display.flip()

    try:
        subprocess.run(cmd,
                       cwd=str(SCRIPT.parent),
                       env=env,
                       capture_output=True,
                       text=True,
                       check=True)
        status_msg = f"✅ {out.name}"
    except subprocess.CalledProcessError as e:
        last = (e.stderr or "").strip().splitlines()[-1]
        status_msg = f"❌ {last}"

# ───────────────── Main Loop ───────────────────────────────────────────────────
while True:
    screen.fill(DARK)

    # ─── State display (top) ───────────────────────────────────────────────────
    draw_text(f"Model      : {sel_model}",            X1, Y_STATE_START, GREEN)
    draw_text(f"Input File : {Path(input_path).name}", X1, Y_STATE_START + 1*Y_LINE_SPACING, WHITE)
    draw_text(f"Output Dir : {Path(output_dir).name}", X1, Y_STATE_START + 2*Y_LINE_SPACING, WHITE)
    draw_text(f"UseIdx     : {user_idx_flag}",          X1, Y_STATE_START + 3*Y_LINE_SPACING, WHITE)
    draw_text(f"Index      : {user_index_value}",     X1, Y_STATE_START + 4*Y_LINE_SPACING, WHITE)
    draw_text(f"Length %   : {length_pct:.2f}",       X1, Y_STATE_START + 5*Y_LINE_SPACING, WHITE)
    draw_text(f"ReTrain    : {retrain_flag}",         X1, Y_STATE_START + 6*Y_LINE_SPACING, WHITE)
    col = GREEN if status_msg.startswith("✅") else RED if status_msg.startswith("❌") else WHITE
    draw_text(status_msg,                             X1, Y_STATE_START + 7*Y_LINE_SPACING, col)

    # ─── Buttons (below state) ─────────────────────────────────────────────────
    btns = []
    btns += [button("Cycle Model",   X1, Y_BTN_ROW1, BTN_W, BTN_H, cycle_model)]
    btns += [button("Select Input",  X2, Y_BTN_ROW1, BTN_W, BTN_H, set_input)]
    btns += [button("Select Output", X3, Y_BTN_ROW1, BTN_W, BTN_H, set_output)]
    btns += [button("Toggle Index",  X1, Y_BTN_ROW2, BTN_W, BTN_H, toggle_index)]
    btns += [button("Set Index",     X2, Y_BTN_ROW2, BTN_W, BTN_H, set_index)]
    btns += [button("Set Length",    X3, Y_BTN_ROW2, BTN_W, BTN_H, set_length)]
    btns += [button("ReTrain",       X4, Y_BTN_ROW2, BTN_W, BTN_H, toggle_retrain, GREY)]  # ← moved here
    btns += [button("RUN TCN",       X1, Y_RUN_BTN,  RUN_W, RUN_H, run_tcn, GREEN)]

    # ─── Event handling ───────────────────────────────────────────────────────
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
