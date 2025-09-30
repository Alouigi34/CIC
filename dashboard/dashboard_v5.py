#!/usr/bin/env python3
"""
AI Composition Dashboard — GPT-4All edition
==========================================

Browse & preview audio/MIDI files, edit metadata, launch generators,
and chat locally with GPT-4All.

Changes in this version (2025-06-25)
------------------------------------
* Chat input is now a **multi-line `Text` widget** (never hidden).
* **Return (↵)** sends the message; **Shift + Return** inserts a newline.
* LLM generation runs in a **background thread**—the GUI never freezes.
* Small visual tweaks (padding, widget heights).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import pygame.mixer as mixer
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Circle
from matplotlib.widgets import SpanSelector
import matplotlib.pyplot as plt

# ───────────────────────────────────────────────────────────────────────────
# Optional sheet-music preview
# ───────────────────────────────────────────────────────────────────────────
try:
    from music21 import converter
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

# ───────────────────────────────────────────────────────────────────────────
# GPT-4All (local LLM)
# ───────────────────────────────────────────────────────────────────────────
from gpt4all import GPT4All

MODEL_ID = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
#MODEL_ID = "Mistral-7B-Instruct.Q2_K.gguf" 
try:
    gpt_model = GPT4All(MODEL_ID, allow_download=True,device="gpu")  
except Exception as exc:
    # If download failed or no internet, ask for a model file.
    _dlg_root = tk.Tk(); _dlg_root.withdraw()
    mdl_path = filedialog.askopenfilename(
        parent=_dlg_root,
        title="Select .gguf model",
        filetypes=[("GGUF", "*.gguf")],
    )
    _dlg_root.destroy()
    if not mdl_path:
        raise RuntimeError(
            "No model chosen; download one from https://gpt4all.io/models/ and retry."
        ) from exc
    gpt_model = GPT4All(model_path=mdl_path)

# ───────────────────────────────────────────────────────────────────────────
# PATHS & CONSTANTS
# ───────────────────────────────────────────────────────────────────────────
BASE          = Path("/home/alels_star/Desktop/AI_composition_assistant/v0.01")
COMMON_DIR    = BASE / "common_data_space"
METADATA_PATH = BASE / "metadata" / "metadata.json"
AUDIO_EXTS    = {".wav", ".mp3", ".flac"}
RUN_TARGETS: dict[str, dict[str, str]] = {
    "MusicGen (wav continuation)": {
        "env": "musicgen_",
        "script": str(BASE / "musicgen_wav_generator" / "gen_chroma.py"),
    },
    "Magenta (MIDI continuation)": {
        "env": "magenta_",
        "script": str(BASE / "magenta_midi_generator" / "generate.py"),
    },
}

plt.rcParams.update({"figure.facecolor": "white"})


# ───────────────────────────────────────────────────────────────────────────
# Dashboard GUI
# ───────────────────────────────────────────────────────────────────────────
class Dashboard(tk.Tk):
    """Main application window."""

    # ────────────────────────────────────────────────────────────────────
    # Construction
    # ────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:
        super().__init__()
        self.title("AI Composition Dashboard")
        self.geometry("1400x1200")

        mixer.init(); mixer.set_num_channels(2)

        # Load (or create) metadata DB
        self.meta_db: dict[str, dict] = (
            json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
        )

        self._build_left()
        self._build_right()

    # ░░ LEFT PANE ░░─────────────────────────────────────────────────────
    def _build_left(self) -> None:
        lf = ttk.Frame(self, width=480)
        lf.pack(side="left", fill="y")
        lf.pack_propagate(False)

        # Filter bar
        top = ttk.Frame(lf); top.pack(fill="x", pady=4, padx=5)
        ttk.Label(top, text="Filter:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.search_var).pack(
            side="left", fill="x", expand=True, padx=4
        )
        ttk.Button(top, text="Go",    command=self.apply_filter ).pack(side="left", padx=4)
        ttk.Button(top, text="Clear", command=self.clear_filter).pack(side="left")

        # Tree of files
        self.tree = ttk.Treeview(lf, show="tree")
        self.tree.column("#0", width=450)
        self.tree.pack(fill="both", expand=True, padx=5, pady=(0, 5))
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self._populate_tree()

    # ░░ RIGHT PANE ░░────────────────────────────────────────────────────
    def _build_right(self) -> None:
        rf = ttk.Frame(self); rf.pack(side="right", fill="both", expand=True)

        # ── Waveform plot ──
        self.fig_wf, self.ax_wf = plt.subplots(figsize=(6, 1.4), dpi=100, constrained_layout=True)
        self.ax_wf.set_title("Waveform")
        self.canvas_wf = FigureCanvasTkAgg(self.fig_wf, master=rf)
        NavigationToolbar2Tk(self.canvas_wf, rf).update()
        self.canvas_wf.get_tk_widget().pack(padx=5, pady=(2, 0))
        self.span = SpanSelector(
            self.ax_wf, self._on_zoom, "horizontal", useblit=True,
            props=dict(alpha=0.3, facecolor="blue")
        )

        # ── Spectrogram / notation ──
        self.fig_sp, self.ax_sp = plt.subplots(figsize=(6, 2.6), dpi=100, constrained_layout=True)
        self.ax_sp.set_title("Spectrogram / Notation")
        self.canvas_sp = FigureCanvasTkAgg(self.fig_sp, master=rf)
        NavigationToolbar2Tk(self.canvas_sp, rf).update()
        self.canvas_sp.get_tk_widget().pack(padx=5, pady=(2, 5))

        # ── Transport controls ──
        ctl = ttk.Frame(rf); ctl.pack(pady=5)
        for txt, fn in (("Play", self.play_current),
                        ("Stop", mixer.music.stop),
                        ("Reset Zoom", self.reset_zoom)):
            ttk.Button(ctl, text=txt, command=fn).pack(side="left", padx=4)

        # ── Metadata editor ──
        mbox = ttk.LabelFrame(rf, text="Metadata"); mbox.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Label(mbox, text="Tags (comma-sep):").pack(anchor="w")
        self.tags_var = tk.StringVar(); ttk.Entry(mbox, textvariable=self.tags_var).pack(fill="x")
        ttk.Label(mbox, text="Notes:").pack(anchor="w")
        self.notes = tk.Text(mbox, height=4); self.notes.pack(fill="x")
        ttk.Label(mbox, text="Provenance:").pack(anchor="w")
        self.prov_var = tk.StringVar(); ttk.Entry(mbox, textvariable=self.prov_var, state="readonly").pack(fill="x")
        ttk.Button(mbox, text="Save Metadata", command=self.save_metadata).pack(pady=5)

        # ── Generator launcher ──
        runf = ttk.Frame(rf); runf.pack(pady=5)
        ttk.Label(runf, text="Run with:").pack(side="left")
        self.run_var = tk.StringVar()
        ttk.Combobox(runf, values=list(RUN_TARGETS), textvariable=self.run_var,
                     state="readonly").pack(side="left", padx=4)
        ttk.Button(runf, text="Go", command=self.run_target).pack(side="left")

        # ── Chat area ──
        # chat = ttk.LabelFrame(rf, text="Chat with Agent"); chat.pack(fill="both", expand=True, padx=5, pady=5)
        # # Conversation log (read-only)
        # self.chat_log = tk.Text(chat, state="disabled", wrap="word")
        # self.chat_log.pack(fill="both", expand=True, pady=(0, 4))
        
        
        # ── Chat area ──
        chat = ttk.LabelFrame(rf, text="Chat with Agent")
        chat.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Conversation log (read-only)
        self.chat_log = tk.Text(
            chat,
            state="disabled",
            wrap="word",
            height=10     # ← 10 text lines tall; pick any number you like
        )
        self.chat_log.pack(fill="x", expand=False, pady=(0, 4))


        # Input box (multi-line)
        ef = ttk.Frame(chat); ef.pack(fill="x")
        self.input_box = tk.Text(ef, height=3, wrap="word")
        self.input_box.pack(side="left", fill="x", expand=True)
        self.input_box.focus_set()
        self.input_box.bind("<Return>", self._send_on_return)
        self.input_box.bind("<Shift-Return>", lambda e: None)  # allow newline with Shift+↵
        ttk.Button(ef, text="Send", command=self._chat_send).pack(side="left", padx=4)

    # ────────────────────────────────────────────────────────────────────
    # Tree helpers
    # ────────────────────────────────────────────────────────────────────
    def _populate_tree(self) -> None:
        """Fill tree widget with folder contents."""
        self.tree.delete(*self.tree.get_children())

        def walk(parent: str, folder: Path) -> None:
            for name in sorted(os.listdir(folder)):
                p = folder / name
                node = self.tree.insert(parent, "end", text=name)
                if p.is_dir():
                    walk(node, p)

        walk("", COMMON_DIR)

    def apply_filter(self) -> None:
        term = self.search_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        if not term:
            self._populate_tree(); return
        for rel, md in self.meta_db.items():
            hay = " ".join([
                rel,
                *md.get("tags", []),
                md.get("notes", ""),
                md.get("provenance", ""),
            ]).lower()
            if term in hay:
                self.tree.insert("", "end", text=rel)

    def clear_filter(self) -> None:
        self.search_var.set("")
        self._populate_tree()

    # ────────────────────────────────────────────────────────────────────
    # Selection handling
    # ────────────────────────────────────────────────────────────────────
    def on_select(self, _) -> None:
        sel = self.tree.selection()
        if len(sel) != 1:
            return
        # Build path from tree hierarchy
        parts, cur = [], sel[0]
        while cur:
            parts.append(self.tree.item(cur, "text"))
            cur = self.tree.parent(cur)
        path = COMMON_DIR.joinpath(*reversed(parts))
        self.current_path = path
        self.current_rel  = str(path.relative_to(COMMON_DIR))
        self._display(path)

        md = self.meta_db.get(self.current_rel, {})
        self.tags_var.set(",".join(md.get("tags", [])))
        self.notes.delete("1.0", "end"); self.notes.insert("1.0", md.get("notes", ""))
        self.prov_var.set(md.get("provenance", ""))

    # ────────────────────────────────────────────────────────────────────
    # Display
    # ────────────────────────────────────────────────────────────────────
    def _display(self, path: Path) -> None:
        ext = path.suffix.lower()
        if ext in AUDIO_EXTS:
            self._show_audio(path)
        elif ext in {".mid", ".midi"}:
            self._show_midi(path)
        else:
            self.ax_wf.clear(); self.ax_sp.clear()
            self.canvas_wf.draw(); self.canvas_sp.draw()

    def _show_audio(self, path: Path) -> None:
        y, sr = librosa.load(path, sr=None)
        t = np.arange(len(y)) / sr
        # Waveform
        self.ax_wf.clear()
        self.ax_wf.plot(t, y, linewidth=0.5)
        self.ax_wf.set_xlabel("s")
        self.full_xlim = (0, t[-1])
        self.canvas_wf.draw()

        # Spectrogram
        S = librosa.stft(y, n_fft=1024, hop_length=512)
        Sdb = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        freqs = np.linspace(0, sr/2, Sdb.shape[0])
        ts = np.arange(Sdb.shape[1]) * 512 / sr
        self.ax_sp.clear()
        self.ax_sp.imshow(Sdb, origin="lower", aspect="auto", cmap="magma",
                          extent=[ts[0], ts[-1], freqs[0], freqs[-1]])
        self.ax_sp.set_ylabel("Hz"); self.ax_sp.set_xlabel("s")
        self.full_xlim_sp = (ts[0], ts[-1])
        self.canvas_sp.draw()

    def _show_midi(self, path: Path) -> None:
        pm = pretty_midi.PrettyMIDI(str(path)); end_t = pm.get_end_time()

        # Piano-roll style in waveform axis
        self.ax_wf.clear()
        notes = sum((inst.notes for inst in pm.instruments), [])
        for note in notes:
            self.ax_wf.hlines(note.pitch, note.start, note.end,
                              linewidth=4, alpha=0.6)
        self.ax_wf.set_ylabel("MIDI pitch"); self.ax_wf.set_xlabel("s")
        self.full_xlim = (0, end_t); self.canvas_wf.draw()

        # Either render sheet with music21, or a simple staff overlay
        self.ax_sp.clear()
        if MUSIC21_AVAILABLE:
            try:
                score = converter.parse(path)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                png = score.write("musicxml.png", fp=tmp.name); tmp.close()
                img = plt.imread(png); os.unlink(png)
                self.ax_sp.imshow(img); self.ax_sp.axis("off")
                self.canvas_sp.draw(); return
            except Exception as e:
                print("music21 failed:", e)

        # Fallback: draw staff + note heads
        staff = [64, 67, 71, 74, 77]
        for y in staff:
            self.ax_sp.hlines(y, 0, end_t, color="black", linewidth=1)
        for note in notes:
            if staff[0]-12 <= note.pitch <= staff[-1]+12:
                c = Circle((note.start, note.pitch), radius=0.15, fc="black")
                self.ax_sp.add_patch(c)
        self.ax_sp.set_xlim(0, end_t); self.ax_sp.set_ylim(staff[0]-5, staff[-1]+5)
        self.ax_sp.axis("off"); self.canvas_sp.draw()
        self.full_xlim_sp = (0, end_t)

    # ────────────────────────────────────────────────────────────────────
    # Zoom
    # ────────────────────────────────────────────────────────────────────
    def _on_zoom(self, x0: float, x1: float) -> None:
        self.ax_wf.set_xlim(x0, x1)
        self.ax_sp.set_xlim(x0, x1)
        self.canvas_wf.draw(); self.canvas_sp.draw()

    def reset_zoom(self) -> None:
        if hasattr(self, "full_xlim"):
            self.ax_wf.set_xlim(*self.full_xlim)
        if hasattr(self, "full_xlim_sp"):
            self.ax_sp.set_xlim(*self.full_xlim_sp)
        self.canvas_wf.draw(); self.canvas_sp.draw()

    # ────────────────────────────────────────────────────────────────────
    # Transport
    # ────────────────────────────────────────────────────────────────────
    def play_current(self) -> None:
        if getattr(self, "current_path", None) and self.current_path.suffix.lower() in AUDIO_EXTS:
            mixer.music.load(self.current_path); mixer.music.play()

    # ────────────────────────────────────────────────────────────────────
    # Metadata
    # ────────────────────────────────────────────────────────────────────
    def save_metadata(self) -> None:
        md = {
            "tags": [t.strip() for t in self.tags_var.get().split(",") if t.strip()],
            "notes": self.notes.get("1.0", "end").strip(),
            "provenance": self.prov_var.get(),
        }
        self.meta_db[self.current_rel] = md
        METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.write_text(json.dumps(self.meta_db, indent=2))
        messagebox.showinfo("Saved", "Metadata saved.")

    # ────────────────────────────────────────────────────────────────────
    # External generator launcher
    # ────────────────────────────────────────────────────────────────────
    def run_target(self) -> None:
        tgt = self.run_var.get(); cfg = RUN_TARGETS.get(tgt)
        if not cfg or not getattr(self, "current_path", None):
            return
        cmd = [
            "conda", "run", "-n", cfg["env"],
            "python", cfg["script"],
            "--input", str(self.current_path),
        ]
        threading.Thread(target=subprocess.run, args=(cmd,), daemon=True).start()
        messagebox.showinfo("Launched", f"Started {tgt} on\n{self.current_path}")

    # ────────────────────────────────────────────────────────────────────
    # Chat helpers
    # ────────────────────────────────────────────────────────────────────
    # ↵ in the Text widget
    def _send_on_return(self, event: tk.Event) -> str:
        # Shift+↵ should insert a newline, so only send if Shift not held
        if event.state & 0x0001:   # Shift key bitmask
            return ""  # let Tkinter add newline
        self._chat_send()
        return "break"  # block default newline

    def _chat_send(self) -> None:
        """Grab user text, clear box, spawn generation thread."""
        q = self.input_box.get("1.0", "end-1c").strip()
        if not q:
            return
        self.input_box.delete("1.0", "end")
        self._chat_log("You", q)

        # Build prompt with a tiny sample of DB metadata
        sample = {
            k: {kk: (vv[:120] + "…" if isinstance(vv, str) and len(vv) > 120 else vv)
                for kk, vv in v.items()}
            for k, v in list(self.meta_db.items())[:5]
        }
        sys_msg = "You are an assistant for a music-generation metadata DB."
        prompt = (
            f"{sys_msg}\nDB sample:\n{json.dumps(sample, indent=2)}\n"
            f"User: {q}\nAssistant:"
        )

        threading.Thread(target=self._generate_and_log, args=(prompt,), daemon=True).start()

    # def _generate_and_log(self, prompt: str) -> None:
    #     ans = self._safe_generate(prompt)
    #     self._chat_log("Assistant", ans)
        
        
        
    def _generate_and_log(self, prompt: str) -> None:
        def append(token: str) -> None:              # ▶ callback runs on each token
            self._chat_log("Assistant", token, live=True)
    
        # streaming generator yields tokens one-by-one
        with gpt_model.chat_session() as chat:
            for _ in chat.generate(prompt,
                                   max_tokens=256,          # tweak if you like
                                   streaming=True):         # ▶
                append(_)                                   # ▶
        # final newline so the next message starts cleanly
        append("\n")                                       # ▶
           
            

    def _safe_generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Run the LLM, retrying with a shorter prompt if too long."""
        try:
            with gpt_model.chat_session() as chat:
                return chat.generate(prompt, max_tokens=max_tokens).strip()
        except ValueError:
            short = prompt.split("DB sample:")[0] + prompt.split("User:")[-1]
            with gpt_model.chat_session() as chat:
                return chat.generate(short, max_tokens=max_tokens).strip()

    # def _chat_log(self, who: str, txt: str) -> None:
    #     self.chat_log.configure(state="normal")
    #     self.chat_log.insert("end", f"{who}: {txt}\n\n")
    #     self.chat_log.see("end")
    #     self.chat_log.configure(state="disabled")

    def _chat_log(self, who: str, txt: str, live: bool = False) -> None:
        self.chat_log.configure(state="normal")
        if live and who == "Assistant":
            self.chat_log.insert("end", txt)               # no extra \n\n
        else:
            self.chat_log.insert("end", f"{who}: {txt}\n\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")


# ───────────────────────────────────────────────────────────────────────────
# Entrypoint
# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE.mkdir(parents=True, exist_ok=True)
    COMMON_DIR.mkdir(parents=True, exist_ok=True)
    Dashboard().mainloop()
