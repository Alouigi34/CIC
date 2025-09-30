#!/usr/bin/env python3
import os, json, subprocess, threading, tempfile
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import librosa, numpy as np
import pretty_midi
import pygame.mixer as mixer
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# try to import music21 for sheet rendering
try:
    from music21 import converter
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

# ─── CONFIG ────────────────────────────────────────────────────────────────
COMMON_DIR    = "/home/alels_star/Desktop/AI_composition_assistant/v0.01/common_data_space"
METADATA_PATH = os.path.join(COMMON_DIR, "metadata.json")

RUN_TARGETS = {
    "MusicGen (wav continuation)": {
        "env":    "musicgen_",
        "script": "/home/alels_star/Desktop/AI_composition_assistant/v0.01/musicgen_wav_generator/gen_chroma.py"
    },
    "Magenta (MIDI continuation)": {
        "env":    "magenta_",
        "script": "/home/alels_star/Desktop/AI_composition_assistant/v0.01/magenta_midi_generator/generate.py"
    },
    # … add more as needed …
}
# ────────────────────────────────────────────────────────────────────────────

class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Composition Dashboard")
        self.geometry("1400x1200")

        # # ─── left pane: file tree
        # left = ttk.Frame(self, width=300)
        # left.pack(side="left", fill="y")
        # self.tree = ttk.Treeview(left)
        # self.tree.pack(fill="both", expand=True)
        # self.tree.bind("<<TreeviewSelect>>", self.on_select)
        
        
        # ─── left pane: file tree (make it wider & bigger‐font) ─────────────────────
        left = ttk.Frame(self, width=500)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)   # honour the fixed width
        
        # enlarge the Treeview font & row height
        style = ttk.Style(self)
        style.configure("Big.Treeview",
                        font=("Arial", 10),
                        rowheight=24)
        style.configure("Big.Treeview.Heading",
                        font=("Arial", 10))
        
        # make your tree use that style and give it a wider column
        self.tree = ttk.Treeview(left,
                                 style="Big.Treeview",
                                 columns=(),            # no extra columns
                                 show="tree")
        self.tree.column("#0", width=380, stretch=False)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_select)

        

        # ─── right pane: plots + metadata
        right = ttk.Frame(self)
        right.pack(side="right", fill="both", expand=True)

        # ── Waveform plot + toolbar (narrower width)
        self.fig_wf, self.ax_wf = plt.subplots(figsize=(6, 1.4), dpi=100)
        self.canvas_wf = FigureCanvasTkAgg(self.fig_wf, master=right)
        self.toolbar_wf = NavigationToolbar2Tk(self.canvas_wf, right)
        self.toolbar_wf.update()
        self.canvas_wf.get_tk_widget().pack(padx=5, pady=(2,0))   # no fill="x"
        self.fig_wf.tight_layout()



        # ── Sheet/MIDI plot + toolbar (narrower width)
        self.fig_sp, self.ax_sp = plt.subplots(figsize=(6, 2.4), dpi=100)
        self.canvas_sp = FigureCanvasTkAgg(self.fig_sp, master=right)
        self.toolbar_sp = NavigationToolbar2Tk(self.canvas_sp, right)
        self.toolbar_sp.update()
        self.canvas_sp.get_tk_widget().pack(padx=5, pady=(2,5))  # no fill="x"
        self.fig_sp.tight_layout()

        # ─── playback controls
        ctl = ttk.Frame(right)
        ctl.pack(pady=5)
        ttk.Button(ctl, text="Play",  command=self.play).pack(side="left", padx=5)
        ttk.Button(ctl, text="Stop",  command=mixer.music.stop).pack(side="left", padx=5)

        # ─── metadata editor
        meta = ttk.LabelFrame(right, text="Metadata")
        meta.pack(fill="both", expand=True, padx=5, pady=5)
        ttk.Label(meta, text="Tags (comma-sep):").pack(anchor="w")
        self.tags_var = tk.StringVar()
        ttk.Entry(meta, textvariable=self.tags_var).pack(fill="x")
        ttk.Label(meta, text="Notes:").pack(anchor="w")
        self.notes = tk.Text(meta, height=4); self.notes.pack(fill="x")
        ttk.Label(meta, text="Provenance:").pack(anchor="w")
        self.prov_var = tk.StringVar()
        ttk.Entry(meta, textvariable=self.prov_var, state="readonly").pack(fill="x")
        ttk.Button(meta, text="Save Metadata", command=self.save_metadata).pack(pady=5)

        # ─── “Run with…” dropdown
        runf = ttk.Frame(right); runf.pack(pady=5)
        ttk.Label(runf, text="Run with:").pack(side="left")
        self.run_var = tk.StringVar()
        opts = list(RUN_TARGETS.keys())
        self.run_cb = ttk.Combobox(runf, values=opts, textvariable=self.run_var, state="readonly")
        self.run_cb.pack(side="left", padx=5)
        ttk.Button(runf, text="Go", command=self.run_target).pack(side="left")

        # init mixer & metadata DB
        mixer.init()
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                self.meta_db = json.load(f)
        else:
            self.meta_db = {}

        self._build_tree()

    def _build_tree(self):
        self.tree.delete(*self.tree.get_children())
        def rec(parent, path):
            for fn in sorted(os.listdir(path)):
                full = os.path.join(path, fn)
                node = self.tree.insert(parent, "end", text=fn, open=False)
                if os.path.isdir(full):
                    rec(node, full)
        rec("", COMMON_DIR)

    def on_select(self, _):
        sel = self.tree.selection()[0]
        parts, cur = [], sel
        while cur:
            parts.append(self.tree.item(cur, "text"))
            cur = self.tree.parent(cur)
        path = os.path.join(COMMON_DIR, *reversed(parts))
        ext  = os.path.splitext(path)[1].lower()

        if ext in [".wav", ".mp3", ".flac"]:
            self._show_audio(path)
        elif ext in [".mid", ".midi"]:
            self._show_midi(path)
        else:
            for ax, canvas in ((self.ax_wf, self.canvas_wf),(self.ax_sp, self.canvas_sp)):
                ax.clear(); canvas.draw()

        # metadata
        rel = os.path.relpath(path, COMMON_DIR)
        md  = self.meta_db.get(rel, {})
        self.tags_var.set(",".join(md.get("tags", [])))
        self.notes.delete("1.0","end"); self.notes.insert("1.0", md.get("notes",""))
        prov = Path(rel).parts[0] if "/" in rel else ""
        self.prov_var.set(prov)

        self.current_path = path
        self.current_rel  = rel

    def _show_audio(self, path):
        y, sr = librosa.load(path, sr=None)
        # waveform
        self.ax_wf.clear()
        times = np.arange(len(y)) / sr
        self.ax_wf.plot(times, y, linewidth=0.5)
        self.ax_wf.set_ylabel("amplitude")
        self.ax_wf.set_xlim(0, times[-1])
        self.ax_wf.set_xlabel("Time (s)")
        self.canvas_wf.draw()

        # spectrogram with real axes
        self.ax_sp.clear()
        S   = librosa.stft(y, n_fft=1024, hop_length=512)
        Sdb = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        hop     = 512
        freqs   = np.linspace(0, sr/2, Sdb.shape[0])
        times_sp = np.arange(Sdb.shape[1]) * hop / sr
        self.ax_sp.imshow(
            Sdb,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[times_sp[0], times_sp[-1], freqs[0], freqs[-1]]
        )
        self.ax_sp.set_ylabel("Frequency (Hz)")
        self.ax_sp.set_xlabel("Time (s)")
        self.canvas_sp.draw()

    def _show_midi(self, path):
        pm = pretty_midi.PrettyMIDI(path)

        # — Top: piano-roll timeline
        self.ax_wf.clear()
        for inst in pm.instruments:
            for note in inst.notes:
                self.ax_wf.hlines(note.pitch, note.start, note.end, linewidth=4, alpha=0.6)
        self.ax_wf.set_ylabel("MIDI pitch")
        self.ax_wf.set_xlabel("Time (s)")
        self.ax_wf.set_xlim(0, pm.get_end_time())
        self.canvas_wf.draw()

        # — Bottom: notation via music21 if available
        self.ax_sp.clear()
        if MUSIC21_AVAILABLE:
            try:
                score = converter.parse(path)
                tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                fn  = score.write('musicxml.png', fp=tmp.name)
                tmp.close()
                img = plt.imread(fn)
                self.ax_sp.imshow(img)
                self.ax_sp.axis('off')
                os.unlink(fn)
                self.canvas_sp.draw()
                return
            except Exception as e:
                print("music21 render failed:", e)

        # — Fallback: simple staff + note‐heads
        staff = [64,67,71,74,77]  # E4 G4 B4 D5 F5
        t_end = pm.get_end_time()
        for y in staff:
            self.ax_sp.hlines(y, 0, t_end, color="black", linewidth=1)
        for inst in pm.instruments:
            for note in inst.notes:
                if staff[0]-12 <= note.pitch <= staff[-1]+12:
                    circ = Circle((note.start, note.pitch), radius=0.15,
                                  facecolor="black", edgecolor="black")
                    self.ax_sp.add_patch(circ)
        self.ax_sp.set_xlim(0, t_end)
        self.ax_sp.set_ylim(staff[0]-5, staff[-1]+5)
        self.ax_sp.axis("off")
        self.canvas_sp.draw()

    def play(self):
        try:
            mixer.music.load(self.current_path)
            mixer.music.play()
        except Exception as e:
            messagebox.showerror("Playback error", str(e))

    def save_metadata(self):
        md = {
            "tags":       [t.strip() for t in self.tags_var.get().split(",") if t.strip()],
            "notes":      self.notes.get("1.0","end").strip(),
            "provenance": self.prov_var.get()
        }
        self.meta_db[self.current_rel] = md
        with open(METADATA_PATH, "w") as f:
            json.dump(self.meta_db, f, indent=2)
        messagebox.showinfo("Saved", "Metadata saved!")

    def run_target(self):
        tgt = self.run_var.get()
        if not tgt or tgt not in RUN_TARGETS:
            return
        cfg = RUN_TARGETS[tgt]
        cmd = [
            "conda", "run", "-n", cfg["env"],
            "python", cfg["script"],
            "--input", self.current_path
        ]
        threading.Thread(target=subprocess.run, args=(cmd,)).start()
        messagebox.showinfo("Launched", f"Started {tgt} on:\n{self.current_path}")

if __name__ == "__main__":
    Dashboard().mainloop()
