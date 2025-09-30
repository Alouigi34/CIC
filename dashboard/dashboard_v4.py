#!/usr/bin/env python3
import os
import re
import json
import subprocess
import threading
import tempfile
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import pygame.mixer as mixer
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk
)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import SpanSelector

# try to import music21 for sheet rendering
try:
    from music21 import converter
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False

# ─── CONFIG ────────────────────────────────────────────────────────────────
COMMON_DIR    = "/home/alels_star/Desktop/AI_composition_assistant/v0.01/common_data_space"
#METADATA_PATH = os.path.join(COMMON_DIR, "metadata.json")
METADATA_PATH = "/home/alels_star/Desktop/AI_composition_assistant/v0.01/metadata/metadata.json"
AUDIO_EXTS    = {".wav", ".mp3", ".flac"}

RUN_TARGETS = {
    "MusicGen (wav continuation)": {
        "env":    "musicgen_",
        "script": "/home/alels_star/Desktop/AI_composition_assistant/v0.01/musicgen_wav_generator/gen_chroma.py"
    },
    "Magenta (MIDI continuation)": {
        "env":    "magenta_",
        "script": "/home/alels_star/Desktop/AI_composition_assistant/v0.01/magenta_midi_generator/generate.py"
    },
}
# ────────────────────────────────────────────────────────────────────────────

class Dashboard(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Composition Dashboard")
        self.geometry("1200x900")

        mixer.init()
        mixer.set_num_channels(2)
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r") as f:
                self.meta_db = json.load(f)
        else:
            self.meta_db = {}

        left = ttk.Frame(self, width=500)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        sf = ttk.Frame(left); sf.pack(fill="x", pady=(5,2), padx=5)
        ttk.Label(sf, text="Filter:").pack(side="left")
        self.search_var = tk.StringVar()
        ttk.Entry(sf, textvariable=self.search_var).pack(side="left", fill="x", expand=True, padx=(5,0))
        ttk.Button(sf, text="Go",    command=self.apply_filter).pack(side="left", padx=5)
        ttk.Button(sf, text="Clear", command=self.clear_filter).pack(side="left")

        style = ttk.Style(self)
        style.configure("Big.Treeview", font=("Arial",10), rowheight=24)
        style.configure("Big.Treeview.Heading", font=("Arial",10))
        self.tree = ttk.Treeview(left, style="Big.Treeview", show="tree")
        self.tree.column("#0",width=450, stretch=False)
        self.tree.pack(fill="both", expand=True, pady=(0,5), padx=5)
        self.tree.config(selectmode="extended")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.build_full_tree()

        right = ttk.Frame(self); right.pack(side="right", fill="both", expand=True)

        self.fig_wf, self.ax_wf = plt.subplots(figsize=(6,1.4),dpi=100,constrained_layout=True)
        self.ax_wf.set_title("Waveform",pad=10)
        self.canvas_wf = FigureCanvasTkAgg(self.fig_wf, master=right)
        self.toolbar_wf = NavigationToolbar2Tk(self.canvas_wf, right)
        self.toolbar_wf.update()
        self.canvas_wf.get_tk_widget().pack(padx=5,pady=(2,0))

        self.span = SpanSelector(self.ax_wf, self.on_select_region,'horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor='blue'))

        self.fig_sp, self.ax_sp = plt.subplots(figsize=(6,2.4),dpi=100,constrained_layout=True)
        self.ax_sp.set_title("Spectrogram / Notation",pad=10)
        self.canvas_sp = FigureCanvasTkAgg(self.fig_sp, master=right)
        self.toolbar_sp = NavigationToolbar2Tk(self.canvas_sp, right)
        self.toolbar_sp.update()
        self.canvas_sp.get_tk_widget().pack(padx=5,pady=(2,5))

        ctl = ttk.Frame(right); ctl.pack(pady=5)
        for txt, cmd in [("Play",self.play),("Play Both",self.play_both),
                         ("Stop",mixer.music.stop),("Reset Zoom",self.reset_zoom),
                         ("Compare",self.compare)]:
            ttk.Button(ctl, text=txt, command=cmd).pack(side="left",padx=5)

        meta = ttk.LabelFrame(right, text="Metadata"); meta.pack(fill="both",expand=True,padx=5,pady=5)
        ttk.Label(meta, text="Tags (comma-sep):").pack(anchor="w")
        self.tags_var = tk.StringVar(); ttk.Entry(meta, textvariable=self.tags_var).pack(fill="x")
        ttk.Label(meta, text="Notes:").pack(anchor="w")
        self.notes = tk.Text(meta,height=4); self.notes.pack(fill="x")
        ttk.Label(meta, text="Provenance:").pack(anchor="w")
        self.prov_var = tk.StringVar(); ttk.Entry(meta, textvariable=self.prov_var, state="readonly").pack(fill="x")
        ttk.Button(meta, text="Save Metadata", command=self.save_metadata).pack(pady=5)

        runf = ttk.Frame(right); runf.pack(pady=5)
        ttk.Label(runf, text="Run with:").pack(side="left")
        self.run_var = tk.StringVar()
        self.run_cb = ttk.Combobox(runf, values=list(RUN_TARGETS), textvariable=self.run_var, state="readonly")
        self.run_cb.pack(side="left",padx=5)
        ttk.Button(runf, text="Go", command=self.run_target).pack(side="left")

    def build_full_tree(self):
        self.tree.delete(*self.tree.get_children())
        def rec(parent,path):
            for fn in sorted(os.listdir(path)):
                full=os.path.join(path,fn)
                node=self.tree.insert(parent,"end",text=fn,open=False)
                if os.path.isdir(full): rec(node,full)
        rec("",COMMON_DIR)

    def apply_filter(self):
        term=self.search_var.get().strip().lower()
        if not term: return
        self.tree.delete(*self.tree.get_children())
        matches=[]
        for root,_,files in os.walk(COMMON_DIR):
            for fn in files:
                rel=os.path.relpath(os.path.join(root,fn),COMMON_DIR)
                md=self.meta_db.get(rel,{});
                hay=" ".join([fn,*md.get("tags",[]),md.get("notes",""),md.get("provenance","")]).lower()
                if term in hay: matches.append(rel)
        for rel in sorted(matches): self.tree.insert("","end",text=rel)

    def clear_filter(self): self.search_var.set(""); self.build_full_tree()

    def on_select(self,_):
        sels=self.tree.selection()
        if len(sels)==1: self._view_single(sels[0])

    def _view_single(self,sel):
        parts,cur=[],sel
        while cur: parts.append(self.tree.item(cur,"text")); cur=self.tree.parent(cur)
        path=os.path.join(COMMON_DIR,*reversed(parts))
        self.current_path=path; self.current_rel=os.path.relpath(path,COMMON_DIR)
        ext=os.path.splitext(path)[1].lower()
        if ext in AUDIO_EXTS: self._show_audio(path)
        elif ext in {".mid",".midi"}: self._show_midi(path)
        else:
            self.ax_wf.clear(); self.canvas_wf.draw()
            self.ax_sp.clear(); self.canvas_sp.draw()
        md=self.meta_db.get(self.current_rel,{})
        self.tags_var.set(",".join(md.get("tags",[])))
        self.notes.delete("1.0","end"); self.notes.insert("1.0",md.get("notes",""))
        prov=Path(self.current_rel).parts[0] if "/" in self.current_rel else ""
        self.prov_var.set(prov)

    def _show_audio(self,path):
        y,sr=librosa.load(path,sr=None)
        times=np.arange(len(y))/sr
        self.ax_wf.clear(); self.ax_wf.plot(times,y,linewidth=0.5,label=os.path.basename(path))
        self.ax_wf.set_ylabel("amplitude"); self.ax_wf.set_xlabel("Time (s)")
        self.full_xlim=(0,times[-1]); self.ax_wf.set_xlim(*self.full_xlim); self.ax_wf.legend(); self.canvas_wf.draw()
        S=librosa.stft(y,n_fft=1024,hop_length=512)
        Sdb=librosa.amplitude_to_db(np.abs(S),ref=np.max)
        freqs=np.linspace(0,sr/2,Sdb.shape[0]); ts=np.arange(Sdb.shape[1])*512/sr
        self.ax_sp.clear(); self.ax_sp.imshow(Sdb,origin="lower",aspect="auto",cmap="magma",extent=[ts[0],ts[-1],freqs[0],freqs[-1]])
        self.ax_sp.set_ylabel("Frequency (Hz)"); self.ax_sp.set_xlabel("Time (s)")
        self.full_xlim_sp=(ts[0],ts[-1]); self.canvas_sp.draw()

    def _show_midi(self,path):
        pm=pretty_midi.PrettyMIDI(path); end_t=pm.get_end_time()
        self.ax_wf.clear()
        for note in sum((inst.notes for inst in pm.instruments),[]):
            self.ax_wf.hlines(note.pitch,note.start,note.end,linewidth=4,alpha=0.6)
        self.ax_wf.set_ylabel("MIDI pitch"); self.ax_wf.set_xlabel("Time (s)")
        self.full_xlim=(0,end_t); self.ax_wf.set_xlim(*self.full_xlim); self.canvas_wf.draw()
        self.ax_sp.clear()
        if MUSIC21_AVAILABLE:
            try:
                score=converter.parse(path)
                tmp=tempfile.NamedTemporaryFile(suffix=".png",delete=False); fn=score.write('musicxml.png',fp=tmp.name); tmp.close()
                img=plt.imread(fn); os.unlink(fn)
                self.ax_sp.imshow(img); self.ax_sp.axis("off"); self.canvas_sp.draw(); return
            except Exception as e:
                print("music21 failed:",e)
        staff=[64,67,71,74,77]
        for y in staff: self.ax_sp.hlines(y,0,end_t,color="black",linewidth=1)
        for note in sum((inst.notes for inst in pm.instruments),[]):
            if staff[0]-12<=note.pitch<=staff[-1]+12:
                c=Circle((note.start,note.pitch),radius=0.15,fc="black"); self.ax_sp.add_patch(c)
        self.ax_sp.set_xlim(0,end_t); self.ax_sp.set_ylim(staff[0]-5,staff[-1]+5)
        self.ax_sp.axis("off"); self.full_xlim_sp=(0,end_t); self.canvas_sp.draw()

    def compare(self):
        sels=self.tree.selection()
        if len(sels)!=2: messagebox.showerror("Compare","Select exactly two files to compare."); return
        paths=[]
        for sel in sels:
            parts,cur=[],sel
            while cur: parts.append(self.tree.item(cur,"text")); cur=self.tree.parent(cur)
            paths.append(os.path.join(COMMON_DIR,*reversed(parts)))
        exts=[os.path.splitext(p)[1].lower() for p in paths]
        self.ax_wf.clear()
        for idx,p in enumerate(paths):
            label=os.path.basename(p)
            if exts[idx] in AUDIO_EXTS:
                y,sr=librosa.load(p,sr=None); t=np.arange(len(y))/sr
                self.ax_wf.plot(t,y,linewidth=0.8,alpha=0.7,label=label)
            else:
                pm=pretty_midi.PrettyMIDI(p)
                for note in sum((inst.notes for inst in pm.instruments),[]):
                    self.ax_wf.hlines(note.pitch+idx*4,note.start,note.end,linewidth=4,alpha=0.7,label=label if note is pm.instruments[0].notes[0] else None)
        self.ax_wf.set_ylabel("Signal / pitch"); self.ax_wf.set_xlabel("Time (s)")
        if hasattr(self,'full_xlim'): self.ax_wf.set_xlim(*self.full_xlim)
        self.ax_wf.legend(); self.canvas_wf.draw()
        self.ax_sp.clear()
        if all(e in AUDIO_EXTS for e in exts):
            for idx,p in enumerate(paths):
                y,sr=librosa.load(p,sr=None)
                S=librosa.stft(y,n_fft=1024,hop_length=512)
                Sdb=librosa.amplitude_to_db(np.abs(S),ref=np.max)
                freqs=np.linspace(0,sr/2,Sdb.shape[0]); ts=np.arange(Sdb.shape[1])*512/sr
                self.ax_sp.imshow(Sdb if idx==0 else np.roll(Sdb,50,axis=0),origin="lower",aspect="auto",cmap="magma",extent=[ts[0],ts[-1],freqs[0],freqs[-1]],alpha=0.6)
            self.ax_sp.set_ylabel("Frequency (Hz)"); self.ax_sp.set_xlabel("Time (s)")
        else:
            for idx,p in enumerate(paths):
                pm=pretty_midi.PrettyMIDI(p)
                for note in sum((inst.notes for inst in pm.instruments),[]):
                    circ=Circle((note.start,note.pitch),radius=0.15,alpha=0.7,facecolor=("blue" if idx==0 else "red"))
                    self.ax_sp.add_patch(circ)
            self.ax_sp.axis("off")
        if hasattr(self,'full_xlim_sp'): self.ax_sp.set_xlim(*self.full_xlim_sp)
        self.canvas_sp.draw()

    def play(self):
        path=getattr(self,"current_path",None)
        if not path: return
        ext=os.path.splitext(path)[1].lower()
        if ext in AUDIO_EXTS:
            mixer.music.load(path); mixer.music.play()
        else:
            messagebox.showwarning("Play","Can't play MIDI here.")

    def play_both(self):
        if not hasattr(self,"compare_paths"): messagebox.showerror("Play Both","First hit Compare on two audio files."); return
        p1,p2=self.compare_paths
        if os.path.splitext(p1)[1] not in AUDIO_EXTS or os.path.splitext(p2)[1] not in AUDIO_EXTS:
            messagebox.showerror("Play Both","Both must be audio files."); return
        s1=mixer.Sound(p1); s2=mixer.Sound(p2)
        mixer.Channel(0).play(s1); mixer.Channel(1).play(s2)

    def on_select_region(self,xmin,xmax):
        self.ax_wf.set_xlim(xmin,xmax); self.ax_sp.set_xlim(xmin,xmax)
        self.canvas_wf.draw(); self.canvas_sp.draw()

    def reset_zoom(self):
        if hasattr(self,"full_xlim"): self.ax_wf.set_xlim(*self.full_xlim)
        if hasattr(self,"full_xlim_sp"): self.ax_sp.set_xlim(*self.full_xlim_sp)
        self.canvas_wf.draw(); self.canvas_sp.draw()

    def save_metadata(self):
        md={
            "tags":[t.strip() for t in self.tags_var.get().split(",") if t.strip()],
            "notes":self.notes.get("1.0","end").strip(),
            "provenance":self.prov_var.get()
        }
        self.meta_db[self.current_rel]=md
        with open(METADATA_PATH,"w") as f: json.dump(self.meta_db,f,indent=2)
        messagebox.showinfo("Saved","Metadata saved!")

    def run_target(self):
        tgt=self.run_var.get()
        if not tgt or tgt not in RUN_TARGETS: return
        cfg=RUN_TARGETS[tgt]
        cmd=["conda","run","-n",cfg["env"],"python",cfg["script"],"--input",self.current_path]
        threading.Thread(target=subprocess.run,args=(cmd,)).start()
        messagebox.showinfo("Launched",f"Started {tgt} on:\n{self.current_path}")

if __name__ == "__main__":
    Dashboard().mainloop()
