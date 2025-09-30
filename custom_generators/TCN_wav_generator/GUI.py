import sys
import os
# os.environ['GDK_SCALE'] = '2'  # Increase from 1 to 2 or higher based on your requirements

import tkinter as tk
from tkinter import filedialog, messagebox

###############################################################################
#                     TKINTER GUI WRAPPER
###############################################################################
class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Sound Prediction and Transformation GUI")

        # Existing Variables
        self.audio_file_var = tk.StringVar()
        self.output_folder_var = tk.StringVar()
        self.use_index_var = tk.BooleanVar(value=False)
        self.index_check_var = tk.StringVar(value="2980")
        # --- NEW: Variable for selected trained model ---
        self.selected_model_var = tk.StringVar(value="model1")  # Default selection

        # --- NEW: RAVE model file path ---
        self.rave_model_var = tk.StringVar()

        # --- NEW: Variable for length percentage from slider ---
        self.length_percentage = tk.DoubleVar(value=1.0)

        # --- NEW: Variable for organ selection for DDSP ---
        self.organ_var = tk.StringVar(value="Violin")  # Default organ for DDSP

        # Will hold the list of generated files (both .wav and .mid)
        self.generated_files = []
        
        # --- NEW: Status message variable
        self.status_var = tk.StringVar(value="")

        # --- NEW: SF2 path variable for Fluidsynth ---
        self.sf2_path_var = tk.StringVar(value="")  # You can put a default path here if desired

        # --- NEW: Instrument selection for MIDI synthesis
        self.midi_organ_var = tk.StringVar(value="Church Organ")  # default
        # Common GM instruments + "Sine Waves" (which uses pm.synthesize)
        self.instrument_map = {
            "Sine Waves": None,         # We'll handle this separately
            "Acoustic Grand Piano": 0,
            "Church Organ": 19,
            "Piano": 73,   #PIANO
            "Inst1": 1,
            "Inst2": 2,
            "Inst5": 5,
            "Inst6": 6,
            "Inst9": 9,
            "Inst10": 10,
            "Inst14": 14,
            "Inst15": 15,
            "Inst16": 16,
            "Inst27": 27,
            "Inst38": 38,
            "Inst44": 44
            
        }

        # Dictionary storing short descriptions about each model
        self.MODEL_INFOS = {
            "model1": "MFCC TCN model trained by song 1.",
            "model2": "MFCC TCN model trained by song 4.",
            "model3": "MFCC TCN model trained by songs 1 and 2.",
            "model4": "MFCC TCN model trained by songs 1 and 2.",
            "model5": "MFCC TCN model trained by songs 1 and 2.",
            "model6": "CQT TCN model trained by song 1.",
            "model7": "CQT TCN model trained by song 1.",
            "model8": "CQT TCN model trained by song 1.",
            "model9": "CQT TCN model trained by song 7b."
        }

        self.build_gui()

    def build_gui(self):
        # Global font for Ubuntu
        default_font = ("DejaVu Sans", 14)
        self.root.option_add("*Font", default_font)
    
        entry_width = 60
        button_width = 16
        padding_y = 10
        padding_x = 10
 
        # --- Row for choosing a trained model + 'Run' button ---
        row_model = tk.Frame(self.root)
        row_model.pack(padx=padding_x, pady=padding_y, fill='x')
        
        tk.Label(row_model, text="Select Prediction Model:").pack(side='left')
        
        model_options = ["model1", "model2", "model3", "model4", "model5", "model6", "model7", "model8", "model9"]
        self.model_dropdown = tk.OptionMenu(row_model, self.selected_model_var, *model_options)
        self.model_dropdown.config(width=20)
        self.model_dropdown.pack(side='left', padx=5)
        
        btn_run_model = tk.Button(row_model, text="Load model", command=self.on_run_selected_model, width=button_width)
        btn_run_model.pack(side='left', padx=(0, 10))    
 
        row0 = tk.Frame(self.root)
        row0.pack(padx=padding_x, pady=padding_y, fill='x')
        tk.Label(row0, text="Select Audio File:").pack(side='left')
        e_audio = tk.Entry(row0, textvariable=self.audio_file_var, width=entry_width)
        e_audio.pack(side='left', padx=5)
        btn_audio = tk.Button(row0, text="Browse", command=self.browse_audio, width=button_width)
        btn_audio.pack(side='left')
    
        row1 = tk.Frame(self.root)
        row1.pack(padx=padding_x, pady=padding_y, fill='x')
        chk_user_index = tk.Checkbutton(
            row1,
            text="User Defined Time Prediction Index",
            variable=self.use_index_var,
            command=self.toggle_index_entry
        )
        chk_user_index.pack(side='left')
    
        row2 = tk.Frame(self.root)
        row2.pack(padx=padding_x, pady=padding_y, fill='x')
        tk.Label(row2, text="index_check:").pack(side='left')
        self.index_entry = tk.Entry(row2, textvariable=self.index_check_var, width=15, state='disabled')
        self.index_entry.pack(side='left', padx=5)
    
        row3 = tk.Frame(self.root)
        row3.pack(padx=padding_x, pady=padding_y, fill='x')
        tk.Label(row3, text="Output Folder:").pack(side='left')
        e_out = tk.Entry(row3, textvariable=self.output_folder_var, width=entry_width)
        e_out.pack(side='left', padx=5)
        btn_out = tk.Button(row3, text="Browse", command=self.browse_output, width=button_width)
        btn_out.pack(side='left')
    
        # RAVE model row
        row_rave = tk.Frame(self.root)
        row_rave.pack(padx=padding_x, pady=padding_y, fill='x')
        tk.Label(row_rave, text="RAVE Model:").pack(side='left')
        e_rave = tk.Entry(row_rave, textvariable=self.rave_model_var, width=entry_width)
        e_rave.pack(side='left', padx=5)
        btn_rave_browse = tk.Button(row_rave, text="Browse", command=self.browse_rave_model, width=button_width)
        btn_rave_browse.pack(side='left')
        
        # --- Row for Length Percentage slider ---
        row_slider = tk.Frame(self.root)
        row_slider.pack(padx=padding_x, pady=padding_y, fill='x')
        tk.Label(row_slider, text="Length Percentage:").pack(side='left')
        self.slider_length = tk.Scale(
            row_slider,
            variable=self.length_percentage,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient='horizontal',
            length=300
        )
        self.slider_length.pack(side='left', padx=5)
    
        # Buttons Row
        row4 = tk.Frame(self.root)
        row4.pack(padx=padding_x, pady=padding_y, fill='x')
    
        btn_run = tk.Button(row4, text="Run Snippet", command=self.on_run_snippet, width=button_width)
        btn_run.pack(side='left', padx=(0, 10))
    
        btn_rave = tk.Button(row4, text="Generate RAVE", command=self.on_rave_generation, width=button_width)
        btn_rave.pack(side='left', padx=(0, 10))
        
        btn_ddsp = tk.Button(row4, text="Generate DDSP", command=self.on_ddsp_generation, width=button_width)
        btn_ddsp.pack(side='left', padx=(0, 10))
        
        tk.Label(row4, text="Organ:").pack(side='left')
        organ_dropdown = tk.OptionMenu(row4, self.organ_var, "Violin", "Flute")
        organ_dropdown.config(width=10)
        organ_dropdown.pack(side='left', padx=(0, 10))
    
        btn_play = tk.Button(row4, text="Play Selected File", command=self.on_play_file, width=button_width)
        btn_play.pack(side='left')
    
        # Listbox row
        row5 = tk.Frame(self.root)
        row5.pack(padx=padding_x, pady=padding_y, fill='both', expand=True)
        tk.Label(row5, text="Generated Files:").pack(anchor='w')
        self.listbox = tk.Listbox(row5, height=10)
        self.listbox.pack(side='left', fill='both', expand=True)
        scrollbar = tk.Scrollbar(row5, orient='vertical', command=self.listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.listbox.config(yscrollcommand=scrollbar.set)

        # ---- New row for MIDI generation (Basic Pitch) + WAV from MIDI ----
        row6 = tk.Frame(self.root)
        row6.pack(padx=padding_x, pady=padding_y, fill='x')

        btn_gen_midi = tk.Button(row6, text="Gen MIDI", command=self.on_generate_midi, width=button_width)
        btn_gen_midi.pack(side='left', padx=(0, 10))

        btn_gen_wav_from_midi = tk.Button(row6, text="Gen WAV from MIDI", command=self.on_generate_wav_from_midi, width=button_width)
        btn_gen_wav_from_midi.pack(side='left', padx=(0, 10))

        # MIDI Organ label + OptionMenu
        tk.Label(row6, text="MIDI Instrument:").pack(side='left')
        midi_organ_dropdown = tk.OptionMenu(row6, self.midi_organ_var, *self.instrument_map.keys())
        midi_organ_dropdown.config(width=14)
        midi_organ_dropdown.pack(side='left', padx=5)

        # Button + entry for SoundFont
        btn_sf2 = tk.Button(row6, text="Load SF (.sf2)", command=self.browse_sf2, width=button_width)
        btn_sf2.pack(side='left', padx=(20, 5))

        # status_frame at the bottom
        status_frame = tk.Frame(self.root)
        status_frame.pack(side='bottom', fill='x')
        tk.Label(status_frame, textvariable=self.status_var).pack(side='left', padx=5)

    # --- EXISTING UTILITY METHODS ---
    def browse_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )
        if file_path:
            self.audio_file_var.set(file_path)

    def browse_output(self):
        folder_path = filedialog.askdirectory(title="Select Output Folder")
        if folder_path:
            self.output_folder_var.set(folder_path)

    def toggle_index_entry(self):
        if self.use_index_var.get():
            self.index_entry.config(state='normal')
        else:
            self.index_entry.config(state='disabled')

    # --- Browse RAVE model
    def browse_rave_model(self):
        model_path = filedialog.askopenfilename(
            title="Select RAVE Model",
            filetypes=[("Torchscript / TS / PT", "*.ts *.pt *.pth"), ("All Files", "*.*")]
        )
        if model_path:
            self.rave_model_var.set(model_path)

    # --- Browse SF2 SoundFont ---
    def browse_sf2(self):
        sf2_path = filedialog.askopenfilename(
            title="Select SoundFont (.sf2) File",
            filetypes=[("SoundFont Files", "*.sf2"), ("All Files", "*.*")]
        )
        if sf2_path:
            self.sf2_path_var.set(sf2_path)
            messagebox.showinfo("SoundFont Selected", f"SoundFont loaded:\n{sf2_path}")

    def on_run_snippet(self):
        audio_file = self.audio_file_var.get()
        if not audio_file or not os.path.isfile(audio_file):
            messagebox.showerror("Error", "Please select a valid .wav file.")
            return

        out_folder = self.output_folder_var.get()
        if not out_folder:
            messagebox.showerror("Error", "Please select a valid output folder.")
            return

        user_defined = self.use_index_var.get()
        try:
            idx_val = int(self.index_check_var.get())
        except ValueError:
            idx_val = 0

        # Clear old results
        self.listbox.delete(0, tk.END)
        self.generated_files = []

        # Retrieve the current slider value (0.0 to 1.0)
        length_percentage_value = self.length_percentage.get()

        # Call your EXACT snippet code
        try:
            new_files = run_inference(
                _path_=out_folder,
                audio_file_another=audio_file,
                user_defined_time_prediction_index=user_defined,
                index_check=idx_val,
                length_percentage=length_percentage_value
            )
            if new_files:
                self.generated_files = new_files
                for f in new_files:
                    self.listbox.insert(tk.END, f)
                messagebox.showinfo("Success", "Snippet completed!")
            else:
                messagebox.showinfo("Info", "No files generated.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")

    # --- RAVE generation using generate_RAVE.py ---
    def on_rave_generation(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a .wav file from the list first.")
            return
    
        idx = selection[0]
        input_wav = self.generated_files[idx]
        if not input_wav.lower().endswith(".wav"):
            messagebox.showwarning("Warning", "Selected file is not a .wav.")
            return
        if not os.path.isfile(input_wav):
            messagebox.showerror("Error", "Selected file no longer exists.")
            return
    
        rave_model = self.rave_model_var.get()
        if not rave_model or not os.path.isfile(rave_model):
            messagebox.showerror("Error", "Please select a valid RAVE model file.")
            return
    
        import subprocess
        model_basename = os.path.splitext(os.path.basename(rave_model))[0]
        input_basename = os.path.splitext(os.path.basename(input_wav))[0]
        out_folder = self.output_folder_var.get()
        rave_output = os.path.join(out_folder, f"{input_basename}_RAVE_{model_basename}.wav")
    
        cmd = [
            "python",
            "generate_RAVE.py",
            "--model", rave_model,
            "--input", input_wav,
            "--output", rave_output
        ]
        try:
            subprocess.run(cmd, check=True)
            self.generated_files.append(rave_output)
            self.listbox.insert(tk.END, rave_output)
            messagebox.showinfo("Success", f"RAVE generation complete: {rave_output}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"RAVE generation failed:\n{e}")

    # --- DDSP generation using generate_DDSP.py ---
    def on_ddsp_generation(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a .wav file from the list first.")
            return
    
        idx = selection[0]
        input_wav = self.generated_files[idx]
        if not input_wav.lower().endswith(".wav"):
            messagebox.showwarning("Warning", "Selected file is not a .wav.")
            return
        if not os.path.isfile(input_wav):
            messagebox.showerror("Error", "Selected file no longer exists.")
            return

        organ_choice = self.organ_var.get()
        input_basename = os.path.splitext(os.path.basename(input_wav))[0]
        out_folder = self.output_folder_var.get()
        ddsp_output = os.path.join(out_folder, f"{input_basename}_DDSP_{organ_choice}.wav")
    
        import subprocess
        cmd = [
            "python",
            "generate_DDSP.py",
            "--organ", organ_choice,
            "--input", input_wav,
            "--output", ddsp_output
        ]
        try:
            subprocess.run(cmd, check=True)
            self.generated_files.append(ddsp_output)
            self.listbox.insert(tk.END, ddsp_output)
            messagebox.showinfo("Success", f"DDSP generation complete: {ddsp_output}")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"DDSP generation failed:\n{e}")

    # --- NEW: Generate MIDI using Basic Pitch ---
    def on_generate_midi(self):
        """
        1) Takes the currently selected .wav from the Listbox.
        2) Uses Basic Pitch to convert the .wav into a MIDI file.
        3) Saves the .mid into the same output folder with the same base name.
        4) Adds .mid to the Listbox + self.generated_files.
        """
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a .wav file from the list first.")
            return
    
        idx = selection[0]
        input_wav = self.generated_files[idx]
        if not input_wav.lower().endswith(".wav"):
            messagebox.showwarning("Warning", "Selected file is not a .wav.")
            return
        if not os.path.isfile(input_wav):
            messagebox.showerror("Error", "Selected file no longer exists.")
            return

        out_folder = self.output_folder_var.get()
        if not out_folder:
            messagebox.showerror("Error", "Please select a valid output folder first.")
            return

        import soundfile as sf
        try:
            from basic_pitch.inference import predict
            from basic_pitch import ICASSP_2022_MODEL_PATH
        except ImportError as e:
            messagebox.showerror("Error", "Please install basic_pitch:\n pip install basic-pitch")
            return

        input_basename = os.path.splitext(os.path.basename(input_wav))[0]
        midi_output = os.path.join(out_folder, f"{input_basename}.mid")

        try:
            # Run basic pitch
            model_output, midi_data, note_events = predict(input_wav, ICASSP_2022_MODEL_PATH)
            midi_data.write(midi_output)

            self.generated_files.append(midi_output)
            self.listbox.insert(tk.END, midi_output)
            messagebox.showinfo("Success", f"MIDI file generated:\n{midi_output}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating MIDI:\n{e}")

    # --- NEW: Generate WAV from an existing MIDI file ---
    def on_generate_wav_from_midi(self):
        """
        1) Takes the currently selected .mid from the Listbox.
        2) Uses pretty_midi (and fluidsynth if not Sine Waves) to generate a .wav.
        3) Respects the instrument choice from the self.midi_organ_var dropdown.
        4) Saves the new .wav into the same output folder.
        5) Appends to the listbox + generated_files.
        """
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "Please select a .mid file from the list.")
            return

        idx = selection[0]
        input_mid = self.generated_files[idx]
        if not input_mid.lower().endswith(".mid"):
            messagebox.showwarning("Warning", "Selected file is not a .mid.")
            return
        if not os.path.isfile(input_mid):
            messagebox.showerror("Error", "Selected file no longer exists.")
            return

        out_folder = self.output_folder_var.get()
        if not out_folder:
            messagebox.showerror("Error", "Please select a valid output folder first.")
            return

        organ_choice = self.midi_organ_var.get()  # e.g. "Church Organ" or "Sine Waves"
        input_basename = os.path.splitext(os.path.basename(input_mid))[0]
        out_wav = os.path.join(out_folder, f"{input_basename}_{organ_choice.replace(' ', '')}.wav")

        try:
            import pretty_midi
            import soundfile as sf
        except ImportError as e:
            messagebox.showerror("Error", "Please install pretty_midi:\n pip install pretty_midi")
            return

        try:
            pm = pretty_midi.PrettyMIDI(input_mid)

            if organ_choice == "Sine Waves":
                # Basic sine wave synthesis
                audio_data = pm.synthesize(fs=16000)
            else:
                # Fluidsynth approach
                sf2_path = self.sf2_path_var.get()
                if not sf2_path or not os.path.isfile(sf2_path):
                    messagebox.showwarning("Warning", "No valid .sf2 selected, or file not found.\n\n"
                                                     "Please click 'Load SoundFont (.sf2)' to choose one.")
                    return

                program_num = self.instrument_map.get(organ_choice, 19)  # fallback to church organ
                # If multiple instruments exist, set the program of the first
                if pm.instruments:
                    pm.instruments[0].program = program_num

                audio_data = pm.fluidsynth(fs=16000, sf2_path=sf2_path)

            sf.write(out_wav, audio_data, 16000)
            self.generated_files.append(out_wav)
            self.listbox.insert(tk.END, out_wav)
            messagebox.showinfo("Success", f"WAV successfully generated:\n{out_wav}")
        except Exception as e:
            messagebox.showerror("Error", f"Error generating WAV from MIDI:\n{e}")

    def on_play_file(self):
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Info", "No file selected.")
            return

        idx = selection[0]
        file_to_play = self.generated_files[idx]
        if os.path.isfile(file_to_play):
            try:
                from playsound import playsound
                playsound(file_to_play)
            except Exception as e:
                messagebox.showerror("Error", f"Error playing file:\n{e}")
        else:
            messagebox.showerror("Error", "Selected file no longer exists.")

    def on_run_selected_model(self):
        chosen_model = self.selected_model_var.get()
        print(f"You selected: {chosen_model}")
        
        # Set status to "Please wait..." and force an update
        self.status_var.set("Please wait...")
        self.root.update_idletasks() 
            
        try:
            # 1) Put chosen_model into the global namespace 
            globals()["chosen_model"] = chosen_model
            with open("initializations.py", "r") as f:
                code_str = f.read()
            exec(code_str, globals())  # Variables & functions are now defined globally

            print("initializations.py executed successfully!")   
            self.status_var.set("Loaded prediction model.")

            # Show a message box with info about the selected model, if it exists
            model_info = self.MODEL_INFOS.get(chosen_model, "No information available for this model.")
            messagebox.showinfo("Model Information", f"You selected: {chosen_model}\n\n{model_info}")
            
        except Exception as e:
            print("An error occurred in initializations.py:", e)
            self.status_var.set("Error occurred. Check logs.")

def main():
    os.environ['GDK_SCALE'] = '4'
    root = tk.Tk()
    root.title("AI Sound Prediction and Transformation GUI")
    root.geometry("1280x1024")
    root.minsize(900, 600)
    app = PredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
