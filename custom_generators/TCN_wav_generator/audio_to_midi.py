#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:06:10 2025

@author: alels_star
"""

from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import IPython

model_output, midi_data, note_events = predict('/home/alels_star/Desktop/ekpa_diplom/new3/test_GUI/predicted_song_6573_1a.wav')



midi_data.write('/home/alels_star/Desktop/ekpa_diplom/new3/test_GUI/out.mid')




import soundfile as sf
pm = midi_data

fs = 16000

# # For the "sine waves" version
# audio_data_sine = pm.synthesize(fs=fs)
# sf.write('/home/alels_star/Desktop/ekpa_diplom/new3/test_GUI/sine_output.wav', audio_data_sine, fs)

# # For the "cello" version
# audio_data_cello = pm.fluidsynth(fs=fs)
# sf.write('/home/alels_star/Desktop/ekpa_diplom/new3/test_GUI/cello_output.wav', audio_data_cello, fs)


sf2_path = '/home/alels_star/Downloads/ChateauGrand-Plus-Instruments-bs16i-v4.sf2' 
pm.instruments[0].program = 19  # GM standard: 0=Acoustic Grand Piano, 19=Church Organ
audio_data = pm.fluidsynth(sf2_path=sf2_path)
sf.write('/home/alels_star/Desktop/ekpa_diplom/new3/test_GUI/output.wav', audio_data, fs)

