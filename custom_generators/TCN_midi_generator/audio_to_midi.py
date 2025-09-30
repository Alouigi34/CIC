#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 12 16:06:10 2025

@author: alels_star
"""


import pretty_midi
import soundfile as sf


pm = pretty_midi.PrettyMIDI('/home/alels_star/Desktop/ekpa_diplom/piano_notes2/v7_more_notes_prediction_b/reconstructed_final_predicted_numpy_actual.mid')
#pm = pretty_midi.PrettyMIDI('/home/alels_star/Desktop/ekpa_diplom/piano_notes2/v7_more_notes_prediction_b/reconstructed_final_real_numpy_actual.mid')



fs = 16000


sf2_path = '/home/alels_star/Downloads/ChateauGrand-Plus-Instruments-bs16i-v4.sf2' 
pm.instruments[0].program = 19 # GM standard: 0=Acoustic Grand Piano, 19=Church Organ
audio_data = pm.fluidsynth(sf2_path=sf2_path)
sf.write('/home/alels_star/Desktop/ekpa_diplom/piano_notes2/v7_more_notes_prediction_b/output.wav', audio_data, fs)



