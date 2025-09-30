#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_fluidsynth.py

Convert a MIDI file to WAV using pretty_midi + fluidsynth + a SoundFont.
"""

import argparse
import os
import pretty_midi
import soundfile as sf

def convert_midi_to_wav(midi_path, output_wav_path,
                        soundfont_path,
                        fs=16000, instrument_program=0):
    """
    2) Convert a MIDI file to WAV using pretty_midi and fluidsynth
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
        # set program on first instrument
        if pm.instruments:
            pm.instruments[0].program = instrument_program
        # synthesize
        audio_data = pm.fluidsynth(sf2_path=soundfont_path, fs=fs)
        # write out
        sf.write(output_wav_path, audio_data, fs)
    except Exception as e:
        raise ValueError(f"Error converting {midi_path} to WAV: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIDI → WAV via fluidsynth + SoundFont"
    )
    parser.add_argument("-i", "--input",     required=True,
                        help="Path to input .mid")
    parser.add_argument("-o", "--output",    required=True,
                        help="Path to output .wav")
    parser.add_argument("-s", "--soundfont",
                        default=os.path.join(
                            os.path.dirname(__file__),
                            "Fluidsynth_models",
                            "ChateauGrand-Plus-Instruments-bs16i-v4.sf2"
                        ),
                        help="Path to .sf2 SoundFont")
    parser.add_argument("-r", "--rate",      type=int, default=16000,
                        help="Sample rate (Hz)")
    parser.add_argument("-p", "--program",   type=int, default=0,
                        help="MIDI program number")
    args = parser.parse_args()

    convert_midi_to_wav(
        args.input,
        args.output,
        args.soundfont,
        fs=args.rate,
        instrument_program=args.program
    )
