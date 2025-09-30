#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_Basicpitch.py

Convert a WAV file to MIDI using BasicPitch.
"""

import argparse
from basic_pitch.inference import predict

def wav_to_midi(wav_path: str, midi_path: str):
    """
    1) Run BasicPitch prediction on the input WAV
    2) Write out the resulting MIDI
    """
    try:
        # predict returns (model_output, PrettyMIDI object, note_events)
        _, midi_data, _ = predict(wav_path)
        midi_data.write(midi_path)
    except Exception as e:
        raise RuntimeError(f"Error converting {wav_path} → MIDI: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="WAV → MIDI via BasicPitch"
    )
    parser.add_argument(
        "-i","--input", required=True,
        help="Path to input .wav"
    )
    parser.add_argument(
        "-o","--output", required=True,
        help="Path to output .mid"
    )
    args = parser.parse_args()

    wav_to_midi(args.input, args.output)
