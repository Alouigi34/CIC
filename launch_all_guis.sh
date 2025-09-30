



#!/usr/bin/env bash
# Activate your GUI environment
conda activate AI_composer_GUIs_
BASE="$PWD/GUIs"

python "$BASE/GUI_Basicpitch.py"    &
python "$BASE/GUI_DDSP.py"         &
python "$BASE/GUI_Fluidsynth.py"   &
python "$BASE/GUI_magenta.py"      &
python "$BASE/GUI_musicgen.py"     &
python "$BASE/GUI_RAVE.py"         &
python "$BASE/GUI_custom_TCN_midi.py"     &
python "$BASE/GUI_custom_TCN_wav.py"         &

wait
