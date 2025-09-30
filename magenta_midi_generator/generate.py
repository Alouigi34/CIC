# v0.01/magenta_midi_generator/generate.py

#!/usr/bin/env python
"""
performance_rnn_generate.py — now takes flags, defaults match your tree.
"""

import os, time, argparse
import note_seq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior(); tf.get_logger().setLevel("ERROR")

from magenta.models.performance_rnn import (
    performance_model, performance_sequence_generator
)
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2, music_pb2

# ────────── argparse with updated defaults ─────────────────
BASE_DIR    = os.path.dirname(__file__)
COMMON_DATA = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "common_data_space"))

DEFAULT_BUNDLE = os.path.join(BASE_DIR, "models", "performance_with_dynamics.mag")
DEFAULT_RUN    = BASE_DIR
DEFAULT_PRIMER = os.path.join(
    COMMON_DATA,
    "input_data", "magenta", "input_cutted_.mid"
)
DEFAULT_OUT    = os.path.join(COMMON_DATA, "generated_data", "magenta")

parser = argparse.ArgumentParser(description="Generate MIDI with Performance-RNN")
parser.add_argument("--use_pretrained_model", type=lambda s: s.lower()=="true",
                    default=True)
parser.add_argument("--BUNDLE_FILE", default=DEFAULT_BUNDLE)
parser.add_argument("--RUN_DIR",     default=DEFAULT_RUN)
parser.add_argument("--OUTPUT_DIR",  default=DEFAULT_OUT)
parser.add_argument("--PRIMER_MIDI", default=DEFAULT_PRIMER)
parser.add_argument("--NUM_OUTPUTS", type=int,   default=3)
parser.add_argument("--NUM_STEPS",   type=int,   default=3000)
parser.add_argument("--TEMPERATURE", type=float, default=1.0)
parser.add_argument("--CONFIG",      default="performance_with_dynamics")
parser.add_argument("--BEAM_SIZE",     type=int, default=1)
parser.add_argument("--BRANCH_FACTOR", type=int, default=1)
parser.add_argument("--STEPS_PER_ITERATION", type=int, default=1)
parser.add_argument("--DISABLE_CONDITIONING", action="store_true")
args = parser.parse_args()

if args.use_pretrained_model and not args.BUNDLE_FILE:
    parser.error("BUNDLE_FILE required when use_pretrained_model=True")

def read_bundle(path):
    return sequence_generator_bundle.read_bundle_file(path) if path else None

def primer_sequence():
    if args.PRIMER_MIDI:
        return note_seq.midi_file_to_sequence_proto(args.PRIMER_MIDI)
    return music_pb2.NoteSequence(ticks_per_quarter=note_seq.STANDARD_PPQ)

def build_options(gen, primer):
    end_time = args.NUM_STEPS / gen.steps_per_second
    opts = generator_pb2.GeneratorOptions()
    opts.generate_sections.add(start_time=primer.total_time, end_time=end_time)
    opts.args["temperature"].float_value       = args.TEMPERATURE
    opts.args["beam_size"].int_value           = args.BEAM_SIZE
    opts.args["branch_factor"].int_value       = args.BRANCH_FACTOR
    opts.args["steps_per_iteration"].int_value = args.STEPS_PER_ITERATION
    if args.DISABLE_CONDITIONING:
        opts.args["disable_conditioning"].string_value = "True"
    return opts

def make_generator(bundle):
    cfg_id = bundle.generator_details.id if bundle else args.CONFIG
    cfg    = performance_model.default_configs[cfg_id]
    cfg.hparams.batch_size = min(cfg.hparams.batch_size,
                                 args.BEAM_SIZE * args.BRANCH_FACTOR)
    ckpt_dir = (os.path.join(args.RUN_DIR, "train")
                if not args.use_pretrained_model else None)
    return performance_sequence_generator.PerformanceRnnSequenceGenerator(
        performance_model.PerformanceRnnModel(cfg),
        cfg.details,
        cfg.steps_per_second,
        cfg.num_velocity_bins,
        cfg.control_signals,
        cfg.optional_conditioning,
        checkpoint        = ckpt_dir,
        bundle            = bundle,
        note_performance  = cfg.note_performance
    )

def main():
    os.makedirs(args.OUTPUT_DIR, exist_ok=True)
    bundle = read_bundle(args.BUNDLE_FILE) if args.use_pretrained_model else None
    gen    = make_generator(bundle)
    primer = primer_sequence()
    opts   = build_options(gen, primer)
    timestamp = time.strftime("%Y-%m-%d_%H%M%S")
    pad = len(str(args.NUM_OUTPUTS))

    for i in range(args.NUM_OUTPUTS):
        seq   = gen.generate(primer, opts)
        fname = f"{timestamp}_{str(i+1).zfill(pad)}.mid"
        outp  = os.path.join(args.OUTPUT_DIR, fname)
        note_seq.sequence_proto_to_midi_file(seq, outp)
        print("✓", outp)

if __name__ == "__main__":
    main()
