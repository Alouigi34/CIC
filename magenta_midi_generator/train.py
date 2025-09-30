# v0.01/magenta_midi_generator/train.py

#!/usr/bin/env python
"""
performance_rnn_train_from_midi.py — now with CLI flags and updated defaults.
"""

import os, glob, time, shutil, tempfile, argparse
import note_seq
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from magenta.models.performance_rnn import performance_model
from magenta.models.shared import events_rnn_graph, events_rnn_train
from magenta.pipelines import performance_pipeline, pipeline
import magenta

# ───────────────── argparse ───────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
COMMON_DATA   = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "common_data_space"))
DEFAULT_MIDI  = os.path.join(COMMON_DATA, "training_data", "magenta", "training_midis")
DEFAULT_RUN   = BASE_DIR
DEFAULT_BUNDLE= os.path.join(BASE_DIR, "models", "performance_with_dynamics.mag")

parser = argparse.ArgumentParser(description="Train Performance-RNN")
parser.add_argument("--MIDI_DIR",    default=DEFAULT_MIDI,
                    help="Directory of input MIDI files")
parser.add_argument("--RUN_DIR",     default=DEFAULT_RUN,
                    help="Base run directory (checkpoints → RUN_DIR/train)")
parser.add_argument("--BUNDLE_FILE", default=DEFAULT_BUNDLE,
                    help=".mag bundle to fine-tune if no checkpoints exist")
parser.add_argument("--TRAIN_STEPS", type=int, default=200,
                    help="Total training steps")
parser.add_argument("--BATCH_SIZE",  type=int, default=64,
                    help="Batch size")
parser.add_argument("--CONFIG_NAME", default="performance_with_dynamics_compact",
                    help="Which Performance-RNN config to use")
parser.add_argument("--LOG_LEVEL",   default="INFO",
                    choices=["DEBUG","INFO","WARN","ERROR"])
args = parser.parse_args()

tf.logging.set_verbosity(args.LOG_LEVEL)

# ───────────────── helper functions ────────────────────────────────────
def iter_midis(folder):
    for patt in ("**/*.mid","**/*.midi","**/*.mxl"):
        for fn in glob.glob(os.path.join(folder,patt), recursive=True):
            try:
                yield note_seq.midi_file_to_sequence_proto(fn)
            except Exception as e:
                tf.logging.warn("Skipping %s: %s", fn, e)

def build_dataset(midi_dir):
    cfg  = performance_model.default_configs[args.CONFIG_NAME]
    pipe = performance_pipeline.get_pipeline(
        min_events=32, max_events=512, eval_ratio=0.0, config=cfg
    )
    tmpdir = tempfile.mkdtemp(prefix="perf_rnn_ds_")
    pipeline.run_pipeline_serial(pipe, iter_midis(midi_dir), tmpdir)
    tfrec = os.path.join(tmpdir, "training_performances.tfrecord")
    if not tf.gfile.Exists(tfrec):
        raise RuntimeError("No training examples produced!")
    n = magenta.common.count_records([tfrec])
    tf.logging.info("✓ Dataset ready: %d examples", n)
    return tfrec, tmpdir

# ───────────────── main ─────────────────────────────────────────────────
def main():
    t0 = time.time()
    tfrec, workdir = build_dataset(args.MIDI_DIR)

    cfg = performance_model.default_configs[args.CONFIG_NAME]
    cfg.hparams.batch_size = min(cfg.hparams.batch_size, args.BATCH_SIZE)

    build_graph_fn = events_rnn_graph.get_build_graph_fn(
        mode="train",
        config=cfg,
        sequence_example_file_paths=[tfrec]
    )

    train_ckpt_dir = os.path.join(os.path.expanduser(args.RUN_DIR), "train")
    tf.gfile.MakeDirs(train_ckpt_dir)

    ckpts = tf.gfile.Glob(os.path.join(train_ckpt_dir, "model.ckpt-*"))
    warm_bundle = None
    if args.BUNDLE_FILE and not ckpts:
        warm_bundle = args.BUNDLE_FILE
        tf.logging.info("Fine-tuning from bundle %s", warm_bundle)
    elif ckpts:
        tf.logging.info("Resuming from existing checkpoints")
    else:
        tf.logging.info("Training from scratch")

    events_rnn_train.run_training(
        build_graph_fn,
        train_ckpt_dir,
        num_training_steps     = args.TRAIN_STEPS,
        summary_frequency      = 10,
        checkpoints_to_keep    = 5,
        warm_start_bundle_file = warm_bundle
    )

    shutil.rmtree(workdir, ignore_errors=True)
    tf.logging.info("Total wall-clock %.1f s", time.time() - t0)

if __name__ == "__main__":
    main()
