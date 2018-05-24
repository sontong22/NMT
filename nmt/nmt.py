# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TensorFlow NMT model implementation."""
from __future__ import print_function

import os

import tensorflow as tf

from . import train
from .utils import misc_utils as utils
from .utils import vocab_utils

utils.check_tensorflow_version()


def create_hparams():
  """Create training hparams."""
  return tf.contrib.training.HParams(
      # Data
      src="vi",
      tgt="en",
      train_prefix="/tmp/nmt_data/train",
      dev_prefix="/tmp/nmt_data/tst2012",
      test_prefix="/tmp/nmt_data/tst2013",
      vocab_prefix="/tmp/nmt_data/vocab",
      embed_prefix=None,
      out_dir="/tmp/nmt_3STC_model",

      # Networks
      num_units=32,
      num_layers=2,  # Compatible
      num_encoder_layers=2,
      num_decoder_layers=2,
      dropout=0.2,
      unit_type="lstm",
      encoder_type="uni",
      residual=False,
      time_major=True,
      num_embeddings_partitions=0,

      # Attention mechanisms
      attention="",
      attention_architecture="standard",
      output_attention=True,
      pass_hidden_state=True,

      # Train
      optimizer="sgd",
      num_train_steps=12000,
      batch_size=128,
      init_op="uniform",
      init_weight=0.1,
      max_gradient_norm=5.0,
      learning_rate=1.0,
      warmup_steps=0,
      warmup_scheme="t2t",
      decay_scheme="",
      colocate_gradients_with_ops=True,

      # Data constraints
      num_buckets=5,
      max_train=0,
      src_max_len=50,
      tgt_max_len=50,

      # Inference
      src_max_len_infer=None,
      tgt_max_len_infer=None,
      infer_batch_size=32,
      beam_width=0,
      length_penalty_weight=0.0,
      sampling_temperature=0.0,
      num_translations_per_input=1,

      # Vocab
      sos="<s>",
      eos="</s>",
      subword_option="",
      check_special_token=True,

      # Misc
      forget_bias=1.0,
      num_gpus=1,
      epoch_step=0,  # record where we were within an epoch.
      steps_per_stats=100,
      steps_per_external_eval=None,
      share_vocab=False,
      metrics="bleu".split(","),
      log_device_placement=False,
      random_seed=None,
      override_loaded_hparams=False,
      num_keep_ckpts=5,
      avg_ckpts=False,
      num_intra_threads=0,
      num_inter_threads=0,
  )


def extend_hparams(hparams):
  """Extend training hparams."""

  # Set residual layers
  num_encoder_residual_layers = 0
  num_decoder_residual_layers = 0
  hparams.add_hparam("num_encoder_residual_layers", num_encoder_residual_layers)
  hparams.add_hparam("num_decoder_residual_layers", num_decoder_residual_layers)

  # Vocab
  src_vocab_file = hparams.vocab_prefix + "." + hparams.src
  tgt_vocab_file = hparams.vocab_prefix + "." + hparams.tgt

  # Source vocab
  src_vocab_size, src_vocab_file = vocab_utils.check_vocab(
      src_vocab_file,
      hparams.out_dir,
      check_special_token=hparams.check_special_token,
      sos=hparams.sos,
      eos=hparams.eos,
      unk=vocab_utils.UNK)

  # Target vocab
  tgt_vocab_size, tgt_vocab_file = vocab_utils.check_vocab(
        tgt_vocab_file,
        hparams.out_dir,
        check_special_token=hparams.check_special_token,
        sos=hparams.sos,
        eos=hparams.eos,
        unk=vocab_utils.UNK)
  hparams.add_hparam("src_vocab_size", src_vocab_size)
  hparams.add_hparam("tgt_vocab_size", tgt_vocab_size)
  hparams.add_hparam("src_vocab_file", src_vocab_file)
  hparams.add_hparam("tgt_vocab_file", tgt_vocab_file)

  # Pretrained Embeddings:
  hparams.add_hparam("src_embed_file", "")
  hparams.add_hparam("tgt_embed_file", "")

  # Evaluation
  for metric in hparams.metrics:
    hparams.add_hparam("best_" + metric, 0)  # larger is better
    best_metric_dir = os.path.join(hparams.out_dir, "best_" + metric)
    hparams.add_hparam("best_" + metric + "_dir", best_metric_dir)
    tf.gfile.MakeDirs(best_metric_dir)

  return hparams


def create_or_load_hparams(out_dir, default_hparams):
  """Create hparams."""
  hparams = default_hparams
  hparams = extend_hparams(hparams)

  # Print HParams
  utils.print_hparams(hparams)
  return hparams


def run_main(default_hparams, train_fn, target_session=""):
  """Run main."""

  """Create hparams."""
  hparams = default_hparams
  hparams = extend_hparams(hparams)

  # Print HParams
  utils.print_hparams(hparams)

  train_fn(hparams, target_session=target_session)


def main(unused_argv):
  default_hparams = create_hparams()
  train_fn = train.train
  run_main(default_hparams, train_fn)


if __name__ == "__main__":
  print("Start running!")
  tf.app.run(main=main)
