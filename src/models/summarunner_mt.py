# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.contrib.rnn import GRUCell
from random import random

from summarunner_abs import SummaRuNNerAbs
import lib
import memory

# NB: batch_size could be unspecified (None) in decode mode
HParams = namedtuple("HParams", "mode, min_lr, lr, batch_size,"
                     "num_sentences, num_words_sent, rel_pos_max_idx,"
                     "dec_timesteps, enc_num_hidden, dec_num_hidden,"
                     "emb_dim, pos_emb_dim, doc_repr_dim,"
                     "word_conv_k_sizes, word_conv_filter,"
                     "min_num_input_sents, min_num_words_sent, min_output_len,"
                     "max_grad_norm, num_softmax_samples, extract_topk,"
                     "trg_weight_norm, ext_loss_weight, abs_loss_weight")


def CreateHParams(flags):
  """Create Hyper-parameters from tf.app.flags.FLAGS"""
  word_conv_k_sizes = tuple(
      np.fromstring(flags.word_conv_k_sizes, dtype=np.int32, sep=","))
  hps = HParams(
      mode=flags.mode,  # train, eval, decode
      lr=flags.lr,
      min_lr=flags.min_lr,
      batch_size=flags.batch_size,
      num_sentences=flags.num_sentences,  # number of sentences in a document
      num_words_sent=flags.num_words_sent,  # number of words in a sentence
      rel_pos_max_idx=flags.rel_pos_max_idx,
      dec_timesteps=flags.dec_timesteps,
      enc_num_hidden=flags.enc_num_hidden,  # for rnn cell
      dec_num_hidden=flags.dec_num_hidden,
      emb_dim=flags.emb_dim,
      pos_emb_dim=flags.pos_emb_dim,
      doc_repr_dim=flags.doc_repr_dim,
      word_conv_k_sizes=word_conv_k_sizes,
      word_conv_filter=flags.word_conv_filter,
      min_num_input_sents=flags.min_num_input_sents,
      min_num_words_sent=flags.min_num_words_sent,
      min_output_len=flags.min_output_len,
      max_grad_norm=4.0,
      extract_topk=flags.extract_topk,
      num_softmax_samples=
      flags.num_softmax_samples,  # If 0, no sampled softmax.
      trg_weight_norm=flags.trg_weight_norm,
      ext_loss_weight=flags.ext_loss_weight,
      abs_loss_weight=flags.abs_loss_weight)
  return hps


def TrainLoop(model, sess, batchers, valid_batchers, summary_writer, flags):
  """Runs model training."""
  step = 0

  assert len(batchers) == 2
  assert len(valid_batchers) == 2

  ext_batcher, abs_batcher = batchers
  ext_val_batcher, abs_val_batcher = valid_batchers

  ext_prob = flags.ext_data_prop
  assert ext_prob < 1.0
  ext_losses, abs_losses = [], []

  while step < flags.max_run_steps:
    if random() < ext_prob:  # Train on one extractive batch
      next_batch = ext_batcher.next()
      ext_loss, train_step = model.run_ext_train_step(sess, next_batch)
      ext_losses.append(ext_loss)
    else:  # Train on one abstractive batch
      next_batch = abs_batcher.next()
      abs_loss, train_step = model.run_abs_train_step(sess, next_batch)
      abs_losses.append(abs_loss)

    # summary_writer.add_summary(summaries, train_step)
    step += 1

    # Display current training loss
    if step % flags.display_freq == 0:
      avg_ext_loss = lib.compute_avg(ext_losses, summary_writer, "avg_ext_loss",
                                     train_step)
      avg_abs_loss = lib.compute_avg(abs_losses, summary_writer, "avg_abs_loss",
                                     train_step)

      tf.logging.info("Train step %d: ext_loss %f abs_loss %f" %
                      (train_step, avg_ext_loss, avg_abs_loss))
      ext_losses, abs_losses = [], []
      summary_writer.flush()

    # Run evaluation on validation set
    if step % flags.valid_freq == 0:
      ext_val_losses, abs_val_losses = [], []
      num_ext_val_batch = int(flags.num_valid_batch * ext_prob)

      for _ in xrange(num_ext_val_batch):
        next_batch = ext_val_batcher.next()
        ext_val_loss, step = model.run_ext_eval_step(sess, next_batch)
        ext_val_losses.append(ext_val_loss)

      for _ in xrange(flags.num_valid_batch - num_ext_val_batch):
        next_batch = abs_val_batcher.next()
        abs_val_loss, step = model.run_abs_eval_step(sess, next_batch)
        abs_val_losses.append(abs_val_loss)

      avg_ext_val_loss = lib.compute_avg(ext_val_losses, summary_writer,
                                         "avg_ext_val_loss", step)
      avg_abs_val_loss = lib.compute_avg(abs_val_losses, summary_writer,
                                         "avg_abs_val_loss", step)

      tf.logging.info("\tValid step %d: ext_loss %f abs_loss %f" %
                      (step, avg_ext_val_loss, avg_abs_val_loss))
      summary_writer.flush()


class SummaRuNNerMT(SummaRuNNerAbs):
  """ Implements extractive + abstractive summarization model with
  Multi-task Learning. The following works are referenced:

  [1] Cheng, J., & Lapata, M. (2016). Neural Summarization by Extracting Sentences
  and Words. arXiv:1603.07252 [Cs]. Retrieved from http://arxiv.org/abs/1603.07252
  [2] Nallapati, R., Zhai, F., & Zhou, B. (2016). SummaRuNNer: A Recurrent Neural
  Network based Sequence Model for Extractive Summarization of Documents.
  arXiv:1611.04230 [Cs]. Retrieved from http://arxiv.org/abs/1611.04230

  This is the multi-task version of SummaRuNNer.
  """

  def _add_loss(self):
    hps = self._hps
    self._extract_loss = self._add_extract_loss() * hps.ext_loss_weight
    self._abstract_loss = self._add_abstract_loss() * hps.abs_loss_weight
    # No overall loss is defined

  def _add_grad_op(self, loss, tvars, name=""):
    hps = self._hps
    grads, norm = tf.clip_by_global_norm(
        tf.gradients(loss, tvars), hps.max_grad_norm)

    tf.summary.scalar(name + "_norm", norm)
    return grads

  def _add_train_op(self):
    """Sets self._train_op, op to run for training."""
    hps = self._hps
    self._lr_rate = tf.maximum(
        hps.min_lr,  # minimum learning rate.
        tf.train.exponential_decay(hps.lr, self.global_step, 30000, 0.98))
    tf.summary.scalar("learning_rate", self._lr_rate)

    # Compute gradients
    tvars = tf.trainable_variables()
    with tf.device(self._device_2):
      ext_grads = self._add_grad_op(self._extract_loss, tvars, name="ext_grads")
      abs_grads = self._add_grad_op(
          self._abstract_loss, tvars, name="abs_grads")

      # Create optimizer and train ops
      optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)

      self._ext_train_op = optimizer.apply_gradients(
          zip(ext_grads, tvars),
          global_step=self.global_step,
          name="ext_train_step")
      self._abs_train_op = optimizer.apply_gradients(
          zip(abs_grads, tvars),
          global_step=self.global_step,
          name="abs_train_step")

  def run_ext_train_step(self, sess, batch):
    (enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, extract_targets,
     target_weights, _) = batch

    to_return = [self._ext_train_op, self._extract_loss, self.global_step]
    results = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos,
            self._extract_targets: extract_targets,
            self._target_weights: target_weights
        })
    return results[1:]

  def run_abs_train_step(self, sess, batch):
    (enc_batch, dec_batch, target_batch, enc_doc_lens, enc_sent_lens,
     sent_rel_pos, dec_lens, _, _) = batch

    to_return = [self._abs_train_op, self._abstract_loss, self.global_step]
    results = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos,
            self._decoder_inputs: dec_batch,
            self._decoder_targets: target_batch,
            self._decoder_lens: dec_lens
        })
    return results[1:]

  def run_ext_eval_step(self, sess, batch):
    (enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, extract_targets,
     target_weights, _) = batch

    to_return = [self._extract_loss, self.global_step]
    results = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos,
            self._extract_targets: extract_targets,
            self._target_weights: target_weights
        })
    return results

  def run_abs_eval_step(self, sess, batch):
    (enc_batch, dec_batch, target_batch, enc_doc_lens, enc_sent_lens,
     sent_rel_pos, dec_lens, _, _) = batch

    to_return = [self._abstract_loss, self.global_step]
    results = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos,
            self._decoder_inputs: dec_batch,
            self._decoder_targets: target_batch,
            self._decoder_lens: dec_lens
        })
    return results

  def run_valid_steps(self, sess, data_batcher, num_valid_batch,
                      summary_writer):
    raise NotImplementedError()

  def get_extract_probs(self, sess, batch):
    (enc_batch, _, _, enc_doc_lens, enc_sent_lens, sent_rel_pos, _, _,
     _) = batch

    to_return = self._extract_probs
    results = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos
        })
    return results
