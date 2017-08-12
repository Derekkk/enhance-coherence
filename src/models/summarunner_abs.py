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
from summarunner import SummaRuNNer

import lib
import memory

# NB: batch_size could be unspecified (None) in decode mode
HParams = namedtuple("HParams", "mode, min_lr, lr, batch_size,"
                     "num_sentences, num_words_sent, rel_pos_max_idx,"
                     "dec_timesteps, enc_num_hidden, dec_num_hidden,"
                     "emb_dim, pos_emb_dim, doc_repr_dim,"
                     "word_conv_k_sizes, word_conv_filter,"
                     "min_num_input_sents, min_num_words_sent, min_output_len,"
                     "max_grad_norm, num_softmax_samples, extract_topk")


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
      flags.num_softmax_samples)  # If 0, no sampled softmax.
  return hps


def TrainLoop(model, sess, batcher, valid_batcher, summary_writer, flags):
  """Runs model training."""
  step, losses = 0, []
  while step < flags.max_run_steps:
    next_batch = batcher.next()
    summaries, loss, train_step = model.run_train_step(sess, next_batch)
    losses.append(loss)
    summary_writer.add_summary(summaries, train_step)
    step += 1

    # Display current training loss
    if step % flags.display_freq == 0:
      avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss", train_step)
      tf.logging.info("Train step %d: avg_loss %f" % (train_step, avg_loss))
      losses = []
      summary_writer.flush()

    # Run evaluation on validation set
    if step % flags.valid_freq == 0:
      model.run_valid_steps(sess, valid_batcher, flags.num_valid_batch,
                            summary_writer)
      summary_writer.flush()


class SummaRuNNerAbs(SummaRuNNer):
  """ Implements extractive summarization model based on the following works:

  [1] Cheng, J., & Lapata, M. (2016). Neural Summarization by Extracting Sentences
  and Words. arXiv:1603.07252 [Cs]. Retrieved from http://arxiv.org/abs/1603.07252
  [2] Nallapati, R., Zhai, F., & Zhou, B. (2016). SummaRuNNer: A Recurrent Neural
  Network based Sequence Model for Extractive Summarization of Documents.
  arXiv:1611.04230 [Cs]. Retrieved from http://arxiv.org/abs/1611.04230

  This is the abstractive version of SummaRuNNer.
  """

  def __init__(self, hps, input_vocab, output_vocab, num_gpus=0):
    super(SummaRuNNerAbs, self).__init__(hps, input_vocab, None, num_gpus)
    self._output_vocab = output_vocab

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    super(SummaRuNNerAbs, self)._add_placeholders()

    hps = self._hps
    # Output sequence
    self._decoder_inputs = tf.placeholder(
        tf.int32, [hps.batch_size, hps.dec_timesteps], name="decoder_inputs")
    self._decoder_targets = tf.placeholder(
        tf.int32, [hps.batch_size, hps.dec_timesteps], name="decoder_targets")
    self._decoder_lens = tf.placeholder(
        tf.int32, [hps.batch_size], name="decoder_lens")

  def _add_seq2seq(self):
    super(SummaRuNNerAbs, self)._add_seq2seq()

    hps = self._hps
    batch_size_ts = hps.batch_size  # batch size Tensor
    if hps.batch_size is None:
      batch_size_ts = tf.shape(inputs)[0]

    with tf.variable_scope("seq2seq", reuse=None):
      # Decoder
      with tf.variable_scope("decoder"), tf.device(self._device_1):
        # Extract the weighted summaries for decoder to generate summary
        hist_summary_indices = tf.stack(
            [tf.range(0, batch_size_ts), self._input_doc_lens - 1],
            axis=1)  # [batch_size, 2]
        extracted_summary = tf.gather_nd(
            self._hist_summaries,
            hist_summary_indices)  # [batch_size, enc_num_hidden*2]

        # Create decoder
        self._dec_outputs = self._add_decoder(extracted_summary)
        # _dec_outputs are decoder outputs before projection.
        # _dec_outputs: [dec_timesteps, batch_size, dec_num_hidden]

        # Output projection
        dec_output_rsp = tf.reshape(self._dec_outputs, [-1, hps.dec_num_hidden])
        # dec_output_rsp: [dec_timesteps * batch_size, dec_num_hidden]
        self._proj_dec_output = lib.linear(
            dec_output_rsp, hps.emb_dim, True, scope="lstm2word")
        # _proj_dec_output: [dec_timesteps * batch_size, emb_dim]

  def _add_embeddings(self):
    super(SummaRuNNerAbs, self)._add_embeddings()

    hps = self._hps
    output_vsize = self._output_vocab.NumIds

    with tf.device(self._device_1):
      # Output embedding and bias
      self._output_embed = tf.get_variable(
          "output_embed", [output_vsize, hps.emb_dim],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))
      self._out_vocab_bias = tf.get_variable(
          "out_vocab_bias", [output_vsize],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))

  def _add_decoder(self, extracted_summary, transpose_output=False):
    """
    extracted_summary: a [batch_size, enc_num_hidden * 2] Tensor
    """
    hps = self._hps
    decoder_inputs_t = tf.transpose(self._decoder_inputs)
    emb_dec_inputs_t = tf.nn.embedding_lookup(
        self._output_embed,
        decoder_inputs_t)  # [dec_timesteps, batch_size, emb_dim]
    ext_sum_expanded = tf.tile(
        tf.expand_dims(extracted_summary, 0), [hps.dec_timesteps, 1, 1])
    dec_inputs = tf.concat([emb_dec_inputs_t, ext_sum_expanded], 2)
    # dec_inputs: [dec_timesteps, batch_size, emb_dim + enc_num_hidden*2]

    dec_model = cudnn_rnn_ops.CudnnLSTM(1, hps.dec_num_hidden,
                                        hps.emb_dim + hps.enc_num_hidden * 2)
    # Compute the total size of RNN params (Tensor)
    params_size_ts = dec_model.params_size()
    params = tf.Variable(
        tf.random_uniform([params_size_ts], minval=-0.1, maxval=0.1),
        validate_shape=False,
        name="decoder_cudnn_lstm_var")

    batch_size_ts = tf.shape(self._decoder_inputs)[0]  # batch size Tensor
    init_state = tf.zeros(tf.stack([1, batch_size_ts, hps.dec_num_hidden]))
    init_c = tf.zeros(tf.stack([1, batch_size_ts, hps.dec_num_hidden]))

    # Call the CudnnLSTM
    dec_output, _, _ = dec_model(
        input_data=dec_inputs,
        input_h=init_state,
        input_c=init_c,
        params=params)  # [dec_timesteps, batch_size, dec_num_hidden]

    if transpose_output:
      dec_output = tf.transpose(
          dec_output, [1, 0, 2])  # [batch_size, dec_timesteps, dec_num_hidden]

    return dec_output

  def _add_loss(self):
    self._loss = self._add_abstract_loss()
    tf.summary.scalar("loss", self._loss)

  def _add_abstract_loss(self):
    hps = self._hps
    output_vsize = self._output_vocab.NumIds

    with tf.variable_scope("loss"), tf.device(self._device_2):
      targets_t_rsp = tf.expand_dims(
          tf.reshape(tf.transpose(self._decoder_targets), [-1]),
          1)  # [dec_timesteps * batch_size, 1]

      all_losses = tf.nn.sampled_softmax_loss(
          self._output_embed,
          self._out_vocab_bias,
          labels=targets_t_rsp,
          inputs=self._proj_dec_output,
          num_sampled=hps.num_softmax_samples,
          num_classes=output_vsize)  # [dec_timesteps * batch_size]

      # Masking the loss
      loss_weights_t = tf.transpose(
          tf.sequence_mask(
              self._decoder_lens, maxlen=hps.dec_timesteps,
              dtype=tf.float32))  # [dec_timesteps, batch_size]
      loss_weights_t_rsp = tf.reshape(loss_weights_t,
                                      [-1])  # [dec_timesteps * batch_size]
      abstract_loss = tf.reduce_mean(all_losses * loss_weights_t_rsp)
      # tf.summary.scalar("abstract_loss", abstract_loss)

    return abstract_loss

  def run_train_step(self, sess, batch):
    (enc_batch, dec_batch, target_batch, enc_doc_lens, enc_sent_lens,
     sent_rel_pos, dec_lens, _, _) = batch

    to_return = [self._train_op, self._summaries, self._loss, self.global_step]
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

  def run_eval_step(self, sess, batch):
    (enc_batch, dec_batch, target_batch, enc_doc_lens, enc_sent_lens,
     sent_rel_pos, dec_lens, _, _) = batch

    to_return = [self._loss, self.global_step]
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

