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
import lib

# NB: batch_size is not given (None) when deployed as a critic.
HParams = namedtuple("HParams", "mode, min_lr, lr, dropout, batch_size,"
                     "max_num_sents, max_sent_len, emb_dim, gru_num_hidden,"
                     "conv_filters, conv_heights, conv_widths, maxpool_widths,"
                     "fc_num_hiddens, max_grad_norm, decay_step, decay_rate")


def CreateHParams(flags):
  """Create Hyper-parameters from tf.app.flags.FLAGS"""
  hps = HParams(
      mode=flags.mode,  # train, eval, decode
      lr=flags.lr,
      min_lr=flags.min_lr,
      dropout=flags.dropout,
      batch_size=flags.batch_size,
      max_num_sents=flags.max_num_sents,
      max_sent_len=flags.max_sent_len,
      emb_dim=flags.emb_dim,
      gru_num_hidden=flags.gru_num_hidden,
      conv_filters=lib.parse_list_str(flags.conv_filters),
      conv_heights=lib.parse_list_str(flags.conv_heights),
      conv_widths=lib.parse_list_str(flags.conv_widths),
      maxpool_widths=lib.parse_list_str(flags.maxpool_widths),
      fc_num_hiddens=lib.parse_list_str(flags.fc_num_hiddens),
      max_grad_norm=flags.max_grad_norm,
      decay_step=flags.decay_step,
      decay_rate=flags.decay_rate)
  return hps


class CoherenceModel(object):
  """ Implements the a local coherence model that is based on:

  [1] Li, J., & Hovy, E. H. (2014). A Model of Coherence Based on
       Distributed Sentence Representation. In EMNLP (pp. 2039-2048).

  [2] Nguyen, D. T., & Joty, S. (2017). A Neural Local Coherence Model.
  """

  def __init__(self, hps, vocab, num_gpus=1):
    self._hps = hps
    self._vocab = vocab

    if num_gpus < 1:
      raise ValueError("Current implementation requires at least one GPU.")
    elif num_gpus > 1:
      tf.logging.warn("Current implementation uses at only one GPU.")
    self._device_0 = "/gpu:0"

  def build_graph(self):
    self._add_placeholders()
    self._build_model()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self._hps.mode == 'train':
      self._add_loss()
      self._add_train_op()

    self._summaries = tf.summary.merge_all()

  def _add_placeholders(self):
    hps = self._hps
    self._sent_inputs = tf.placeholder(
        tf.int32, [hps.batch_size, hps.max_num_sents, hps.max_sent_len])
    self._sent_lengths = tf.placeholder(tf.int32,
                                        [hps.batch_size, hps.max_num_sents])
    # self._doc_length = tf.placeholder(tf.int32, [hps.batch_size])
    self._targets = tf.placeholder(tf.int32, [hps.batch_size])

  def _build_model(self):
    hps = self._hps
    vsize = self._vocab.NumIds

    with tf.variable_scope("coherence"):
      with tf.variable_scope('embeddings'), tf.device(self._device_0):
        embedding = tf.get_variable(
            'embedding', [vsize, hps.emb_dim],
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

        sent_inputs_embed = tf.nn.embedding_lookup(
            embedding,
            self._sent_inputs)  #[bs, max_num_sents, max_sent_len, emb_dim]

      with tf.variable_scope('coherence_net'), tf.device(self._device_0):
        # Encoder all the sentences with GRU
        with tf.variable_scope('sentence_gru'):
          gru_input = tf.reshape(sent_inputs_embed, [
              -1, hps.max_num_sents * hps.max_sent_len, hps.emb_dim
          ])

          gru_output, _ = lib.cudnn_rnn_wrapper(
              input_data=gru_input,
              rnn_mode='gru',
              num_layers=1,
              num_units=hps.gru_num_hidden,
              input_size=hps.emb_dim,
              variable_name="sent_gru_variable",
              direction="bidirectional",
              dropout=
              hps.dropout)  # [max_num_sents * max_sent_len, bs, num_hidden*2]
          gru_output = tf.transpose(
              gru_output,
              [1, 0, 2])  # [bs, max_num_sents * max_sent_len, num_hidden*2]
          gru_output = tf.reshape(gru_output, [
              -1, hps.max_num_sents, hps.max_sent_len, hps.gru_num_hidden * 2
          ])  # [bs, max_num_sents, max_sent_len, num_hidden*2]

          gru_mask = tf.sequence_mask(
              tf.reshape(self._sent_lengths, [-1]), hps.max_sent_len,
              tf.float32)  # [bs * max_num_sents, max_sent_len]
          gru_mask = tf.reshape(gru_mask,
                                [-1, hps.max_num_sents, hps.max_sent_len,
                                 1])  # [bs, max_num_sents, max_sent_len, 1]
          gru_output *= gru_mask

        with tf.variable_scope('doc_classifier'):
          conv_input = gru_output  # [bs, max_num_sents, max_sent_len, num_hidden*2]

          # Convolutional layers
          for i, (cf, ch, cw, mw) in enumerate(
              zip(hps.conv_filters, hps.conv_heights, hps.conv_widths,
                  hps.maxpool_widths)):
            conv_output = tf.layers.conv2d(
                conv_input,
                cf, (ch, cw),
                padding="same",
                activation=tf.nn.relu,
                kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                name="conv_layer_%d" % (i + 1))

            if mw > 0:
              maxpool_output = tf.layers.max_pooling2d(
                  conv_output, (1, mw), (1, mw),
                  name="maxpool_layer_%d" % (i + 1))
              conv_input = maxpool_output
            else:
              conv_input = conv_output

          # Fully-connected layers
          fc_input = tf.contrib.layers.flatten(conv_input)
          for i, num_hidden in enumerate(hps.fc_num_hiddens):
            fc_input = tf.contrib.layers.fully_connected(
                fc_input,
                num_hidden,
                activation_fn=tf.nn.relu,  # tf.tanh/tf.sigmoid
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                scope="fc_layer_%d" % (i + 1))

          self._output_logit = tf.squeeze(
              tf.contrib.layers.fully_connected(
                  fc_input,
                  1,
                  activation_fn=None,
                  weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                  scope="fc_output"))  # [batch_size]
          self._output_prob = tf.sigmoid(self._output_logit)  # [batch_size]

  def _add_loss(self):
    with tf.variable_scope("loss"), tf.device(self._device_0):
      batch_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.to_float(self._targets), logits=self._output_logit)
      loss = tf.reduce_mean(batch_loss)

      # Accuracy: threshold at 0.5
      accuracy = tf.reduce_mean(
          tf.to_float(
              tf.equal(tf.to_int32(self._output_prob > 0.5), self._targets)))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    self._loss = loss
    self._accuracy = accuracy

  def _add_train_op(self):
    hps = self._hps

    self._lr_rate = tf.maximum(
        hps.min_lr,  # minimum learning rate.
        tf.train.exponential_decay(hps.lr, self.global_step, hps.decay_step,
                                   hps.decay_rate))
    tf.summary.scalar("learning_rate", self._lr_rate)

    tvars = tf.trainable_variables()
    with tf.device(self._device_0):
      # Compute gradients
      grads, global_norm = tf.clip_by_global_norm(
          tf.gradients(self._loss, tvars), hps.max_grad_norm)
      tf.summary.scalar("global_norm", global_norm)

      # Create optimizer and train ops
      optimizer = tf.train.GradientDescentOptimizer(self._lr_rate)
      self._train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=self.global_step, name="train_step")

  def run_train_step(self, sess, batch):
    sent_inputs, sent_lens, targets = batch

    to_return = [
        self._train_op, self._summaries, self._loss, self._accuracy,
        self.global_step
    ]
    results = sess.run(
        to_return,
        feed_dict={
            self._sent_inputs: sent_inputs,
            self._sent_lengths: sent_lens,
            self._targets: targets
        })

    return results[1:]

  def run_eval_step(self, sess, batch):
    sent_inputs, sent_lens, targets = batch

    to_return = [self._loss, self._accuracy]
    return sess.run(
        to_return,
        feed_dict={
            self._sent_inputs: sent_inputs,
            self._sent_lengths: sent_lens,
            self._targets: targets
        })

  def train_loop(self, sess, batcher, valid_batcher, summary_writer, flags):
    """Runs model training."""
    step, losses, accuracies = 0, [], []
    while step < flags.max_run_steps:
      next_batch = batcher.next()
      summaries, loss, accuracy, train_step = self.run_train_step(
          sess, next_batch)

      losses.append(loss)
      accuracies.append(accuracy)
      summary_writer.add_summary(summaries, train_step)
      step += 1

      # Display current training loss
      if step % flags.display_freq == 0:
        avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss",
                                   train_step)
        avg_acc = lib.compute_avg(accuracies, summary_writer, "avg_acc",
                                  train_step)
        tf.logging.info("Train step %d: avg_loss %f avg_acc %f" %
                        (train_step, avg_loss, avg_acc))
        losses, accuracies = [], []
        summary_writer.flush()

      # Run evaluation on validation set
      if step % flags.valid_freq == 0:
        valid_losses, valid_accs = [], []
        for _ in xrange(flags.num_valid_batch):
          next_batch = valid_batcher.next()
          valid_loss, valid_acc = self.run_eval_step(sess, next_batch)
          valid_losses.append(valid_loss)
          valid_accs.append(valid_acc)

        gstep = self.get_global_step(sess)
        avg_valid_loss = lib.compute_avg(valid_losses, summary_writer,
                                         "valid_loss", gstep)
        avg_valid_acc = lib.compute_avg(valid_accs, summary_writer,
                                        "valid_accuracy", gstep)
        tf.logging.info("\tValid step %d: avg_loss %f avg_acc %f" %
                        (step, avg_valid_loss, avg_valid_acc))
        summary_writer.flush()

  def get_global_step(self, sess):
    """Get the current number of training steps."""
    return sess.run(self.global_step)
