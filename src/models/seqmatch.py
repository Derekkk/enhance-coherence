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
import re

# NB: batch_size is not given (None) when deployed as a critic.
HParams = namedtuple("HParams", "mode, min_lr, lr, dropout, batch_size,"
                     "max_sent_len, emb_dim, num_hidden, conv_filters,"
                     "conv_width, maxpool_width, max_grad_norm, decay_step,"
                     "decay_rate")


def parse_list_str(list_str):
  l = [int(x) for x in re.split("[\[\]\s,]", list_str) if x]
  if not l:
    raise ValueError("List is empty.")
  return l


def CreateHParams(flags):
  """Create Hyper-parameters from tf.app.flags.FLAGS"""
  hps = HParams(
      mode=flags.mode,  # train, eval, decode
      lr=flags.lr,
      min_lr=flags.min_lr,
      dropout=flags.dropout,
      batch_size=flags.batch_size,
      max_sent_len=flags.max_sent_len,
      emb_dim=flags.emb_dim,
      num_hidden=flags.num_hidden,
      conv_filters=parse_list_str(flags.conv_filters),
      conv_width=parse_list_str(flags.conv_width),
      maxpool_width=parse_list_str(flags.maxpool_width),
      max_grad_norm=flags.max_grad_norm,
      decay_step=flags.decay_step,
      decay_rate=flags.decay_rate)
  return hps


def TrainLoop(model, sess, batcher, valid_batcher, summary_writer, flags):
  """Runs model training."""
  step, losses, accuracies = 0, [], []
  while step < flags.max_run_steps:
    next_batch = batcher.next()
    summary, loss, accuracy, train_step = model.run_train_step(sess, next_batch)
    losses.append(loss)
    accuracies.append(accuracy)
    summary_writer.add_summary(summary, train_step)
    step += 1

    # Display current training loss
    if step % flags.display_freq == 0:
      avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss", train_step)
      avg_acc = lib.compute_avg(accuracies, summary_writer, "avg_acc",
                                train_step)
      tf.logging.info("Train step %d: avg_loss %f avg_acc %f" %
                      (train_step, avg_loss, avg_acc))
      losses, accuracies = [], []
      summary_writer.flush()

    # Run evaluation on validation set
    if step % flags.valid_freq == 0:
      model.run_valid_steps(sess, valid_batcher, flags.num_valid_batch,
                            summary_writer)
      summary_writer.flush()


class SeqMatchNet(object):
  """ Implements the sequential matching network based on the following works:

  [1] Wu, Y., Wu, W., Xing, C., Zhou, M., & Li, Z. (2016). Sequential Matching
      Network: A New Architecture for Multi-turn Response Selection in
      Retrieval-based Chatbots. arXiv:1612.01627 [Cs].

  [2] Hu, B., Lu, Z., Li, H., & Chen, Q. (2014). Convolutional neural network
      architectures for matching natural language sentences. In Advances in neural
      information processing systems (pp. 2042-2050).
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
    self._sents_A = tf.placeholder(tf.int32, [None, hps.max_sent_len])
    self._sents_B = tf.placeholder(tf.int32, [None, hps.max_sent_len])
    self._lengths_A = tf.placeholder(tf.int32, [None])
    self._lengths_B = tf.placeholder(tf.int32, [None])
    self._targets = tf.placeholder(tf.int32, [None])

  def _build_model(self):
    hps = self._hps
    vsize = self._vocab.NumIds

    with tf.variable_scope("seq_match"):
      with tf.variable_scope('embeddings'), tf.device(self._device_0):
        embedding = tf.get_variable(
            'embedding', [vsize, hps.emb_dim],
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

        sent_A_embed = tf.nn.embedding_lookup(
            embedding, self._sents_A)  #[?,max_sent_len,emb_dim]
        sent_B_embed = tf.nn.embedding_lookup(
            embedding, self._sents_B)  #[?,max_sent_len,emb_dim]

      with tf.variable_scope('seq_match_net'), tf.device(self._device_0):
        # Encoder the first sentence with GRU
        with tf.variable_scope('gru_A'):
          gru_A_output, _ = lib.cudnn_rnn_wrapper(
              input_data=sent_A_embed,
              rnn_mode='gru',
              num_layers=1,
              num_units=hps.num_hidden,
              input_size=hps.emb_dim,
              variable_name="gru_A_variable",
              direction="bidirectional",
              dropout=hps.dropout)  # [max_sent_len, ?, num_hidden*2]
          gru_A_output = tf.transpose(
              gru_A_output, [1, 0, 2])  # [?, max_sent_len, num_hidden*2]
          gru_A_mask = tf.expand_dims(
              tf.sequence_mask(self._lengths_A, hps.max_sent_len, tf.float32),
              2)  # [?, max_sent_len, 1]
          gru_A_output *= gru_A_mask

        # Encoder the second sentence with GRU
        with tf.variable_scope('gru_B'):
          gru_B_output, _ = lib.cudnn_rnn_wrapper(
              input_data=sent_B_embed,
              rnn_mode='gru',
              num_layers=1,
              num_units=hps.num_hidden,
              input_size=hps.emb_dim,
              variable_name="gru_B_variable",
              direction="bidirectional",
              dropout=hps.dropout)  # [max_sent_len, ?, num_hidden*2]
          gru_B_output = tf.transpose(
              gru_B_output, [1, 2, 0])  # [?, num_hidden*2, max_sent_len]
          gru_B_mask = tf.expand_dims(
              tf.sequence_mask(self._lengths_B, hps.max_sent_len, tf.float32),
              1)  # [?, 1, max_sent_len]
          gru_B_output *= gru_B_mask

        with tf.variable_scope('layer_1'):
          M1 = tf.matmul(sent_A_embed, tf.transpose(
              sent_B_embed, [0, 2, 1]))  # [?, max_sent_len, max_sent_len]
          M2 = tf.matmul(gru_A_output,
                         gru_B_output)  # [?, max_sent_len, max_sent_len]

          # Compute the third feature map M3
          W3 = tf.get_variable(
              'W3', [hps.num_hidden * 2, hps.num_hidden * 2],
              tf.float32,
              initializer=tf.random_uniform_initializer(-0.1, 0.1))
          A_feat = tf.matmul(
              tf.reshape(gru_A_output, [-1, hps.num_hidden * 2]),
              W3)  # [? * max_sent_len, num_hidden*2]
          A_feat_rsp = tf.reshape(A_feat,
                                  [-1, hps.max_sent_len, hps.num_hidden * 2])
          M3 = tf.matmul(A_feat_rsp,
                         gru_B_output)  # [?, max_sent_len, max_sent_len]

          L1 = tf.stack([M1, M2, M3], axis=3)  # [?,max_sent_len,max_sent_len,3]

        # Layer 2
        conv_input = L1
        for i, (cf, cw, mw) in enumerate(
            zip(hps.conv_filters, hps.conv_width, hps.maxpool_width)):
          conv_output = tf.layers.conv2d(
              conv_input,
              cf,
              cw,
              padding="valid",
              activation=tf.nn.relu,
              kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
              name="conv_layer_%d" % (i + 2))
          maxpool_output = tf.layers.max_pooling2d(
              conv_output, mw, mw, name="maxpool_layer_%d" % (i + 2))
          conv_input = maxpool_output

        # Layer 3
        self._output_logit = tf.squeeze(
            tf.contrib.layers.fully_connected(
                tf.contrib.layers.flatten(maxpool_output),
                1,
                activation_fn=None,
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                scope="fc_layer"))  # [?]
        self._output_prob = tf.sigmoid(self._output_logit)  # [?]

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
    sents_A, sents_B, lengths_A, lengths_B, targets = batch

    to_return = [
        self._train_op, self._summaries, self._loss, self._accuracy,
        self.global_step
    ]
    results = sess.run(
        to_return,
        feed_dict={
            self._sents_A: sents_A,
            self._sents_B: sents_B,
            self._lengths_A: lengths_A,
            self._lengths_B: lengths_B,
            self._targets: targets
        })

    return results[1:]

  def run_eval_step(self, sess, batch):
    sents_A, sents_B, lengths_A, lengths_B, targets = batch

    to_return = [self._loss, self._accuracy]
    return sess.run(
        to_return,
        feed_dict={
            self._sents_A: sents_A,
            self._sents_B: sents_B,
            self._lengths_A: lengths_A,
            self._lengths_B: lengths_B,
            self._targets: targets
        })

  def run_valid_steps(self, sess, data_batcher, num_valid_batch,
                      summary_writer):
    losses, accuracies = [], []
    for _ in xrange(num_valid_batch):
      next_batch = data_batcher.next()
      loss, accuracy = self.run_eval_step(sess, next_batch)
      losses.append(loss)
      accuracies.append(accuracy)

    step = self.get_global_step(sess)
    valid_loss = lib.compute_avg(losses, summary_writer, "valid_loss", step)
    valid_acc = lib.compute_avg(accuracies, summary_writer, "valid_accuracy",
                                step)
    tf.logging.info("\tValid step %d: avg_loss %f avg_acc %f" %
                    (step, valid_loss, valid_acc))

  def get_global_step(self, sess):
    """Get the current number of training steps."""
    return sess.run(self.global_step)
