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
from base import BaseModel

FLAGS = tf.app.flags.FLAGS
# NB: batch_size is not given (None) when deployed as a critic.
HParams = namedtuple("HParams", "mode, min_lr, lr, dropout, batch_size,"
                     "seqmatch_type,max_sent_len,sm_emb_dim,sm_gru_num_units,"
                     "sm_margin, sm_conv1d_filter, sm_conv1d_width,"
                     "sm_conv_filters, sm_conv_heights, sm_conv_widths,"
                     "sm_maxpool_widths, sm_fc_num_units,"
                     "max_grad_norm, decay_step, decay_rate")


def CreateHParams():
  """Create Hyper-parameters from tf.app.flags.FLAGS"""
  hps = HParams(
      mode=FLAGS.mode,  # train, eval, decode
      lr=FLAGS.lr,
      min_lr=FLAGS.min_lr,
      dropout=FLAGS.dropout,
      batch_size=FLAGS.batch_size,
      seqmatch_type=FLAGS.seqmatch_type,
      max_sent_len=FLAGS.max_sent_len,
      sm_emb_dim=FLAGS.sm_emb_dim,
      sm_gru_num_units=FLAGS.sm_gru_num_units,
      sm_margin=FLAGS.sm_margin,
      sm_conv1d_filter=FLAGS.sm_conv1d_filter,
      sm_conv1d_width=FLAGS.sm_conv1d_width,
      sm_conv_filters=lib.parse_list_str(FLAGS.sm_conv_filters),
      sm_conv_heights=lib.parse_list_str(FLAGS.sm_conv_heights),
      sm_conv_widths=lib.parse_list_str(FLAGS.sm_conv_widths),
      sm_maxpool_widths=lib.parse_list_str(FLAGS.sm_maxpool_widths),
      sm_fc_num_units=lib.parse_list_str(FLAGS.sm_fc_num_units),
      max_grad_norm=FLAGS.max_grad_norm,
      decay_step=FLAGS.decay_step,
      decay_rate=FLAGS.decay_rate)
  return hps


class SeqMatchNet(BaseModel):
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
    self._lengths_A = tf.placeholder(tf.int32, [None])

    self._sents_B_pos = tf.placeholder(tf.int32, [None, hps.max_sent_len])
    self._lengths_B_pos = tf.placeholder(tf.int32, [None])

    self._sents_B_neg = tf.placeholder(tf.int32, [None, hps.max_sent_len])
    self._lengths_B_neg = tf.placeholder(tf.int32, [None])

  def _build_model(self):
    hps = self._hps
    vsize = self._vocab.NumIds

    with tf.variable_scope("seq_match"), tf.device(self._device_0):
      with tf.variable_scope('embeddings'):
        embedding = tf.get_variable(
            'embedding', [vsize, hps.sm_emb_dim],
            initializer=tf.truncated_normal_initializer(stddev=1e-4))

        sent_A_embed = tf.nn.embedding_lookup(
            embedding, self._sents_A)  #[?,max_sent_len,sm_emb_dim]
        sent_B_pos_embed = tf.nn.embedding_lookup(
            embedding, self._sents_B_pos)  #[?,max_sent_len,sm_emb_dim]
        sent_B_neg_embed = tf.nn.embedding_lookup(
            embedding, self._sents_B_neg)  #[?,max_sent_len,sm_emb_dim]

      if hps.seqmatch_type == "smn":
        self._output_pos = self._add_smn(sent_A_embed, sent_B_pos_embed,
                                         self._lengths_A, self._lengths_B_pos)
        self._output_neg = self._add_smn(sent_A_embed, sent_B_neg_embed,
                                         self._lengths_A, self._lengths_B_neg,
                                         True)
      elif hps.seqmatch_type == "conv_match":
        self._output_pos = self._add_conv_match(sent_A_embed, sent_B_pos_embed,
                                                self._lengths_A,
                                                self._lengths_B_pos)
        self._output_neg = self._add_conv_match(sent_A_embed, sent_B_neg_embed,
                                                self._lengths_A,
                                                self._lengths_B_neg, True)
      else:
        raise ValueError("Invalid seqmatch_type %s" % hps.seqmatch_type)

  def _add_smn(self,
               sent_A_embed,
               sent_B_embed,
               lengths_A,
               lengths_B,
               reuse=None):
    """ This method implements the following work:

      Wu, Y., Wu, W., Xing, C., Zhou, M., & Li, Z. (2016). Sequential Matching
      Network: A New Architecture for Multi-turn Response Selection in
      Retrieval-based Chatbots. arXiv:1612.01627 [Cs].

    """
    hps = self._hps

    with tf.variable_scope('seq_match_net', reuse=reuse):
      # Part 1: encoder the first sentence with GRU
      with tf.variable_scope('gru_A'):
        gru_A_output, _ = lib.cudnn_rnn_wrapper(
            input_data=sent_A_embed,
            rnn_mode='gru',
            num_layers=1,
            num_units=hps.sm_gru_num_units,
            input_size=hps.sm_emb_dim,
            variable_name="gru_A_variable",
            direction="bidirectional",
            dropout=hps.dropout)  # [max_sent_len, ?, sm_gru_num_units*2]
        gru_A_output = tf.transpose(
            gru_A_output, [1, 0, 2])  # [?, max_sent_len, sm_gru_num_units*2]
        gru_A_mask = tf.expand_dims(
            tf.sequence_mask(lengths_A, hps.max_sent_len, tf.float32),
            2)  # [?, max_sent_len, 1]
        gru_A_output *= gru_A_mask

      # Encoder the second sentence with GRU
      with tf.variable_scope('gru_B'):
        gru_B_output, _ = lib.cudnn_rnn_wrapper(
            input_data=sent_B_embed,
            rnn_mode='gru',
            num_layers=1,
            num_units=hps.sm_gru_num_units,
            input_size=hps.sm_emb_dim,
            variable_name="gru_B_variable",
            direction="bidirectional",
            dropout=hps.dropout)  # [max_sent_len, ?, sm_gru_num_units*2]
        gru_B_output = tf.transpose(
            gru_B_output, [1, 2, 0])  # [?, sm_gru_num_units*2, max_sent_len]
        gru_B_mask = tf.expand_dims(
            tf.sequence_mask(lengths_B, hps.max_sent_len, tf.float32),
            1)  # [?, 1, max_sent_len]
        gru_B_output *= gru_B_mask

      with tf.variable_scope('layer_1'):
        M1 = tf.matmul(sent_A_embed, tf.transpose(
            sent_B_embed, [0, 2, 1]))  # [?, max_sent_len, max_sent_len]
        M2 = tf.matmul(gru_A_output,
                       gru_B_output)  # [?, max_sent_len, max_sent_len]

        # Compute the third feature map M3
        W3 = tf.get_variable(
            'W3', [hps.sm_gru_num_units * 2, hps.sm_gru_num_units * 2],
            tf.float32,
            initializer=tf.random_uniform_initializer(-0.1, 0.1))
        A_feat = tf.matmul(
            tf.reshape(gru_A_output, [-1, hps.sm_gru_num_units * 2]),
            W3)  # [? * max_sent_len, sm_gru_num_units*2]
        A_feat_rsp = tf.reshape(
            A_feat, [-1, hps.max_sent_len, hps.sm_gru_num_units * 2])
        M3 = tf.matmul(A_feat_rsp,
                       gru_B_output)  # [?, max_sent_len, max_sent_len]

        L1 = tf.stack([M1, M2, M3], axis=3)  # [?,max_sent_len,max_sent_len,3]

      # Part 2: conv2d
      conv_feats = L1
      for i, (cf, ch, cw, mw) in enumerate(
          zip(hps.sm_conv_filters, hps.sm_conv_heights, hps.sm_conv_widths,
              hps.sm_maxpool_widths)):
        conv_feats = tf.layers.conv2d(
            conv_feats,
            cf, (ch, cw),
            padding="valid",
            activation=tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            name="conv_layer_%d" % (i + 1))

        if mw:
          conv_feats = tf.layers.max_pooling2d(
              conv_feats, mw, mw, name="maxpool_layer_%d" % (i + 1))

      # Part 3: fully-connected
      mlp_hidden = tf.contrib.layers.flatten(conv_feats)
      for i, n in enumerate(hps.sm_fc_num_units):
        mlp_hidden = tf.contrib.layers.fully_connected(
            mlp_hidden,
            n,
            activation_fn=tf.nn.relu,  # tf.tanh/tf.sigmoid
            weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            scope="fc_layer_%d" % (i + 1))

      prob = tf.squeeze(
          tf.contrib.layers.fully_connected(
              mlp_hidden,
              1,
              activation_fn=tf.sigmoid,
              weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
              scope="fc_layer_output"),
          axis=1)  # [?]

    return prob

  def _add_conv_match(self,
                      sent_A_embed,
                      sent_B_embed,
                      lengths_A,
                      lengths_B,
                      reuse=None):
    """ This method implements the following work:

      Hu, B., Lu, Z., Li, H., & Chen, Q. (2014). Convolutional neural network
      architectures for matching natural language sentences. In Advances in neural
      information processing systems (pp. 2042-2050).

    """
    hps = self._hps

    with tf.variable_scope('conv_match', reuse=reuse):
      # Part 1: conv1d
      with tf.variable_scope('conv1d'):
        # First sentence with conv-1D in layer 1
        sent_A_mask = tf.expand_dims(
            tf.sequence_mask(lengths_A, hps.max_sent_len, tf.float32),
            2)  # [?, max_sent_len, 1]
        sent_A_embed *= sent_A_mask

        sent_A_conv1d_out = tf.layers.conv1d(
            sent_A_embed,
            hps.sm_conv1d_filter,
            hps.sm_conv1d_width,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            name="sent_A_conv1d_layer1")

        # Second sentence with conv-1D in layer 1
        sent_B_mask = tf.expand_dims(
            tf.sequence_mask(lengths_B, hps.max_sent_len, tf.float32),
            2)  # [?, max_sent_len, 1]
        sent_B_embed *= sent_B_mask

        sent_B_conv1d_out = tf.layers.conv1d(
            sent_B_embed,
            hps.sm_conv1d_filter,
            hps.sm_conv1d_width,
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            name="sent_B_conv1d_layer1")

        # Extend and concat the feature maps into 2D
        sent_B_conv1d_out_list = tf.unstack(sent_B_conv1d_out, axis=1)
        sent_AB_concat_feats = []
        for x in sent_B_conv1d_out_list:  #[?, sm_conv1d_filter]
          tiled_x = tf.tile(tf.expand_dims(x, 1), [1, hps.max_sent_len, 1])
          concat_feats = tf.concat(
              [sent_A_conv1d_out, tiled_x],
              axis=2)  #[?, max_sent_len, sm_conv1d_filter*2]
          sent_AB_concat_feats.append(concat_feats)
        sent_AB_2D_feats = tf.stack(
            sent_AB_concat_feats,
            axis=2)  #[?, max_sent_len, max_sent_len, sm_conv1d_filter*2]

      # Part 2: conv2d
      with tf.variable_scope('conv2d'):
        conv2d_feats = sent_AB_2D_feats
        for i, (cf, ch, cw, mw) in enumerate(
            zip(hps.sm_conv_filters, hps.sm_conv_heights, hps.sm_conv_widths,
                hps.sm_maxpool_widths)):
          conv2d_feats = tf.layers.conv2d(
              conv2d_feats,
              cf, (ch, cw),
              padding="valid",
              activation=tf.nn.relu,
              kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1),
              name="conv2d_layer_%d" % (i + 1))

          if mw:
            conv2d_feats = tf.layers.max_pooling2d(
                conv2d_feats, mw, mw, name="maxpool_layer_%d" % (i + 1))

      # Part 3: fully-connected
      with tf.variable_scope('fc'):
        mlp_hidden = tf.contrib.layers.flatten(conv2d_feats)

        for i, n in enumerate(hps.sm_fc_num_units):
          mlp_hidden = tf.contrib.layers.fully_connected(
              mlp_hidden,
              n,
              activation_fn=tf.nn.relu,  # tf.tanh/tf.sigmoid
              weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
              scope="fc_layer_%d" % (i + 1))

        prob = tf.squeeze(
            tf.contrib.layers.fully_connected(
                mlp_hidden,
                1,
                activation_fn=tf.sigmoid,
                weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                scope="fc_layer_output"),
            axis=1)  # [?]

    return prob

  def _add_loss(self):
    hps = self._hps

    with tf.variable_scope("loss"), tf.device(self._device_0):
      # Implements ranking triplet loss
      batch_loss = tf.nn.relu(hps.sm_margin + self._output_neg -
                              self._output_pos)
      loss = tf.reduce_mean(batch_loss)
      # Accuracy: correct if pos > neg
      accuracy = tf.reduce_mean(
          tf.to_float(self._output_pos > self._output_neg))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    self._loss = loss
    self._accuracy = accuracy

  def run_train_step(self, sess, batch):
    (sents_A, sents_B_pos, sents_B_neg, lengths_A, lengths_B_pos,
     lengths_B_neg) = batch

    to_return = [
        self._train_op, self._summaries, self._loss, self._accuracy,
        self.global_step
    ]
    results = sess.run(
        to_return,
        feed_dict={
            self._sents_A: sents_A,
            self._sents_B_pos: sents_B_pos,
            self._sents_B_neg: sents_B_neg,
            self._lengths_A: lengths_A,
            self._lengths_B_pos: lengths_B_pos,
            self._lengths_B_neg: lengths_B_neg
        })

    return results[1:]

  def run_eval_step(self, sess, batch):
    (sents_A, sents_B_pos, sents_B_neg, lengths_A, lengths_B_pos,
     lengths_B_neg) = batch

    to_return = [self._loss, self._accuracy]
    results = sess.run(
        to_return,
        feed_dict={
            self._sents_A: sents_A,
            self._sents_B_pos: sents_B_pos,
            self._sents_B_neg: sents_B_neg,
            self._lengths_A: lengths_A,
            self._lengths_B_pos: lengths_B_pos,
            self._lengths_B_neg: lengths_B_neg
        })

    return results

  def train_loop(self, sess, batcher, valid_batcher, summary_writer):
    """Runs model training."""
    step, losses, accuracies = 0, [], []
    while step < FLAGS.max_run_steps:
      next_batch = batcher.next()
      summaries, loss, accuracy, train_step = self.run_train_step(
          sess, next_batch)

      losses.append(loss)
      accuracies.append(accuracy)
      summary_writer.add_summary(summaries, train_step)
      step += 1

      # Display current training loss
      if step % FLAGS.display_freq == 0:
        avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss",
                                   train_step)
        avg_acc = lib.compute_avg(accuracies, summary_writer, "avg_acc",
                                  train_step)
        tf.logging.info("Train step %d: avg_loss %f avg_acc %f" %
                        (train_step, avg_loss, avg_acc))
        losses, accuracies = [], []
        summary_writer.flush()

      # Run evaluation on validation set
      if step % FLAGS.valid_freq == 0:
        valid_losses, valid_accs = [], []
        for _ in xrange(FLAGS.num_valid_batch):
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
