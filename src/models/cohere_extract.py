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
import lib

FLAGS = tf.app.flags.FLAGS

# NB: batch_size could be unspecified (None) in decode mode
HParams = namedtuple("HParams", "mode, min_lr, lr, dropout, batch_size,"
                     "num_sents_doc, num_words_sent, rel_pos_max_idx,"
                     "enc_layers, enc_num_hidden, emb_dim, pos_emb_dim,"
                     "doc_repr_dim, word_conv_widths, word_conv_filters,"
                     "mlp_num_hiddens, train_mode, coherence_coef, coh_samples,"
                     "min_num_input_sents, trg_weight_norm,"
                     "max_grad_norm, decay_step, decay_rate")


def CreateHParams():
  """Create Hyper-parameters from tf.app.flags.FLAGS"""

  assert FLAGS.mode in ["train", "decode"], "Invalid mode."
  assert FLAGS.train_mode in ["sl", "coherence",
                              "sl+coherence"], "Invalid train mode."

  hps = HParams(
      mode=FLAGS.mode,
      train_mode=FLAGS.train_mode,
      lr=FLAGS.lr,
      min_lr=FLAGS.min_lr,
      dropout=FLAGS.dropout,
      batch_size=FLAGS.batch_size,
      num_sents_doc=FLAGS.num_sents_doc,  # number of sentences in a document
      num_words_sent=FLAGS.num_words_sent,  # number of words in a sentence
      rel_pos_max_idx=FLAGS.rel_pos_max_idx,  # number of relative positions
      enc_layers=FLAGS.enc_layers,  # number of layers for sentence-level rnn
      enc_num_hidden=FLAGS.enc_num_hidden,  # for sentence-level rnn
      emb_dim=FLAGS.emb_dim,
      pos_emb_dim=FLAGS.pos_emb_dim,
      doc_repr_dim=FLAGS.doc_repr_dim,
      word_conv_widths=lib.parse_list_str(FLAGS.word_conv_widths),
      word_conv_filters=lib.parse_list_str(FLAGS.word_conv_filters),
      mlp_num_hiddens=lib.parse_list_str(FLAGS.mlp_num_hiddens),
      min_num_input_sents=FLAGS.min_num_input_sents,  # for batch reader
      trg_weight_norm=FLAGS.trg_weight_norm,  # for batch reader
      max_grad_norm=FLAGS.max_grad_norm,
      decay_step=FLAGS.decay_step,
      decay_rate=FLAGS.decay_rate,
      coherence_coef=FLAGS.coherence_coef,  # coefficient of coherence loss
      coh_samples=FLAGS.coh_samples)  #samples/instance for coherence model
  return hps


class CoherentExtract(object):
  """ An extractive summarization model based on SummaRuNNer that is enhanced
      with local coherence model. Related works are listed:

  [1] Nallapati, R., Zhai, F., & Zhou, B. (2016). SummaRuNNer: A Recurrent
      Neural Network based Sequence Model for Extractive Summarization of
      Documents. arXiv:1611.04230 [Cs].

  [2] Li, J., & Hovy, E. H. (2014). A Model of Coherence Based on
      Distributed Sentence Representation. In EMNLP (pp. 2039-2048).
  """

  def __init__(self, hps, input_vocab, num_gpus=0):
    if hps.mode not in ["train", "decode"]:
      raise ValueError("Only train and decode mode are supported.")

    self._hps = hps
    self._input_vocab = input_vocab
    self._num_gpus = num_gpus

  def build_graph(self):
    self._allocate_devices()
    self._add_placeholders()
    self._build_model()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self._hps.mode == "train":
      self._add_loss()
      self._add_train_op()

    self._summaries = tf.summary.merge_all()

  def _allocate_devices(self):
    num_gpus = self._num_gpus

    if num_gpus == 0:
      raise ValueError("Current implementation requires at least one GPU.")
    elif num_gpus == 1:
      self._device_0 = "/gpu:0"
      self._device_1 = "/gpu:0"
    elif num_gpus > 1:
      self._device_0 = "/gpu:0"
      self._device_1 = "/gpu:1"

  def _add_placeholders(self):
    """Inputs to be fed to the graph."""
    hps = self._hps
    # Input sequence
    self._inputs = tf.placeholder(
        tf.int32, [hps.batch_size, hps.num_sents_doc, hps.num_words_sent],
        name="inputs")
    self._input_sent_lens = tf.placeholder(
        tf.int32, [hps.batch_size, hps.num_sents_doc], name="input_sent_lens")
    self._input_doc_lens = tf.placeholder(
        tf.int32, [hps.batch_size], name="input_doc_lens")
    self._input_rel_pos = tf.placeholder(
        tf.int32, [hps.batch_size, hps.num_sents_doc], name="input_rel_pos")

    # Output extraction decisions
    self._extract_targets = tf.placeholder(
        tf.int32, [hps.batch_size, hps.num_sents_doc], name="extract_targets")
    # The extraction decisions may be weighted differently
    self._target_weights = tf.placeholder(
        tf.float32, [hps.batch_size, hps.num_sents_doc], name="target_weights")

  def _build_model(self):
    """Construct the deep neural network of SummaRuNNer."""
    hps = self._hps

    with tf.variable_scope("coherent_extract") as self._vs:
      with tf.variable_scope("embeddings"):
        self._add_embeddings()

      # Encoder
      with tf.variable_scope("encoder",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):
        # Hierarchical encoding of input document
        self._sentence_vecs = self._add_encoder(
            self._inputs, self._input_sent_lens, self._input_doc_lens
        )  # [num_sents_doc, batch_size, enc_num_hidden*2]
        # Note: output size of Bi-RNN is double of the enc_num_hidden

        # Document representation
        doc_mean_vec = tf.div(
            tf.reduce_sum(self._sentence_vecs, 0),
            tf.expand_dims(tf.to_float(self._input_doc_lens),
                           1))  # [batch_size, enc_num_hidden*2]
        self._doc_repr = tf.tanh(
            lib.linear(
                doc_mean_vec, hps.doc_repr_dim, True, scope="doc_repr_linear"))

        # Absolute position embedding
        abs_pos_idx = tf.range(0, hps.num_sents_doc)  # [num_sents_doc]
        abs_pos_emb = tf.expand_dims(
            tf.nn.embedding_lookup(self._abs_pos_embed, abs_pos_idx),
            1)  # [num_sents_doc, 1, pos_emb_dim]
        batch_size_ts = hps.batch_size if hps.batch_size else tf.shape(
            self._inputs)[0]  # batch size Tensor
        self._sent_abs_pos_emb = tf.tile(
            abs_pos_emb, tf.stack(
                [1, batch_size_ts,
                 1]))  # [num_sents_doc, batch_size, pos_emb_dim]

        # Relative position embedding
        self._sent_rel_pos_emb = tf.nn.embedding_lookup(
            self._rel_pos_embed,
            self._input_rel_pos)  # [batch_size, num_sents_doc, pos_emb_dim]

        # Unstack the features into list: num_sents_doc * [batch_size, ?]
        sent_vecs_list = tf.unstack(self._sentence_vecs, axis=0)
        abs_pos_emb_list = tf.unstack(self._sent_abs_pos_emb, axis=0)
        rel_pos_emb_list = tf.unstack(self._sent_rel_pos_emb, axis=1)

      # Compute the extraction probability of each sentence
      with tf.variable_scope("extract_sent",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):

        # Initialize the representation of all historical summaries extracted
        hist_summary = tf.zeros_like(sent_vecs_list[0])
        extract_logit_list, extract_prob_list = [], []

        # Loop over the sentences
        for i in xrange(hps.num_sents_doc):
          cur_sent_vec = sent_vecs_list[i]
          cur_abs_pos = abs_pos_emb_list[i]
          cur_rel_pos = rel_pos_emb_list[i]

          if i > 0:  # NB: reusing is important!
            tf.get_variable_scope().reuse_variables()

          extract_logit = self._compute_extract_prob(
              cur_sent_vec, cur_abs_pos, cur_rel_pos, self._doc_repr,
              hist_summary)  # [batch_size, 2]
          extract_logit_list.append(extract_logit)
          extract_prob = tf.nn.softmax(extract_logit)  # [batch_size, 2]
          extract_prob_list.append(extract_prob)

          prob_1 = tf.expand_dims(tf.unstack(extract_prob, axis=1)[1],
                                  1)  # [batch_size, 1] float32
          hist_summary += prob_1 * cur_sent_vec  # [batch_size,enc_num_hidden*2]

        self._extract_logits = tf.stack(
            extract_logit_list, axis=1)  # [batch_size, num_sents_doc, 2]
        self._extract_probs = tf.stack(
            extract_prob_list, axis=1)  # [batch_size, num_sents_doc, 2]

        if hps.train_mode in ["coherence", "sl+coherence"]:
          rsp_extr_logits = tf.reshape(self._extract_logits,
                                       [-1, 2])  #[batch_size*num_sents_doc, 2]

          sampled_extracts = tf.multinomial(
              logits=rsp_extr_logits, num_samples=
              hps.coh_samples)  #[batch_size*num_sents_doc, coh_samples] int32

          self._sampled_extracts = tf.reshape(sampled_extracts, [
              -1, hps.num_sents_doc, hps.coh_samples
          ])  # [batch_size, num_sents_doc, coh_samples] int32

  def _add_embeddings(self):
    hps = self._hps
    input_vsize = self._input_vocab.NumIds

    with tf.device(self._device_0):
      # Input word embeddings
      self._input_embed = tf.get_variable(
          "input_embed", [input_vsize, hps.emb_dim],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))
      # Absolute position embeddings
      self._abs_pos_embed = tf.get_variable(
          "abs_pos_embed", [hps.num_sents_doc, hps.pos_emb_dim],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))
      # Relative position embeddings
      self._rel_pos_embed = tf.get_variable(
          "rel_pos_embed", [hps.rel_pos_max_idx, hps.pos_emb_dim],
          dtype=tf.float32,
          initializer=tf.truncated_normal_initializer(stddev=1e-4))

  def _add_encoder(self, inputs, sent_lens, doc_lens, transpose_output=False):
    hps = self._hps

    # Masking the word embeddings
    sent_lens_rsp = tf.reshape(sent_lens, [-1])  # [batch_size * num_sents_doc]
    word_masks = tf.expand_dims(
        tf.sequence_mask(
            sent_lens_rsp, maxlen=hps.num_words_sent, dtype=tf.float32),
        2)  # [batch_size * num_sents_doc, num_words_sent, 1]

    inputs_rsp = tf.reshape(inputs, [-1, hps.num_words_sent])
    emb_inputs = tf.nn.embedding_lookup(
        self._input_embed,
        inputs_rsp)  # [batch_size * num_sents_doc, num_words_sent, emb_dim]
    emb_inputs = emb_inputs * word_masks

    # Level 1: Add the word-level convolutional neural network
    word_conv_outputs = []

    for num_filter, width in zip(hps.word_conv_filters, hps.word_conv_widths):
      # Create CNNs with different kernel width
      word_conv = tf.layers.conv1d(
          emb_inputs,
          num_filter,
          width,
          padding="same",
          kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))

      word_mean_pool = tf.reduce_mean(
          word_conv, axis=1)  # [batch_size * num_sents_doc, num_filter]
      word_conv_outputs.append(word_mean_pool)

    word_conv_concat = tf.concat(
        word_conv_outputs, axis=1)  # concat the sentence representations
    # Reshape the representations of sentences
    sentence_size = sum(hps.word_conv_filters)
    sentence_repr = tf.reshape(word_conv_concat, [
        -1, hps.num_sents_doc, sentence_size
    ])  # [batch_size, num_sents_doc, sentence_size]

    # Level 2: Add the sentence-level RNN
    sent_rnn_output, _ = lib.cudnn_rnn_wrapper(
        sentence_repr,
        "gru",
        hps.enc_layers,
        hps.enc_num_hidden,
        sentence_size,
        "enc_cudnn_gru_var",
        direction="bidirectional",
        dropout=hps.dropout)  # [num_sents_doc, batch_size, enc_num_hidden*2]

    # Masking the paddings
    sent_out_masks = tf.sequence_mask(doc_lens, hps.num_sents_doc,
                                      tf.float32)  # [batch_size, num_sents_doc]
    sent_out_masks = tf.expand_dims(tf.transpose(sent_out_masks),
                                    2)  # [num_sents_doc, batch_size, 1]
    sent_rnn_output = sent_rnn_output * sent_out_masks  # [num_sents_doc, batch_size, enc_num_hidden*2]

    if transpose_output:
      sent_rnn_output = tf.transpose(sent_rnn_output, [1, 0, 2])
      # [batch_size, num_sents_doc, enc_num_hidden*2]

    return sent_rnn_output

  def _compute_extract_prob(self, sent_vec, abs_pos_emb, rel_pos_emb, doc_repr,
                            hist_summary):
    hps = self._hps

    hist_sum_norm = tf.tanh(hist_summary)  # normalized with tanh
    mlp_hidden = tf.concat(
        [sent_vec, abs_pos_emb, rel_pos_emb, doc_repr, hist_sum_norm], axis=1)

    # Construct an MLP for extraction decisions
    for i, n in enumerate(hps.mlp_num_hiddens):
      mlp_hidden = tf.contrib.layers.fully_connected(
          mlp_hidden,
          n,
          activation_fn=tf.nn.relu,  # tf.tanh/tf.sigmoid
          weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
          scope="mlp_layer_%d" % (i + 1))

    extract_logit = tf.contrib.layers.fully_connected(
        mlp_hidden,
        2,
        activation_fn=None,
        weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
        scope="mlp_output_layer")

    return extract_logit  # [batch_size, 2]

  def _add_loss(self):
    hps = self._hps

    with tf.variable_scope("loss"), tf.device(self._device_0):
      # Masking the loss
      loss_mask = tf.sequence_mask(
          self._input_doc_lens, maxlen=hps.num_sents_doc,
          dtype=tf.float32)  # [batch_size, num_sents_doc]

      if hps.train_mode in ["sl", "sl+coherence"]:  # supervised learning
        xe_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._extract_targets,
            logits=self._extract_logits,
            name="sl_xe_loss")

        batch_loss = tf.div(
            tf.reduce_sum(xe_loss * self._target_weights * loss_mask, 1),
            tf.to_float(self._input_doc_lens))

        sl_loss = tf.reduce_mean(batch_loss)
        tf.summary.scalar("sl_loss", sl_loss)
        loss = sl_loss

    if hps.train_mode in ["coherence", "sl+coherence"]:
      coh_loss = self._add_coherence_loss(loss_mask)
      tf.summary.scalar("coh_loss", coh_loss)
      if hps.train_mode == "coherence":
        loss = coh_loss
      else:
        loss += coh_loss * hps.coherence_coef

    tf.summary.scalar("loss", loss)
    self._loss = loss

  def _add_coherence_loss(self, loss_mask):
    hps = self._hps

    import coherence
    self._coh_hps = coherence.CreateHParams()._replace(batch_size=None)
    assert hps.num_words_sent == self._coh_hps.max_sent_len, \
        "num_words_sent must equal to max_sent_len"

    # Step 1: covert the format
    sent_inputs, sent_lens, states = tf.py_func(
        self._convert_to_coherence, [
            self._sampled_extracts, self._inputs, self._input_doc_lens,
            self._input_sent_lens
        ],
        Tout=[tf.int32, tf.int32, tf.int32],
        stateful=False,
        name="convert_to_coherence_format")

    # Step 2: add the coherence model to computation graph
    coherence_model = coherence.CoherenceModel(
        self._coh_hps, self._input_vocab, num_gpus=1)
    coh_prob, self._coh_vs = coherence_model.inference_graph(
        sent_inputs, sent_lens, device=self._device_1)

    # Step 3: convert the coherence probabilities to rewards and compute loss
    rewards = tf.py_func(
        self._convert_to_reward, [coh_prob, states],
        Tout=tf.float32,
        stateful=False,
        name="convert_to_reward")  # [batch_size, coh_samples]

    # Shape information missing in py_func output Tensors.
    # hps.batch_size must be specified when training.
    rewards.set_shape([hps.batch_size, hps.coh_samples])
    self._avg_reward = tf.reduce_mean(rewards)  # average reward
    # tf.summary.scalar("avg_reward", self._avg_reward)

    # Return is same with reward [batch_size, num_sents_doc, coh_samples]
    returns = tf.tile(tf.expand_dims(rewards, 1), [1, hps.num_sents_doc, 1])

    # Compute the negative log-likelihood of chosen actions
    tiled_logits = tf.tile(
        tf.expand_dims(self._extract_logits, 2), [1, 1, hps.coh_samples, 1])
    # [batch_size, num_sents_doc, coh_samples, 2]
    neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self._sampled_extracts,
        logits=tiled_logits,
        name="coh_neg_log_prob")  # [batch_size, num_sents_doc, coh_samples]

    # Compute the policy loss, average over the samples
    batch_loss = tf.div(
        tf.reduce_sum(
            tf.reduce_mean(neg_log_probs * returns, 2) * loss_mask, 1),
        tf.to_float(self._input_doc_lens))

    return tf.reduce_mean(batch_loss)

  def _convert_to_coherence(self, sampled_extracts, inputs, doc_lens,
                            sent_lens):
    """
    Parameters:
      sampled_extracts: [batch_size, num_sents_doc, coh_samples] 0 or 1
      inputs: [batch_size, num_sents_doc, num_words_sent]
      doc_lens: [batch_size]
      sent_lens: [batch_size, num_sents_doc]

    Returns:
      np_sents: [?, max_num_sents, max_sent_len]
      np_sent_lens: [?, max_num_sents]
      np_states: [?, coh_samples]
    """
    num_samples = self._hps.coh_samples
    assert num_samples == sampled_extracts.shape[2]

    coh_hps = self._coh_hps
    max_num_sents = coh_hps.max_num_sents

    pad_sent = np.array(
        [self._input_vocab.pad_id] * coh_hps.max_sent_len, dtype=np.int32)

    ext_sents_list, states = [], []
    for extracts, doc, doc_len, sent_len in zip(sampled_extracts, inputs,
                                                doc_lens, sent_lens):
      for i in xrange(num_samples):
        extracted_sents = [(doc[j], sent_len[j])
                           for j, ext in enumerate(extracts)
                           if ext[i] and j < doc_len]
        if not extracted_sents:
          states.append(0)  # Failure
        elif len(extracted_sents) <= max_num_sents:
          ext_sents_list.append(extracted_sents)
          states.append(1)  # Success
        else:
          states.append(0)  # Failure

    if ext_sents_list:
      sents_batch, lens_batch = [], []
      for item in ext_sents_list:
        sents = [s for s, _ in item]
        sents += [pad_sent] * (max_num_sents - len(sents))
        sents_batch.append(sents)

        lens = [l for _, l in item]
        lens += [0] * (max_num_sents - len(lens))
        lens_batch.append(lens)

      np_sents = np.array(sents_batch, dtype=np.int32)
      np_sent_lens = np.array(lens_batch, dtype=np.int32)
      np_states = np.reshape(
          np.array(states, dtype=np.int32), [-1, num_samples])
    else:  # No valid extractions, return pseudo output
      np_sents = np.zeros(
          [1, coh_hps.max_num_sents, coh_hps.max_sent_len], dtype=np.int32)
      np_sent_lens = np.zeros([1, coh_hps.max_num_sents], dtype=np.int32)
      np_states = np.reshape(
          np.array(states, dtype=np.int32), [-1, num_samples])

    return np_sents, np_sent_lens, np_states

  def _convert_to_reward(self, extract_probs, states):
    """
    Parameters:
      extract_probs: [?] float32
      states: [batch_size, coh_samples] 0/1
    """
    rewards = np.zeros_like(states, dtype=np.float32)  # zero if fail
    idx = 0
    for i, s in enumerate(states):
      for j, t in enumerate(s):
        if t:
          rewards[i, j] = extract_probs[idx]
          idx += 1

    return rewards

  def _add_train_op(self):
    """Sets self._train_op for training."""
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
    (enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, extract_targets,
     target_weights, others) = batch

    if self._hps.train_mode == "sl":
      to_return = [
          self._train_op, self._summaries, self._loss, self.global_step
      ]
    else:  # coherence or sl+coherence
      to_return = [
          self._train_op, self._summaries, self._loss, self._avg_reward,
          self.global_step
      ]

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

  def run_eval_step(self, sess, batch):
    (enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, extract_targets,
     target_weights, others) = batch

    if self._hps.train_mode == "sl":
      to_return = self._loss
    else:  # coherence or sl+coherence
      to_return = [self._loss, self._avg_reward]

    result = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos,
            self._extract_targets: extract_targets,
            self._target_weights: target_weights
        })

    return result

  def train_loop_sl(self, sess, batcher, valid_batcher, summary_writer):
    """Runs model training."""
    step, losses = 0, []
    while step < FLAGS.max_run_steps:
      next_batch = batcher.next()
      summaries, loss, train_step = self.run_train_step(sess, next_batch)

      losses.append(loss)
      summary_writer.add_summary(summaries, train_step)
      step += 1

      # Display current training loss
      if step % FLAGS.display_freq == 0:
        avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss",
                                   train_step)
        tf.logging.info("Train step %d: avg_loss %f" % (train_step, avg_loss))
        losses = []
        summary_writer.flush()

      # Run evaluation on validation set
      if step % FLAGS.valid_freq == 0:
        valid_losses = []
        for _ in xrange(FLAGS.num_valid_batch):
          next_batch = valid_batcher.next()
          valid_loss = self.run_eval_step(sess, next_batch)
          valid_losses.append(valid_loss)

        gstep = self.get_global_step(sess)
        avg_valid_loss = lib.compute_avg(valid_losses, summary_writer,
                                         "valid_loss", gstep)
        tf.logging.info("\tValid step %d: avg_loss %f" % (gstep,
                                                          avg_valid_loss))

        summary_writer.flush()

  def train_loop_coh(self, sess, batcher, valid_batcher, summary_writer):
    """Runs model training."""
    step, losses, rewards = 0, [], []
    while step < FLAGS.max_run_steps:
      next_batch = batcher.next()
      summaries, loss, reward, train_step = self.run_train_step(
          sess, next_batch)

      losses.append(loss)
      rewards.append(reward)
      summary_writer.add_summary(summaries, train_step)
      step += 1

      # Display current training loss
      if step % FLAGS.display_freq == 0:
        avg_loss = lib.compute_avg(losses, summary_writer, "avg_loss",
                                   train_step)
        avg_reward = lib.compute_avg(rewards, summary_writer, "avg_reward",
                                     train_step)

        tf.logging.info("Train step %d: avg_loss %f avg_reward %f" %
                        (train_step, avg_loss, avg_reward))
        losses, rewards = [], []
        summary_writer.flush()

      # Run evaluation on validation set
      if step % FLAGS.valid_freq == 0:
        valid_losses, valid_rewards = [], []
        for _ in xrange(FLAGS.num_valid_batch):
          next_batch = valid_batcher.next()
          valid_loss, valid_reward = self.run_eval_step(sess, next_batch)
          valid_losses.append(valid_loss)
          valid_rewards.append(valid_reward)

        gstep = self.get_global_step(sess)
        avg_valid_loss = lib.compute_avg(valid_losses, summary_writer,
                                         "valid_loss", gstep)
        avg_valid_reward = lib.compute_avg(valid_rewards, summary_writer,
                                           "valid_reward", gstep)

        tf.logging.info("\tValid step %d: avg_loss %f avg_reward %f" %
                        (gstep, avg_valid_loss, avg_valid_reward))

        summary_writer.flush()

  def train_loop(self, sess, batcher, valid_batcher, summary_writer):
    if self._hps.train_mode == "sl":
      self.train_loop_sl(sess, batcher, valid_batcher, summary_writer)
    else:  # coherence or sl+coherence
      self.train_loop_coh(sess, batcher, valid_batcher, summary_writer)

  def train(self, data_batcher, valid_batcher):
    """Runs model training."""
    hps = self._hps
    assert hps.mode == "train", "This method is only callable in train mode."

    with tf.device("/gpu:0"):  # GPU by default
      self.build_graph()

    # Restore pretrained model if necessary
    if hps.train_mode in ["coherence", "sl+coherence"]:
      coherence_vars = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self._coh_vs.name)
      coh_restorer = tf.train.Saver(
          coherence_vars)  #restore coherence model params
      coh_ckpt_state = tf.train.get_checkpoint_state(FLAGS.coherence_dir)
      if not (coh_ckpt_state and coh_ckpt_state.model_checkpoint_path):
        raise ValueError("No pretrain model found at %s" % FLAGS.coherence_dir)

      extract_vars = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self._vs.name)
      ext_restorer = tf.train.Saver(
          extract_vars)  #restore extraction model params
      ext_ckpt_state = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
      if not (ext_ckpt_state and ext_ckpt_state.model_checkpoint_path):
        ext_restorer = None

      def load_pretrain(sess):
        coh_restorer.restore(sess, coh_ckpt_state.model_checkpoint_path)
        if ext_restorer:
          ext_restorer.restore(sess, ext_ckpt_state.model_checkpoint_path)
    else:  # train_mode == "sl"
      load_pretrain = None

    saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)
    sv = tf.train.Supervisor(
        logdir=FLAGS.ckpt_root,
        saver=saver,
        summary_op=None,
        save_summaries_secs=FLAGS.checkpoint_secs,
        save_model_secs=FLAGS.checkpoint_secs,
        global_step=self.global_step,
        init_fn=load_pretrain)  # TODO: could exploit more Supervisor features

    config = tf.ConfigProto(allow_soft_placement=True)
    # Turn on JIT compilation if necessary
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = sv.prepare_or_wait_for_session(config=config)

    # Summary dir is different from ckpt_root to avoid conflict.
    summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

    # Start the training loop
    self.train_loop(sess, data_batcher, valid_batcher, summary_writer)

    sv.Stop()

  def get_global_step(self, sess):
    """Get the current number of training steps."""
    return sess.run(self.global_step)
