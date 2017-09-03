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
from multiprocessing import Pool
import lib
from base import BaseModel

# Import pythonrouge package
from pythonrouge import PythonROUGE
ROUGE_dir = "/qydata/ywubw/download/RELEASE-1.5.5"
sentence_sep = "</s>"
rouge_weights = [0.4, 1.0, 0.5]  # [0.5, 1.0, 0.6]

rouge = PythonROUGE(
    ROUGE_dir,
    n_gram=2,
    ROUGE_SU4=False,
    ROUGE_L=True,
    stemming=True,
    stopwords=False,
    length_limit=False,
    length=75,
    word_level=False,
    use_cf=True,
    cf=95,
    ROUGE_W=False,
    ROUGE_W_Weight=1.2,
    scoring_formula="average",
    resampling=False,
    samples=1000,
    favor=False,
    p=0.5)


def compute_rouge(item):
  system_sents = item[0]
  reference_sents = item[1]

  rouge_dict = rouge.evaluate(
      [[system_sents]], [[reference_sents]], to_dict=True, f_measure_only=True)
  weighted_rouge = (rouge_dict["ROUGE-1"] * rouge_weights[0] +
                    rouge_dict["ROUGE-2"] * rouge_weights[1] +
                    rouge_dict["ROUGE-L"] * rouge_weights[2]) / 3.0
  return weighted_rouge


FLAGS = tf.app.flags.FLAGS
# NB: batch_size could be unspecified (None) in decode mode
HParams = namedtuple("HParams", "mode, min_lr, lr, dropout, batch_size,"
                     "num_sents_doc, num_words_sent, rel_pos_max_idx,"
                     "enc_layers, enc_num_hidden, emb_dim, pos_emb_dim,"
                     "doc_repr_dim, word_conv_widths, word_conv_filters,"
                     "mlp_num_hiddens, train_mode, coherence_coef,"
                     "rouge_coef, min_num_input_sents, trg_weight_norm,"
                     "max_grad_norm, decay_step, decay_rate, coh_reward_clip,"
                     "hist_repr_dim")


def CreateHParams():
  """Create Hyper-parameters from tf.app.flags.FLAGS"""

  assert FLAGS.mode in ["train", "decode"], "Invalid mode."

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
      rouge_coef=FLAGS.rouge_coef,  # coefficient of ROUGE loss
      coh_reward_clip=FLAGS.coh_reward_clip,  # maximum reward clipping
      hist_repr_dim=FLAGS.hist_repr_dim)  # dimension of history representation
  return hps


class CoherentExtractRF(BaseModel):
  """ An extractive summarization model based on SummaRuNNer that is enhanced
      with REINFORCE by ROUGE and coherence model. Related works are listed:

  [1] Nallapati, R., Zhai, F., & Zhou, B. (2016). SummaRuNNer: A Recurrent
      Neural Network based Sequence Model for Extractive Summarization of
      Documents. arXiv:1611.04230 [Cs].

  [2] Hu, B., Lu, Z., Li, H., & Chen, Q. (2014). Convolutional neural network
      architectures for matching natural language sentences. In Advances in
      neural information processing systems (pp. 2042-2050).
  """

  def __init__(self, hps, input_vocab, num_gpus=0):
    self._hps = hps
    self._input_vocab = input_vocab
    self._num_gpus = num_gpus

    if hps.mode == "train" and not any(
        [x in hps.train_mode for x in ["sl", "rouge", "coherence"]]):
      raise ValueError("Invalid train mode.")

  def build_graph(self):
    self._allocate_devices()
    self._add_placeholders()
    self._build_model()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self._hps.mode == "train":
      if "rouge" in self._hps.train_mode:
        self._pool = Pool(15)
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

    # For ROUGE mode
    self._document_strs = tf.placeholder(
        tf.string, [hps.batch_size], name="document_strs")
    self._summary_strs = tf.placeholder(
        tf.string, [hps.batch_size], name="summary_strs")

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
        # self._doc_repr = tf.tanh(
        #     lib.linear(
        #         doc_mean_vec, hps.doc_repr_dim, True, scope="doc_repr_linear"))
        self._doc_repr = tf.contrib.layers.fully_connected(
            doc_mean_vec,
            hps.doc_repr_dim,
            activation_fn=tf.tanh,
            weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
            scope="sents_to_doc")

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
        targets = tf.unstack(tf.to_float(self._extract_targets), axis=1)

      # Compute the extraction probability of each sentence
      with tf.variable_scope("extract_sent",
          initializer=tf.random_uniform_initializer(-0.1, 0.1)), \
          tf.device(self._device_0):

        if hps.mode == "train":  # train mode
          if "sl" in hps.train_mode:
            # Initialize the representation of all history summaries extracted
            if hps.hist_repr_dim:
              hist_summary = tf.zeros(
                  [hps.batch_size, hps.hist_repr_dim],
                  dtype=tf.float32)  # [batch_size, hist_repr_dim]
            else:
              hist_summary = tf.zeros_like(sent_vecs_list[0])  # [batch_size, ?]

            extract_logit_list, extract_prob_list = [], []

            for i in xrange(hps.num_sents_doc):  # loop over the sentences
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

              target = tf.expand_dims(targets[i], 1)  # [batch_size, 1] float32

              if hps.hist_repr_dim:
                hist_sent_vec = tf.contrib.layers.fully_connected(
                    cur_sent_vec,
                    hps.hist_repr_dim,
                    activation_fn=tf.tanh,
                    weights_initializer=tf.random_uniform_initializer(
                        -0.1, 0.1),
                    scope="sent_to_hist")
                hist_summary += target * hist_sent_vec  #[batch_size, hist_repr_dim]
              else:
                hist_summary += target * cur_sent_vec  #[batch_size, enc_num_hidden*2]

            self._extract_logits = tf.stack(
                extract_logit_list, axis=1)  # [batch_size, num_sents_doc, 2]
            self._extract_probs = tf.stack(
                extract_prob_list, axis=1)  # [batch_size, num_sents_doc, 2]

          if "rouge" in hps.train_mode or "coherence" in hps.train_mode:
            if hps.hist_repr_dim:
              rl_hist_summary = tf.zeros(
                  [hps.batch_size, hps.hist_repr_dim],
                  dtype=tf.float32)  # [batch_size, hist_repr_dim]
            else:
              rl_hist_summary = tf.zeros_like(sent_vecs_list[0])
            rl_extract_logit_list, sampled_extract_list = [], []

            for i in xrange(hps.num_sents_doc):
              cur_sent_vec = sent_vecs_list[i]
              cur_abs_pos = abs_pos_emb_list[i]
              cur_rel_pos = rel_pos_emb_list[i]

              if i > 0:  # NB: reusing is important!
                tf.get_variable_scope().reuse_variables()

              rl_extract_logit = self._compute_extract_prob(
                  cur_sent_vec, cur_abs_pos, cur_rel_pos, self._doc_repr,
                  rl_hist_summary)  # [batch_size, 2]
              rl_extract_logit_list.append(rl_extract_logit)

              sampled_extract = tf.multinomial(
                  logits=rl_extract_logit,
                  num_samples=1)  # [batch_size, 1] int32
              sampled_extract_list.append(sampled_extract)

              if hps.hist_repr_dim:
                hist_sent_vec = tf.contrib.layers.fully_connected(
                    cur_sent_vec,
                    hps.hist_repr_dim,
                    activation_fn=tf.tanh,
                    weights_initializer=tf.random_uniform_initializer(
                        -0.1, 0.1),
                    scope="sent_to_hist")
                rl_hist_summary += tf.to_float(
                    sampled_extract
                ) * hist_sent_vec  #[batch_size, hist_repr_dim]

              else:
                rl_hist_summary += tf.to_float(
                    sampled_extract
                ) * cur_sent_vec  # [batch_size, enc_num_hidden*2]

            self._rl_extract_logits = tf.stack(
                rl_extract_logit_list, axis=1)  # [batch_size, num_sents_doc, 2]
            self._sampled_extracts = tf.concat(
                sampled_extract_list,
                axis=1)  # [batch_size, num_sents_doc] int32

        else:  # decode mode
          self._cur_sent_vec = tf.placeholder(tf.float32,
                                              sent_vecs_list[0].get_shape())
          self._cur_abs_pos = tf.placeholder(tf.float32,
                                             abs_pos_emb_list[0].get_shape())
          self._cur_rel_pos = tf.placeholder(tf.float32,
                                             rel_pos_emb_list[0].get_shape())
          self._hist_summary = tf.placeholder(tf.float32,
                                              sent_vecs_list[0].get_shape())

          extract_logit = self._compute_extract_prob(
              self._cur_sent_vec, self._cur_abs_pos, self._cur_rel_pos,
              self._doc_repr, self._hist_summary)  # [batch_size, 2]
          self._ext_log_prob = tf.log(
              tf.nn.softmax(extract_logit))  # [batch_size, 2]

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

    if hps.hist_repr_dim:
      hist_sum_norm = tf.contrib.layers.fully_connected(
          hist_summary,
          hps.hist_repr_dim,
          activation_fn=tf.tanh,
          weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
          scope="hist_tanh")
    else:
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
    loss = None

    with tf.variable_scope("loss"), tf.device(self._device_0):
      loss_mask = tf.sequence_mask(
          self._input_doc_lens, maxlen=hps.num_sents_doc,
          dtype=tf.float32)  # [batch_size, num_sents_doc]

      if "sl" in hps.train_mode:  # supervised learning
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

      rewards = None
      if "rouge" in hps.train_mode:
        rouge_rewards, rouge_reward_sum = tf.py_func(
            self._get_rouge_rewards, [
                self._sampled_extracts, self._input_doc_lens,
                self._document_strs, self._summary_strs
            ],
            Tout=[tf.float32, tf.float32],
            stateful=False,
            name="rouge_reward")

        rouge_rewards.set_shape([hps.batch_size, hps.num_sents_doc])
        rouge_reward_sum.set_shape([hps.batch_size])

        tf.summary.scalar("rouge_reward", tf.reduce_mean(rouge_reward_sum))
        rewards = rouge_rewards * hps.rouge_coef

    # Exit "loss" scope for definition of coherence model
    if "coherence" in hps.train_mode:
      coherence_rewards, coh_reward_sum = self._get_coherence_rewards()
      coherence_rewards.set_shape([hps.batch_size, hps.num_sents_doc])
      coh_reward_sum.set_shape([hps.batch_size])

      tf.summary.scalar("coherence_reward", tf.reduce_mean(coh_reward_sum))
      if rewards is not None:
        rewards += coherence_rewards * hps.coherence_coef
      else:
        rewards = coherence_rewards * hps.coherence_coef

    with tf.variable_scope("loss"), tf.device(self._device_0):
      if rewards is not None:
        self._avg_reward = tf.reduce_mean(tf.reduce_sum(rewards, 1))

        # Compute the return value by cumulating all the advantages backwards
        rewards_list = tf.unstack(rewards, axis=1)
        rev_returns, cumulator = [], None  # reversed list of returns
        for r in reversed(rewards_list):
          cumulator = r if cumulator is None else cumulator + r  # discount=1
          rev_returns.append(cumulator)
        returns = tf.stack(
            list(reversed(rev_returns)), axis=1)  # [batch_size, num_sents_doc]

        # Compute the negative log-likelihood of chosen actions
        neg_log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self._sampled_extracts,
            logits=self._rl_extract_logits,
            name="neg_log_probs")  # [batch_size, num_sents_doc]

        # Compute the policy loss, average over the samples
        batch_loss = tf.div(
            tf.reduce_sum(neg_log_probs * returns * loss_mask, 1),
            tf.to_float(self._input_doc_lens))
        rl_loss = tf.reduce_mean(batch_loss)
        tf.summary.scalar("rl_loss", rl_loss)
        if loss is not None:
          loss += rl_loss
        else:
          loss = rl_loss

    tf.summary.scalar("loss", loss)
    self._loss = loss

  def _get_rouge_rewards(self, sampled_extracts, doc_lens, doc_strs,
                         summary_strs):
    ext_sents_list, sum_sents_list = [], []
    for extracts, doc_str, summary_str in zip(sampled_extracts, doc_strs,
                                              summary_strs):
      doc_sents = doc_str.split(sentence_sep)
      extract_sents = [s for e, s in zip(extracts, doc_sents) if e]
      ext_sents_list.append(extract_sents)  # system summary

      summary_sents = summary_str.split(sentence_sep)
      sum_sents_list.append(summary_sents)  # reference summary

    rouge_scores = self._pool.map(compute_rouge,
                                  zip(ext_sents_list, sum_sents_list))
    np_scores = np.zeros_like(sampled_extracts, dtype=np.float32)
    for i, j in enumerate(doc_lens):
      np_scores[i, j - 1] = rouge_scores[i]  # index starts with 0
    np_total_scores = np.array(rouge_scores, dtype=np.float32)

    return np_scores, np_total_scores

  def _get_coherence_rewards(self):
    hps = self._hps

    import seqmatch
    self._sm_hps = seqmatch.CreateHParams()._replace(
        batch_size=None, mode="decode")  #NB: not train mode
    assert hps.num_words_sent == self._sm_hps.max_sent_len, \
        "num_words_sent must equal to max_sent_len"

    # Step 1: convert the format
    sents_A, sents_B, lengths_A, lengths_B = tf.py_func(
        self._convert_to_seqmatch, [
            self._sampled_extracts, self._inputs, self._input_doc_lens,
            self._input_sent_lens
        ],
        Tout=[tf.int32, tf.int32, tf.int32, tf.int32],
        stateful=False,
        name="convert_format")

    # Step 2: add the coherence model to computation graph
    seqmatch_model = seqmatch.SeqMatchNet(
        self._sm_hps, self._input_vocab, num_gpus=1)
    sm_output, self._coh_vs = seqmatch_model.inference_graph(
        sents_A, sents_B, lengths_A, lengths_B, device=self._device_1)

    # Step 3: convert the coherence score to rewards
    rewards, reward_sum = tf.py_func(
        self._convert_to_reward, [sm_output, self._sampled_extracts],
        Tout=[tf.float32, tf.float32],
        stateful=False,
        name="convert_to_reward")  # [batch_size]

    return rewards, reward_sum

  def _convert_to_seqmatch(self, sampled_extracts, inputs, doc_lens, sent_lens):
    """
    Parameters:
      sampled_extracts: [batch_size, num_sents_doc] 0 or 1
      inputs: [batch_size, num_sents_doc, num_words_sent/max_sent_len]
      doc_lens: [batch_size]
      sent_lens: [batch_size, num_sents_doc]

    Returns:
      np_sents_A: [?, max_sent_len]
      np_sents_B: [?, max_sent_len]
      np_lens_A: [?]
      np_lens_B: [?]
    """
    hps = self._sm_hps
    max_sent_len = hps.max_sent_len

    start_sent = np.array(
        [self._input_vocab.start_id] * max_sent_len, dtype=np.int32)

    sent_A_list, sent_B_list, len_A_list, len_B_list = [], [], [], []
    for i in xrange(sampled_extracts.shape[0]):
      prev_idx = -1
      for j in xrange(sampled_extracts.shape[1]):
        if sampled_extracts[i, j]:
          if prev_idx < 0:
            sent_A_list.append(start_sent)
            len_A_list.append(max_sent_len)
          else:
            sent_A_list.append(inputs[i, prev_idx])
            len_A_list.append(sent_lens[i, prev_idx])

          sent_B_list.append(inputs[i, j])
          len_B_list.append(sent_lens[i, j])
          prev_idx = j

    if sent_A_list:
      np_sents_A = np.stack(sent_A_list, axis=0)
      np_sents_B = np.stack(sent_B_list, axis=0)
      np_lens_A = np.array(len_A_list, dtype=np.int32)
      np_lens_B = np.array(len_B_list, dtype=np.int32)
    else:  # No valid extractions, return pseudo output
      np_sents_A = np.zeros([1, max_sent_len], dtype=np.int32)
      np_sents_B = np.zeros([1, max_sent_len], dtype=np.int32)
      np_lens_A = np.zeros([1], dtype=np.int32)
      np_lens_B = np.zeros([1], dtype=np.int32)

    return np_sents_A, np_sents_B, np_lens_A, np_lens_B

  def _convert_to_reward(self, scores, sampled_extracts):
    """
    Parameters:
      extract_probs: [?] float32
      sampled_extracts: [batch_size, num_sents_doc] 0 or 1

    Returns:
      rewards: [batch_size, num_sents_doc] float32
      reward_sum: [batch_size] float32
    """
    rewards = np.zeros_like(sampled_extracts, dtype=np.float32)  # zero if fail
    max_reward = self._hps.coh_reward_clip
    idx = 0
    for i in xrange(sampled_extracts.shape[0]):
      for j in xrange(sampled_extracts.shape[1]):
        if sampled_extracts[i, j]:
          rewards[i, j] = scores[idx]
          idx += 1
    reward_sum = np.sum(rewards, axis=1)

    for i, r in enumerate(reward_sum):
      if r > max_reward:
        rewards[i, :] *= max_reward / r
        reward_sum[i] = max_reward

    return rewards, reward_sum

  def run_train_step(self, sess, batch):
    (enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, extract_targets,
     target_weights, others) = batch

    if self._hps.train_mode == "sl":
      to_return = [
          self._train_op, self._summaries, self._loss, self.global_step
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
    else:  # with coherence or rouge
      to_return = [
          self._train_op, self._summaries, self._loss, self._avg_reward,
          self.global_step
      ]
      doc_strs, summary_strs = others

      results = sess.run(
          to_return,
          feed_dict={
              self._inputs: enc_batch,
              self._input_sent_lens: enc_sent_lens,
              self._input_doc_lens: enc_doc_lens,
              self._input_rel_pos: sent_rel_pos,
              self._extract_targets: extract_targets,
              self._target_weights: target_weights,
              self._document_strs: doc_strs,
              self._summary_strs: summary_strs
          })

    return results[1:]

  def run_eval_step(self, sess, batch):
    (enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, extract_targets,
     target_weights, others) = batch

    if self._hps.train_mode == "sl":
      result = sess.run(
          self._loss,
          feed_dict={
              self._inputs: enc_batch,
              self._input_sent_lens: enc_sent_lens,
              self._input_doc_lens: enc_doc_lens,
              self._input_rel_pos: sent_rel_pos,
              self._extract_targets: extract_targets,
              self._target_weights: target_weights
          })
    else:  # with coherence or rouge
      doc_strs, summary_strs = others
      result = sess.run(
          [self._loss, self._avg_reward],
          feed_dict={
              self._inputs: enc_batch,
              self._input_sent_lens: enc_sent_lens,
              self._input_doc_lens: enc_doc_lens,
              self._input_rel_pos: sent_rel_pos,
              self._extract_targets: extract_targets,
              self._target_weights: target_weights,
              self._document_strs: doc_strs,
              self._summary_strs: summary_strs
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

  def train_loop_rl(self, sess, batcher, valid_batcher, summary_writer):
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
      self.train_loop_rl(sess, batcher, valid_batcher, summary_writer)

  def train(self, data_batcher, valid_batcher):
    """Runs model training."""
    hps = self._hps
    assert hps.mode == "train", "This method is only callable in train mode."

    with tf.device("/gpu:0"):  # GPU by default
      self.build_graph()

    if "coherence" in hps.train_mode:  # restore coherence model params
      coherence_vars = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self._coh_vs.name)
      coh_restorer = tf.train.Saver(
          coherence_vars)  #restore coherence model params
      coh_ckpt_state = tf.train.get_checkpoint_state(FLAGS.coherence_dir)
      if not (coh_ckpt_state and coh_ckpt_state.model_checkpoint_path):
        raise ValueError("No pretrain model found at %s" % FLAGS.coherence_dir)

      # Restore pretrained extraction model if necessary
      extract_vars = tf.get_collection(
          tf.GraphKeys.GLOBAL_VARIABLES, scope=self._vs.name)
      ext_restorer = tf.train.Saver(extract_vars)
      ext_ckpt_state = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
      if not (ext_ckpt_state and ext_ckpt_state.model_checkpoint_path):
        ext_restorer = None

      def load_pretrain(sess):
        coh_restorer.restore(sess, coh_ckpt_state.model_checkpoint_path)
        if ext_restorer:
          ext_restorer.restore(sess, ext_ckpt_state.model_checkpoint_path)
    else:
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

  def decode_get_feats(self, sess, enc_batch, enc_doc_lens, enc_sent_lens,
                       sent_rel_pos):
    """Get hidden features for decode mode."""
    if not self._hps.mode == "decode":
      raise ValueError("This method is only for decode mode.")

    to_return = [
        self._sentence_vecs, self._sent_abs_pos_emb, self._sent_rel_pos_emb,
        self._doc_repr
    ]

    results = sess.run(
        to_return,
        feed_dict={
            self._inputs: enc_batch,
            self._input_sent_lens: enc_sent_lens,
            self._input_doc_lens: enc_doc_lens,
            self._input_rel_pos: sent_rel_pos
        })
    return results

  def decode_log_probs(self, sess, sent_vec, abs_pos_embed, rel_pos_embed,
                       doc_repr, hist_summary):
    """Get log probability of extraction given a sentence and its history."""
    if not self._hps.mode == "decode":
      raise ValueError("This method is only for decode mode.")

    # sent_vec, abs_pos_embed, rel_pos_embed, doc_repr, hist_summary = features
    return sess.run(
        self._ext_log_prob,
        feed_dict={
            self._cur_sent_vec: sent_vec,
            self._cur_abs_pos: abs_pos_embed,
            self._cur_rel_pos: rel_pos_embed,
            self._doc_repr: doc_repr,
            self._hist_summary: hist_summary
        })
