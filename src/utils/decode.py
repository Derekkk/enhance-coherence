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
"""Module for decoding."""

import os
import sys
import time
import numpy as np
import tensorflow as tf
# import pdb

FLAGS = tf.app.flags.FLAGS

DECODE_IO_FLUSH_INTERVAL = 30
sentence_sep = "</s>"
sys_tag = "<system>"
ref_tag = "<reference>"


def arg_topk(values, k):
  topk_idx = np.argsort(values)[-k:].tolist()
  return sorted(topk_idx)


class Hypothesis(object):
  """Defines a hypothesis during beam search."""

  def __init__(self, extracts, log_prob, hist_summary):
    """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens, usually 1.
      state: decoder initial states.
    """
    self.extracts = extracts
    self.log_prob = log_prob
    self.hist_summary = hist_summary

  def extend(self, extract, log_prob, sent_vec):
    """Extend the hypothesis with result from latest step.

    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
    if extract == 0:
      return Hypothesis(self.extracts + [extract], self.log_prob + log_prob,
                        self.hist_summary)
    else:
      return Hypothesis(self.extracts + [extract], self.log_prob + log_prob,
                        self.hist_summary + sent_vec)

  @property
  def extract_ids(self):
    return [i for i, x in enumerate(self.extracts) if x]

  def __str__(self):
    return 'Hypothesis(log prob = %.4f, extract_ids = [%s])' % (
        self.log_prob, ", ".join([str(x) for x in self.extract_ids]))


class BeamSearch(object):
  """Beam search."""

  def __init__(self, model, beam_size):
    """Creates BeamSearch object.

    Args:
      model: SummaRuNNerRF model.
      beam_size: int.
    """
    self._model = model
    self._beam_size = beam_size

  def beam_search(self, sess, enc_input, enc_doc_len, enc_sent_len,
                  sent_rel_pos):
    """Performs beam search for decoding.

    Args:
      sess: tf.Session, session
      enc_input: ndarray of shape (1, enc_), the document ids to encode
      enc_seqlen: ndarray of shape (1), the length of the sequnce

    Returns:
      hyps: list of Hypothesis, the best hypotheses found by beam search,
          ordered by score
    """
    beam_size = self._beam_size
    model = self._model

    # # Repeat the inputs by beam_size times
    # enc_batch = np.tile(enc_input, (beam_size, 1, 1))
    # enc_doc_lens = np.tile(enc_doc_len, beam_size)
    # enc_sent_lens = np.tile(enc_sent_len, (beam_size, 1))
    # sent_rel_poses = np.tile(sent_rel_pos, (beam_size, 1))

    # Run the encoder and extract the outputs and final state.
    sent_vecs, abs_pos_embs, rel_pos_embs, doc_repr = model.decode_get_feats(
        sess, enc_input, enc_doc_len, enc_sent_len, sent_rel_pos)
    # NB: sent_vecs.shape=[num_sentences, 1, enc_num_hidden*2]
    # NB: abs_pos_embs.shape=[num_sentences, 1, pos_emb_dim]
    # NB: rel_pos_embs.shape=[1, num_sentences, pos_emb_dim]
    # NB: doc_repr.shape=[1, enc_num_hidden*2]

    # Replicate the initialized hypothesis for the first step.
    hyps = [Hypothesis(
        [], 0.0, np.zeros_like(sent_vecs[0, :, :]))]  # [1, enc_num_hidden*2]
    results = []
    max_steps = enc_doc_len[0]

    for i in xrange(max_steps):
      hyps_len = len(hyps)
      sent_vec_i = sent_vecs[i, :, :]  # [1, enc_num_hidden*2]
      cur_sent_vec = np.tile(sent_vec_i, (hyps_len, 1))
      cur_abs_pos = np.tile(abs_pos_embs[i, :, :], (hyps_len, 1))
      cur_rel_pos = np.tile(rel_pos_embs[:, i, :], (hyps_len, 1))
      cur_doc_repr = np.tile(doc_repr, (hyps_len, 1))
      cur_hist_sum = np.concatenate([h.hist_summary for h in hyps], axis=0)

      ext_log_probs = model.decode_log_probs(sess, cur_sent_vec, cur_abs_pos,
                                             cur_rel_pos, cur_doc_repr,
                                             cur_hist_sum)

      # Extend each hypothesis.
      all_hyps = []
      for j, h in enumerate(hyps):
        all_hyps.append(h.extend(0, ext_log_probs[j, 0], sent_vec_i))
        all_hyps.append(h.extend(1, ext_log_probs[j, 1], sent_vec_i))

      hyps = self._best_hyps(all_hyps)[:beam_size]

    return hyps

  def _best_hyps(self, hyps):
    """Sort the hyps based on log probs.

    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    return sorted(hyps, key=lambda h: h.log_prob, reverse=True)


class SummaRuNNerRFDecoder(object):
  """Beam search decoder for SummaRuNNerRF."""

  def __init__(self, model, beam_size):
    """Beam search decoding.

    Args:
      model: the model object.
      beam_size: int.
    """
    self._model = model
    self._model.build_graph()
    self._beam_size = beam_size

  def decode(self, batch_reader):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()

    # Restore the saved checkpoint model
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.ckpt_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      tf.logging.info('No model to decode at %s', FLAGS.ckpt_root)
      return False

    tf.logging.info('Checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(FLAGS.ckpt_root,
                             os.path.basename(ckpt_state.model_checkpoint_path))
    tf.logging.info('Renamed checkpoint path %s', ckpt_path)
    saver.restore(sess, ckpt_path)

    result_list = []
    bs = BeamSearch(self._model, self._beam_size)

    for next_batch in batch_reader:
      enc_batch, enc_doc_lens, enc_sent_lens, sent_rel_pos, _, _, others = next_batch

      for i in xrange(enc_batch.shape[0]):
        enc_batch_i = enc_batch[i:i + 1]
        enc_doc_len_i = enc_doc_lens[i:i + 1]
        enc_sent_len_i = enc_sent_lens[i:i + 1]
        sent_rel_pos_i = sent_rel_pos[i:i + 1]

        best_beam = bs.beam_search(sess, enc_batch_i, enc_doc_len_i,
                                   enc_sent_len_i, sent_rel_pos_i)[0]

        doc_sents = others[0][i].split(sentence_sep)
        decoded_sents = [doc_sents[x] for x in best_beam.extract_ids]
        decoded_str = sentence_sep.join(decoded_sents)
        summary_str = others[1][i]

        result_list.append(" ".join(
            [sys_tag, decoded_str, ref_tag, summary_str]) + "\n")

    # Name output files by number of train steps and time
    decode_dir = FLAGS.decode_dir
    if not os.path.exists(decode_dir):
      os.makedirs(decode_dir)
    step = self._model.get_global_step(sess)
    timestep = int(time.time())
    output_fn = os.path.join(decode_dir, 'iter_%d_%d' % (step, timestep))

    with open(output_fn, 'w') as f:
      f.writelines(result_list)
    tf.logging.info('Outputs written to %s', output_fn)

    return output_fn


class TopKDecoder(object):
  """Top-k decoder."""

  def __init__(self, model, batch_size):
    """Beam search decoding.

    Args:
      model: the model object.
      batch_size: batch size.
    """
    self._model = model
    self._model.build_graph()
    self._bs = batch_size

  def decode(self, batch_reader, extract_topk):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    saver = tf.train.Saver()

    # Restore the saved checkpoint model
    ckpt_state = tf.train.get_checkpoint_state(FLAGS.ckpt_root)
    if not (ckpt_state and ckpt_state.model_checkpoint_path):
      raise ValueError("No model to decode at %s" % FLAGS.ckpt_root)

    tf.logging.info('Checkpoint path %s', ckpt_state.model_checkpoint_path)
    ckpt_path = os.path.join(FLAGS.ckpt_root,
                             os.path.basename(ckpt_state.model_checkpoint_path))
    saver.restore(sess, ckpt_path)

    model = self._model
    result_list = []
    # pdb.set_trace()
    # Run decoding for data samples
    for next_batch in batch_reader:
      document_strs = next_batch.others[0]
      summary_strs = next_batch.others[1]
      doc_lens = next_batch.enc_doc_lens

      probs = model.get_extract_probs(sess, next_batch)

      for i in xrange(self._bs):
        doc_len = doc_lens[i]
        probs_i = probs[i, :].tolist()[:doc_len]
        decoded_str = self._DecodeTopK(document_strs[i], probs_i, extract_topk)
        summary_str = summary_strs[i]

        result_list.append(" ".join(
            [sys_tag, decoded_str, ref_tag, summary_str]) + "\n")

    # Name output files by number of train steps and time
    decode_dir = FLAGS.decode_dir
    if not os.path.exists(decode_dir):
      os.makedirs(decode_dir)
    step = model.get_global_step(sess)
    timestep = int(time.time())
    output_fn = os.path.join(decode_dir, 'iter_%d_%d' % (step, timestep))

    with open(output_fn, 'w') as f:
      f.writelines(result_list)
    tf.logging.info('Outputs written to %s', output_fn)

    return output_fn

  def _DecodeTopK(self, document, probs, top_k=3):
    """Convert id to words and writing results.

    Args:
      document: a list of original document sentences.
      probs: probabilities of extraction.
      top_k: number of sentence extracted.
    """
    topk_ids = arg_topk(probs, top_k)
    doc_sents = document.split(sentence_sep)
    decoded_output = sentence_sep.join([doc_sents[i] for i in topk_ids])

    return decoded_output
