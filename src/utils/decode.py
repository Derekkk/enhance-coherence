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

  def __init__(self, tokens, log_prob, state):
    """Hypothesis constructor.

    Args:
      tokens: start tokens for decoding.
      log_prob: log prob of the start tokens, usually 1.
      state: decoder initial states.
    """
    self.tokens = tokens
    self.log_prob = log_prob
    self.state = state

  def Extend(self, token, log_prob, new_state):
    """Extend the hypothesis with result from latest step.

    Args:
      token: latest token from decoding.
      log_prob: log prob of the latest decoded tokens.
      new_state: decoder output state. Fed to the decoder for next step.
    Returns:
      New Hypothesis with the results from latest step.
    """
    return Hypothesis(self.tokens + [token], self.log_prob + log_prob,
                      new_state)

  @property
  def latest_token(self):
    return self.tokens[-1]

  def __str__(self):
    return ('Hypothesis(log prob = %.4f, tokens = %s)' % (self.log_prob,
                                                          self.tokens))


class BeamSearch(object):
  """Beam search."""

  def __init__(self, model, beam_size, start_token, end_token, unk_token,
               max_steps):
    """Creates BeamSearch object.

    Args:
      model: Seq2SeqAttentionModel.
      beam_size: int.
      start_token: int, id of the token to start decoding with
      end_token: int, id of the token that completes an hypothesis
      max_steps: int, upper limit on the size of the hypothesis
    """
    self._model = model
    self._beam_size = beam_size
    self._start_token = start_token
    self._end_token = end_token
    self._max_steps = max_steps
    self._unk_token = unk_token

  def BeamSearch(self, sess, enc_input, enc_seqlen):
    """Performs beam search for decoding.

    Args:
      sess: tf.Session, session
      enc_input: ndarray of shape (1, enc_length), the document ids to encode
      enc_seqlen: ndarray of shape (1), the length of the sequnce

    Returns:
      hyps: list of Hypothesis, the best hypotheses found by beam search,
          ordered by score
    """

    # Repeat the input and length by beam_size times
    enc_batch = np.tile(enc_input, (self._beam_size, 1))
    enc_lens = np.tile(enc_seqlen, self._beam_size)

    # Run the encoder and extract the outputs and final state.
    enc_out_states, dec_in_state = self._model.encode_top_state(
        sess, enc_batch, enc_lens)
    # Replicate the initial states K times for the first step.
    hyps = [Hypothesis([self._start_token], 0.0, dec_in_state)
           ] * self._beam_size
    results = []

    steps = 0
    while steps < self._max_steps and len(results) < self._beam_size:
      latest_tokens = [h.latest_token for h in hyps]
      states = [h.state for h in hyps]

      topk_ids, topk_log_probs, new_states = self._model.decode_topk(
          sess, latest_tokens, enc_out_states, states)

      # The first step takes the best K results from first hyps. Following
      # steps take the best K results from K*K hyps.
      num_beam_source = 1 if steps == 0 else len(hyps)

      # Extend each hypothesis.
      all_hyps = []
      for i in xrange(num_beam_source):
        h, ns = hyps[i], new_states[i]
        for j in xrange(topk_ids.shape[1]):
          if topk_ids[i, j] != self._unk_token:
            all_hyps.append(h.Extend(topk_ids[i, j], topk_log_probs[i, j], ns))

      # Filter and collect any hypotheses that have the end token.
      hyps = []
      for h in self._BestHyps(all_hyps):
        if h.latest_token == self._end_token:
          # Pull the hypothesis off the beam if the end token is reached.
          results.append(h)
        else:
          # Otherwise continue to the extend the hypothesis.
          hyps.append(h)
        if len(hyps) == self._beam_size or len(results) == self._beam_size:
          break

      steps += 1

    if steps == self._max_steps:
      results.extend(hyps)

    return self._BestHyps(results, normalize_by_length=True)

  def _BestHyps(self, hyps, normalize_by_length=False):
    """Sort the hyps based on log probs and length.

    Args:
      hyps: A list of hypothesis.
    Returns:
      hyps: A list of sorted hypothesis in reverse log_prob order.
    """
    # This length normalization is only effective for the final results.
    if normalize_by_length:
      return sorted(
          hyps, key=lambda h: h.log_prob / len(h.tokens), reverse=True)
    else:
      return sorted(hyps, key=lambda h: h.log_prob, reverse=True)


class DecodeIO(object):
  """Writes the decoded and references to RKV files for Rouge score.

    See nlp/common/utils/internal/rkv_parser.py for detail about rkv file.
  """

  def __init__(self, outdir):
    self._cnt = 0
    self._outdir = outdir
    if not os.path.exists(self._outdir):
      os.makedirs(self._outdir)
    self._ref_file = None
    self._dec_file = None

  def Write(self, reference, decode):
    """Writes the reference and decoded outputs to RKV files.

    Args:
      reference: The human (correct) result.
      decode: The machine-generated result
    """
    self._ref_file.write(reference + '\n')
    self._dec_file.write(decode + '\n')
    self._cnt += 1
    if self._cnt % DECODE_IO_FLUSH_INTERVAL == 0:
      self._ref_file.flush()
      self._dec_file.flush()

  def ResetFiles(self, step):
    """Resets the output files. Must be called once before Write()."""
    if self._ref_file:
      self._ref_file.close()
    if self._dec_file:
      self._dec_file.close()
    timestamp = int(time.time())

    ref_fn = os.path.join(self._outdir, 'ref_%d_%d' % (step, timestamp))
    dec_fn = os.path.join(self._outdir, 'dec_%d_%d' % (step, timestamp))
    self._ref_file = open(ref_fn, 'w')
    self._dec_file = open(dec_fn, 'w')

    return ref_fn, dec_fn


class BSDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batch_reader, hps, vocab):
    """Beam search decoding.

    Args:
      model: The seq2seq attentional model.
      batch_reader: The batch data reader.
      hps: Hyperparamters.
      vocab: Vocabulary
    """
    self._model = model
    self._model.build_graph()
    self._batch_reader = batch_reader
    self._hps = hps
    self._vocab = vocab

    self._saver = tf.train.Saver()
    self._decode_io = DecodeIO(FLAGS.decode_dir)

  def DecodeLoop(self):
    """Decoding loop for long running process."""
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    filenames = self._Decode(self._saver, sess)
    return filenames

  def _Decode(self, saver, sess):
    """Restore a checkpoint and decode it.

    Args:
      saver: Tensorflow checkpoint saver.
      sess: Tensorflow session.
    Returns:
      ref_fn: path of reference file.
      dec_fn: path of decode file.
    """
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

    # Name output files by number of train steps and time
    step = self._model.get_global_step(sess)
    ref_fn, dec_fn = self._decode_io.ResetFiles(step)

    unk_id = self._vocab.unk_id if FLAGS.exclude_unks else None

    bs = BeamSearch(self._model, self._hps.beam_size, self._vocab.start_id,
                    self._vocab.end_id, unk_id, self._hps.dec_timesteps)

    for next_batch in self._batch_reader:
      (enc_batch, _, _, enc_lens, _, origin_inputs, origin_outputs) = next_batch

      for i in xrange(self._hps.batch_size):
        enc_batch_i = enc_batch[i:i + 1]
        enc_lens_i = enc_lens[i]

        best_beam = bs.BeamSearch(sess, enc_batch_i, enc_lens_i)[0]

        decode_output = [int(t) for t in best_beam.tokens[1:]]
        self._DecodeBatch(origin_inputs[i], origin_outputs[i], decode_output)

    tf.logging.info(
        "Finished decoding into following files:\nreference=%s\ndecode=%s",
        ref_fn, dec_fn)
    return ref_fn, dec_fn

  def _DecodeBatch(self, article, abstract, output_ids):
    """Convert id to words and writing results.

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      output_ids: The abstract word ids output by machine.
    """

    try:
      end_idx = output_ids.index(self._vocab.end_id)  # first occurrence
      output_ids = output_ids[:end_idx]
    except:
      pass  # do nothing when end_id is missing in output_ids
    decoded_output = ' '.join(self._vocab.GetWords(output_ids))

    tf.logging.info('article:  %s', article)
    tf.logging.info('abstract: %s', abstract)
    tf.logging.info('decoded:  %s', decoded_output)
    self._decode_io.Write(abstract, decoded_output.strip())


class SummaRuNNerDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, hps):
    """Beam search decoding.

    Args:
      model: The seq2seq attentional model.
      hps: Hyperparamters.
    """
    self._model = model
    self._model.build_graph()
    self._hps = hps

  def Decode(self, batch_reader, extract_topk):
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

    # Name output files by number of train steps and time
    model = self._model
    hps = self._hps

    result_list = []
    # Run decoding for data samples
    for next_batch in batch_reader:
      document_sents = next_batch.origin_inputs
      summary_sents = next_batch.origin_outputs
      doc_lens = next_batch.enc_doc_lens

      probs = model.get_extract_probs(sess, next_batch)

      for i in xrange(hps.batch_size):
        doc_len = doc_lens[i]
        probs_i = probs[i, :].tolist()[:doc_len]
        decoded_str = self._DecodeTopK(document_sents[i], probs_i, extract_topk)
        summary_str = sentence_sep.join(summary_sents[i])

        result_list.append(" ".join(
            [sys_tag, decoded_str, ref_tag, summary_str]) + "\n")

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
    extracted_sents = [document[i] for i in topk_ids]
    decoded_output = sentence_sep.join(extracted_sents)

    return decoded_output
