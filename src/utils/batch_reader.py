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
"""Batch reader to seq2seq attention model, with bucketing support."""

from collections import namedtuple
import Queue
from random import shuffle
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import glob
import re
import math
# import pdb

AbsModelInput = namedtuple('AbsModelInput',
                           'enc_input dec_input target enc_doc_len '
                           'enc_sent_len sent_rel_pos dec_len '
                           'origin_input origin_output')

AbsModelBatch = namedtuple('AbsModelBatch',
                           'enc_batch dec_batch target_batch enc_doc_lens '
                           'enc_sent_lens sent_rel_pos dec_lens '
                           'origin_inputs origin_outputs')

ExtModelInput = namedtuple('ExtModelInput',
                           'enc_input, enc_doc_len, enc_sent_len,'
                           'sent_rel_pos, extract_target, target_weight,'
                           'origin_input')

ExtModelBatch = namedtuple('ExtModelBatch',
                           'enc_batch, enc_doc_lens, enc_sent_lens,'
                           'sent_rel_pos, extract_targets, target_weights,'
                           'origin_inputs')

QUEUE_NUM_BATCH = 100  # Number of batches kept in the queue
BUCKET_NUM_BATCH = 10  # Number of batches per bucketing iteration fetches
GET_TIMEOUT = 60

# Sentence and paragraph separator
sentence_sep = "<s>"
para_sep = "</para>"
sent_para_sep = "<s>|</para>"
# Field separators for abstractive batcher
summary_key = "<summary>"
content_key = "<content>"
sum_sent_count_key = "<sum_sent_count>"
sum_sent_lens_key = "<sum_sent_lens>"
content_sent_count_key = "<content_sent_count>"
content_sent_lens_key = "<content_sent_lens>"
abs_key_list = [
    summary_key, content_key, sum_sent_count_key, sum_sent_lens_key,
    content_sent_count_key, content_sent_lens_key
]
# Field separators for extractive batcher
summary_key = "<summary_ids>"
count_key = "<count>"
ret_sum_key = "<ret_summary>"
ret_sum_count_key = "<ret_sum_count>"
ext_key_list_1 = [content_key, summary_key, count_key]
ext_key_list_23 = [
    content_key, summary_key, count_key, ret_sum_key, ret_sum_count_key
]


class Batcher(object):
  """Batch reader with shuffling and bucketing support."""

  def __init__(self,
               data_path,
               enc_vocab,
               dec_vocab,
               hps,
               field_key_list=abs_key_list,
               bucketing=False,
               truncate_input=True,
               num_epochs=None,
               shuffle_batches=True):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      enc_vocab: Encoder vocabulary.
      dec_vocab: Decoder vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      bucketing: Whether bucket inputs of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
      shuffle_batches: True if the examples would be randomly shuffled.
    """
    if not data_path:
      raise ValueError("Data path must be specified.")
    self._data_path = data_path
    self._enc_vocab = enc_vocab
    self._dec_vocab = dec_vocab
    self._hps = hps
    self._field_key_list = field_key_list
    self._bucketing = bucketing
    self._truncate_input = truncate_input
    self._num_epochs = num_epochs
    self._shuffle_batches = shuffle_batches

    self._input_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
    self._bucket_input_queue = Queue.Queue(QUEUE_NUM_BATCH)

    # Get data file list
    filelist = glob.glob(self._data_path)
    assert filelist, 'Empty filelist.'
    if self._shuffle_batches:
      shuffle(filelist)
    self._filelist = filelist

    # Create input reading threads
    self._input_threads = []
    for f in filelist:
      self._input_threads.append(Thread(target=self._FillInputQueue, args=(f,)))
      self._input_threads[-1].daemon = True
      self._input_threads[-1].start()

    # Create bucketing threads
    self._bucketing_threads = []
    for _ in xrange(max(1, len(filelist) / 2)):
      self._bucketing_threads.append(Thread(target=self._FillBucketInputQueue))
      self._bucketing_threads[-1].daemon = True
      self._bucketing_threads[-1].start()

    # Create watch threads
    if self._hps.mode == 'train':
      # Keep input threads running in train mode,
      # but they are not needed in eval and decode mode.
      self._watch_thread = Thread(target=self._WatchThreads)
      self._watch_thread.daemon = True
      self._watch_thread.start()

  def __iter__(self):
    return self

  def next(self):
    """Returns a batch of inputs for seq2seq attention model.

        Returns:
          batch: a AbsModelBatch object.
        """

    try:
      batch = self._bucket_input_queue.get(timeout=GET_TIMEOUT)
      # batch = self._bucket_input_queue.get_nowait()
    except Queue.Empty as e:
      raise StopIteration('bucket_input_queue.get() timeout: %s' % e)

    return batch

  def _FillInputQueue(self, data_path):
    """Fill input queue with AbsModelInput."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    dec_vocab = self._dec_vocab
    enc_pad_id = enc_vocab.pad_id
    dec_start_id = dec_vocab.start_id
    dec_end_id = dec_vocab.end_id
    enc_empty_sent = [enc_pad_id] * hps.num_words_sent
    rel_pos_max_float = float(hps.rel_pos_max_idx - 1)

    example_gen = self._TextGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for example_tuple in example_gen:
      (summary_str, content_str, _, _, _, _) = example_tuple

      # Content as enc_input
      content_sents = re.split(sent_para_sep, content_str)
      enc_input = [(i, enc_vocab.GetIds(s))
                   for i, s in enumerate(content_sents)]

      # Summary as dec_input
      summary_str = summary_str.replace(sentence_sep, "")
      summary_ids = dec_vocab.GetIds(summary_str)
      # Use SENTENCE_START as the <GO> symbol for decoder inputs.
      dec_input = [dec_start_id] + summary_ids

      # Filter out too-short input
      enc_input = [(i, s) for i, s in enc_input
                   if len(s) > hps.min_num_words_sent]
      if (len(enc_input) < hps.min_num_input_sents or
          len(dec_input) < hps.min_output_len):
        continue

      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        enc_input = [(i, s) for i, s in enc_input
                     if len(s) <= hps.num_words_sent]
        if (len(enc_input) > hps.num_sentences or
            len(dec_input) > hps.dec_timesteps):
          continue
      # If we are truncating input, do so if necessary
      else:
        if len(enc_input) > hps.num_sentences:
          enc_input = enc_input[:hps.num_sentences]
        enc_input = [(i, s[:hps.num_words_sent]) for i, s in enc_input]

        if len(dec_input) > hps.dec_timesteps:
          dec_input = dec_input[:hps.dec_timesteps]

      # Now enc_input should fit in 2-D matrix [num_sentences, num_words_sent]
      # and len(dec_input) <= dec_timesteps.
      enc_sent_len = [len(s) for i, s in enc_input]
      enc_doc_len = len(enc_input)
      dec_len = len(dec_input)

      # Compute the relative position. 0 is reserved for padding.
      rel_pos_coef = rel_pos_max_float / enc_doc_len
      sent_rel_pos = [int(i * rel_pos_coef) + 1 for i in range(enc_doc_len)]

      # Pad enc_input if necessary
      padded_enc_input = [
          s + [enc_pad_id] * (hps.num_words_sent - l)
          for (i, s), l in zip(enc_input, enc_sent_len)
      ]
      padded_enc_input += [enc_empty_sent] * (hps.num_sentences - enc_doc_len)
      np_enc_input = np.array(padded_enc_input, dtype=np.int32)

      # Pad dec_input and target if necessary
      dec_input += [dec_end_id] * (hps.dec_timesteps - dec_len)
      # target is dec_input without <GO> at beginning, plus end id at the end
      target = dec_input[1:] + [dec_end_id]
      np_dec_input = np.array(dec_input, dtype=np.int32)
      np_target = np.array(target, dtype=np.int32)

      # Pad the lengths
      pad_enc_sent_len = enc_sent_len + [0] * (hps.num_sentences - enc_doc_len)
      padded_rel_pos = sent_rel_pos + [0] * (hps.num_sentences - enc_doc_len)
      np_enc_sent_len = np.array(pad_enc_sent_len, dtype=np.int32)
      np_rel_pos = np.array(padded_rel_pos, dtype=np.int32)

      # Get the filtered content sentences
      filt_content_sents = [content_sents[i] for i, s in enc_input]

      element = AbsModelInput(np_enc_input, np_dec_input, np_target,
                              enc_doc_len, np_enc_sent_len, np_rel_pos, dec_len,
                              filt_content_sents, summary_str)
      self._input_queue.put(element)

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    hps = self._hps
    while True:
      inputs = []
      for _ in xrange(hps.batch_size * BUCKET_NUM_BATCH):
        inputs.append(self._input_queue.get())

      if self._bucketing:
        inputs = sorted(inputs, key=lambda inp: inp.enc_doc_len)

      batches = []
      for i in xrange(0, len(inputs), hps.batch_size):
        batches.append(inputs[i:i + hps.batch_size])

      if self._shuffle_batches:
        shuffle(batches)

      for b in batches:
        self._bucket_input_queue.put(self._PackBatch(b))

  def _WatchThreads(self):
    """Watch the daemon input threads and restart if dead."""
    while True:
      time.sleep(60)
      input_threads = []
      for i, t in enumerate(self._input_threads):
        if t.is_alive():
          input_threads.append(t)
        else:
          tf.logging.error('Found input thread dead.')
          new_t = Thread(target=self._FillInputQueue, args=(self._filelist[i],))
          input_threads.append(new_t)
          input_threads[-1].daemon = True
          input_threads[-1].start()
      self._input_threads = input_threads

      bucketing_threads = []
      for t in self._bucketing_threads:
        if t.is_alive():
          bucketing_threads.append(t)
        else:
          tf.logging.error('Found bucketing thread dead.')
          new_t = Thread(target=self._FillBucketInputQueue)
          bucketing_threads.append(new_t)
          bucketing_threads[-1].daemon = True
          bucketing_threads[-1].start()
      self._bucketing_threads = bucketing_threads

  def _TextGenerator(self, path, num_epochs=None):
    """Generates input and output text."""
    epoch = 0
    while True:
      if num_epochs is not None and epoch >= num_epochs:
        return

      f = open(path, 'r')
      for line in f:
        e = line.strip()
        if not e:
          continue
        try:
          example_tuple = self._ExtractExample(e)
        except ValueError:
          tf.logging.error('Failed to read fields from example')
          continue
        yield example_tuple

      epoch += 1

  def _ExtractExample(self, ex):
    """Extract text for a feature from tf.Example.

        Args:
          ex: string of one example.
        Returns:
          input, output: a pair of input and output string extracted.
        """
    field_key_list = self._field_key_list
    idx_list = [ex.index(k) for k in field_key_list]
    for i in xrange(len(idx_list) - 1):
      assert idx_list[i] < idx_list[i + 1]

    field_list = []
    for i in xrange(len(field_key_list) - 1):
      field_str = ex[idx_list[i] + len(field_key_list[i]):idx_list[i +
                                                                   1]].strip()
      field_list.append(field_str)
    last_field = ex[idx_list[-1] + len(field_key_list[-1]):].strip()
    field_list.append(last_field)

    return tuple(field_list)

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

        Returns:
            model_batch: AbsModelBatch
        """
    hps = self._hps
    field_lists = [[], [], [], [], [], [], []]
    origin_inputs, origin_outputs = [], []

    for ex in batch:
      # (enc_input, dec_input, target, enc_doc_len, enc_sent_len, sent_rel_pos,
      #  dec_len, content_str, summary_str) = ex
      for i in range(7):
        field_lists[i].append(ex[i])

      origin_inputs.append(ex[-2])
      origin_outputs.append(ex[-1])

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return AbsModelBatch(stacked_fields[0], stacked_fields[1],
                         stacked_fields[2], stacked_fields[3],
                         stacked_fields[4], stacked_fields[5],
                         stacked_fields[6], origin_inputs, origin_outputs)


class ExtractiveBatcher(Batcher):
  """Batch reader for extractive summarization data."""

  def __init__(self,
               data_path,
               enc_vocab,
               hps,
               read_mode=1,
               bucketing=False,
               truncate_input=True,
               num_epochs=None,
               shuffle_batches=True):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      enc_vocab: Encoder vocabulary.
      hps: Seq2SeqAttention model hyperparameters.
      read_mode: mode of data reading, 1 for raw only, 2 for post-processing
        result only, 3 for both.
      bucketing: Whether bucket inputs of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
      shuffle_batches: True if the examples would be randomly shuffled.
    """
    if read_mode == 1:
      key_list = ext_key_list_1
    elif read_mode == 2 or read_mode == 3:
      key_list = ext_key_list_23
    else:
      raise ValueError("Invalid read mode.")
    self._read_mode = read_mode

    super(ExtractiveBatcher,
          self).__init__(data_path, enc_vocab, None, hps, key_list, bucketing,
                         truncate_input, num_epochs, shuffle_batches)

  def _FillInputQueue(self, data_path):
    """Fill input queue with ExtModelInput."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    enc_pad_id = enc_vocab.pad_id
    enc_empty_sent = [enc_pad_id] * hps.num_words_sent
    rel_pos_max_float = float(hps.rel_pos_max_idx - 1)

    example_gen = self._TextGenerator(data_path, self._num_epochs)
    read_mode = self._read_mode

    # pdb.set_trace()
    for example_tuple in example_gen:
      if read_mode == 1:
        (content_str, summary_ids_str, count_str) = example_tuple
      elif read_mode == 2:
        (content_str, _, _, summary_ids_str2, count_str2) = example_tuple
      elif read_mode == 3:
        (content_str, summary_ids_str, count_str, summary_ids_str2,
         count_str2) = example_tuple

      # Content as enc_input
      content_sents = re.split(sent_para_sep, content_str)
      enc_input = [(i, enc_vocab.GetIds(s))
                   for i, s in enumerate(content_sents)]

      # Filter out too-short input
      if len(enc_input) < hps.min_num_input_sents:
        continue

      # If we're not truncating input, throw out too-long input
      if not self._truncate_input:
        if len(enc_input) > hps.num_sentences:
          continue
      else:
        # If we are truncating input, do so if necessary
        if len(enc_input) > hps.num_sentences:
          enc_input = enc_input[:hps.num_sentences]
        enc_input = [(i, s[:hps.num_words_sent]) for i, s in enc_input]

      # Now enc_input should fit in 2-D matrix [num_sentences, num_words_sent]
      enc_sent_len = [len(s) for i, s in enc_input]
      enc_doc_len = len(enc_input)

      # Compute the relative position. 0 is reserved for padding.
      rel_pos_coef = rel_pos_max_float / enc_doc_len
      sent_rel_pos = [int(i * rel_pos_coef) + 1 for i in range(enc_doc_len)]

      # Pad enc_input if necessary
      padded_enc_input = [
          s + [enc_pad_id] * (hps.num_words_sent - l)
          for (i, s), l in zip(enc_input, enc_sent_len)
      ]
      padded_enc_input += [enc_empty_sent] * (hps.num_sentences - enc_doc_len)
      np_enc_input = np.array(padded_enc_input, dtype=np.int32)

      # Pad the input lengths and positions
      pad_enc_sent_len = enc_sent_len + [0] * (hps.num_sentences - enc_doc_len)
      padded_rel_pos = sent_rel_pos + [0] * (hps.num_sentences - enc_doc_len)
      np_enc_sent_len = np.array(pad_enc_sent_len, dtype=np.int32)
      np_rel_pos = np.array(padded_rel_pos, dtype=np.int32)

      # Get indices of the summary sentences, and corresponding weights
      summary_ids, counts = [], []
      if read_mode == 1 or read_mode == 3:
        summary_ids += [
            int(a) for a in re.split(r"[\(\s\)]", summary_ids_str) if a
        ]
        counts += [int(a) for a in count_str.split()]

      if read_mode == 2 or read_mode == 3:
        summary_ids2 = [a for a in re.split(r"[\(\s\)]", summary_ids_str2) if a]
        counts2 = [int(a) for a in count_str2.split()]
        for s, c in zip(summary_ids2, counts2):
          ids = [int(x) for x in s.split(",")]
          summary_ids += ids
          counts += [c] * len(ids)

      if len(summary_ids) == 0:
        continue  # Skip those with no summaries

      np_target = np.zeros([hps.num_sentences], dtype=np.int32)
      for i in summary_ids:
        if i < hps.num_sentences:
          np_target[i] = 1

      if hps.trg_weight_norm > 0:
        total_count = float(sum(counts))
        weight_norm = hps.trg_weight_norm / (total_count + 0.01)
        weights = [weight_norm * c for c in counts]  # normalize the weights

        # Convert to numpy vector and pad with 0s or 1s
        np_weight_sum = np.zeros([hps.num_sentences], dtype=np.float32)
        for i, w in zip(summary_ids, weights):
          if i < hps.num_sentences:
            np_weight_sum[i] += w

        np_weights = np.ones([hps.num_sentences], dtype=np.float32)
        for i in xrange(hps.num_sentences):
          if np_target[i] == 1:
            np_weights[i] = np_weight_sum[i]

      else:
        np_weights = np.ones([hps.num_sentences], dtype=np.float32)

      # Get the filtered content sentences
      filt_content_sents = [content_sents[i] for i, s in enc_input]

      element = ExtModelInput(np_enc_input, enc_doc_len, np_enc_sent_len,
                              np_rel_pos, np_target, np_weights,
                              filt_content_sents)
      self._input_queue.put(element)

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

        Returns:
            model_batch: ExtModelBatch
    """
    hps = self._hps
    field_lists = [[], [], [], [], [], []]
    origin_inputs = []

    for ex in batch:
      for i in range(6):
        field_lists[i].append(ex[i])
      origin_inputs.append(ex[-1])

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return ExtModelBatch(stacked_fields[0], stacked_fields[1],
                         stacked_fields[2], stacked_fields[3],
                         stacked_fields[4], stacked_fields[5], origin_inputs)
