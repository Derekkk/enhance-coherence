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

import Queue
from random import shuffle, sample, randint
from threading import Thread
import time
import numpy as np
import tensorflow as tf
import glob
import cPickle as pkl
from collections import namedtuple
# import pdb

ExtractiveSample = namedtuple('ExtractiveSample',
                              'enc_input, enc_doc_len, enc_sent_len,'
                              'sent_rel_pos, extract_target, target_weight,'
                              'origin_input origin_output')

ExtractiveBatch = namedtuple('ExtractiveBatch',
                             'enc_batch, enc_doc_lens, enc_sent_lens,'
                             'sent_rel_pos, extract_targets, target_weights,'
                             'others')  # others=(origin_inputs,origin_outputs)

# Note(jimmycode): refer to run.py for definition of DocSummary, DocSummaryCount

QUEUE_NUM_BATCH = 100  # Number of batches kept in the queue
BUCKET_NUM_BATCH = 10  # Number of batches per bucketing iteration fetches
GET_TIMEOUT = 120

def neg_samp_pairs(sequence, size=0):
  if len(sequence) < 2:
    return []
  elif len(sequence) == 2:
    return [(sequence[1], sequence[0])]

  if not size:
    size = len(sequence)

  neg_samples_set = set()
  seq_end_idx = len(sequence) - 1
  while len(neg_samples_set) < size:
    id_1 = randint(0, seq_end_idx)
    id_2 = randint(0, seq_end_idx)
    if id_1 != id_2 and id_1 + 1 != id_2:
      neg_samples_set.add((id_1, id_2))

  neg_samples = [(sequence[i], sequence[j]) for i, j in neg_samples_set]
  return neg_samples

class ExtractiveBatcher(object):
  """Batch reader for extractive summarization data."""

  def __init__(self,
               data_path,
               enc_vocab,
               hps,
               bucketing=False,
               truncate_input=True,
               num_epochs=None,
               shuffle_batches=True):
    """Batcher constructor.

    Args:
      data_path: tf.Example filepattern.
      enc_vocab: Encoder vocabulary.
      hps: model hyperparameters.
      bucketing: Whether bucket inputs of similar length into the same batch.
      truncate_input: Whether to truncate input that is too long. Alternative is
        to discard such examples.
      shuffle_batches: True if the examples would be randomly shuffled.
    """
    if not data_path:
      raise ValueError("Data path must be specified.")
    self._data_path = data_path
    self._enc_vocab = enc_vocab
    self._hps = hps
    self._bucketing = bucketing
    self._truncate_input = truncate_input
    self._num_epochs = num_epochs
    self._shuffle_batches = shuffle_batches

    self._sample_queue = Queue.Queue(QUEUE_NUM_BATCH * self._hps.batch_size)
    self._batch_queue = Queue.Queue(QUEUE_NUM_BATCH)

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
      batch = self._batch_queue.get(timeout=GET_TIMEOUT)
    except Queue.Empty as e:
      raise StopIteration('batch_queue.get() timeout: %s' % e)

    return batch

  def _FillInputQueue(self, data_path):
    """Fill input queue with ExtractiveSample."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    enc_pad_id = enc_vocab.pad_id
    enc_empty_sent = [enc_pad_id] * hps.num_words_sent
    rel_pos_max_float = float(hps.rel_pos_max_idx - 1)

    data_generator = self._DataGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for data_sample in data_generator:
      document = data_sample.document
      extract_ids = data_sample.extract_ids

      # Content as enc_input
      enc_input = [enc_vocab.GetIds(s) for s in document]

      # Filter out too-short input
      if len(enc_input) < hps.min_num_input_sents:
        continue

      if self._truncate_input:
        enc_input = [
            s[:hps.num_words_sent] for s in enc_input[:hps.num_sentences]
        ]
      else:
        if len(enc_input) > hps.num_sentences:
          continue  # throw away too long inputs

      # Now enc_input should fit in 2-D matrix [num_sentences, num_words_sent]
      enc_sent_len = [len(s) for s in enc_input]
      enc_doc_len = len(enc_input)

      # Pad enc_input if necessary
      padded_enc_input = [
          s + [enc_pad_id] * (hps.num_words_sent - l)
          for s, l in zip(enc_input, enc_sent_len)
      ]
      padded_enc_input += [enc_empty_sent] * (hps.num_sentences - enc_doc_len)
      np_enc_input = np.array(padded_enc_input, dtype=np.int32)

      # Compute the relative position. 0 is reserved for padding.
      rel_pos_coef = rel_pos_max_float / enc_doc_len
      sent_rel_pos = [int(i * rel_pos_coef) + 1 for i in range(enc_doc_len)]

      # Pad the input lengths and positions
      pad_enc_sent_len = enc_sent_len + [0] * (hps.num_sentences - enc_doc_len)
      pad_rel_pos = sent_rel_pos + [0] * (hps.num_sentences - enc_doc_len)
      np_enc_sent_len = np.array(pad_enc_sent_len, dtype=np.int32)
      np_rel_pos = np.array(pad_rel_pos, dtype=np.int32)

      # Skip those with no extractive summaries
      if len(extract_ids) == 0:
        continue

      np_target = np.zeros([hps.num_sentences], dtype=np.int32)
      for i in extract_ids:
        if i < hps.num_sentences:
          np_target[i] = 1

      if hps.trg_weight_norm > 0:
        counts = data_sample.extract_counts
        total_count = float(sum(counts))
        weight_norm = hps.trg_weight_norm / (total_count + 0.01)
        weights = [weight_norm * c for c in counts]  # normalize the weights

        np_weights = np.ones([hps.num_sentences], dtype=np.float32)
        for i, w in zip(extract_ids, weights):
          if i < hps.num_sentences:
            np_weights[i] = w
      else:
        np_weights = np.ones([hps.num_sentences], dtype=np.float32)

      try:
        summary = data_sample.summary
      except:
        summary = [""]
      sample = ExtractiveSample(np_enc_input, enc_doc_len, np_enc_sent_len,
                                np_rel_pos, np_target, np_weights, document,
                                summary)
      self._sample_queue.put(sample)

  def _DataGenerator(self, path, num_epochs=None):
    """An (infinite) iterator that outputs data samples."""
    epoch = 0
    with open(path, 'r') as f:
      dataset = pkl.load(f)

    while True:
      if num_epochs is not None and epoch >= num_epochs:
        return

      for d in dataset:
        yield d

      epoch += 1

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    hps = self._hps
    while True:
      samples = []
      for _ in xrange(hps.batch_size * BUCKET_NUM_BATCH):
        samples.append(self._sample_queue.get())

      if self._bucketing:
        samples = sorted(samples, key=lambda inp: inp.enc_doc_len)

      batches = []
      for i in xrange(0, len(samples), hps.batch_size):
        batches.append(samples[i:i + hps.batch_size])

      if self._shuffle_batches:
        shuffle(batches)

      for b in batches:
        self._batch_queue.put(self._PackBatch(b))

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

    Returns:
        model_batch: ExtractiveBatch
    """
    hps = self._hps
    field_lists = [[], [], [], [], [], []]
    origin_inputs, origin_outputs = [], []

    for ex in batch:
      for i in range(6):
        field_lists[i].append(ex[i])
      origin_inputs.append(ex.origin_input)
      origin_outputs.append(ex.origin_output)

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return ExtractiveBatch(stacked_fields[0], stacked_fields[1],
                           stacked_fields[2], stacked_fields[3],
                           stacked_fields[4], stacked_fields[5],\
                           (origin_inputs, origin_outputs))

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


class SentencePairBatcher(ExtractiveBatcher):
  """Batch reader for sequence matching model."""

  def _FillInputQueue(self, data_path):
    """Fill input queue with ExtractiveSample."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    enc_pad_id = enc_vocab.pad_id
    enc_empty_sent = [enc_pad_id] * hps.num_words_sent
    rel_pos_max_float = float(hps.rel_pos_max_idx - 1)

    data_generator = self._DataGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for data_sample in data_generator:  # type(data_sample): DocSummary
      document = data_sample.document
      summary = data_sample.summary
      if not summary:
        continue

      # Step 1: sample positive/negative pairs
      positive_pairs = []

      # Positive 1 (25%): summary
      summary_length = len(summary)
      for i in range(summary_length - 1):
        positive_pairs.append((summary[i], summary[i+1]))

      # Positive 2 (25%): document
      doc_length = len(document)
      if doc_length > summary_length:
        sampled_ids = sample(range(doc_length - 1), summary_length)
        for i in sampled_ids:
          positive_pairs.append((document[i], document[i+1]))
        
      # Negative (50%)
      if summary_length > 1:
        summary_neg = neg_samp_pairs(summary)
      document_neg = neg_samp_pairs(document, len(positive_pairs) - len(summary_neg))
      negative_pairs = summary_neg + document_neg

      # Content as enc_input
      enc_input = [enc_vocab.GetIds(s) for s in document]

      # Filter out too-short input
      if len(enc_input) < hps.min_num_input_sents:
        continue

      if self._truncate_input:
        enc_input = [
            s[:hps.num_words_sent] for s in enc_input[:hps.num_sentences]
        ]
      else:
        if len(enc_input) > hps.num_sentences:
          continue  # throw away too long inputs

      # Now enc_input should fit in 2-D matrix [num_sentences, num_words_sent]
      enc_sent_len = [len(s) for s in enc_input]
      enc_doc_len = len(enc_input)

      # Pad enc_input if necessary
      padded_enc_input = [
          s + [enc_pad_id] * (hps.num_words_sent - l)
          for s, l in zip(enc_input, enc_sent_len)
      ]
      padded_enc_input += [enc_empty_sent] * (hps.num_sentences - enc_doc_len)
      np_enc_input = np.array(padded_enc_input, dtype=np.int32)

      # Compute the relative position. 0 is reserved for padding.
      rel_pos_coef = rel_pos_max_float / enc_doc_len
      sent_rel_pos = [int(i * rel_pos_coef) + 1 for i in range(enc_doc_len)]

      # Pad the input lengths and positions
      pad_enc_sent_len = enc_sent_len + [0] * (hps.num_sentences - enc_doc_len)
      pad_rel_pos = sent_rel_pos + [0] * (hps.num_sentences - enc_doc_len)
      np_enc_sent_len = np.array(pad_enc_sent_len, dtype=np.int32)
      np_rel_pos = np.array(pad_rel_pos, dtype=np.int32)

      # Skip those with no extractive summaries
      if len(extract_ids) == 0:
        continue

      np_target = np.zeros([hps.num_sentences], dtype=np.int32)
      for i in extract_ids:
        if i < hps.num_sentences:
          np_target[i] = 1

      if hps.trg_weight_norm > 0:
        counts = data_sample.extract_counts
        total_count = float(sum(counts))
        weight_norm = hps.trg_weight_norm / (total_count + 0.01)
        weights = [weight_norm * c for c in counts]  # normalize the weights

        np_weights = np.ones([hps.num_sentences], dtype=np.float32)
        for i, w in zip(extract_ids, weights):
          if i < hps.num_sentences:
            np_weights[i] = w
      else:
        np_weights = np.ones([hps.num_sentences], dtype=np.float32)

      try:
        summary = data_sample.summary
      except:
        summary = [""]
      sample = ExtractiveSample(np_enc_input, enc_doc_len, np_enc_sent_len,
                                np_rel_pos, np_target, np_weights, document,
                                summary)
      self._sample_queue.put(sample)

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    hps = self._hps
    while True:
      samples = []
      for _ in xrange(hps.batch_size * BUCKET_NUM_BATCH):
        samples.append(self._sample_queue.get())

      if self._bucketing:
        samples = sorted(samples, key=lambda inp: inp.enc_doc_len)

      batches = []
      for i in xrange(0, len(samples), hps.batch_size):
        batches.append(samples[i:i + hps.batch_size])

      if self._shuffle_batches:
        shuffle(batches)

      for b in batches:
        self._batch_queue.put(self._PackBatch(b))

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

    Returns:
        model_batch: ExtractiveBatch
    """
    hps = self._hps
    field_lists = [[], [], [], [], [], []]
    origin_inputs, origin_outputs = [], []

    for ex in batch:
      for i in range(6):
        field_lists[i].append(ex[i])
      origin_inputs.append(ex.origin_input)
      origin_outputs.append(ex.origin_output)

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return ExtractiveBatch(stacked_fields[0], stacked_fields[1],
                           stacked_fields[2], stacked_fields[3],
                           stacked_fields[4], stacked_fields[5],\
                           (origin_inputs, origin_outputs))

