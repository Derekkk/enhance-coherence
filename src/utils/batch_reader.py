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
import random
from random import shuffle, randint
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

SentencePair = namedtuple('SentencePair',
                          'sent_A, sent_B, length_A, length_B, label')
SentencePairBatch = namedtuple('SentencePairBatch',
                               'sents_A, sents_B, lengths_A, lengths_B,'
                               'labels')

SentenceTriplet = namedtuple('SentenceTriplet',
                             'sent_A, sent_B_pos, sent_B_neg,'
                             'length_A, length_B_pos, length_B_neg')
SentenceTripletBatch = namedtuple('SentenceTripletBatch',
                                  'sents_A, sents_B_pos, sents_B_neg,'
                                  'lengths_A, lengths_B_pos, lengths_B_neg')

SentenceBlock = namedtuple('SentenceBlock', 'sent_input, length, label')
SentenceBlockBatch = namedtuple('SentenceBlockBatch',
                                'sent_inputs, lengths, labels')

# Note(jimmycode): refer to run.py for definition of DocSummary, DocSummaryCount

QUEUE_NUM_BATCH = 100  # Number of batches kept in the queue
BUCKET_NUM_BATCH = 20  # Number of batches per bucketing iteration fetches
GET_TIMEOUT = 240
SAMPLE_PATIENCE = 10  # Maximum number of sample failures, to avoid infinite loop
sentence_sep = "</s>"


def neg_samp_pairs(sequence, size=0):
  if len(sequence) < 2:
    return []
  elif len(sequence) == 2:
    return [(sequence[1], sequence[0])]

  if not size:
    size = len(sequence)

  neg_samples_set = set()
  seq_end_idx = len(sequence) - 1
  fail_count = 0  # number of sample failures
  while len(neg_samples_set) < size and fail_count < SAMPLE_PATIENCE:
    id_1 = randint(0, seq_end_idx)
    id_2 = randint(0, seq_end_idx)

    if id_1 != id_2 and id_1 + 1 != id_2:
      pair = (id_1, id_2)
      if pair not in neg_samples_set:
        neg_samples_set.add(pair)
      else:
        fail_count += 1
    else:
      fail_count += 1

  neg_samples = [(sequence[i], sequence[j]) for i, j in neg_samples_set]
  return neg_samples


def sample_triplets(sequence, seq_len=1000, window_size=None):
  """
  Returns:
    triplets: a list of three-element tuples
  """
  triplets = []
  seq_len = min(seq_len, len(sequence))
  if not window_size:
    window_size = seq_len - 1  # full size

  for i in xrange(-1, seq_len):
    if i + 1 < seq_len:
      for j in xrange(i + 2, i + window_size + 2):
        if j < seq_len:
          triplets.append((i, i + 1, j))

      if i > 0:
        triplets.append((i, i + 1, i))

  return triplets


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
            s[:hps.num_words_sent] for s in enc_input[:hps.num_sents_doc]
        ]
      else:
        if len(enc_input) > hps.num_sents_doc:
          continue  # throw away too long inputs

      # Now enc_input should fit in 2-D matrix [num_sents_doc, num_words_sent]
      enc_sent_len = [len(s) for s in enc_input]
      enc_doc_len = len(enc_input)

      # Pad enc_input if necessary
      padded_enc_input = [
          s + [enc_pad_id] * (hps.num_words_sent - l)
          for s, l in zip(enc_input, enc_sent_len)
      ]
      padded_enc_input += [enc_empty_sent] * (hps.num_sents_doc - enc_doc_len)
      np_enc_input = np.array(padded_enc_input, dtype=np.int32)

      # Compute the relative position. 0 is reserved for padding.
      rel_pos_coef = rel_pos_max_float / enc_doc_len
      sent_rel_pos = [int(i * rel_pos_coef) + 1 for i in range(enc_doc_len)]

      # Pad the input lengths and positions
      pad_enc_sent_len = enc_sent_len + [0] * (hps.num_sents_doc - enc_doc_len)
      pad_rel_pos = sent_rel_pos + [0] * (hps.num_sents_doc - enc_doc_len)
      np_enc_sent_len = np.array(pad_enc_sent_len, dtype=np.int32)
      np_rel_pos = np.array(pad_rel_pos, dtype=np.int32)

      # Skip those with no extractive summaries
      if len(extract_ids) == 0:
        continue

      np_target = np.zeros([hps.num_sents_doc], dtype=np.int32)
      for i in extract_ids:
        if i < hps.num_sents_doc:
          np_target[i] = 1

      if hps.trg_weight_norm > 0:
        counts = data_sample.extract_counts
        total_count = float(sum(counts))
        weight_norm = hps.trg_weight_norm / (total_count + 0.01)
        weights = [weight_norm * c for c in counts]  # normalize the weights

        np_weights = np.ones([hps.num_sents_doc], dtype=np.float32)
        for i, w in zip(extract_ids, weights):
          if i < hps.num_sents_doc:
            np_weights[i] = w
      else:
        np_weights = np.ones([hps.num_sents_doc], dtype=np.float32)

      try:
        summary = data_sample.summary
      except:
        summary = []

      doc_str = sentence_sep.join(document)
      summary_str = sentence_sep.join(summary)

      element = ExtractiveSample(np_enc_input, enc_doc_len, np_enc_sent_len,
                                 np_rel_pos, np_target, np_weights, doc_str,
                                 summary_str)
      self._sample_queue.put(element)

  def _DataGenerator(self, path, num_epochs=None):
    """An (infinite) iterator that outputs data samples."""
    epoch = 0
    with open(path, 'r') as f:
      dataset = pkl.load(f)

    while True:
      if num_epochs is not None and epoch >= num_epochs:
        return
      if self._shuffle_batches:
        shuffle(dataset)

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
    np_origin_inputs = np.array(origin_inputs, dtype=np.str)
    np_origin_outputs = np.array(origin_outputs, dtype=np.str)

    return ExtractiveBatch(stacked_fields[0], stacked_fields[1],
                           stacked_fields[2], stacked_fields[3],
                           stacked_fields[4], stacked_fields[5],\
                           (np_origin_inputs, np_origin_outputs))

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
    data_generator = self._DataGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for data_sample in data_generator:  # type(data_sample): DocSummary
      document = data_sample.document
      summary = data_sample.summary
      if not summary:
        continue

      # Step 1: sample positive/negative pairs
      positive_pairs = []

      # Positive (50% * 1/3): summary
      summary_length = len(summary)
      for i in range(summary_length - 1):
        positive_pairs.append((summary[i], summary[i + 1]))

      # Positive (50% * 2/3): document
      doc_length = len(document)
      sample_size = min(2 * summary_length, doc_length - 1)
      sampled_ids = random.sample(range(doc_length - 1), sample_size)
      for i in sampled_ids:
        positive_pairs.append((document[i], document[i + 1]))

      # Negative (50%)
      if summary_length > 1:
        summary_neg = neg_samp_pairs(summary)
      document_neg = neg_samp_pairs(document,
                                    len(positive_pairs) - len(summary_neg))
      negative_pairs = summary_neg + document_neg

      # Combine them with labels, 1 for positive, 0 for negative
      all_pairs = []
      for p in positive_pairs:
        all_pairs.append((p[0], p[1], 1))
      for p in negative_pairs:
        all_pairs.append((p[0], p[1], 0))
      shuffle(all_pairs)

      # Convert format and add to queue
      for sent_A_str, sent_B_str, label in all_pairs:
        sent_A_ids = enc_vocab.GetIds(sent_A_str)
        sent_B_ids = enc_vocab.GetIds(sent_B_str)
        length_A = len(sent_A_ids)
        length_B = len(sent_B_ids)

        if length_A >= hps.max_sent_len:
          sent_A_ids = sent_A_ids[:hps.max_sent_len]
        else:
          sent_A_ids += [enc_pad_id] * (hps.max_sent_len - length_A)

        if length_B >= hps.max_sent_len:
          sent_B_ids = sent_B_ids[:hps.max_sent_len]
        else:
          sent_B_ids += [enc_pad_id] * (hps.max_sent_len - length_B)

        np_sent_A = np.array(sent_A_ids, dtype=np.int32)
        np_sent_B = np.array(sent_B_ids, dtype=np.int32)

        element = SentencePair(np_sent_A, np_sent_B, length_A, length_B, label)
        self._sample_queue.put(element)

  def _FillBucketInputQueue(self):
    """Fill bucketed batches into the bucket_input_queue."""
    hps = self._hps
    while True:
      samples = []
      for _ in xrange(hps.batch_size * BUCKET_NUM_BATCH):
        samples.append(self._sample_queue.get())

      # Bucketing ignored
      # if self._bucketing:
      #   samples = sorted(samples, key=lambda inp: inp.enc_doc_len)

      if self._shuffle_batches:
        shuffle(samples)

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
    field_lists = [[], [], [], [], []]

    for ex in batch:
      for i in range(5):
        field_lists[i].append(ex[i])

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return SentencePairBatch(stacked_fields[0], stacked_fields[1],
                             stacked_fields[2], stacked_fields[3],
                             stacked_fields[4])


class SentenceTripletBatcher(SentencePairBatcher):
  """Batch reader for sequence matching model."""

  def _FillInputQueue(self, data_path):
    """Fill input queue with ExtractiveSample."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    enc_pad_id = enc_vocab.pad_id
    enc_start_id = enc_vocab.start_id
    start_sent = [enc_start_id] * hps.max_sent_len

    data_generator = self._DataGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for data_sample in data_generator:  # type(data_sample): DocSummary
      document = data_sample.document
      summary = data_sample.summary

      summary_triplets = sample_triplets(summary)
      summary_triplets = random.sample(summary_triplets,
                                       min(2, len(summary_triplets)))

      doc_triplets = sample_triplets(document, window_size=9)
      doc_triplets = random.sample(doc_triplets, min(14, len(doc_triplets)))

      all_triplets = []
      for t in summary_triplets:
        all_triplets.append((0, t))
      for t in doc_triplets:
        all_triplets.append((1, t))
      shuffle(all_triplets)

      # Convert format and add to queue
      for src_id, (sent_A_id, sent_B_pos_id, sent_B_neg_id) in all_triplets:
        source = document if src_id else summary

        if sent_A_id < 0:
          sent_A_wids = start_sent
        else:
          sent_A_wids = enc_vocab.GetIds(source[sent_A_id])

        sent_B_pos_wids = enc_vocab.GetIds(source[sent_B_pos_id])
        sent_B_neg_wids = enc_vocab.GetIds(source[sent_B_neg_id])

        length_A = len(sent_A_wids)
        length_B_pos = len(sent_B_pos_wids)
        length_B_neg = len(sent_B_neg_wids)

        if length_A >= hps.max_sent_len:
          sent_A_wids = sent_A_wids[:hps.max_sent_len]
        else:
          sent_A_wids += [enc_pad_id] * (hps.max_sent_len - length_A)

        if length_B_pos >= hps.max_sent_len:
          sent_B_pos_wids = sent_B_pos_wids[:hps.max_sent_len]
        else:
          sent_B_pos_wids += [enc_pad_id] * (hps.max_sent_len - length_B_pos)

        if length_B_neg >= hps.max_sent_len:
          sent_B_neg_wids = sent_B_neg_wids[:hps.max_sent_len]
        else:
          sent_B_neg_wids += [enc_pad_id] * (hps.max_sent_len - length_B_neg)

        np_sent_A = np.array(sent_A_wids, dtype=np.int32)
        np_sent_B_pos = np.array(sent_B_pos_wids, dtype=np.int32)
        np_sent_B_neg = np.array(sent_B_neg_wids, dtype=np.int32)

        element = SentenceTriplet(np_sent_A, np_sent_B_pos, np_sent_B_neg,
                                  length_A, length_B_pos, length_B_neg)
        self._sample_queue.put(element)

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

    Returns:
        model_batch: ExtractiveBatch
    """
    hps = self._hps
    field_lists = [[], [], [], [], [], []]

    for ex in batch:
      for i in range(6):
        field_lists[i].append(ex[i])

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return SentenceTripletBatch(stacked_fields[0], stacked_fields[1],
                                stacked_fields[2], stacked_fields[3],
                                stacked_fields[4], stacked_fields[5])


class SentenceBlockBatcher(SentencePairBatcher):
  """Batch reader for sequence matching model."""

  def _FillInputQueue(self, data_path):
    """Fill input queue with ExtractiveSample."""
    hps = self._hps
    enc_vocab = self._enc_vocab
    enc_pad_id = enc_vocab.pad_id
    max_num_sents = hps.max_num_sents
    max_sent_len = hps.max_sent_len
    pad_sent = [enc_pad_id] * max_sent_len
    data_generator = self._DataGenerator(data_path, self._num_epochs)

    # pdb.set_trace()
    for data_sample in data_generator:  # type(data_sample): DocSummary
      document = data_sample.document
      summary = data_sample.summary
      if not summary:
        continue

      # Sample negative blocks
      if len(summary) > max_num_sents:
        if self._truncate_input:
          summary = summary[:max_num_sents]
        else:
          continue
      summary_len = len(summary)

      negative_blocks = []
      if summary_len > 1:
        # 1. Wrong order
        while True:
          neg_1 = random.sample(summary, summary_len)
          if neg_1 != summary:
            break

        # 2. Remove one sentence
        del_idx = randint(0, summary_len - 1)
        neg_2 = [s for i, s in enumerate(summary) if i != del_idx]
        negative_blocks += [neg_1, neg_2]

      # 3. Replace one sentence
      rep_idx = randint(0, summary_len - 1)
      neg_3 = summary[:]
      neg_3[rep_idx] = random.choice(document)

      # 4. Random permutation from document
      neg_4 = random.sample(document, min(summary_len, len(document)))
      negative_blocks += [neg_3, neg_4]
      negative_blocks = random.sample(negative_blocks, 2)

      # Combine them with labels, 1 for positive, 0 for negative
      all_blocks = [(summary, 1)]  # positive sample (1/3)
      for b in negative_blocks:  # negative samples (2/3)
        all_blocks.append((b, 0))
      shuffle(all_blocks)

      # Convert format and add to queue
      for sentences, label in all_blocks:
        # pdb.set_trace()
        sent_ids = [enc_vocab.GetIds(s) for s in sentences]
        sent_ids = [x[:max_sent_len] for x in sent_ids]  # truncate inputs
        sent_lens = [len(x)
                     for x in sent_ids] + [0] * (max_num_sents - len(sent_ids))

        for ids in sent_ids:
          ids += [enc_pad_id] * (max_sent_len - len(ids))
        sent_ids += [pad_sent] * (max_num_sents - len(sent_ids))

        np_sent_input = np.array(sent_ids, dtype=np.int32)
        np_length = np.array(sent_lens, dtype=np.int32)

        element = SentenceBlock(np_sent_input, np_length, label)
        self._sample_queue.put(element)

  def _PackBatch(self, batch):
    """ Pack the batch into numpy arrays.

    Returns:
        model_batch: ExtractiveBatch
    """
    hps = self._hps
    field_lists = [[], [], []]

    for ex in batch:
      for i in range(3):
        field_lists[i].append(ex[i])

    stacked_fields = [np.stack(field, axis=0) for field in field_lists]

    return SentenceBlockBatch(stacked_fields[0], stacked_fields[1],
                              stacked_fields[2])
