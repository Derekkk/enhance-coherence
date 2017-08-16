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
import random

# Special tokens
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
UNKNOWN_TOKEN = '<UNK>'
PAD_TOKEN = '<PAD>'
special_symbols = [PAD_TOKEN, UNKNOWN_TOKEN, SENTENCE_START, SENTENCE_END]


class Vocab(object):
  """Vocabulary class for mapping words and ids."""

  def __init__(self,
               vocab_file,
               max_size,
               random_unk=False,
               enable_full_vocab=False):
    self._word_to_id, self._id_to_word = {}, {}
    self._id_to_meta, self._tag_to_id = {}, {}
    self._count = 0
    self._random_unk = random_unk  # True if UNK is replaced by a random word id
    self._enable_full_vocab = enable_full_vocab  # When True, may output word ids beyond max_size
    assert max_size > 0, "max_size must be greater than 0."
    self._max_size = max_size

    with open(vocab_file, 'r') as vocab_f:
      for line in vocab_f:
        token, freq = line.split()

        if token in self._word_to_id:
          raise ValueError('Duplicated word: %s.' % token)

        self._word_to_id[token] = self._count
        self._id_to_word[self._count] = token

        self._count += 1
        if not self._enable_full_vocab and self._count >= self._max_size:
          break

    # Check whether special symbols are in the vocab
    for tok in special_symbols:
      assert tok in self._word_to_id, "%s missing." % tok

    # Set ids for special symbols
    self.start_id = self.WordToId(SENTENCE_START)
    self.end_id = self.WordToId(SENTENCE_END)
    self.unk_id = self.WordToId(UNKNOWN_TOKEN)
    self.pad_id = self.WordToId(PAD_TOKEN)

  def WordToId(self, word):
    if word not in self._word_to_id:
      if self._random_unk:
        return random.randint(5, self.NumIds - 1)  # Skip special symbols
      else:
        return self._word_to_id[UNKNOWN_TOKEN]

    return self._word_to_id[word]

  def IdToWord(self, word_id):
    if word_id not in self._id_to_word:
      raise ValueError("id not found in vocab: %d." % word_id)
    return self._id_to_word[word_id]

  @property
  def NumIds(self):
    return self._max_size

  def GetIds(self, text):
    """Get ids corresponding to words in text.
    Assumes tokens separated by space.

    Args:
      text: a string with tokens separated by space.

    Returns:
      A list of ints representing word ids.
    """
    return [self.WordToId(w) for w in text.split()]

  def GetWords(self, ids_list):
    """Get words from ids.

    Args:
      ids_list: list of int32

    Returns:
      List of words corresponding to ids.
    """
    assert isinstance(ids_list, list), '%s is not a list' % ids_list
    return [self.IdToWord(i) for i in ids_list]
