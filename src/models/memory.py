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
# ============================================================================
"""seq2seq library codes copied from elsewhere for customization."""

import lib
import tensorflow as tf
from tensorflow.python.util import nest


class MemoryWrapper(object):

  def __init__(self,
               content,
               num_heads=1,
               query_size=None,
               output_is_list=True,
               scope=None,
               **kwargs):
    if content.get_shape().ndims != 3:
      raise ValueError("Content tensor must has 3 dimensions.")
    self._content = content

    # Acquiring shape information
    self._batch_size, self._memory_length,\
        self._memory_size = content.get_shape().as_list()
    if self._batch_size is None:
      self._batch_size = tf.unstack(tf.shape(content))[0]

    if query_size is None:
      self._query_size = self._memory_size
    elif query_size > 0:
      self._query_size = query_size
    else:
      raise ValueError("Size of query vector must be positive.")

    if num_heads < 1:
      raise ValueError("Must have at least one read head.")
    self._num_heads = num_heads

    self._output_is_list = output_is_list  # True if query output is a list
    self._scope_reuse = False  # True if read head parameters are reused

    # Other optional arguments
    if "content_lengths" in kwargs:
      self._content_lengths = kwargs["content_lengths"]
    else:
      self._content_lengths = None  # None if mask is not needed.

    with tf.variable_scope(
        scope or "MemoryWrapper",
        initializer=tf.random_uniform_initializer(-0.1, 0.1)) as self._scope:
      # Create zero state
      self._zero_state = self._create_zero_state()

      # Create addressing mechanism
      self._address_func = self._get_address_func()

  def query(self, q):
    with tf.variable_scope(
        self._scope, reuse=True if self._scope_reuse else None):
      self._scope_reuse = True  # Update reuse flag

      outputs = self._address_func(q)

    if isinstance(outputs, list):
      if self._output_is_list:
        return outputs
      else:
        return tf.concat(outputs, 1)
    else:
      if self._output_is_list:
        return nest.flatten(outputs)
      else:
        return outputs

  def _create_zero_state(self):
    if self._output_is_list:
      zero_state = [
          tf.zeros(
              tf.stack([self.batch_size, self.memory_size]), dtype=tf.float32)
          for i in xrange(self._num_heads)
      ]
    else:
      zero_state = tf.zeros(
          tf.stack([self.batch_size, self.memory_size * self._num_heads]),
          dtype=tf.float32)
    return zero_state

  @property
  def zero_state(self):
    return self._zero_state

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def memory_size(self):
    """Memory size for external use, output size of query() and zero_state."""
    return self._memory_size

  def _get_address_func(self):
    """Content-based addressing mechanism

    Returns:
      A function that accepts query tensor and outputs the read content.

    """
    num_heads = self._num_heads
    query_size = self._query_size
    memory_size = self._memory_size

    content = self._content  # [B, L, M]
    if self._content_lengths is not None:
      memory_masks = tf.sequence_mask(
          self._content_lengths, maxlen=self._memory_length,
          dtype=tf.float32)  # [B, L]
    else:  # No mask would be added in content_lengths is None
      memory_masks = None

    # Create parameter variables for each read head
    hidden_features = []
    for i in xrange(num_heads):
      k = tf.get_variable(
          "content_k_%d" % i, [1, memory_size, query_size], dtype=tf.float32)
      h = tf.nn.conv1d(content, k, 1, "SAME")  # [B, L, Q]
      hidden_features.append(h)

    def address_func(query):
      ds = []  # Results of reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        query_list = nest.flatten(query)
        for q in query_list:  # Check that ndims == 2 if specified.
          ndims = q.get_shape().ndims
          if ndims:
            assert ndims == 2
        query = tf.concat(query_list, 1)

      for i in xrange(num_heads):
        with tf.variable_scope("content_head_%d" % i):
          y = tf.expand_dims(lib.linear(query, query_size, True),
                             1)  # [B, 1, Q]

          # Compute the addressing weights
          s = tf.reduce_sum(hidden_features[i] * y, 2)  # [B, L]
          w = tf.nn.softmax(s)  # [B, L]

          # Mask the weigths if needed
          if memory_masks is not None:
            w = w * memory_masks  # [B, L]

          # Now calculate the attention-weighted vector d.
          d = tf.reduce_sum(tf.expand_dims(w, 2) * content, 1)  # [B, M]
          ds.append(d)
      return ds

    return address_func


class AttenOverAtten(MemoryWrapper):
  """Implements attention over attention memory module."""

  def __init__(self,
               memories,
               num_heads=1,
               query_size=None,
               output_is_list=True,
               scope=None,
               **kwargs):
    if not len(memories) > 1:
      raise ValueError("Must have more than one memory to attend on.")
    for m in memories:
      if type(m) != MemoryWrapper:
        raise TypeError(
            "Type of m must be MemoryWrapper (not including its subclasses).")
    self._memories = memories

    # Acquire shape information, and check compatability
    batch_size_list, memory_size_list = [], []
    for m in self._memories:
      bs = m.batch_size
      ms = m.memory_size
      batch_size_list.append(bs)
      memory_size_list.append(ms)
      assert ms == memory_size_list[0]
    self._batch_size = batch_size_list[0]
    self._memory_size = memory_size_list[0]  # internal memory size

    # Ignoring memory_lengths
    if query_size is None:
      self._query_size = self._memory_size
    elif query_size > 0:
      self._query_size = query_size
    else:
      raise ValueError("Size of query vector must be positive.")

    if num_heads < 1:
      raise ValueError("Must have at least one read head.")
    self._num_heads = num_heads

    self._output_is_list = output_is_list  # True if query output is a list
    self._scope_reuse = False  # True if read head parameters are reused

    # Optional argument
    if "concat_reads" in kwargs:
      self._concat_reads = kwargs["concat_reads"]
    else:
      self._concat_reads = False

    with tf.variable_scope(
        scope or "AttenOverAtten",
        initializer=tf.random_uniform_initializer(-0.1, 0.1)) as self._scope:
      # Create zero state
      self._zero_state = self._create_zero_state()

      # Create addressing mechanism
      self._address_func = self._get_address_func()

  @property
  def memory_size(self):
    """Memory size for external use, output size of query() and zero_state."""
    if self._concat_reads:
      return len(self._memories) * self._memory_size
    else:
      return self._memory_size

  def _get_address_func(self):
    """Content-based addressing mechanism

    Returns:
      A function that accepts query tensor and outputs the read content.

    """
    num_heads = self._num_heads
    query_size = self._query_size
    memory_size = self._memory_size
    memories = self._memories
    memory_len = len(memories)

    def address_func(query):
      ds = []  # Results of reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        assert len(query) > 0
        if len(query) == 1:
          query = query[0]
        else:
          query_list = nest.flatten(query)
          for q in query_list:  # Check that ndims == 2 if specified.
            ndims = q.get_shape().ndims
            if ndims:
              assert ndims == 2
          query = tf.concat(query_list, 1)

      # Query the memories, first layer of attention
      memory_reads = nest.flatten([m.query(query) for m in memories])
      content = tf.stack(memory_reads, 1)  # [B, L, M]
      hidden_features = []

      # Create parameter variables for each read head
      for i in xrange(num_heads):
        k = tf.get_variable(
            "content_k_%d" % i, [1, memory_size, query_size], dtype=tf.float32)
        h = tf.nn.conv1d(content, k, 1, "SAME")  # [B, L, Q]
        hidden_features.append(h)

      for i in xrange(num_heads):
        with tf.variable_scope("content_head_%d" % i):
          y = tf.expand_dims(lib.linear(query, query_size, True),
                             1)  # [B, 1, Q]

          # Compute the addressing weights
          s = tf.reduce_sum(hidden_features[i] * y, 2)  # [B, L]
          w = tf.nn.softmax(s)  # [B, L]

          # Now calculate the attention-weighted vector d.
          if self._concat_reads:
            d = tf.reshape(
                tf.expand_dims(w, 2) * content,
                [-1, memory_len * memory_size])  # [B, L * M]
          else:
            d = tf.reduce_sum(tf.expand_dims(w, 2) * content, 1)  # [B, M]
          ds.append(d)
      return ds

    return address_func


class RecurrentConvAttention(MemoryWrapper):
  """Implements recurrent convolutional kernel attention module."""

  def __init__(self,
               content,
               num_heads=1,
               query_size=None,
               output_is_list=True,
               scope=None,
               **kwargs):
    if "kernel_size" in kwargs:
      self._kernel_size = kwargs["kernel_size"]
    else:
      self._kernel_size = 3

    super(RecurrentConvAttention, self).__init__(
        content,
        num_heads,
        query_size,
        output_is_list,
        scope=scope or "RecurrentConvAtten",
        **kwargs)

  def _create_zero_state(self):
    zero_state = tf.zeros(
        tf.stack([self._batch_size, self._memory_size * self._num_heads]),
        dtype=tf.float32)

    if self._output_is_list:
      return [zero_state]
    else:
      return zero_state

  def _get_address_func(self):
    """Recurrent convolutional addressing mechanism

    Returns:
      A function that accepts query tensor and outputs the read content.

    """
    num_heads = self._num_heads
    memory_size = self._memory_size
    kernel_size = self._kernel_size
    content = self._content  # [B, L, M]
    rsp_content = tf.expand_dims(content, 0)  # [1, B, L, M]
    exp_dim_content = tf.expand_dims(content, 2)  # [B, L, 1, M]

    # Create mask
    if self._content_lengths is not None:
      memory_masks = tf.expand_dims(
          tf.sequence_mask(
              self._content_lengths,
              maxlen=self._memory_length,
              dtype=tf.float32), 2)  # [B, L, 1]
    else:  # No mask would be added in content_lengths is None
      memory_masks = None

    def address_func(query):
      ds = []  # Results of reads will be stored here.
      if nest.is_sequence(query):  # If the query is a tuple, flatten it.
        assert len(query) > 0
        if len(query) == 1:
          query = query[0]
        else:
          query_list = nest.flatten(query)
          for q in query_list:  # Check that ndims == 2 if specified.
            ndims = q.get_shape().ndims
            if ndims:
              assert ndims == 2
          query = tf.concat(query_list, 1)

      query_map = tf.contrib.layers.fully_connected(
          inputs=query,
          num_outputs=kernel_size * memory_size * num_heads,
          activation_fn=None,  # tf.tanh
          weights_initializer = \
              tf.random_uniform_initializer(-0.1, 0.1),
          biases_initializer = \
              tf.random_uniform_initializer(-0.1, 0.1),
          scope="rec_conv_kern_fc")  # [B, K * M * H]
      query_kernel = tf.reshape(
          query_map, [-1, kernel_size, memory_size, num_heads])  # [B, K, M, H]

      # Compute the addressing weights
      rec_conv_output = tf.nn.conv2d(
          rsp_content,
          query_kernel, [1, 1, 1, 1],
          "SAME",
          name="recurrent_conv")  # [1, B, L, H]
      rec_conv_weight = tf.nn.softmax(
          tf.squeeze(rec_conv_output, [0]), dim=1)  # [B, L, H]

      # Mask the weigths if needed
      if memory_masks is not None:
        rec_conv_weight = rec_conv_weight * memory_masks

      # Now calculate the attention-weighted vector d.
      exp_dim_weight = tf.expand_dims(rec_conv_weight, 3)  # [B, L, H, 1]
      output = tf.reduce_sum(exp_dim_content * exp_dim_weight, 1)  # [B, H, M]
      output = tf.reshape(output, [-1, num_heads * memory_size])
      return output

    return address_func
