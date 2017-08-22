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

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops


def compute_avg(values, summary_writer, tag_name, step):
  """Calculate the average of losses and write to summary."""
  avg_value = float(np.mean(values))
  avg_summary = tf.Summary()
  avg_summary.value.add(tag=tag_name, simple_value=avg_value)
  summary_writer.add_summary(avg_summary, step)
  return avg_value


def sequence_loss_by_example(inputs,
                             targets,
                             weights,
                             loss_function,
                             average_across_timesteps=True,
                             name=None):
  """Sampled softmax loss for a sequence of inputs (per example).
     Adapted to support sampled_softmax loss function, which accepts
     activations instead of logits.

  Args:
    inputs: List of 2D Tensors of shape [batch_size x hid_dim].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    loss_function: Sampled softmax function (inputs, labels) -> loss
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    name: Optional name for this operation, default: 'sequence_loss_by_example'.

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(inputs) is different from len(targets) or len(weights).
  """
  if len(targets) != len(inputs) or len(weights) != len(inputs):
    raise ValueError('Lengths of logits, weights, and targets must be the same '
                     '%d, %d, %d.' % (len(inputs), len(weights), len(targets)))
  with tf.name_scope(name, 'sequence_loss_by_example',
                     inputs + targets + weights):
    log_perp_list = []
    for inp, target, weight in zip(inputs, targets, weights):
      crossent = loss_function(inp, target)
      log_perp_list.append(crossent * weight)
    log_perps = tf.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = tf.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(inputs,
                  targets,
                  weights,
                  loss_function,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    inputs: List of 2D Tensors of shape [batch_size x hid_dim].
    targets: List of 1D batch-sized int32 Tensors of the same length as inputs.
    weights: List of 1D batch-sized float-Tensors of the same length as inputs.
    loss_function: Sampled softmax function (inputs, labels) -> loss
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    name: Optional name for this operation, defaults to 'sequence_loss'.

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(inputs) is different from len(targets) or len(weights).
  """
  with tf.name_scope(name, 'sequence_loss', inputs + targets + weights):
    cost = tf.reduce_sum(
        sequence_loss_by_example(
            inputs,
            targets,
            weights,
            loss_function,
            average_across_timesteps=average_across_timesteps))
    if average_across_batch:
      batch_size = tf.shape(targets[0])[0]
      return cost / tf.cast(batch_size, tf.float32)
    else:
      return cost


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or tf.get_variable_scope()) as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)


def create_cudnn_rnn(input_data,
                     rnn_mode,
                     num_layers,
                     num_units,
                     input_size,
                     variable_name,
                     direction="unidirectional",
                     time_major=False,
                     dropout=0.0):

  if rnn_mode == "lstm":
    model = cudnn_rnn_ops.CudnnLSTM(
        num_layers, num_units, input_size, direction=direction, dropout=dropout)
  elif rnn_mode == "gru":
    model = cudnn_rnn_ops.CudnnGRU(
        num_layers, num_units, input_size, direction=direction, dropout=dropout)
  else:
    raise ValueError("Invalid rnn_mode: %s" % rnn_mode)

  # Compute the total size of RNN params (Tensor)
  params_size_ts = model.params_size()
  params = tf.Variable(
      tf.random_uniform([params_size_ts], minval=-0.1, maxval=0.1),
      validate_shape=False,
      name=variable_name)

  if not time_major:
    batch_size_ts = tf.shape(input_data)[0]  # batch size Tensor
    input_data = tf.transpose(input_data, [1, 0, 2])
  else:
    batch_size_ts = tf.shape(input_data)[1]  # batch size Tensor
  # NB: input_data should has shape [batch_size, num_timestep, d]

  if direction == "unidirectional":
    dir_count = 1
  elif direction == "bidirectional":
    dir_count = 2
  else:
    raise ValueError("Invalid direction: %s" % direction)

  init_h = tf.zeros(
      tf.stack([num_layers * dir_count, batch_size_ts, num_units]))
  has_input_c = (rnn_mode == "lstm")

  # Call the CudnnRNN
  if has_input_c:
    init_c = tf.zeros(
        tf.stack([num_layers * dir_count, batch_size_ts, num_units]))
    output, output_h, output_c = model(
        input_data=input_data, input_h=init_h, input_c=init_c, params=params)
  else:
    output, output_h = model(
        input_data=input_data, input_h=init_h, params=params)

  # output: [num_timestep, batch_size, num_units*dir_count]
  # output_h/c: [batch_size, num_units*dir_count]
  return output, output_h


def embed_loop_func(embedding, update_embedding=True):
  """Get a loop_function that return embedding of previous symbol.

  Args:
    embedding: embedding tensor for symbols.
    update_embedding: Boolean; if False, the gradients will not propagate
      through the embeddings.

  Returns:
    A loop function.
  """

  def loop_function(prev_symbol, i):
    """function that feed previous model output rather than ground truth."""
    # Note that gradients will not propagate through the second parameter of
    # embedding_lookup.
    emb_prev = tf.nn.embedding_lookup(embedding, prev_symbol)
    if not update_embedding:
      emb_prev = tf.stop_gradient(emb_prev)
    return emb_prev

  return loop_function


def no_op_action():

  def loop_function(output):
    null_symbols = tf.zeros([tf.shape(output)[0]], dtype=tf.int64)
    return output, null_symbols

  return loop_function


def stochastic_action(w, b):

  def loop_function(output):
    proj_output = tf.matmul(output, w, transpose_b=True) + b
    # output_logp = tf.nn.log_softmax(proj_output) #bug!
    # Randomly sample actions from the distribution.
    # Note: logits do NOT need log_softmax transformation
    output_symbol = tf.squeeze(
        tf.multinomial(logits=proj_output, num_samples=1))
    return proj_output, output_symbol

  return loop_function


def greedy_action(w, b):

  def loop_function(output):
    proj_output = tf.matmul(output, w, transpose_b=True) + b
    output_symbol = tf.argmax(proj_output, 1)
    return proj_output, output_symbol

  return loop_function


def memory_augmented_rnn(cell,
                         initial_state,
                         inputs,
                         memories,
                         output_size,
                         input_size=None,
                         in_loop_function=None,
                         out_loop_function=None,
                         dtype=tf.float32,
                         scope=None,
                         initial_state_attention=False):
  """Wrapping RNN with attention mechanism over memories. Useful for
  sequence-to-sequence model.


  Args:
    cell: rnn_cell.RNNCell defining the cell function and size.
    initial_state: 2D Tensor [batch_size x cell.state_size].
    inputs: A list of 2D Tensors [batch_size x emb_size].
    memories: A list/tuple of instances of MemoryWrapper class.
    in_loop_function: If not None, this function will be applied to i-th output
      in order to generate i+1-th input, and inputs will be ignored,
      except for the first element ("GO" symbol). This can be used for decoding,
      but also for training to emulate http://arxiv.org/abs/1506.03099.
      Signature -- loop_function(prev, i) = next
        * prev is a 2D Tensor of shape [batch_size x output_size],
        * i is an integer, the step number (when advanced control is needed),
        * next is a 2D Tensor of shape [batch_size x emb_size].
    dtype: The dtype to use for the RNN initial state (default: tf.float32).
    scope: VariableScope for the created subgraph; default: "attention_decoder".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states -- useful when we wish to resume decoding from a previously
      stored decoder state and attention states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as inputs of 2D Tensors of
        shape [batch_size x output_size]. These represent the generated outputs.
        Output i is computed from input i (which is either the i-th element
        of inputs or loop_function(output {i-1}, i)) as follows.
        First, we run the cell on a combination of the input and previous
        attention masks:
          cell_output, new_state = cell(linear(input, prev_attn), prev_state).
        Then, we calculate new attention masks:
          new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
        and then we calculate the output:
          output = linear(cell_output, new_attn).
      state: The state of each decoder cell the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].

  Raises:
    ValueError: when num_heads is not positive, there are no inputs, shapes
      of attention_states are not set, or input size cannot be inferred
      from the input.
  """
  if input_size is None:
    # input_size = cell.state_size * 2
    input_size = cell.state_size * (len(memories) + 1)

  if out_loop_function is None:
    out_loop_function = no_op_action()

  for m in memories:
    assert isinstance(m, MemoryWrapper)

  with tf.variable_scope(scope or "memory_augmented_rnn", dtype=dtype):
    memory_list = nest.flatten(memories)

    # Intialize read vector with zeros
    reads = nest.flatten([m.zero_state for m in memory_list])

    if initial_state_attention:
      reads = nest.flatten([m.query(initial_state) for m in memory_list])

    state = initial_state
    outputs, output_symbols, states, prev = [], [], [], None
    for i, inp in enumerate(inputs):
      if i > 0:
        tf.get_variable_scope().reuse_variables()

      # If in_loop_function is set, we use it instead of decoder_inputs.
      if in_loop_function is not None and prev is not None:
        with tf.variable_scope("in_loop_func", reuse=True):
          inp = in_loop_function(prev, i)

      # Merge input and previous reads into one vector of the right size.
      x = linear([inp] + reads, input_size, True, scope="cell_input")

      # Run the RNN.
      cell_output, state = cell(x, state)
      states.append(state)

      # Run the attention mechanism.
      if i == 0 and initial_state_attention:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
          reads = nest.flatten([m.query(state) for m in memory_list])
      else:
        reads = nest.flatten([m.query(state) for m in memory_list])

      output = linear(
          [cell_output] + reads, output_size, True, scope="cell_output")

      with tf.variable_scope("out_loop_func", reuse=True):
        output, prev = out_loop_function(output)
        outputs.append(output)
        output_symbols.append(prev)

    # output_symbols = tf.stack(output_symbols, 1)
  return outputs, output_symbols, states
