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
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
# NB: batch_size is not given (None) when deployed as a critic.
HParams = namedtuple("HParams", "mode, min_lr, lr, dropout, batch_size,"
                     "max_grad_norm, decay_step, decay_rate")


def CreateHParams():
  """Create Hyper-parameters from tf.app.flags.FLAGS"""
  hps = HParams(
      mode=FLAGS.mode,  # train, eval, decode
      lr=FLAGS.lr,
      min_lr=FLAGS.min_lr,
      dropout=FLAGS.dropout,
      batch_size=FLAGS.batch_size,
      max_grad_norm=FLAGS.max_grad_norm,
      decay_step=FLAGS.decay_step,
      decay_rate=FLAGS.decay_rate)
  return hps


class BaseModel(object):

  def __init__(self, hps, vocab, num_gpus=1):
    self._hps = hps
    self._vocab = vocab

    if num_gpus > 0:
      self._device_0 = "/gpu:0"
    else:
      self._device_0 = "/cpu:0"

  def build_graph(self):
    self._add_placeholders()
    self._build_model()
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    if self._hps.mode == 'train':
      self._add_loss()
      self._add_train_op()

    self._summaries = tf.summary.merge_all()

  def _add_placeholders(self):
    raise NotImplementedError()

  def _build_model(self):
    raise NotImplementedError()

  def _add_loss(self):
    raise NotImplementedError()
    self._loss = 0

  def _add_train_op(self, optimizer_class=tf.train.GradientDescentOptimizer):
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
      optimizer = optimizer_class(self._lr_rate)
      self._train_op = optimizer.apply_gradients(
          zip(grads, tvars), global_step=self.global_step, name="train_step")

  def run_train_step(self, sess, batch):
    raise NotImplementedError()

  def run_eval_step(self, sess, batch):
    raise NotImplementedError()

  def train_loop(self, sess, batcher, valid_batcher, summary_writer):
    """Runs model training."""
    raise NotImplementedError()

  def train(self, data_batcher, valid_batcher):
    """Runs model training."""
    with tf.device(self._device_0):
      restorer = self.build_graph()

    # Restore pretrained model if necessary
    # if FLAGS.restore_pretrain and restorer is not None:
    if restorer is not None:
      ckpt_state = tf.train.get_checkpoint_state(FLAGS.pretrain_dir)
      if not (ckpt_state and ckpt_state.model_checkpoint_path):
        raise ValueError("No pretrain model found at %s" % FLAGS.pretrain_dir)

      def load_pretrain(sess):
        tf.logging.info("Restoring pretrained model from %s" %
                        ckpt_state.model_checkpoint_path)
        restorer.restore(sess, ckpt_state.model_checkpoint_path)
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
        init_fn=load_pretrain)

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
