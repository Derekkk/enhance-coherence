"""Script for training and testing models."""
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import batch_reader, vocab, evaluate
from utils.decode import SummaRuNNerDecoder

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model", "", "The name of model runned.")
tf.app.flags.DEFINE_string("data_path", "", "Path expression to data file.")
tf.app.flags.DEFINE_string("input_vocab", "", "Path to input vocabulary file.")
tf.app.flags.DEFINE_string("output_vocab", "",
                           "Path to output vocabulary file.")
tf.app.flags.DEFINE_integer("input_vsize", 0,
                            "Number of words in input vocabulary.")
tf.app.flags.DEFINE_integer("output_vsize", 0,
                            "Number of words in output vocabulary.")
tf.app.flags.DEFINE_string("ckpt_root", "", "Directory for checkpoint root.")
tf.app.flags.DEFINE_string("summary_dir", "", "Directory for summary files.")
tf.app.flags.DEFINE_string("mode", "train", "train/decode mode")
tf.app.flags.DEFINE_integer("batch_size", 32, "Size of minibatch.")
# ----------- Train mode related flags ------------------
tf.app.flags.DEFINE_float("lr", 0.15, "Initial learning rate.")
tf.app.flags.DEFINE_float("min_lr", 0.01, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_grad_norm", 4.0,
                          "Maximum gradient norm for gradient clipping.")
tf.app.flags.DEFINE_integer("decay_step", 30000, "Exponential decay step.")
tf.app.flags.DEFINE_float("decay_rate", 0.1, "Exponential decay rate.")
tf.app.flags.DEFINE_integer("max_run_steps", 1000000,
                            "Maximum number of run steps.")
tf.app.flags.DEFINE_float("dropout", 0.0, "Dropout rate.")
tf.app.flags.DEFINE_string("valid_path", "",
                           "Path expression to validation set.")
tf.app.flags.DEFINE_integer("valid_freq", 1000, "How often to run eval.")
tf.app.flags.DEFINE_integer("num_valid_batch", 30,
                            "Number valid batches in each _Valid step.")
tf.app.flags.DEFINE_integer("checkpoint_secs", 1200, "How often to checkpoint.")
tf.app.flags.DEFINE_integer("max_to_keep", None,
                            "Maximum number of checkpoints to keep. "
                            "Keep all by default")
tf.app.flags.DEFINE_integer("display_freq", 200, "How often to print.")
tf.app.flags.DEFINE_integer("verbosity", 20,
                            "tf.logging verbosity (default INFO).")
# ----------- Data reading related flags ------------------
tf.app.flags.DEFINE_bool("use_bucketing", False,
                         "Whether bucket articles of similar length.")
tf.app.flags.DEFINE_bool("truncate_input", False,
                         "Truncate inputs that are too long. If False, "
                         "examples that are too long are discarded.")
# ----------- Decode mode related flags ------------------
tf.app.flags.DEFINE_integer("beam_size", 10,
                            "beam size for beam search decoding.")
tf.app.flags.DEFINE_string("decode_dir", "", "Directory for decode summaries.")
tf.app.flags.DEFINE_integer("extract_topk", 3,
                            "Number of sentence extracted in decode mode.")
# ----------- general flags ----------------
tf.app.flags.DEFINE_integer("emb_dim", 128, "Dim of word embedding.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus used.")
# ----------- summarunner related flags ----------------
tf.app.flags.DEFINE_integer("enc_layers", 1, "Number of encoder layers.")
tf.app.flags.DEFINE_integer("num_sentences", 100,
                            "Maximum number of sentences in a document.")
tf.app.flags.DEFINE_integer("num_words_sent", 50,
                            "Maximum number of words in a sentence.")
tf.app.flags.DEFINE_integer("rel_pos_max_idx", 11,
                            "Maximum index of relative position embedding.")
tf.app.flags.DEFINE_integer("enc_num_hidden", 512,
                            "Number of hidden units in encoder RNN.")
tf.app.flags.DEFINE_integer("pos_emb_dim", 64,
                            "Dimension of positional embedding.")
tf.app.flags.DEFINE_integer("doc_repr_dim", 512,
                            "Dimension of document representation.")
tf.app.flags.DEFINE_string("word_conv_k_sizes", "3,5,7",
                           "Kernel sizes of word-level CNN.")
tf.app.flags.DEFINE_integer("word_conv_filter", 128,
                            "Number of output filters of all kernel sizes.")
tf.app.flags.DEFINE_integer("min_num_input_sents", 0,
                            "Minimum number of sentences in input docuement.")
tf.app.flags.DEFINE_integer("min_num_words_sent", 0,
                            "Ignore sentences shorter than this threshold.")
tf.app.flags.DEFINE_integer("trg_weight_norm", 0,
                            "Normalize the extraction target weights. "
                            "No normalization if it is not positive.")
# ----------- seqmatch related flags ----------------
tf.app.flags.DEFINE_integer("num_hidden", 256,
                            "Number of hidden units in encoder RNN.")
tf.app.flags.DEFINE_integer("max_sent_len", 50, "Maximum length of sentences.")
tf.app.flags.DEFINE_string("conv_filters", 256, "Number of filters in the CNN.")
tf.app.flags.DEFINE_string("conv_width", 3, "Width of convolution kernel.")
tf.app.flags.DEFINE_string("maxpool_width", 2, "Width of max-pooling.")

# DocSummary = namedtuple('DocSummary', 'document summary extract_ids rouge_2')

DocSummary = namedtuple('DocSummary',
                        'url document summary extract_ids rouge_2')

# DocSummaryCount = namedtuple('DocSummaryCount',
#                              'url document extract_ids count')


def _Train(model, data_batcher, valid_batcher, train_loop):
  """Runs model training."""
  with tf.device("/gpu:0"):  # GPU by default
    restorer = model.build_graph()

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
      global_step=model.global_step,
      init_fn=load_pretrain)  # TODO: could exploit more Supervisor features

  config = tf.ConfigProto(allow_soft_placement=True)
  # Turn on JIT compilation if necessary
  config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  sess = sv.prepare_or_wait_for_session(config=config)

  # Summary dir is different from ckpt_root to avoid conflict.
  summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

  # Start the training loop
  train_loop(model, sess, data_batcher, valid_batcher, summary_writer, FLAGS)

  sv.Stop()


def main():
  # Configure the enviroment
  tf.logging.set_verbosity(FLAGS.verbosity)

  # Import model
  model_type = FLAGS.model
  if model_type == "summarunner":
    from models.summarunner import CreateHParams, TrainLoop
    from models.summarunner import SummaRuNNer as Model
  elif model_type == "seqmatch":
    from models.seqmatch import CreateHParams, TrainLoop
    from models.seqmatch import SeqMatchNet as Model
  else:
    raise ValueError("%s model NOT defined." % model_type)
  tf.logging.info("Using model %s." % model_type.upper())

  # Build vocabs
  input_vocab = vocab.Vocab(FLAGS.input_vocab, FLAGS.input_vsize)

  # Create model hyper-parameters
  hps = CreateHParams(FLAGS)
  tf.logging.info("Using the following hyper-parameters:\n%r" % str(hps))

  if FLAGS.mode == "train":
    num_epochs = None  # infinite loop
    shuffle_batches = True
  else:
    num_epochs = 1  # only go through test set once
    shuffle_batches = False  # do not shuffle the batches
    hps._replace(batch_size=1)  # ensure all examples are used
    batch_reader.BUCKET_NUM_BATCH = 1  # ensure all examples are used

  # Create data reader
  if model_type == "summarunner":
    batcher = batch_reader.ExtractiveBatcher(
        FLAGS.data_path,
        input_vocab,
        hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
    if FLAGS.mode == "train":
      # Create validation data reader
      valid_batcher = batch_reader.ExtractiveBatcher(
          FLAGS.valid_path,
          input_vocab,
          hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)
  elif model_type == "seqmatch":
    batcher = batch_reader.SentencePairBatcher(
        FLAGS.data_path,
        input_vocab,
        hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
    if FLAGS.mode == "train":
      # Create validation data reader
      valid_batcher = batch_reader.SentencePairBatcher(
          FLAGS.valid_path,
          input_vocab,
          hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)
  else:
    raise NotImplementedError()

  if FLAGS.mode == "train":
    model = Model(hps, input_vocab, num_gpus=FLAGS.num_gpus)
    _Train(model, batcher, valid_batcher, TrainLoop)  # start training
  elif FLAGS.mode == "decode":
    if model_type == "summarunner":
      model = Model(hps, input_vocab, num_gpus=FLAGS.num_gpus)
      decoder = SummaRuNNerDecoder(model, hps)
      output_fn = decoder.Decode(batcher, FLAGS.extract_topk)
      evaluate.eval_rouge(output_fn)
    else:
      raise NotImplementedError()
  else:
    raise ValueError("Invalid mode %s. Try train/decode instead." % FLAGS.mode)


if __name__ == "__main__":
  main()
