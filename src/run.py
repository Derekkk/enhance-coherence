"""Script for training and testing models."""
import sys
import numpy as np
import tensorflow as tf

from utils import batch_reader, vocab
from utils.decode import BSDecoder, BSDemoDecoder, SummaRuNNerDecoder

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
tf.app.flags.DEFINE_string("article_key", "<input>",
                           "tf.Example feature key for article.")
tf.app.flags.DEFINE_string("abstract_key", "<output>",
                           "tf.Example feature key for abstract.")
tf.app.flags.DEFINE_string("ckpt_root", "", "Directory for checkpoint root.")
tf.app.flags.DEFINE_string("summary_dir", "", "Directory for summary files.")
# tf.app.flags.DEFINE_string("eval_dir", "", "Directory for eval.")
tf.app.flags.DEFINE_string("mode", "train", "train/decode mode")
tf.app.flags.DEFINE_integer("enc_timesteps", 83, "Max number of encoder steps.")
tf.app.flags.DEFINE_integer("dec_timesteps", 18, "Max number of decoder steps.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Size of minibatch.")
tf.app.flags.DEFINE_integer("enc_layers", 1, "Number of encoder layers.")
tf.app.flags.DEFINE_integer("num_hidden", 256, "Number of hidden units in RNN.")
tf.app.flags.DEFINE_integer("emb_dim", 128, "Dim of word embedding.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus used.")
# ----------- Train mode related flags ------------------
tf.app.flags.DEFINE_float("lr", 0.15, "Initial learning rate.")
tf.app.flags.DEFINE_float("min_lr", 0.01, "Minimum learning rate.")
tf.app.flags.DEFINE_integer("max_run_steps", 1000000,
                            "Maximum number of run steps.")
tf.app.flags.DEFINE_string("valid_path", "",
                           "Path expression to validation set.")
tf.app.flags.DEFINE_integer("valid_freq", 1000, "How often to run eval.")
tf.app.flags.DEFINE_integer("num_valid_batch", 20,
                            "Number valid batches in each _Valid step.")
tf.app.flags.DEFINE_integer("checkpoint_secs", 300, "How often to checkpoint.")
tf.app.flags.DEFINE_integer("max_to_keep", None,
                            "Maximum number of checkpoints to keep. "
                            "Keep all by default")
tf.app.flags.DEFINE_integer("display_freq", 500, "How often to print.")
# ----------- Data reading related flags ------------------
tf.app.flags.DEFINE_bool("use_bucketing", False,
                         "Whether bucket articles of similar length.")
tf.app.flags.DEFINE_bool("truncate_input", False,
                         "Truncate inputs that are too long. If False, "
                         "examples that are too long are discarded.")
tf.app.flags.DEFINE_bool("random_unk", False,
                         "True if UNKs in output vocab are "
                         "replaced by random word ids.")
tf.app.flags.DEFINE_bool("enable_full_vocab", False,
                         "True if output vocab could access full vocab"
                         "so that it could output ids beyond max_size.")
# ----------- Decode mode related flags ------------------
tf.app.flags.DEFINE_integer("beam_size", 10,
                            "beam size for beam search decoding.")
tf.app.flags.DEFINE_string("decode_dir", "", "Directory for decode summaries.")
tf.app.flags.DEFINE_bool("exclude_unks", False,
                         "True if <UNK> is forbidden when beam search.")
tf.app.flags.DEFINE_integer("decode_verbosity", 20,
                            "Verbosity when decoding (default INFO).")
# ----------- seq2seq related flags ----------------
tf.app.flags.DEFINE_integer("num_softmax_samples", 1024,
                            "Number of samples in sampled cross-entropy.")
# ----------- summarunner related flags ----------------
tf.app.flags.DEFINE_integer("num_sentences", 200,
                            "Maximum number of sentences in a document.")
tf.app.flags.DEFINE_integer("num_words_sent", 60,
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
tf.app.flags.DEFINE_integer("min_num_input_sents", 5,
                            "Minimum number of sentences in input docuement.")
tf.app.flags.DEFINE_integer("min_num_words_sent", 3,
                            "Ignore sentences shorter than this threshold.")
tf.app.flags.DEFINE_integer("extract_topk", 3,
                            "Number of sentence extracted in decode mode.")
tf.app.flags.DEFINE_integer("trg_weight_norm", 5,
                            "Normalize the extraction target weights. "
                            "No normalization if it is not positive.")
tf.app.flags.DEFINE_integer("read_mode", 1,
                            "Mode of data reading, 1 for raw only, "
                            "2 for post-processing result only, 3 for both.")
# ----------- summarunner_abs related flags ----------------
tf.app.flags.DEFINE_integer("dec_num_hidden", 256,
                            "Number of hidden units in decoder RNN.")
tf.app.flags.DEFINE_integer("min_output_len", 10,
                            "Minimum number of words in output summary.")
# ----------- cnn2seq related flags ----------------
tf.app.flags.DEFINE_string("cnn_hparams", "",
                           "Hyper-parameters of CNN encoder in string.")
tf.app.flags.DEFINE_integer("dec_input_size", 0,
                            "Size of decoder input after transformation.")
tf.app.flags.DEFINE_bool("cnn_input_mask", True,
                         "Whether mask the input using length information.")
tf.app.flags.DEFINE_bool("att_over_att", False,
                         "Whether use attention over attention mechanism.")
tf.app.flags.DEFINE_bool("concat_reads", False,
                         "Whether concatanate memory reads from different "
                         "layers, used only when att_over_att is True.")
tf.app.flags.DEFINE_string("memory_hparams", "",
                           "Hyper-parameters for memories.")
# ----------- structure_net related flags ----------------
tf.app.flags.DEFINE_integer("memory_length", 30,
                            "Max number of words in memory.")
# ----------- structure_net_rf related flags -------------
tf.app.flags.DEFINE_bool("restore_pretrain", True,
                         "Whether or not restore from pretrain model.")
tf.app.flags.DEFINE_string("pretrain_dir", "",
                           "Directory of pretrained models.")
tf.app.flags.DEFINE_float("discount_factor", 1.0,
                          "Discount factor for computing return.")
tf.app.flags.DEFINE_float("rouge_beta", 1.0,
                          "Beta parameter for computing ROUGE F-score.")
tf.app.flags.DEFINE_bool("with_baseline", False, "True if baseline is used.")
# ----------- separation_net related flags ---------------
tf.app.flags.DEFINE_integer("sep_emb_dim", 128,
                            "Dim size of word embedding for separator.")
tf.app.flags.DEFINE_integer("sep_num_hidden", 256,
                            "Number of units in separator RNN.")
tf.app.flags.DEFINE_integer("num_options", 2,
                            "Number of decision options for separator.")
tf.app.flags.DEFINE_integer("num_sep_train_steps", 10000,
                            "Number of train steps for separator alone, "
                            "before end-to-end training of separation_net.")
tf.app.flags.DEFINE_integer("sep_train_type", 0,
                            "Type of separator pretrain process. "
                            "0 for REINFORCE loss, 1 for SL loss.")
tf.app.flags.DEFINE_bool("sep_reuse_embed", True,
                         "True if separator will reuse input embedding "
                         "with encoder, False otherwise. ")


def _Train(model, data_batcher, valid_batcher, train_loop):
  """Runs model training."""
  with tf.device("/cpu:0"):  # CPU by default
    restorer = model.build_graph()

  if FLAGS.restore_pretrain and restorer is not None:
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
      init_fn=load_pretrain)

  config = tf.ConfigProto(allow_soft_placement=True)
  # Turn on JIT compilation
  # config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  sess = sv.prepare_or_wait_for_session(config=config)

  # Summary dir is different from ckpt_root to avoid conflict.
  summary_writer = tf.summary.FileWriter(FLAGS.summary_dir)

  # Start the training loop
  train_loop(model, sess, data_batcher, valid_batcher, summary_writer, FLAGS)

  sv.Stop()


def main():
  # Configure enviroments
  tf.logging.set_verbosity(tf.logging.INFO)

  # Import model
  model_type = FLAGS.model
  if model_type == "seq2seq":
    from models.seq2seq import CreateHParams, TrainLoop
    from models.seq2seq import Seq2seq as Model
  elif model_type == "summarunner":
    from models.summarunner import CreateHParams, TrainLoop
    from models.summarunner import SummaRuNNer as Model
  elif model_type == "summarunner_abs":
    from models.summarunner_abs import CreateHParams, TrainLoop
    from models.summarunner_abs import SummaRuNNerAbs as Model
  elif model_type == "cnn2seq":
    from models.cnn2seq import CreateHParams, TrainLoop
    from models.cnn2seq import CNN2Seq as Model
  elif model_type == "structure_net":
    from models.structure_net import CreateHParams, TrainLoop
    from models.structure_net import StructureNet as Model
  elif model_type == "structure_net_rf":
    from models.structure_net_rf import CreateHParams, TrainLoop
    from models.structure_net_rf import RFStructureNet as Model
  elif model_type == "separation_net":
    from models.separation_net import CreateHParams, TrainLoop
    from models.separation_net import SeparationNet as Model
  else:
    raise ValueError("%s model NOT defined." % model_type)
  tf.logging.info("Using model %s." % model_type.upper())

  # Create model hyper-parameters
  hps = CreateHParams(FLAGS)
  tf.logging.info("Using the following hyper-parameters:\n%r" % str(hps))

  # Build vocabs
  input_vocab = vocab.Vocab(FLAGS.input_vocab, FLAGS.input_vsize)
  if model_type == "summarunner":
    output_vocab = None
  else:
    output_vocab = vocab.Vocab(
        FLAGS.output_vocab,
        FLAGS.output_vsize,
        random_unk=FLAGS.random_unk,
        enable_full_vocab=FLAGS.enable_full_vocab)

  # Create data reader
  if FLAGS.mode in ["eval", "decode"]:
    FLAGS.batch_size = 1  # read one by one so that no test example will be dropped
    num_epochs = 1  # only go through test set once
    shuffle_batches = False  # do not shuffle the batches
  else:
    num_epochs = None
    shuffle_batches = True

  if model_type == "summarunner":
    batcher = batch_reader.ExtractiveBatcher(
        FLAGS.data_path,
        input_vocab,
        hps,
        read_mode=FLAGS.read_mode,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
  else:
    batcher = batch_reader.Batcher(
        FLAGS.data_path,
        input_vocab,
        output_vocab,
        hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)

  if hps.mode == "train":  # Train the model
    model = Model(hps, input_vocab, output_vocab, num_gpus=FLAGS.num_gpus)

    # Create validation data reader
    if model_type == "summarunner":
      valid_batcher = batch_reader.ExtractiveBatcher(
          FLAGS.valid_path,
          input_vocab,
          hps,
          read_mode=FLAGS.read_mode,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)
    else:
      valid_batcher = batch_reader.Batcher(
          FLAGS.valid_path,
          input_vocab,
          output_vocab,
          hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)

    # Start training the model
    _Train(model, batcher, valid_batcher, TrainLoop)

  elif hps.mode == "decode":  # Decode with beam search
    tf.logging.set_verbosity(FLAGS.decode_verbosity)  # Avoid long printing
    model = Model(hps, input_vocab, output_vocab, num_gpus=FLAGS.num_gpus)
    decoder = SummaRuNNerDecoder(model, batcher, hps)
    ref_fn, dec_fn = decoder.DecodeLoop()
    # evaluate_files(ref_fn, dec_fn)

  elif hps.mode == "demo":  # Run demo in console
    decode_mdl_hps = hps._replace(
        dec_timesteps=1, batch_size=None, mode="decode")
    model = structure_net.Seq2SeqAttentionModel(
        decode_mdl_hps, vocab, num_gpus=FLAGS.num_gpus)
    decoder = BSDemoDecoder(model, hps, vocab)
    decoder.DecodeLoop()

  else:
    raise ValueError("Invalid mode %s." % hps.mode)


if __name__ == "__main__":
  main()
