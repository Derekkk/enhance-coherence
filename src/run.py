"""Script for training and testing models."""
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple

from utils import batch_reader, vocab, evaluate

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
tf.app.flags.DEFINE_integer("batch_size", 1,
                            "Size of minibatch. Beam size in decode mode.")
# ----------- Train mode related flags ------------------
tf.app.flags.DEFINE_float("lr", 0.15, "Initial learning rate.")
tf.app.flags.DEFINE_float("min_lr", 0.01, "Minimum learning rate.")
tf.app.flags.DEFINE_float("max_grad_norm", 1.0,
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
tf.app.flags.DEFINE_string("decode_dir", "", "Directory for decode summaries.")
tf.app.flags.DEFINE_integer("extract_topk", 3,
                            "Number of sentence extracted in decode mode.")
# ----------- general flags ----------------
tf.app.flags.DEFINE_integer("emb_dim", 128, "Dim of word embedding.")
tf.app.flags.DEFINE_integer("num_gpus", 1, "Number of gpus used.")
# ----------- summarunner related flags ----------------
tf.app.flags.DEFINE_integer("enc_layers", 1, "Number of encoder layers.")
tf.app.flags.DEFINE_integer("num_sents_doc", 100,
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
# ----------- summarunner_rf related flags ----------------
tf.app.flags.DEFINE_string("train_mode", "sl",
                           "Kernel sizes of word-level CNN.")
tf.app.flags.DEFINE_string("mlp_num_hiddens", "256",
                           "Kernel sizes of word-level CNN.")
tf.app.flags.DEFINE_float("rl_coef", 1.0,
                          "Coefficient for RL loss in SL+RL mode.")
# ----------- cohere_extract related flags ----------------
tf.app.flags.DEFINE_string("word_conv_widths", "3,5,7",
                           "Kernel sizes of word-level CNN.")
tf.app.flags.DEFINE_string("word_conv_filters", "128,256,256",
                           "Number of output filters for each kernel size.")
tf.app.flags.DEFINE_float("coherence_coef", 1.0,
                          "Coefficient of coherence loss in REINFORCE.")
tf.app.flags.DEFINE_integer("coh_samples", 1, "Number of samples/instance.")
tf.app.flags.DEFINE_string("coherence_dir", "", "Directory of coherence model.")
tf.app.flags.DEFINE_string("pretrain_dir", "",
                           "Directory for pretrained model.")
# ----------- cohere_extract_rf related flags ----------------
tf.app.flags.DEFINE_float("rouge_coef", 1.0,
                          "Coefficient of ROUGE loss in REINFORCE.")
tf.app.flags.DEFINE_string("temp_dir", "/ssd/data/ywubw/tmp",
                           "Temporary directory for ROUGE script.")
tf.app.flags.DEFINE_float("coh_reward_clip", 3.0,
                          "Maximum coherence reward for reward clipping.")
tf.app.flags.DEFINE_integer("hist_repr_dim", 0,
                            "Dimension of history summary representation.")
# ----------- seqmatch related flags ----------------
tf.app.flags.DEFINE_string("seqmatch_type", "conv_match",
                           "Type of sequence matching architecture.")
tf.app.flags.DEFINE_integer("max_sent_len", 50, "Maximum length of sentences.")
tf.app.flags.DEFINE_integer("sm_emb_dim", 128, "Dim of word embedding.")
tf.app.flags.DEFINE_integer("sm_gru_num_units", 256,
                            "Number of hidden units in encoder RNN.")
tf.app.flags.DEFINE_float("sm_margin", 0.5, "Margin for ranking loss")
tf.app.flags.DEFINE_integer("sm_conv1d_filter", 128,
                            "Number of filters for conv1d.")
tf.app.flags.DEFINE_integer("sm_conv1d_width", 3, "Width of conv1d.")
tf.app.flags.DEFINE_string("sm_conv_filters", "256",
                           "Number of filters in the CNN.")
tf.app.flags.DEFINE_string("sm_conv_heights", "3",
                           "Width of convolution kernel.")
tf.app.flags.DEFINE_string("sm_conv_widths", "3",
                           "Width of convolution kernel.")
tf.app.flags.DEFINE_string("sm_maxpool_widths", "2", "Width of max-pooling.")
tf.app.flags.DEFINE_string("sm_fc_num_units", "256",
                           "Number of units in FC layers.")
# ----------- coherence related flags ----------------
tf.app.flags.DEFINE_integer("coh_emb_dim", 128, "Dim of word embedding.")
tf.app.flags.DEFINE_integer("max_num_sents", 6, "Maximum number of sentences.")
tf.app.flags.DEFINE_integer("gru_num_hidden", "256",
                            "Number of hidden units in the GRU.")
tf.app.flags.DEFINE_string("conv_heights", "3", "Height of convolution kernel.")
tf.app.flags.DEFINE_string("conv_widths", "3", "Width of convolution kernel.")
tf.app.flags.DEFINE_string("maxpool_widths", "2", "Width of max-pooling.")
tf.app.flags.DEFINE_string("fc_num_hiddens", "256,128",
                           "Number of hidden units in the final FC layers.")
# DocSummary = namedtuple('DocSummary', 'document summary extract_ids rouge_2')

DocSummary = namedtuple('DocSummary',
                        'url document summary extract_ids rouge_2')

# DocSummaryCount = namedtuple('DocSummaryCount',
#                              'url document extract_ids count')


def main():
  # Configure the enviroment
  tf.logging.set_verbosity(FLAGS.verbosity)

  # Import model
  model_type = FLAGS.model
  if model_type == "summarunner":
    from models.summarunner import CreateHParams, TrainLoop  #TODO: update API
    from models.summarunner import SummaRuNNer as Model
  elif model_type == "summarunner_rf":
    from models.summarunner_rf import CreateHParams
    from models.summarunner_rf import SummaRuNNerRF as Model
  elif model_type == "cohere_extract":
    from models.cohere_extract import CreateHParams
    from models.cohere_extract import CoherentExtract as Model
  elif model_type == "cohere_extract_rf":
    from models.cohere_extract_rf import CreateHParams
    from models.cohere_extract_rf import CoherentExtractRF as Model
  elif model_type == "seqmatch":
    from models.seqmatch import CreateHParams
    from models.seqmatch import SeqMatchNet as Model
  elif model_type == "coherence":
    from models.coherence import CreateHParams
    from models.coherence import CoherenceModel as Model
  else:
    raise ValueError("%s model NOT defined." % model_type)
  tf.logging.info("Using model %s." % model_type.upper())

  # Build vocabs
  input_vocab = vocab.Vocab(FLAGS.input_vocab, FLAGS.input_vsize)

  # Create model hyper-parameters
  hps = CreateHParams()
  tf.logging.info("Using the following hyper-parameters:\n%r" % str(hps))

  if FLAGS.mode == "train":
    num_epochs = None  # infinite loop
    shuffle_batches = True
    batcher_hps = hps
  else:
    num_epochs = 1  # only go through test set once
    shuffle_batches = False  # do not shuffle the batches
    batcher_hps = hps._replace(batch_size=1)  # ensure all examples are used
    batch_reader.BUCKET_NUM_BATCH = 1  # ensure all examples are used
    batch_reader.GET_TIMEOUT = 60

  # Create data reader
  if model_type in [
      "summarunner", "summarunner_rf", "cohere_extract", "cohere_extract_rf"
  ]:
    batcher = batch_reader.ExtractiveBatcher(
        FLAGS.data_path,
        input_vocab,
        batcher_hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
    if FLAGS.mode == "train":
      # Create validation data reader
      valid_batcher = batch_reader.ExtractiveBatcher(
          FLAGS.valid_path,
          input_vocab,
          batcher_hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)

  elif model_type == "seqmatch":
    batcher = batch_reader.SentenceTripletBatcher(
        FLAGS.data_path,
        input_vocab,
        batcher_hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
    if FLAGS.mode == "train":
      # Create validation data reader
      valid_batcher = batch_reader.SentenceTripletBatcher(
          FLAGS.valid_path,
          input_vocab,
          batcher_hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)

  elif model_type == "coherence":
    batcher = batch_reader.SentenceBlockBatcher(
        FLAGS.data_path,
        input_vocab,
        batcher_hps,
        bucketing=FLAGS.use_bucketing,
        truncate_input=FLAGS.truncate_input,
        num_epochs=num_epochs,
        shuffle_batches=shuffle_batches)
    if FLAGS.mode == "train":
      # Create validation data reader
      valid_batcher = batch_reader.SentenceBlockBatcher(
          FLAGS.valid_path,
          input_vocab,
          batcher_hps,
          bucketing=FLAGS.use_bucketing,
          truncate_input=FLAGS.truncate_input,
          num_epochs=num_epochs,
          shuffle_batches=shuffle_batches)

  else:
    raise NotImplementedError()

  if FLAGS.mode == "train":
    model = Model(hps, input_vocab, num_gpus=FLAGS.num_gpus)
    model.train(batcher, valid_batcher)  # start training
  elif FLAGS.mode == "decode":
    if model_type in ["summarunner", "cohere_extract"]:
      from utils.decode import TopKDecoder
      hps = hps._replace(batch_size=1)
      model = Model(hps, input_vocab, num_gpus=FLAGS.num_gpus)
      decoder = TopKDecoder(model, batch_size=1)
      output_fn = decoder.decode(batcher, FLAGS.extract_topk)
      evaluate.eval_rouge(output_fn)
    elif model_type in ["summarunner_rf", "cohere_extract_rf"]:
      from utils.decode import SummaRuNNerRFDecoder
      model = Model(
          hps._replace(batch_size=None),  # to allow variable batch_size
          input_vocab,
          num_gpus=FLAGS.num_gpus)
      decoder = SummaRuNNerRFDecoder(model, hps)  # use batch_size as beam_size
      output_fn = decoder.decode(batcher)
      evaluate.eval_rouge(output_fn)
    else:
      raise NotImplementedError()
  else:
    raise ValueError("Invalid mode %s. Try train/decode instead." % FLAGS.mode)


if __name__ == "__main__":
  main()
