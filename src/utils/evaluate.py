""" Evaluate performance. """
import pdb
import argparse
from nltk.tokenize import sent_tokenize
import time

# Import pythonrouge package
from pythonrouge import PythonROUGE
ROUGE_dir = "/qydata/ywubw/download/RELEASE-1.5.5"

# Input data format
sentence_sep = "</s>"
sys_tag = "<system>"
ref_tag = "<reference>"


def eval_rouge(in_path):
  print "Using pythonrouge package for evaluation."
  rouge = PythonROUGE(
      ROUGE_dir,
      n_gram=2,
      ROUGE_SU4=False,
      ROUGE_L=True,
      stemming=True,
      stopwords=False,
      length_limit=False,
      length=75,
      word_level=False,
      use_cf=True,
      cf=95,
      ROUGE_W=False,
      ROUGE_W_Weight=1.2,
      scoring_formula="average",
      resampling=False,
      samples=1000,
      favor=False,
      p=0.5)

  # pdb.set_trace()
  num_samples = 0
  summary, reference = [], []

  with open(in_path, "r") as in_file:
    for l in in_file.readlines():
      sys_start = l.find(sys_tag) + len(sys_tag)
      sys_end = l.find(ref_tag)
      sys_str = l[sys_start:sys_end].strip()

      ref_start = sys_end + len(ref_tag)
      ref_str = l[ref_start:].strip()

      sys_sent_list = sys_str.split(sentence_sep)
      ref_sent_list = ref_str.split(sentence_sep)

      summary.append([sys_sent_list])
      reference.append([ref_sent_list])
      num_samples += 1

  start_time = time.time()
  # Evaluate ROUGE using pythonrouge package
  print rouge.evaluate(summary, reference)
  total_time = time.time() - start_time
  time_per_eval = total_time / num_samples
  print "Takes %f seconds to evaluate %d samples, avg %fs." % (total_time,
                                                               num_samples,
                                                               time_per_eval)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate ROUGE.')
  parser.add_argument('in_path', type=str, help='Path of input data file.')
  args = parser.parse_args()

  eval_rouge(args.in_path)
