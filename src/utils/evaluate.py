""" Evaluate performance of Lead-3. """
import pdb
import argparse
from nltk.tokenize import sent_tokenize
import time

# Import pythonrouge package
from pythonrouge.pythonrouge import Pythonrouge
ROUGE_path = "/qydata/ywubw/download/RELEASE-1.5.5/ROUGE-1.5.5.pl"
data_path = "/qydata/ywubw/download/RELEASE-1.5.5/data"

# Input data format
sentence_sep = "</s>"
input_tag = "<hypothesis>"
output_tag = "<reference>"


def pyrouge_eval(summary, reference, rouge):
  setting_file = rouge.setting(
      files=False, summary=summary, reference=reference)
  results = rouge.eval_rouge(
      setting_file,
      f_measure_only=False,
      ROUGE_path=ROUGE_path,
      data_path=data_path)

  print results


def eval_rouge(in_path):
  in_file = open(in_path, "r")
  print "Using pythonrouge package for evaluation."
  rouge = Pythonrouge(
      n_gram=2,
      ROUGE_SU4=False,
      ROUGE_L=True,
      stemming=True,
      stopwords=False,
      word_level=False,
      length_limit=False,
      length=75,
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
  input_start = len(input_tag)
  num_samples = 0
  summary, reference = [], []

  for l in in_file.readlines():
    input_end = l.find(output_tag)
    output_start = input_end + len(output_tag)
    input_str = l[input_start:input_end].strip()
    output_str = l[output_start:].strip()

    input_sent_list = input_str.split(sentence_sep)
    output_sent_list = output_str.split(sentence_sep)

    summary.append(input_sent_list)
    reference.append([output_sent_list])
    num_samples += 1

  start_time = time.time()
  # Evaluate ROUGE using pythonrouge package
  pyrouge_eval(summary, reference, rouge)

  total_time = time.time() - start_time
  time_per_eval = total_time / num_samples
  print "Takes %f seconds to evaluate %d samples, avg %fs." % (total_time,
                                                               num_samples,
                                                               time_per_eval)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Evaluate ROUGE of Lead-3.')
  parser.add_argument('in_path', type=str, help='Path of input data file.')
  args = parser.parse_args()

  eval_rouge(args.in_path)
