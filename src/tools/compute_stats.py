import argparse
import pdb
import numpy as np
import matplotlib.pyplot as plt

inp_key = '<input>'
out_key = '<output>'
summary_key = "<summary>"
content_key = "<content>"
sum_sent_count_key = "<sum_sent_count>"


def compute_stats(input_file):
  input_count, output_count = [], []

  with open(input_file, 'r') as f:
    for l in f:
      summary_idx = l.index(summary_key)
      content_start_idx = l.index(content_key)
      content_end_idx = l.index(sum_sent_count_key)

      if content_start_idx > content_end_idx or summary_idx > content_start_idx:
        raise ValueError("Invalid example string %s." % l)

      out_str = l[summary_idx + len(summary_key):content_start_idx].strip()
      inp_str = l[content_start_idx + len(content_key):content_end_idx].strip()

      inp_tokens = inp_str.split()
      input_count.append(len(inp_tokens))
      output_count.append(len(out_str.split()))

  print 'Input: max %d min %d mean %f std %f' % (max(input_count),
                                                 min(input_count),
                                                 np.mean(input_count),
                                                 np.std(input_count))
  # plot_hist(input_count, 'Input count')

  print 'Output: max %d min %d mean %f std %f' % (max(output_count),
                                                  min(output_count),
                                                  np.mean(output_count),
                                                  np.std(output_count))
  # plot_hist(output_count, 'Output count')


def plot_hist(count, title):
  plt.hist(count, bins='auto')
  plt.title(title)
  plt.show()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build vocabularies from LCSTS.')
  parser.add_argument(
      '--input_file', default='data/train', help='path to input data file')

  args = parser.parse_args()
  compute_stats(args.input_file)
