import argparse
import pdb
import numpy as np
# import matplotlib.pyplot as plt

A_key = '<A>'
B_pos_key = '<B+>'
B_neg_key = '<B->'
prob_key = "<prob>"


def evaluate(input_file):
  all_pos_prob, all_neg_probs = [], []

  with open(input_file, 'r') as f:
    pos_prob, neg_prob_list = 0.0, []
    # pdb.set_trace()
    for l in f:
      line = l.strip()
      if not line:
        all_pos_prob.append(pos_prob)
        all_neg_probs.append(neg_prob_list)
      elif A_key in line:
        pos_prob, neg_prob_list = 0.0, []
      elif B_pos_key in line:
        prob_idx = line.index(prob_key) + len(prob_key)
        pos_prob = float(line[prob_idx:])
      elif B_neg_key in line:
        prob_idx = line.index(prob_key) + len(prob_key)
        neg_prob = float(line[prob_idx:])
        neg_prob_list.append(neg_prob)
      else:
        raise ValueError("Invalid line")

  # Compute accuracy and P@1
  TP, total = 0, 0
  for pos_prob, neg_probs in zip(all_pos_prob, all_neg_probs):
    for n_prob in neg_probs:
      if pos_prob > n_prob:
        TP += 1
      total += 1
  print "Accuracy: %f, total %d" % (float(TP) / total, total)

  TP, total = 0, 0
  for pos_prob, neg_probs in zip(all_pos_prob, all_neg_probs):
    if pos_prob > max(neg_probs):
      TP += 1
    total += 1
  print "P@1: %f, total %d" % (float(TP) / total, total)


# def plot_hist(count, title):
#   plt.hist(count, bins='auto')
#   plt.title(title)
#   plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Evaluate output of SeqMatch model.')
  parser.add_argument('--input_file', help='Decode file to evaluate.')

  args = parser.parse_args()
  evaluate(args.input_file)
