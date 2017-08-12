import argparse
import operator
import pdb
import os, sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)

summary_key = "<summary>"
content_key = "<content>"
sum_sent_count_key = "<sum_sent_count>"


def update_vocab(vocab, word):
  if word in vocab:
    vocab[word] += 1
  else:
    vocab[word] = 1


def sort_vocab(vocab):
  return sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)


def write_vocab(vocab, path):
  with open(path, 'w') as f:
    for word, freq in vocab:
      f.write('%s %d\n' % (word, freq))
  print 'Finished writing %s.' % path


def build_vocab(input_file, output_prefix):
  print 'Start building vocabulary'
  input_vocab, output_vocab = {}, {}

  with open(input_file, 'r') as f:
    for i, l in enumerate(f):
      summary_idx = l.index(summary_key)
      content_start_idx = l.index(content_key)
      content_end_idx = l.index(sum_sent_count_key)

      if content_start_idx > content_end_idx or summary_idx > content_start_idx:
        raise ValueError("Invalid example string %s." % l)

      summary_str = l[summary_idx + len(summary_key):content_start_idx].strip()
      content_str = l[content_start_idx + len(content_key):
                      content_end_idx].strip()

      for t in content_str.split():
        update_vocab(input_vocab, t)

      for t in summary_str.split():
        update_vocab(output_vocab, t)

  # Sort the vocabularies in descending order of frequency
  input_vocab = sort_vocab(input_vocab)
  output_vocab = sort_vocab(output_vocab)

  write_vocab(input_vocab, output_prefix + '.content')
  write_vocab(output_vocab, output_prefix + '.summary')

  print 'Finished building vocabulary'
  return input_vocab, output_vocab


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Build vocabularies for dataset.')
  parser.add_argument(
      '--input_file', default='data/data.train', help='path to input data file')
  parser.add_argument(
      '--output_prefix',
      default='data/train_vocab',
      help='filename prefix of output file')

  args = parser.parse_args()
  build_vocab(args.input_file, args.output_prefix)
