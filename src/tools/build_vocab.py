import argparse
import operator
import pdb
import cPickle as pkl
from collections import namedtuple

DocSummary = namedtuple('DocSummary', 'document summary extract_ids rouge_2')
DocSummaryCount = namedtuple('DocSummary', 'document summary extract_ids count')


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


def build_vocab(in_path, out_path):
  print 'Start building vocabulary'
  vocab = {}

  with open(in_path, 'r') as f:
    dataset = pkl.load(f)

    for d in dataset:
      for s in d.document:
        for w in s.split():
          update_vocab(vocab, w)

  # Sort the vocabularies in descending order of frequency
  vocab = sort_vocab(vocab)

  write_vocab(vocab, out_path)
  print 'Finished building vocabulary'


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Build vocabularies for dataset.')
  parser.add_argument('in_path', help='Path of input data file.')
  parser.add_argument('out_path', help='Filename of output vocab file')
  args = parser.parse_args()

  build_vocab(args.in_path, args.out_path)
