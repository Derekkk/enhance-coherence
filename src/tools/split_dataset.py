import argparse
import numpy as np
import random
import codecs


def split_dataset(input_file, output_prefix, ratios):
  assert len(ratios) == 3, 'Invalid ratios.'

  with codecs.open(input_file, 'r', 'utf-8') as f:
    docs = f.readlines()
  if not docs[-1].endswith('\n'):
    docs[-1] += "\n"

  total_length = len(docs)
  dev_test_ratio = ratios[1] + ratios[2]
  dev_test_len = int(total_length * dev_test_ratio)
  test_len = int(total_length * ratios[2])

  random.shuffle(docs)

  train_set = docs[:-dev_test_len]
  dev_set = docs[-dev_test_len:-test_len]
  test_set = docs[-test_len:]

  write_file(output_prefix + ".train", train_set)
  write_file(output_prefix + ".dev", dev_set)
  write_file(output_prefix + ".test", test_set)


def write_file(filename, dataset):
  with codecs.open(filename, "w", "utf-8") as f:
    f.write("".join(dataset))


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Split dataset into train/valid/test.')
  parser.add_argument(
      '--input_file', default='data/raw_data', help='Path to input data file')
  parser.add_argument(
      '--output_prefix',
      default='data/data_split',
      help='Filename prefix of output files')
  parser.add_argument(
      '--split_ratio', default='data/train_vocab', help='Ratio of each set.')

  args = parser.parse_args()
  ratios = list(np.fromstring(args.split_ratio, dtype=np.float32, sep=","))
  split_dataset(args.input_file, args.output_prefix, ratios)
