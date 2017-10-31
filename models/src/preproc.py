import os
from tqdm import tqdm
import numpy as np
import random
from util import pad


DIM = 100
PAD = 0
PAD_KEY = '----PAD----'
PAD_V = [0.] * DIM
UNK = 1
UNK_KEY = '----UNK----'
UNK_V = [random.random() for i in range(DIM)] 

TAGS = {
  'O':      0,
  'B-PER':  1,
  'I-PER':  2,
  'B-ORG':  3,
  'I-ORG':  4,
  'B-LOC':  5,
  'I-LOC':  6,
  'B-MISC': 7,
  'I-MISC': 8,
}


def _get_glove(path):
  d = {}
  with open(path, 'r') as f:
    pbar = tqdm(total=os.path.getsize(path))
    for line in tqdm(f):
      pbar.update(len(line))
      s = line.split()
      d[s[0]] = s[1:]
  return d


def _line_has_data(line):
  return '-DOCSTART-' not in line and line.strip()


def _get_word_dict(data_dir):
  word_dict = {}
  idx = 2 # 0 is PAD, 1 is UNK
  with open('{}/word_list'.format(data_dir), 'r') as f:
    for line in f:
      w, _ = line.split()
      word_dict[w] = idx
      idx += 1
  return word_dict
  

def build_glove_and_word_list(data_dir):
  full_glove = _get_glove('{}/glove.6B.100d.txt'.format(data_dir))
  train_path = '{}/en.train'.format(data_dir)
  word_dict = {}
  with open(train_path, 'r') as f:
    for line in tqdm(f):
      if _line_has_data(line):
        word, _, _, tag = line.split()
        word = word.lower()
        if word in full_glove:
          if word in word_dict:
            word_dict[word] += 1
          else:
            word_dict[word] = 1
    word_list = sorted(
        [[v, k] for k, v in word_dict.items()],
        key=lambda x: x[0],
        reverse=True)
  glove = [PAD_V, UNK_V]
  with open('{}/word_list'.format(data_dir), 'w') as g:
    for w in word_list:
      g.write('{} {}\n'.format(w[1], w[0]))
      glove.append([float(x) for x in full_glove[w[1]]])
  np.save('{}/glove'.format(data_dir), np.array(glove, dtype=np.float32))


def build_train_data(data_dir):
  word_dict = _get_word_dict(data_dir)
  max_len = 120
  word_arr = []
  tag_arr = []
  with open('{}/en.train'.format(data_dir), 'r') as f:
    with open('{}/train'.format(data_dir), 'w') as out:
      words = []
      tags = []
      for line in f:
        if not _line_has_data(line):
          if len(words) > 0:
            out.write('{}\n'.format(' '.join([str(w) for w in words])))
            out.write('{}\n'.format(' '.join([str(t) for t in tags])))
            word_arr.append(words)
            tag_arr.append(tags)
            if len(words) > max_len:
              max_len = len(words)
          words = []
          tags = []
          continue
        w, _, _, t = line.split()
        if w in word_dict:
          words.append(word_dict[w])
        else:
          words.append(UNK)
        tags.append(TAGS[t])
  np.save(
      '{}/input.train'.format(data_dir),
      np.array(pad(word_arr, max_len), dtype=np.int32))
  np.save(
      '{}/output.train'.format(data_dir),
      np.array(pad(tag_arr, max_len), dtype=np.int32))
  print('Max sequence length is: {}'.format(max_len))


if __name__ == '__main__':
  DATA_DIR = '../data'
  build_glove_and_word_list(DATA_DIR)
  build_train_data(DATA_DIR)





