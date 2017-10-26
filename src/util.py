import numpy as np


def _minibatch(X, Y, indices):
  return X[indices], Y[indices]


def get_minibatches(X, Y, batch_size, shuffle=True):
  size = len(X)
  indices = np.arange(size)
  if shuffle:
    np.random.shuffle(indices)
  for i in np.arange(0, size, batch_size):
    yield _minibatch(X, Y, indices[i:i+batch_size])


def _pad(A, seq_len):
  if seq_len < len(A):
    raise ValueError('Sequence length must be longer than list.')
  return A + [0]*(seq_len - len(A))


def pad(A, seq_len):
  if type(A) == list:
    return [_pad(a, seq_len) for a in A]
  else:
    return _pad(A, seq_len)
