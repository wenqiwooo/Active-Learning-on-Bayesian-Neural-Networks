import math
from edward.models import Categorical, Normal
import numpy as np
import tensorflow as tf
import edward as ed
from tqdm import tqdm


def mini_batch(batch_size, x, y=None, shuffle=False):
  assert y is None or x.shape[0] == y.shape[0]

  N = x.shape[0]
  indices = np.arange(N)
  if shuffle:
    np.random.shuffle(indices)

  for i in range(0, N, batch_size):
    batch_indices = indices[i : i + batch_size]
    batch_x = x[batch_indices]
    if y is not None:
      batch_y = y[batch_indices]
      yield batch_x, batch_y
    else:
      yield batch_x


def CNN(f1, b1, f2, b2, f3, b3, fc_w1, fc_b1, fc_w2, fc_b2, X):
  conv1 = tf.nn.bias_add(
      tf.nn.conv2d(X, f1, (1, 1, 1, 1), 'SAME', name='conv1'), b1)
  conv1 = tf.nn.relu(conv1)
  maxpool1 = tf.nn.max_pool(
      conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool1')
  # maxpool1 = tf.nn.dropout(maxpool1, 0.8)
  # 16 x 16 x 32

  conv2 = tf.nn.bias_add(
      tf.nn.conv2d(maxpool1, f2, (1, 1, 1, 1), 'SAME', name='conv2'), b2)
  conv2 = tf.nn.relu(conv2)
  maxpool2 = tf.nn.max_pool(
      conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool2')
  # maxpool2 = tf.nn.dropout(maxpool2, 0.8)
  # 8 x 8 x 32

  conv3 = tf.nn.bias_add(
      tf.nn.conv2d(maxpool2, f3, (1, 1, 1, 1), 'SAME', name='conv3'), b3)
  conv3 = tf.nn.relu(conv3)
  maxpool3 = tf.nn.max_pool(
      conv3, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool3')
  # maxpool3 = tf.nn.dropout(maxpool3, 0.8)
  # 4 x 4 x 64

  conv_out = tf.reshape(maxpool3, (-1, 1024))
  fc1 = tf.matmul(conv_out, fc_w1) + fc_b1
  fc1 = tf.nn.relu(fc1)
  # fc1 = tf.nn.dropout(fc1, 0.8)

  fc2 = tf.matmul(fc1, fc_w2) + fc_b2

  return fc2


class Cifar10CNN(object):
  def __init__(self):
    self.global_step = tf.Variable(
        initial_value=0, name='global_step', trainable=False)

    self.X = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    self.Y = tf.placeholder(tf.int32, shape=(None,))

    f1_shape = (5, 5, 3, 32)
    f2_shape = (5, 5, 32, 32)
    f3_shape = (5, 5, 32, 64)
    fc_w1_shape = (1024, 64)
    fc_w2_shape = (64, 10)

    f1 = tf.get_variable('f1', f1_shape, dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [f1_shape[-1]], dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    f2 = tf.get_variable('f2', f2_shape, dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [f2_shape[-1]], dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    f3 = tf.get_variable('f3', f3_shape, dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [f3_shape[-1]], dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    fc_w1 = tf.get_variable('fc_w1', fc_w1_shape, dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    fc_b1 = tf.get_variable('fc_b1', [fc_w1_shape[-1]], dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    fc_w2 = tf.get_variable('fc_w2', fc_w2_shape, dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    fc_b2 = tf.get_variable('fc_b2', [fc_w2_shape[-1]], dtype=tf.float32,
        initializer=tf.constant_initializer(0.))

    self.nn = CNN(f1, b1, f2, b2, f3, b3, fc_w1, fc_b1, fc_w2, fc_b2, self.X)

    self.pred = tf.nn.softmax(self.nn)

    self.loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.nn, labels=self.Y))

    self.train = tf.train.RMSPropOptimizer(
        learning_rate=1e-4).minimize(self.loss, global_step=self.global_step)

  def optimize(self, sess, X, Y, epochs, batch_size, X_test=None, Y_test=None):
    for epoch in range(epochs):
      print('Optimizing epoch {}'.format(epoch+1))
      epoch_loss = 0.
      pbar = tqdm(total=len(X))
      for i, (X_batch, Y_batch) in enumerate(
          mini_batch(batch_size, X, Y, shuffle=True)):
        pbar.update(len(X_batch))
        _, loss, _ = sess.run([self.pred, self.loss, self.train], 
            feed_dict={
              self.X: X_batch,
              self.Y: Y_batch
            })
        epoch_loss += loss
      pbar.close()
      print('Loss: {}'.format(epoch_loss))
      if X_test is not None and Y_test is not None:
        score = self.validate(sess, X_test, Y_test, batch_size)
        print('Validation score: {}'.format(score))

  def validate(self, sess, X_test, Y_test, batch_size):
    score = 0
    for i, (X_batch, Y_batch) in enumerate(
        mini_batch(batch_size, X_test, Y_test)):
      pred = sess.run(self.pred, feed_dict={self.X: X_batch})
      pred = np.argmax(pred, axis=1)
      score += (pred == Y_batch).sum()
    return score / len(X_test)










