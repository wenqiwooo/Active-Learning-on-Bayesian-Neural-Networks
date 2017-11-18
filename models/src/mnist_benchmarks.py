import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Normal
from tqdm import tqdm

class MnistPlain(object):
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=250, 
      batch_size=100):
    self.global_step = tf.Variable(
        initial_value=0, name='global_step', trainable=False)

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.iterations = iterations
    self.batch_size = batch_size

    self.X_placeholder = tf.placeholder(tf.float32, (None, self.input_dim))
    self.Y_placeholder = tf.placeholder(tf.int32, (None,))

    w_shape = (input_dim, output_dim)
    w = tf.get_variable('w', w_shape, dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable('b', [w_shape[-1]], dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.nn = tf.matmul(self.X_placeholder, w) + b

    self.pred = tf.nn.softmax(self.nn)

    self.loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.nn, labels=self.Y_placeholder))

    self.train = tf.train.AdamOptimizer(
        learning_rate=1e-3).minimize(self.loss, global_step=self.global_step)

  def optimize(self, sess, mnist):
    X_batches = []
    Y_batches = []
    for i in range(3):
      X, Y = mnist.train.next_batch(50)
      X_batches.append(X)
      Y_batches.append(Y)
    for i in range(self.iterations):
      # X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
      X_batch = X_batches[i % 3]
      Y_batch = Y_batches[i % 3]
      sess.run(self.train, feed_dict={
          self.X_placeholder: X_batch,
          self.Y_placeholder: Y_batch
        })

  def validate(self, sess, mnist):
    X_test = mnist.test.images[:1000]
    Y_test = mnist.test.labels[:1000]
    pred = sess.run(self.pred, feed_dict={
        self.X_placeholder: X_test,
        self.Y_placeholder: Y_test
      })
    pred = np.argmax(pred, axis=1)
    return (pred == Y_test).mean() * 100

