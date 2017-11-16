import math
from edward.models import Categorical, Normal, Bernoulli, Beta
import numpy as np
import tensorflow as tf
import edward as ed
from tqdm import tqdm


def lenet_dropout(w1, b1, w2, b2, w3, b3, w4, b4, d, x):
  h = tf.reshape(x, (-1, 28, 28, 1))
  h = tf.nn.bias_add(tf.nn.conv2d(h, w1, (1, 1, 1, 1), 'SAME'), b1)
  h = tf.nn.max_pool(h, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
  h = tf.nn.bias_add(tf.nn.conv2d(h, w2, (1, 1, 1, 1), 'SAME'), b2)
  h = tf.nn.max_pool(h, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME')
  h = tf.reshape(h, (-1, 7*7*50))
  h = tf.matmul(h, w3) + b3
  h = tf.nn.relu(h)
  h = tf.nn.dropout(h, d)
  h = tf.matmul(h, w4) + b4
  return h


class MnistBetaDropout(object):
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=10000, 
      batch_size=100):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.iterations = iterations
    self.batch_size = batch_size

    self.global_step = tf.Variable(
      initial_value=0, name='global_step', trainable=False)
    self.x = tf.placeholder(tf.float32, shape=(None, 784))
    self.y = tf.placeholder(tf.int32, shape=(None,))

    self.w1 = tf.get_variable('w1', (5, 5, 1, 20), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.b1 = tf.get_variable('b1', (20, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.w2 = tf.get_variable('w2', (5, 5, 20, 50), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.b2 = tf.get_variable('b2', (50, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.w3 = tf.get_variable('w3', (7*7*50, 500), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.b3 = tf.get_variable('b3', (500, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.w4 = tf.get_variable('w4', (500, 10), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.b4 = tf.get_variable('b4', (10, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))

    # Prior distribution
    self.d = Beta(20., 20.)

    self.qd = Beta(
        tf.Variable(20., tf.float32, name='qd_a'), 
        tf.Variable(20., tf.float32, name='qd_b'))

    self.nn = lenet_dropout(
        self.w1, self.b1, self.w2, self.b2, 
        self.w3, self.b3, self.w4, self.b4, 
        self.d, self.x)

    self.categorical = Categorical(self.nn)

    self.inference = ed.KLqp({
        self.d: self.qd,
      }, data={self.categorical: self.y})

    self.lr = tf.train.exponential_decay(
        1e-3, self.global_step, 10000, 0.95, staircase=True)

    self.optimizer = tf.train.AdamOptimizer(self.lr)

    self.inference.initialize(
        n_iter=self.iterations, optimizer=self.optimizer, 
        global_step=self.global_step)
    # self.inference.initialize(n_iter=self.iterations, 
    #     scale={self.categorical: mnist.train.num_examples / self.batch_size})

  def optimize(self, mnist):
    variables_names =['qd_a:0', 'qd_b:0']
    sess = ed.get_session()

    qd_a, qd_b = sess.run(variables_names)
    print('Prior >> alpha: {}   beta: {}'.format(qd_a, qd_b))

    for _ in range(self.inference.n_iter):
      X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
      info_dict = self.inference.update(feed_dict={
          self.x: X_batch,
          self.y: Y_batch
        })
      self.inference.print_progress(info_dict)

    qd_a, qd_b = sess.run(variables_names)
    print('Posterior >> alpha: {}   beta: {}'.format(qd_a, qd_b))

  def validate(self, mnist, n_samples):
    X_test = mnist.test.images
    Y_test = mnist.test.labels
    probs = []
    for _ in range(n_samples):
      prob = tf.nn.softmax(self.realize_network(X_test))
      probs.append(prob.eval())
    accuracies = []
    for prob in probs:
      pred = np.argmax(prob, axis=1)
      acc = (pred == Y_test).mean() * 100
      accuracies.append(acc)
    return accuracies

  def realize_network(self, x):
    sd = self.qd.sample()
    return tf.nn.softmax(lenet_dropout(
        self.w1, self.b1, self.w2, self.b2, 
        self.w3, self.b3, self.w4, self.b4, 
        sd, x))
