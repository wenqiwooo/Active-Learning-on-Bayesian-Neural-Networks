import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Normal
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
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=5000, 
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


def MLP(w1, b1, w2, b2, w3, b3, w4, b4, X):
  h = tf.nn.relu(tf.matmul(X, w1) + b1)
  h = tf.nn.relu(tf.matmul(h, w2) + b2)
  h = tf.nn.relu(tf.matmul(h, w3) + b3)
  h = tf.nn.relu(tf.matmul(h, w4) + b4)
  return h


class MnistMLP(object):
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=10000, 
      batch_size=100):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.iterations = iterations
    self.batch_size = batch_size

    self.X_placeholder = tf.placeholder(tf.float32, (None, self.input_dim))
    self.Y_placeholder = tf.placeholder(tf.int32, (None,))

    # Prior distribution
    self.w1_shape = (784, 32)
    self.w2_shape = (32, 10)
    self.w3_shape = (10, 10)
    self.w4_shape = (16, 10)

    scale = 1 / (784*64 + 64*10)
    scale_root = math.sqrt(scale)

    self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
    self.b1 = Normal(
        loc=tf.zeros(self.w1_shape[-1]),scale=tf.ones(self.w1_shape[-1]))

    self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
    self.b2 = Normal(
        loc=tf.zeros(self.w2_shape[-1]), scale=tf.ones(self.w2_shape[-1]))

    self.w3 = Normal(
        loc=tf.zeros(self.w3_shape), scale=tf.ones(self.w3_shape))
    self.b3 = Normal(
        loc=tf.zeros(self.w3_shape[-1]), scale=tf.ones(self.w3_shape[-1]))

    self.w4 = Normal(
        loc=tf.zeros(self.w4_shape), scale=tf.ones(self.w4_shape))
    self.b4 = Normal(
        loc=tf.zeros(self.w4_shape[-1]), scale=tf.ones(self.w4_shape[-1]))

    self.nn = MLP(self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, 
        self.w4, self.b4, self.X_placeholder)
    self.categorical = Categorical(self.nn)

    # Q distribution
    self.qw1 = Normal(
        loc=tf.Variable(tf.random_normal(self.w1_shape), name='qw1_loc'),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal(self.w1_shape), name='qw1_scale')))
    self.qb1 = Normal(
        loc=tf.Variable(tf.random_normal([self.w1_shape[-1]]), name='qb1_loc'),
        scale=tf.nn.softplus(
            tf.Variable(
                tf.random_normal([self.w1_shape[-1]]), name='qb1_scale')))

    self.qw2 = Normal(
        loc=tf.Variable(tf.random_normal(self.w2_shape), name='qw2_loc'),
        scale=tf.nn.softplus(tf.Variable(
            tf.random_normal(self.w2_shape), name='qw2_scale')))
    self.qb2 = Normal(
        loc=tf.Variable(tf.random_normal([self.w2_shape[-1]]), name='qb2_loc'),
        scale=tf.nn.softplus(
            tf.Variable(
                tf.random_normal([self.w2_shape[-1]]), name='qb2_scale')))

    self.qw3 = Normal(
        loc=tf.Variable(tf.random_normal(self.w3_shape), name='qw3_loc'),
        scale=tf.nn.softplus(tf.Variable(
            tf.random_normal(self.w3_shape), name='qw3_scale')))
    self.qb3 = Normal(
        loc=tf.Variable(tf.random_normal([self.w3_shape[-1]]), name='qb3_loc'),
        scale=tf.nn.softplus(
            tf.Variable(
                tf.random_normal([self.w3_shape[-1]]), name='qb3_scale')))

    self.qw4 = Normal(
        loc=tf.Variable(tf.random_normal(self.w4_shape), name='qw4_loc'),
        scale=tf.nn.softplus(tf.Variable(
            tf.random_normal(self.w4_shape), name='qw4_scale')))
    self.qb4 = Normal(
        loc=tf.Variable(tf.random_normal([self.w4_shape[-1]]), name='qb4_loc'),
        scale=tf.nn.softplus(
            tf.Variable(
                tf.random_normal([self.w4_shape[-1]]), name='qb4_scale')))

    self.inference = ed.KLqp({
        self.w1: self.qw1, self.b1: self.qb1,
        self.w2: self.qw2, self.b2: self.qb2,
        self.w3: self.qw3, self.b3: self.qb3,
        self.w4: self.qw4, self.b4: self.qb4,
      }, data={self.categorical: self.Y_placeholder})

    self.inference.initialize(n_iter=self.iterations, 
        scale={self.categorical: mnist.train.num_examples / self.batch_size})

  def optimize(self, mnist):
    for _ in range(self.inference.n_iter):
      X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
      info_dict = self.inference.update(feed_dict={
          self.X_placeholder: X_batch,
          self.Y_placeholder: Y_batch
        })
      self.inference.print_progress(info_dict)

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

  def realize_network(self, X):
    sw1 = self.qw1.sample()
    sb1 = self.qb1.sample()
    sw2 = self.qw2.sample()
    sb2 = self.qb2.sample()
    sw3 = self.qw3.sample()
    sb3 = self.qb3.sample()
    sw4 = self.qw4.sample()
    sb4 = self.qb4.sample()
    return tf.nn.softmax(MLP(sw1, sb1, sw2, sb2, sw3, sb3,
        sw4, sb4, X))


def BCNN(f1, b1, f2, b2, fc_w1, fc_b1, fc_w2, fc_b2, X):
  X = tf.reshape(X, (-1, 28, 28, 1))

  conv1 = tf.nn.bias_add(
      tf.nn.conv2d(X, f1, (1, 1, 1, 1), 'SAME', name='conv1'), b1)
  conv1 = tf.nn.relu(conv1)
  maxpool1 = tf.nn.max_pool(
      conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool1')
  # 14 x 14 x 16

  conv2 = tf.nn.bias_add(
      tf.nn.conv2d(maxpool1, f2, (1, 1, 1, 1), 'SAME', name='conv2'), b2)
  conv2 = tf.nn.relu(conv2)
  maxpool2 = tf.nn.max_pool(
      conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool2')
  # 7 x 7 x 32

  conv_out = tf.reshape(maxpool2, (-1, 784))
  fc1 = tf.matmul(conv_out, fc_w1) + fc_b1
  fc1 = tf.nn.relu(fc1)
  fc2 = tf.matmul(fc1, fc_w2) + fc_b2

  return fc1


class MnistCNN(object):
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=20000, 
      batch_size=100):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.iterations = iterations
    self.batch_size = batch_size

    self.X_placeholder = tf.placeholder(tf.float32, (None, self.input_dim))
    self.Y_placeholder = tf.placeholder(tf.int32, (None,))

    # Prior distribution
    self.f1_shape = (5, 5, 1, 8)
    self.f1 = Normal(loc=tf.zeros(self.f1_shape), scale=tf.ones(self.f1_shape))
    self.b1 = Normal(
        loc=tf.zeros(self.f1_shape[-1]),scale=tf.ones(self.f1_shape[-1]))

    self.f2_shape = (5, 5, 8, 16)
    self.f2 = Normal(loc=tf.zeros(self.f2_shape), scale=tf.ones(self.f2_shape))
    self.b2 = Normal(
        loc=tf.zeros(self.f2_shape[-1]), scale=tf.ones(self.f2_shape[-1]))

    self.fc_w1_shape = (784, 64)
    self.fc_w1 = Normal(
        loc=tf.zeros(self.fc_w1_shape), scale=tf.ones(self.fc_w1_shape))
    self.fc_b1 = Normal(
        loc=tf.zeros(self.fc_w1_shape[-1]), scale=tf.ones(self.fc_w1_shape[-1]))

    self.fc_w2_shape = (64, 10)
    self.fc_w2 = Normal(
        loc=tf.zeros(self.fc_w2_shape), scale=tf.ones(self.fc_w2_shape))
    self.fc_b2 = Normal(
        loc=tf.zeros(self.fc_w2_shape[-1]), scale=tf.ones(self.fc_w2_shape[-1]))

    self.nn = BCNN(self.f1, self.b1, self.f2, self.b2,
        self.fc_w1, self.fc_b1, self.fc_w2, self.fc_b2, self.X_placeholder)
    self.categorical = Categorical(self.nn)

    # Q distribution
    self.qf1 = Normal(
        loc=tf.Variable(tf.random_normal(self.f1_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.f1_shape))))
    self.qb1 = Normal(
        loc=tf.Variable(tf.random_normal([self.f1_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.f1_shape[-1]]))))

    self.qf2 = Normal(
        loc=tf.Variable(tf.random_normal(self.f2_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.f2_shape))))
    self.qb2 = Normal(
        loc=tf.Variable(tf.random_normal([self.f2_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.f2_shape[-1]]))))

    self.qfc_w1 = Normal(
        loc=tf.Variable(tf.random_normal(self.fc_w1_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.fc_w1_shape))))
    self.qfc_b1 = Normal(
        loc=tf.Variable(tf.random_normal([self.fc_w1_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.fc_w1_shape[-1]]))))

    self.qfc_w2 = Normal(
        loc=tf.Variable(tf.random_normal(self.fc_w2_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.fc_w2_shape))))
    self.qfc_b2 = Normal(
        loc=tf.Variable(tf.random_normal([self.fc_w2_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.fc_w2_shape[-1]]))))

    self.inference = ed.KLqp({
        self.f1: self.qf1, self.b1: self.qb1,
        self.f2: self.qf2, self.b2: self.qb2,
        self.fc_w1: self.qfc_w1, self.fc_b1: self.qfc_b1,
        self.fc_w2: self.qfc_w2, self.fc_b2: self.qfc_b2,
      }, data={self.categorical: self.Y_placeholder})

    self.inference.initialize(n_iter=self.iterations, 
        scale={self.categorical: mnist.train.num_examples / self.batch_size})

  def optimize(self, mnist):
    for _ in range(self.inference.n_iter):
      X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
      info_dict = self.inference.update(feed_dict={
          self.X_placeholder: X_batch,
          self.Y_placeholder: Y_batch
        })
      self.inference.print_progress(info_dict)

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

  def realize_network(self, X):
    sf1 = self.qf1.sample()
    sb1 = self.qb1.sample()
    sf2 = self.qf2.sample()
    sb2 = self.qb2.sample()
    sfc_w1 = self.qfc_w1.sample()
    sfc_b1 = self.qfc_b1.sample()
    sfc_w2 = self.qfc_w2.sample()
    sfc_b2 = self.qfc_b2.sample()
    return tf.nn.softmax(BCNN(sf1, sb1, sf2, sb2, sfc_w1, sfc_b1,
        sfc_w2, sfc_b2, X))


class MnistModel(object):
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=250, 
      batch_size=100):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.iterations = iterations
    self.batch_size = batch_size

    self.X_placeholder = tf.placeholder(tf.float32, (None, self.input_dim))
    self.Y_placeholder = tf.placeholder(tf.int32, (None,))

    w_shape = (input_dim, output_dim)
    self.w = Normal(loc=tf.zeros(w_shape), scale=tf.ones(w_shape))
    self.b = Normal(loc=tf.zeros(w_shape[-1]), scale=tf.ones(w_shape[-1]))
    self.pred = Categorical(tf.matmul(self.X_placeholder, self.w) + self.b)

    self.qw = Normal(loc=tf.Variable(tf.random_normal(w_shape)), 
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(w_shape))))
    self.qb = Normal(loc=tf.Variable(tf.random_normal([w_shape[-1]])),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal([w_shape[-1]]))))

    self.inference = ed.KLqp({
        self.w: self.qw,
        self.b: self.qb
      }, data={self.pred: self.Y_placeholder})

    self.inference.initialize(n_iter=self.iterations, 
        scale={self.pred: mnist.train.num_examples / self.batch_size})

  def optimize(self, mnist):
    X_batches = []
    Y_batches = []
    for i in range(3):
      X, Y = mnist.train.next_batch(50)
      X_batches.append(X)
      Y_batches.append(Y)
    for i in range(self.inference.n_iter):
      # X_batch, Y_batch = mnist.train.next_batch(self.batch_size)
      X_batch = X_batches[i % 3]
      Y_batch = Y_batches[i % 3]
      info_dict = self.inference.update(feed_dict={
          self.X_placeholder: X_batch,
          self.Y_placeholder: Y_batch
        })
      self.inference.print_progress(info_dict)

  def validate(self, mnist, n_samples):
    X_test = mnist.test.images[:1000]
    Y_test = mnist.test.labels[:1000]
    probs = np.zeros((1000, 10))
    for _ in tqdm(range(n_samples)):
      sw = self.qw.sample()
      sb = self.qb.sample()
      prob = tf.nn.softmax(tf.matmul(X_test, sw) + sb)
      probs += prob.eval()
    pred = np.argmax(probs, axis=1)
    accuracies = (pred == Y_test).mean() * 100
    return accuracies

