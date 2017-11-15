import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Normal
from tqdm import tqdm


def MLP(w1, b1, w2, b2, w3, b3, w4, b4, X):
  fc1 = tf.nn.relu(tf.matmul(X, w1) + b1)
  fc2 = tf.nn.relu(tf.matmul(fc1, w2) + b2)
  # fc3 = tf.nn.relu(tf.matmul(fc2, w3) + b3)
  # fc4 = tf.matmul(fc3, w4) + b4
  fc3 = tf.matmul(fc2, w3) + b3
  return fc2


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
    self.w1_shape = (784, 64)
    self.w2_shape = (64, 32)
    self.w3_shape = (32, 10)
    self.w4_shape = (16, 10)

    scale = 1 / math.sqrt(784*64 + 64*32)

    self.w1 = Normal(loc=tf.zeros(self.w1_shape), scale=tf.ones(self.w1_shape))
    self.b1 = Normal(
        loc=tf.zeros(self.w1_shape[-1]),scale=tf.ones(self.w1_shape[-1]))

    self.w2 = Normal(loc=tf.zeros(self.w2_shape), scale=tf.ones(self.w2_shape))
    self.b2 = Normal(
        loc=tf.zeros(self.w2_shape[-1]), scale=tf.ones(self.w2_shape[-1]))

    self.w3 = Normal(
        loc=tf.zeros(self.w3_shape), scale=tf.ones(self.w3_shape))
    self.b3 = Normal(
        loc=tf.zeros(self.w3_shape[-1]), scale=tf.ones(self.w3_shape[-1]) * scale)

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
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w1_shape), name='qw1_scale')))
    self.qb1 = Normal(
        loc=tf.Variable(tf.random_normal([self.w1_shape[-1]]), name='qb1_loc'),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.w1_shape[-1]]), name='qb1_scale')))

    self.qw2 = Normal(
        loc=tf.Variable(tf.random_normal(self.w2_shape), name='qw2_loc'),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w2_shape), name='qw2_scale')))
    self.qb2 = Normal(
        loc=tf.Variable(tf.random_normal([self.w2_shape[-1]]), name='qb2_loc'),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.w2_shape[-1]]), name='qb2_scale')))

    self.qw3 = Normal(
        loc=tf.Variable(tf.random_normal(self.w3_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w3_shape))))
    self.qb3 = Normal(
        loc=tf.Variable(tf.random_normal([self.w3_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.w3_shape[-1]]))))

    self.qw4 = Normal(
        loc=tf.Variable(tf.random_normal(self.w4_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.w4_shape))))
    self.qb4 = Normal(
        loc=tf.Variable(tf.random_normal([self.w4_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.w4_shape[-1]]))))

    self.inference = ed.KLqp({
        self.w1: self.qw1, self.b1: self.qb1,
        self.w2: self.qw2, self.b2: self.qb2,
        self.w3: self.qw3, self.b3: self.qb3,
        # self.w4: self.qw4, self.b4: self.qb4,
      }, data={self.categorical: self.Y_placeholder})

    self.global_step = tf.Variable(
      initial_value=0, name='global_step', trainable=False)
    self.optimizer = tf.train.AdamOptimizer(1e-3)
    self.inference.initialize(optimizer=self.optimizer, global_step=self.global_step)

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
      variables_names =['qw1_loc:0', 'qw1_scale:0']
      sess = ed.get_session()
      qw1_loc, qw1_scale = sess.run(variables_names)
      qw1_scale = np.log(np.exp(qw1_scale) + 1)
      print(np.amax(qw1_loc / qw1_scale))

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

    self.inference.initialize(n_iter=self.iterations)
    # self.inference.initialize(n_iter=self.iterations, 
    #     scale={self.categorical: mnist.train.num_examples / self.batch_size})

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
    #probs = []
    probs = np.zeros((1000, 10))
    for _ in tqdm(range(n_samples)):
      sw = self.qw.sample()
      sb = self.qb.sample()
      prob = tf.nn.softmax(tf.matmul(X_test, sw) + sb)
      #probs.append(prob.eval())
      probs += prob.eval()
    # accuracies = []
    # for prob in probs:
    #   pred = np.argmax(prob, axis=1)
    #   acc = (pred == Y_test).mean() * 100
    #   accuracies.append(acc)
    
    pred = np.argmax(probs, axis=1)
    accuracies = (pred == Y_test).mean() * 100
    return accuracies

