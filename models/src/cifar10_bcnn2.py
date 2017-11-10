import math
from edward.models import Categorical, Normal, Bernoulli, Beta
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


def DropoutCNN(f1, b1, f2, b2, fc_w1, fc_b1, fc_w2, fc_b2, fc_w3, fc_b3, 
    d1, d2, d3, d4, X):
  conv1 = tf.nn.bias_add(
      tf.nn.conv2d(X, f1, (1, 1, 1, 1), 'SAME', name='conv1'), b1)
  conv1 = tf.nn.max_pool(
      conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool1')
  conv1 = tf.nn.dropout(tf.nn.relu(conv1), d1)

  conv2 = tf.nn.bias_add(
      tf.nn.conv2d(conv1, f2, (1, 1, 1, 1), 'SAME', name='conv2'), b2)
  conv2 = tf.nn.max_pool(
      conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool2')
  conv2 = tf.nn.dropout(tf.nn.relu(conv2), d2)

  fc1 = tf.reshape(conv2, (-1, 16*8*8))
  fc1 = tf.matmul(fc1, fc_w1) + fc_b1
  fc1 = tf.nn.dropout(tf.nn.relu(fc1), d3)

  fc2 = tf.matmul(fc1, fc_w2) + fc_b2
  fc2 = tf.nn.dropout(tf.nn.relu(fc2), d4)

  fc3 = tf.matmul(fc2, fc_w3) + fc_b3
  return fc3


class BayesianDropout(object):
  def __init__(self, epochs, data_size, batch_size):
    self.epochs = epochs
    self.data_size = data_size
    self.batch_size = batch_size

    self.global_step = tf.Variable(
      initial_value=0, name='global_step', trainable=False)
    self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    self.y = tf.placeholder(tf.int32, shape=(None,))

    # Prior distribution
    self.d1 = Beta(8., 3.)
    self.d2 = Beta(8., 3.)
    self.d3 = Beta(8., 3.)
    self.d4 = Beta(8., 3.)

    self.qd1 = Beta(
        tf.Variable(8., tf.float32, name='qd1_a'), 
        tf.Variable(3., tf.float32, name='qd1_b'))
    self.qd2 = Beta(
        tf.Variable(8., tf.float32, name='qd2_a'), 
        tf.Variable(3., tf.float32, name='qd2_b'))
    self.qd3 = Beta(
        tf.Variable(8., tf.float32, name='qd3_a'), 
        tf.Variable(3., tf.float32, name='qd3_b'))
    self.qd4 = Beta(
        tf.Variable(8., tf.float32, name='qd4_a'), 
        tf.Variable(3., tf.float32, name='qd4_b'))

    self.f1 = tf.get_variable('f1', (5, 5, 3, 6), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.b1 = tf.get_variable('b1', (6, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.f2 = tf.get_variable('f2', (5, 5, 6, 16), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.b2 = tf.get_variable('b2', (16, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.fc_w1 = tf.get_variable('fc_w1', (16*8*8, 120), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.fc_b1 = tf.get_variable('fc_b1', (120, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.fc_w2 = tf.get_variable('fc_w2', (120, 84), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.fc_b2 = tf.get_variable('fc_b2', (84, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))
    self.fc_w3 = tf.get_variable('fc_w3', (84, 10), dtype=tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    self.fc_b3 = tf.get_variable('fc_b3', (10, ), dtype=tf.float32,
        initializer=tf.constant_initializer(0.))

    self.nn = DropoutCNN(self.f1, self.b1, self.f2, self.b2, 
        self.fc_w1, self.fc_b1, self.fc_w2, self.fc_b2, self.fc_w3, self.fc_b3,
        self.d1, self.d2, self.d3, self.d4, self.x)

    # self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=self.y, logits=self.nn)

    # self.train = tf.train.AdamOptimizer(
    #     learning_rate=1e-3).minimize(self.loss, global_step=self.global_step)

    self.categorical = Categorical(self.nn)

    self.inference = ed.KLqp({
        self.d1: self.qd1,
        self.d2: self.qd2,
        self.d3: self.qd3,
        self.d4: self.qd4
      }, data={self.categorical: self.y})

    self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    iterations = self.epochs * math.ceil(self.data_size / self.batch_size)
    self.inference.initialize(
        n_iter=iterations, optimizer=self.optimizer, global_step=self.global_step)

  def optimize(self, X, Y, epochs, batch_size, 
      X_test=None, Y_test=None, n_samples=10, sess=None):
    print('Optimizing {} training examples'.format(self.data_size))
    for i in range(1, epochs+1):
      print('Optimizing for epoch {}'.format(i+1))
      for X_batch, Y_batch in mini_batch(batch_size, X, Y, shuffle=True):
        # sess.run(self.train, feed_dict={
        #     self.x: X_batch,
        #     self.y: Y_batch
        #   })
        info_dict = self.inference.update(feed_dict={
            self.x: X_batch,
            self.y: Y_batch
          })
        self.inference.print_progress(info_dict)

      if X_test is not None and Y_test is not None:
        acc = self.validate(X_test[:1000], Y_test[:1000], batch_size, n_samples)
        print(acc)

  def validate(self, X_test, Y_test, batch_size, n_samples):
    X = tf.convert_to_tensor(X_test, np.float32)
    probs = []
    for i in tqdm(range(n_samples)):
      prob = self.realize_network(X)
      probs.append(prob.eval())
    acc = 0
    for prob in probs:
      pred = np.argmax(prob, axis=1)
      acc += (pred == Y_test).sum()
    return acc / (len(X_test) * n_samples)

  def predict(self, X, batch_size):
    predictions = []
    pbar = tqdm(total=len(x)//batch_size+1)
    for X_batch in mini_batch(batch_size, X):
      pred = self.realize_network(X_batch).eval()
      predictions.append(pred)
      pbar.update()
    pbar.close()
    return np.array(predictions)

  def realize_network(self, X):
    sd1 = self.qd1.sample()
    sd2 = self.qd2.sample()
    sd3 = self.qd3.sample()
    sd4 = self.qd4.sample()
    return tf.nn.softmax(DropoutCNN(
        self.f1, self.b1, self.f2, self.b2,
        self.fc_w1, self.fc_b1, self.fc_w2, self.fc_b2, self.fc_w3, self.fc_b3,
        sd1, sd2, sd3, sd4, X))


    # self.conv1 = nn.Conv2d(3, 6, 5)
    # self.d1 = nn.Dropout(p=0.2)
    # self.conv2 = nn.Conv2d(6, 16, 5)
    # self.d2 = nn.Dropout(p=0.2)
    # self.fc1   = nn.Linear(16*5*5, 120)
    # self.d3 = nn.Dropout(p=0.2)
    # self.fc2   = nn.Linear(120, 84)
    # self.d4 = nn.Dropout(p=0.2)
    # self.fc3   = nn.Linear(84, 10)    



def BCNN(f1, b1, f2, b2, f3, b3, fc_w1, fc_b1, fc_w2, fc_b2, X):
  conv1 = tf.nn.bias_add(
      tf.nn.conv2d(X, f1, (1, 1, 1, 1), 'SAME', name='conv1'), b1)
  conv1 = tf.nn.relu(conv1)
  maxpool1 = tf.nn.max_pool(
      conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool1')
  # 16 x 16 x 32

  conv2 = tf.nn.bias_add(
      tf.nn.conv2d(maxpool1, f2, (1, 1, 1, 1), 'SAME', name='conv2'), b2)
  conv2 = tf.nn.relu(conv2)
  maxpool2 = tf.nn.max_pool(
      conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool2')
  # 8 x 8 x 32

  conv3 = tf.nn.bias_add(
      tf.nn.conv2d(maxpool2, f3, (1, 1, 1, 1), 'SAME', name='conv3'), b3)
  conv3 = tf.nn.relu(conv3)
  maxpool3 = tf.nn.max_pool(
      conv3, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool3')
  # 4 x 4 x 64

  conv_out = tf.reshape(maxpool3, (-1, 1024))
  fc1 = tf.matmul(conv_out, fc_w1) + fc_b1
  fc1 = tf.nn.relu(fc1)
  fc2 = tf.matmul(fc1, fc_w2) + fc_b2

  return fc2


class Cifar10BCNN(object):
  def __init__(self, epochs, data_size, batch_size):
    self.epochs = epochs
    self.data_size = data_size
    self.batch_size = batch_size

    self.global_step = tf.Variable(
        initial_value=0, name='global_step', trainable=False)

    self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    self.y = tf.placeholder(tf.int32, shape=(None,))

    # Prior distribution
    self.f1_shape = (5, 5, 3, 32)
    self.f1 = Normal(loc=tf.zeros(self.f1_shape), scale=tf.ones(self.f1_shape))
    self.b1 = Normal(
        loc=tf.zeros(self.f1_shape[-1]),scale=tf.ones(self.f1_shape[-1]))

    self.f2_shape = (5, 5, 32, 32)
    self.f2 = Normal(loc=tf.zeros(self.f2_shape), scale=tf.ones(self.f2_shape))
    self.b2 = Normal(
        loc=tf.zeros(self.f2_shape[-1]), scale=tf.ones(self.f2_shape[-1]))

    self.f3_shape = (5, 5, 32, 64)
    self.f3 = Normal(loc=tf.zeros(self.f3_shape), scale=tf.ones(self.f3_shape))
    self.b3 = Normal(
        loc=tf.zeros(self.f3_shape[-1]), scale=tf.ones(self.f3_shape[-1]))

    self.fc_w1_shape = (1024, 64)
    self.fc_w1 = Normal(
        loc=tf.zeros(self.fc_w1_shape), scale=tf.ones(self.fc_w1_shape))
    self.fc_b1 = Normal(
        loc=tf.zeros(self.fc_w1_shape[-1]), scale=tf.ones(self.fc_w1_shape[-1]))

    self.fc_w2_shape = (64, 10)
    self.fc_w2 = Normal(
        loc=tf.zeros(self.fc_w2_shape), scale=tf.ones(self.fc_w2_shape))
    self.fc_b2 = Normal(
        loc=tf.zeros(self.fc_w2_shape[-1]), scale=tf.ones(self.fc_w2_shape[-1]))

    self.scores = BCNN(self.f1, self.b1, self.f2, self.b2, self.f3, self.b3,
        self.fc_w1, self.fc_b1, self.fc_w2, self.fc_b2, self.x)
    self.categorical = Categorical(self.scores)

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

    self.qf3 = Normal(
        loc=tf.Variable(tf.random_normal(self.f3_shape)),
        scale=tf.nn.softplus(tf.Variable(tf.random_normal(self.f3_shape))))
    self.qb3 = Normal(
        loc=tf.Variable(tf.random_normal([self.f3_shape[-1]])),
        scale=tf.nn.softplus(
            tf.Variable(tf.random_normal([self.f3_shape[-1]]))))

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
        self.f3: self.qf3, self.b3: self.qb3,
        self.fc_w1: self.qfc_w1, self.fc_b1: self.qfc_b1,
        self.fc_w2: self.qfc_w2, self.fc_b2: self.qfc_b2,
      }, data={self.categorical: self.y})

    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

    iterations = self.epochs * math.ceil(self.data_size / self.batch_size)
    self.inference.initialize(
        n_iter=iterations, optimizer=optimizer, global_step=self.global_step)
    # self.inference.initialize(
    #     n_iter=iterations, 
    #     scale={self.categorical: self.data_size / self.batch_size})


  def optimize(self, X, Y, epochs, batch_size, 
      X_test=None, Y_test=None, n_samples=10, saver=saver):
    print('Optimizing {} training examples'.format(self.data_size))
    for i in range(1, epochs+1):
      for X_batch, Y_batch in mini_batch(batch_size, X, Y, shuffle=True):
        info_dict = self.inference.update(feed_dict={
            self.x: X_batch,
            self.y: Y_batch
          })
        self.inference.print_progress(info_dict)
      if saver is not None:
        ed.get_session()
        saver.save(sess, '../checkpoint/beta_dropout.ckpt')
      # if X_test is not None and Y_test is not None:
      #   acc = self.validate(X_test, Y_test, batch_size, n_samples)
      #   print(acc)


  def validate(self, X_test, Y_test, batch_size, n_samples):
    X = tf.convert_to_tensor(X_test, np.float32)
    probs = []
    for i in tqdm(range(n_samples)):
      prob = self.realize_network(X)
      probs.append(prob.eval())
    acc = 0
    for prob in probs:
      pred = np.argmax(prob, axis=1)
      acc += (pred == Y_test).sum()
    return acc / (len(X_test) * n_samples)


  def predict(self, X, batch_size):
    predictions = []
    pbar = tqdm(total=len(x)//batch_size+1)
    for X_batch in mini_batch(batch_size, X):
      pred = self.realize_network(X_batch).eval()
      predictions.append(pred)
      pbar.update()
    pbar.close()
    return np.array(predictions)


  def realize_network(self, X):
    sf1 = self.qf1.sample()
    sb1 = self.qb1.sample()
    sf2 = self.qf2.sample()
    sb2 = self.qb2.sample()
    sf3 = self.qf3.sample()
    sb3 = self.qb3.sample()
    sfc_w1 = self.qfc_w1.sample()
    sfc_b1 = self.qfc_b1.sample()
    sfc_w2 = self.qfc_w2.sample()
    sfc_b2 = self.qfc_b2.sample()
    return tf.nn.softmax(BCNN(sf1, sb1, sf2, sb2, sf3, sb3, sfc_w1, sfc_b1,
        sfc_w2, sfc_b2, X))





