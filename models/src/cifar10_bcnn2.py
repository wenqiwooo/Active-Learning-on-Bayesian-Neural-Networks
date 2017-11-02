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


  def optimize(self, X, Y, epochs, batch_size):
    print('Optimizing {} training examples'.format(self.data_size))
    for i in range(1, epochs+1):
      for X_batch, Y_batch in mini_batch(batch_size, X, Y, shuffle=True):
        info_dict = self.inference.update(feed_dict={
            self.x: X_batch,
            self.y: Y_batch
          })
        self.inference.print_progress(info_dict)


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
    return acc / len(X_test)


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





