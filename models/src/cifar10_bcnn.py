from tqdm import tqdm
from edward.models import Categorical, Normal
import numpy as np
import tensorflow as tf
import edward as ed

def mini_batch(x, y = None, shuffle = True, batch_size = 128):
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


class Cifar10BCNN(object):
  def __init__(self):
    self.setup_placeholders()
    self.setup_system()
    self.setup_variables()
    self.setup_inference()


  def setup_placeholders(self):
    with tf.variable_scope('placeholders') as scope:
      self.x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
      self.y = tf.placeholder(tf.int32, shape=(None,))


  def setup_system(self):
    with tf.variable_scope('bcnn'):
      self.filter1_size = 3 * 3 * 3 * 64
      self.filter1 = Normal(loc=tf.zeros(self.filter1_size), scale=tf.ones(self.filter1_size))
      self.bias1 = Normal(loc=tf.zeros(64), scale=tf.ones(64))

      kernel1 = tf.reshape(self.filter1, (3, 3, 3, 64))

      conv1 = tf.nn.bias_add(tf.nn.conv2d(self.x, kernel1, (1, 1, 1, 1), 'SAME', name='conv1'), self.bias1)
      conv1 = tf.nn.relu(conv1)
      maxpool1 = tf.nn.max_pool(conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool1')

      self.filter2_size = 3 * 3 * 64 * 128
      self.filter2 = Normal(loc=tf.zeros(self.filter2_size), scale=tf.ones(self.filter2_size))
      self.bias2 = Normal(loc=tf.zeros(128), scale=tf.ones(128))

      kernel2 = tf.reshape(self.filter2, (3, 3, 64, 128))

      conv2 = tf.nn.bias_add(tf.nn.conv2d(maxpool1, kernel2, (1, 1, 1, 1), 'SAME', name='conv2'), self.bias2)
      conv2 = tf.nn.relu(conv2)
      maxpool2 = tf.nn.max_pool(conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool2')

      conv_out = tf.reshape(maxpool2, (-1, 8192))

      self.fc_weights = Normal(loc=tf.zeros([8192, 10]), scale=tf.ones([8192, 10]))
      self.fc_bias = Normal(loc=tf.zeros(10), scale=tf.ones(10))
      self.scores = tf.matmul(conv_out, self.fc_weights) + self.fc_bias

      self.categorical = Categorical(self.scores)


  def setup_variables(self):
    with tf.variable_scope('variables'):
      self.qf1 = Normal(
          loc=tf.Variable(tf.random_normal([self.filter1_size])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.filter1_size]))),
      )
      self.qb1 = Normal(
          loc=tf.Variable(tf.random_normal([64])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([64]))),
      )

      self.qf2 = Normal(
          loc=tf.Variable(tf.random_normal([self.filter2_size])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.filter2_size]))),
      )
      self.qb2 = Normal(
          loc=tf.Variable(tf.random_normal([128])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([128]))),
      )

      self.qfcw = Normal(
          loc=tf.Variable(tf.random_normal([8192, 10])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([8192, 10]))),
      )
      self.qfcb = Normal(
          loc=tf.Variable(tf.random_normal([10])),
          scale=tf.nn.softplus(tf.Variable(tf.random_normal([10]))),
      )


  def setup_inference(self):
    with tf.variable_scope('inference'):
      latent_vars = {
        self.filter1: self.qf1,
        self.bias1: self.qb1,
        self.filter2: self.qf2,
        self.bias2: self.qb2,
        self.fc_weights: self.qfcw,
        self.fc_bias: self.qfcb,
      }
      data = {
        self.categorical: self.y
      }
      self.inference = ed.KLqp(latent_vars, data=data)
      self.inference.initialize()


  def optimize(self, session, x, y, epochs, batch_size):
    print('Optimizing %s training examples' % x.shape[0])
    pbar = tqdm(total=epochs)
    for i in tqdm(range(1, epochs + 1)):
      epoch_loss = 0
      for batch_x, batch_y in mini_batch(x, y, shuffle=True, batch_size=batch_size):
        epoch_loss += self.optimize_batch(batch_x, batch_y)
      pbar.update()
      pbar.set_postfix(loss=epoch_loss / len(x), refresh=False)   


  def optimize_batch(self, batch_x, batch_y):
    feed_dict = {
      self.x: batch_x,
      self.y: batch_y,
    }
    info = self.inference.update(feed_dict=feed_dict)
    return info['loss']


  def validate(self, sess, x, y, batch_size):
    pred = self.predict(sess, x, batch_size)
    pred_results = np.argmax(pred, axis=0)
    return (pred_results == y).sum() / len(x)


  def predict(self, sess, x, batch_size):
    predictions = []
    pbar = tqdm(total=len(x) // batch_size + 1)
    for batch_x in mini_batch(x, shuffle=False, batch_size=batch_size):
      predicted_probs = sess.run(self.predict_batch(),
          feed_dict={
            self.x: batch_x
          })
      predictions.extend(predicted_probs)
      pbar.update() 
    return predictions


  def predict_batch(self):
    kernel1 = tf.reshape(self.qf1.sample(), (3, 3, 3, 64))
    bias1 = self.qb1.sample()
    conv1 = tf.nn.bias_add(tf.nn.conv2d(self.x, kernel1, (1, 1, 1, 1), 'SAME', name='conv1'), bias1)
    conv1 = tf.nn.relu(conv1)
    maxpool1 = tf.nn.max_pool(conv1, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool1')

    kernel2 = tf.reshape(self.qf2.sample(), (3, 3, 64, 128))
    bias2 = self.qb2.sample()

    conv2 = tf.nn.bias_add(tf.nn.conv2d(maxpool1, kernel2, (1, 1, 1, 1), 'SAME', name='conv2'), bias2)
    conv2 = tf.nn.relu(conv2)
    maxpool2 = tf.nn.max_pool(conv2, (1, 2, 2, 1), (1, 2, 2, 1), 'SAME', name='maxpool2')

    conv_out = tf.reshape(maxpool2, (-1, 8192))

    fc_weights = self.qfcw.sample()
    fc_bias = self.qfcb.sample()
    probs = tf.nn.softmax(tf.matmul(conv_out, fc_weights) + fc_bias)

    return probs
