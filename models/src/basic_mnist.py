import tensorflow as tf
from tqdm import tqdm
from util import get_minibatches
from ops import fc, cnn, rnn, bi_rnn


class MnistModel(object):
  """Adapted from Tensorflow repo"""
  def __init__(self, input_dim, output_dim, lr):
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.lr = lr
    self.X_placeholder = tf.placeholder(tf.float32, [None, self.input_dim])
    self.Y_placeholder = tf.placeholder(tf.float32, [None, self.output_dim])
    self.prediction = fc('fc1', self.X_placeholder, self.output_dim)
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self.prediction, labels=self.Y_placeholder))
    self.train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

  def optimize(self, sess, mnist, epochs, batch_size):
    losses = []
    pbar = tqdm(total=epochs)
    for epoch in range(epochs):
      pbar.update()
      epoch_loss = 0.
      x, y = mnist.train.next_batch(batch_size)
      _, loss, train = sess.run([self.prediction, self.loss, self.train], 
          feed_dict={
            self.X_placeholder: x,
            self.Y_placeholder: y,
          })
      # print('\nLoss is {}\n'.format(epoch_loss))
    return losses









