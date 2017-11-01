import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Normal
import matplotlib.pyplot as plt
from tqdm import tqdm


class MnistModel(object):
  def __init__(self, mnist, input_dim=784, output_dim=10, iterations=5000, 
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
      sw = self.qw.sample()
      sb = self.qb.sample()
      prob = tf.nn.softmax(tf.matmul(X_test, sw) + sb)
      probs.append(prob.eval())
    accuracies = []
    for prob in probs:
      pred = np.argmax(prob, axis=1)
      acc = (pred == Y_test).mean() * 100
      accuracies.append(acc)
    return accuracies

