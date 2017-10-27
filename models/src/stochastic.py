import math
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _checked_scope


def gaussian_mixture_nll(samples, mixing_weights, mean1, mean2, std1, std2):
  """
  Computes the NLL from a mixture of two gaussian distributions with the given
  means and standard deviations, mixing weights and samples.
  """
  gaussian1 = (1.0/tf.sqrt(2.0 * std1 * math.pi)) * \
      tf.exp(- tf.square(samples - mean1) / (2.0 * std1))
  gaussian2 = (1.0/tf.sqrt(2.0 * std2 * math.pi)) * \
      tf.exp(- tf.square(samples - mean2) / (2.0 * std2))
  mixture = (mixing_weights[0] * gaussian1) + (mixing_weights[1] * gaussian2)
  return - tf.log(mixture)


def compute_kl_divergence(gaussian1, gaussian2):
  mean1, sigma1 = gaussian1
  mean2, sigma2 = gaussian2
  # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
  kl = tf.log(sigma2) - tf.log(sigma1) + \
      ((tf.square(sigma1) + tf.square(mean1 - mean2)) / \
      (2 * tf.square(sigma2))) - 0.5
  return tf.reduce_mean(kl)


def inverse_softplus(sd, shape):
  return tf.log(tf.exp(sd) - 1.) * tf.ones(shape)


def get_random_normal_var(name, mean, sd, shape, dtype):
  mean = tf.get_variable('{}_mean'.format(name), shape,
      initializer=tf.constant_initializer(mean),
      dtype=dtype)
  sd = inverse_softplus(sd, shape)
  sd = tf.get_variable('{}_sd'.format(name), initializer=sd, dtype=dtype)
  sd = tf.nn.softplus(sd)
  W = mean + (tf.random_normal(shape, 0., 1., dtype) * sd)
  return W, mean, sd


class ExternallyParameterizedLSTM(BasicLSTMCell):
  """
  A simple extension of an LSTM in which the weights are passed in to the class,
  rather than being automatically generated inside the cell when it is called.
  This allows us to parameterize them in other, funky ways.
  """
  def __init__(self, weight, bias, **kwargs):
    self.weight = weight
    self.bias = bias
    super(ExternallyParameterizedLSTM, self).__init__(**kwargs)

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "basic_lstm_cell", reuse=self._reuse):
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

      all_inputs = tf.concat([inputs, h], 1)

      # W is [embedding_size + hidden_size, 4 * hidden_size]
      # b is [4 * hidden_size]
      concat = tf.nn.bias_add(tf.matmul(all_inputs, self.weight), self.bias)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

      new_c = (c * tf.sigmoid(f + self._forget_bias) + tf.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat([new_c, new_h], 1)
      return new_h, new_state







