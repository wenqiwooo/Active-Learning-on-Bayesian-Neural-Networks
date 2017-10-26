import tensorflow as tf


def fc(name, X, out_dim):
  X_dim = X.get_shape().as_list()
  W = tf.get_variable(
      'W_{}'.format(name),
      [X_dim[2] , out_dim],
      dtype=tf.float32,
      initializer=tf.contrib.layers.xavier_initializer())
  b = tf.get_variable(
      'b_{}'.format(name),
      [out_dim, ],
      dtype=tf.float32,
      initializer=tf.constant_initializer(0.))
  fc = tf.map_fn(lambda x: tf.matmul(x, W) + b, X, dtype=tf.float32)
  return fc


def cnn(name, X, filter_dims, strides, out_channels):
  X_dim = X.get_shape().as_list()
  W = tf.get_variable(
      'W_{}'.format(name),
      [filter_dims[0], filter_dims[1] ,X_dim[3], out_channels],
      dtype=tf.float32,
      initializer=tf.contrib.layers.xavier_initializer())
  return tf.nn.conv2d(X, W, strides, padding='SAME', name=name)


def rnn(name, X, state_size):
  cell = tf.nn.rnn_cell.LSTMCell(state_size)
  return tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)


def bi_rnn(name, X, state_size):
  fw_cell = tf.nn.rnn_cell.LSTMCell(state_size)
  bw_cell = tf.nn.rnn_cell.LSTMCell(state_size)
  return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, X, dtype=tf.float32)
