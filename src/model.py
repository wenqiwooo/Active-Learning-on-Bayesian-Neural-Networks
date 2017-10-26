import tensorflow as tf
from tqdm import tqdm
from util import get_minibatches
from ops import fc, cnn, rnn, bi_rnn


class NerModel(object):
  def __init__(self, embeddings, dim_size, seq_len, hidden_size,
      output_size, lr, max_grad_norm):
    self.dim_size = dim_size
    self.seq_len = seq_len
    self.hidden_size = hidden_size
    self.output_size = output_size
    self.lr = lr
    self.max_grad_norm = max_grad_norm

    self.X_placeholder = tf.placeholder(tf.int32, [None, self.seq_len], name='X')
    self.Y_placeholder = tf.placeholder(tf.int32, [None, self.seq_len], name='Y')
    inputs = tf.nn.embedding_lookup(
        tf.Variable(embeddings, trainable=False),
        self.X_placeholder)

    (output_fw, output_bw), _ = bi_rnn('rnn1', inputs, self.hidden_size)
    rnn1 = tf.tanh(tf.concat([output_fw, output_bw], 2))
    fc1 = fc('fc1', rnn1, self.output_size)
    self.prediction = tf.nn.relu(fc1)

    self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=self.prediction,
        labels=self.Y_placeholder))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.loss, tvars),
        self.max_grad_norm)
    self.train = optimizer.apply_gradients(zip(grads, tvars))


  def optimize(self, sess, X, Y, epochs, batch_size):
    for epoch in range(epochs):
      print('Optimizing: Epoch {}/{}'.format(epoch+1, epochs))
      epoch_loss = 0.
      pbar = tqdm(total=len(X))
      for i, (x, y) in tqdm(enumerate(get_minibatches(X, Y, batch_size))):
        pbar.update(len(x))      
        _, loss, train = sess.run([self.prediction, self.loss, self.train],
          feed_dict={
            self.X_placeholder: x,
            self.Y_placeholder: y,
          })
        epoch_loss += loss
      pbar.close()
      print('\nLoss is {}\n'.format(epoch_loss))













