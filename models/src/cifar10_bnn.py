import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import edward as ed
from edward.models import Categorical, Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from cifar10 import load_training_data, load_test_data


def mini_batch(x, batch_size, y=None, shuffle=False):
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


mnist = input_data.read_data_sets('../data', one_hot=False)

EPOCHS = 100
N = 100
D = 3072
K = 10

images, classes, _ = load_test_data()
images = images.reshape((-1, D))
test_images = images[:1000]
test_classes = classes[:1000]

# test_images, test_classes = mnist.train.next_batch(N)

x = tf.placeholder(tf.float32, [None, D])
w = Normal(loc=tf.zeros([D, K]), scale=tf.ones([D, K]))
b = Normal(loc=tf.zeros(K), scale=tf.ones(K))
y = Categorical(tf.matmul(x, w) + b)

qw = Normal(loc=tf.Variable(tf.random_normal([D, K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([D, K]))))
qb = Normal(loc=tf.Variable(tf.random_normal([K])),
              scale=tf.nn.softplus(tf.Variable(tf.random_normal([K]))))

y_placeholder = tf.placeholder(tf.int32, [None, ])

inference = ed.KLqp({w: qw, b: qb}, data={y: y_placeholder})

#inference.initialize(n_iter=5000)
inference.initialize(
    n_iter=EPOCHS*100, n_print=100, scale={y: float(10000) / N})

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for _ in range(EPOCHS):
  for X_batch, Y_batch in mini_batch(x=images, y=classes, batch_size=N):
    info_dict = inference.update(feed_dict={x: X_batch, y_placeholder: Y_batch})
    inference.print_progress(info_dict)

X_test = tf.convert_to_tensor(test_images, np.float32)
Y_test = tf.convert_to_tensor(test_classes, np.float32)

n_samples = 10
prob_lst = []
samples = []
w_samples = []
b_samples = []
for _ in tqdm(range(n_samples)):
  w_samp = qw.sample()
  b_samp = qb.sample()
  w_samples.append(w_samp)
  b_samples.append(b_samp)
  prob = tf.nn.softmax(tf.matmul(X_test, w_samp) + b_samp)
  prob_lst.append(prob.eval())
  # sample = tf.concat([tf.reshape(w_samp, [-1]), b_samp], 0)
  # samples.append(sample.eval())

# Compute the accuracy of the model. 
# For each sample we compute the predicted class and compare with the test labels.
# Predicted class is defined as the one which as maximum proability.
# We perform this test for each (w,b) in the posterior giving us a set of accuracies
# Finally we make a histogram of accuracies for the test data.
accy_test = []
for prob in prob_lst:
  y_trn_prd = np.argmax(prob, axis=1).astype(np.float32)
  acc = np.mean(y_trn_prd == Y_test) * 100
  accy_test.append(acc)

plt.hist(accy_test)
plt.title("Histogram of prediction accuracies in the MNIST test data")
plt.xlabel("Accuracy")
plt.ylabel("Frequency")
plt.show()

