import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.datasets import cifar10
from cifar10_bcnn2 import Cifar10BCNN, BayesianDropout

SAVE_DIR = '../checkpoint'
DATA_DIR = '../data'
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

sess = tf.InteractiveSession()
model = BayesianDropout(epochs=50, data_size=50000, batch_size=50)
saver = tf.train.Saver()
saver.restore(sess, "../checkpoint/beta_dropout.ckpt")

test_img = np.load('../data/test_sample.npy')

pred = model.predict(test_img, batch_size=1, n_samples=100)
print(pred)
