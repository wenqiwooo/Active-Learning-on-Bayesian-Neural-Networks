import os
import sys
import datetime

import click
import numpy as np
np.random.seed(42)

from scipy.special import gamma,psi
from scipy.linalg import det
from numpy import pi
from sklearn.neighbors import NearestNeighbors

import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from inference_dropout import InferenceDropout as Dropout

num_classes = 10
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


class BayesianCNN(object):
    """
    Wrapper around Keras model.
    """

    def __init__(self, loss, optimizer):
        """
        Initializes the Keras model.
        """
        self._saved = None
        self._loss = loss
        self._optimizer = optimizer
        self.init_model()


    def optimize(self, x, y, epochs=12, batch_size=32):
        """
        Fits the model given x and y.

        Params:
            - x: input data
            - y: label data
            - epochs: times to train the neural network
            - batch_size: how big of a batch to train each time

        Returns: None
        """

        self._model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=0)

    def optimize_batch(self, batch_x, batch_y):
        """
        Fits the model to a single batch of x and y.

        Params:
            - batch_x: a single batch of input data
            - batch_y: a single batch of output data

        Returns: None
        """

        self._model.train_on_batch(batch_x, batch_y)

    def predict(self, x, batch_size=32):
        """
        Returns prediction given input data.

        Params:
            - x: input data
            - batch_size: how big of a batch to predict each iteration

        Returns: 2D array of predictions, one row for each row of input x
        """

        return self._model.predict(x, batch_size=batch_size)

    def predict_batch(self, x):
        """
        Returns prediction given a single batch of input data.

        Params:
            - x: input data

        Returns: 2D array of predictions, one row for each row of input x
        """

        return self._model.predict_on_batch(x)

    def evaluate(self, x, y):
        """
        Returns the loss and accuracy of the model.

        Params:
            - x: test input data
            - y: test label data

        Returns:
            - loss: loss value
            - accuracy: accuracy value
        """

        return self._model.evaluate(x, y, verbose=0)

    def sample(self, x, num_samples=10):
        """
        Gets the predictive mean and predictive variance of the neural network.
        This works because of how Dropout layers can be seen as placing a prior
        on the weights of a Neural Network.

        Params:
            - x: input data x

        Returns (2D array, 2D array):
            - predictive_mean
            - predictive_variance
        """

        probs = []
        for _ in range(num_samples):
            probs += [self._model.predict(x)]

        predictive_mean = np.mean(probs, axis=0)
        predictive_variance = np.var(probs, axis=0)

        return predictive_mean, predictive_variance

    def load_model(self, filename):
        """
        Loads a model from a saved pre-trained Keras model.

        Params:
            - filename: .h5 or equivalent file saved by Keras

        Returns: None
        """

        self._model = load_model(
            filename,
            custom_objects={'InferenceDropout': Dropout},
        )
        print(f'Keras model loaded from {filename}.')

    def save_model(self, save_dir, filename):
        """
        Saves our Keras model as a h5 file. NOTE: Requires h5py.

        Params:
            - save_dir: directory to save the model in
            - filename: name to store the model as
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        model_path = os.path.join(save_dir, filename)
        self._model.save(model_path)
        print('Saved trained model at %s ' % model_path)

    def init_model(self):
        if self._saved:
            K.clear_session()
            self.load_model(self._saved)
        else:
            model = Sequential()

            # Conv1
            model.add(Conv2D(20,
                            kernel_size=(5, 5),
                            input_shape=input_shape))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Conv2
            model.add(Conv2D(50, (5, 5)))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            # Fully connected
            model.add(Flatten())
            model.add(Dense(500, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss=self._loss,
                        optimizer=self._optimizer,
                        metrics=['accuracy'])

            self._model = model


def sum_of_mean_square_errors(var):
    """
    Returns the sum of mean square errors criterion.
    TODO: Figure out the crietrion here!

    Params:
        - var: predictive variance from classifier

    Returns: criterion value
    """

    return np.sum(var, axis=1)


def nearest_distances(X, k=1):
    '''
    X = array(N,M)
    N = number of points
    M = number of dimensions
    returns the distance to the kth nearest neighbor for every point in X
    '''
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    d, _ = knn.kneighbors(X) # the first nearest neighbor is itself
    return d[:, -1] # returns the distance to the kth nearest neighbor


def entropy(X, k=1):
    ''' Returns the entropy of the X.
    Parameters
    ===========
    X : array-like, shape (n_samples, n_features)
        The data the entropy of which is computed
    k : int, optional
        number of nearest neighbors for density estimation
    Notes
    ======
    Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
    of a random vector. Probl. Inf. Transm. 23, 95-101.
    See also: Evans, D. 2008 A computationally efficient estimator for
    mutual information, Proc. R. Soc. A 464 (2093), 1203-1215.
    and:
    Kraskov A, Stogbauer H, Grassberger P. (2004). Estimating mutual
    information. Phys Rev E 69(6 Pt 2):066138.
    '''

    # Distance to kth nearest neighbor
    r = nearest_distances(X, k) # squared distances
    n, d = X.shape
    volume_unit_ball = (pi**(.5*d)) / gamma(.5*d + 1)
    '''
    F. Perez-Cruz, (2008). Estimation of Information Theoretic Measures
    for Continuous Random Variables. Advances in Neural Information
    Processing Systems 21 (NIPS). Vancouver (Canada), December.
    return d*mean(log(r))+log(volume_unit_ball)+log(n-1)-log(k)
    '''
    return (d*np.mean(np.log(r + np.finfo(X.dtype).eps))
            + np.log(volume_unit_ball) + psi(n) - psi(k))


def active_learn_random(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10, evaluate=None):
    """
    Starts an active learning process to train the model using the mean squared
    error criterion, a.k.a our variance.
    """
    for i in range(iters):
        print(f'Running active learning iteration {i+1}')

        _, var = model.sample(unobserved_x)

        # Get the data points with top k variance values
        idx = np.random.choice(len(unobserved_x), k, replace=False)

        # Add that to our training data
        init_x = np.append(init_x, np.take(unobserved_x, idx, axis=0), axis=0)
        init_y = np.append(init_y, np.take(unobserved_y, idx, axis=0), axis=0)

        print(f'Total data used so far: {init_x.shape[0]}')

        # Optimize the model again
        model.init_model()
        model.optimize(init_x, init_y)

        # Remove from unobserved data
        unobserved_x = np.delete(unobserved_x, idx, 0)
        unobserved_y = np.delete(unobserved_y, idx, 0)


def active_learn_mse(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10, evaluate=None):
    """
    Starts an active learning process to train the model using the mean squared
    error criterion, a.k.a our variance.
    """
    for i in range(iters):
        print(f'Running active learning iteration {i+1}')

        _, var = model.sample(unobserved_x)

        # Get the data points with top k variance values
        top_k = np.argpartition(sum_of_mean_square_errors(var), -k)[-k:]

        # Add that to our training data
        init_x = np.append(init_x, np.take(unobserved_x, top_k, axis=0), axis=0)
        init_y = np.append(init_y, np.take(unobserved_y, top_k, axis=0), axis=0)

        print(f'Total data used so far: {init_x.shape[0]}')

        # Optimize the model again
        model.init_model()
        model.optimize(init_x, init_y)

        # Remove from unobserved data
        unobserved_x = np.delete(unobserved_x, top_k, 0)
        unobserved_y = np.delete(unobserved_y, top_k, 0)


def active_learn_var_ratio(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10, evaluate=None):
    """
    Starts an active learning process to train the model using the maximum myopic
    entropy criterion
    """
    for i in range(iters):
        print(f'Running active learning iteration {i+1}')

        pred, _ = model.sample(unobserved_x)

        # Get the data points with top k variance values
        top_k = np.argpartition(1 - np.amax(pred, axis=1), -k)[-k:]

        # Add that to our training data
        init_x = np.append(init_x, np.take(unobserved_x, top_k, axis=0), axis=0)
        init_y = np.append(init_y, np.take(unobserved_y, top_k, axis=0), axis=0)

        print(f'Total data used so far: {init_x.shape[0]}')

        # Optimize the model again
        model.init_model()
        model.optimize(init_x, init_y)

        # Remove from unobserved data
        unobserved_x = np.delete(unobserved_x, top_k, 0)
        unobserved_y = np.delete(unobserved_y, top_k, 0)


def active_learn_max_entropy(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10, evaluate=None):
    """
    Starts an active learning process to train the model using the maximum
    entropy criterion
    """

    for i in range(iters):
        print(f'Running active learning iterations {i+1}')
        pred = np.zeros(unobserved_y.shape)
        idx = None

        for _ in range(10):
            pred += model.predict(unobserved_x)

        pred /= iters
        entropy = np.sum(-1 * pred * np.log(pred + 1e-9), axis=1)
        idx = np.argpartition(entropy, -k)[-k:]

        print(f'Total data used so far: {init_x.shape[0]}')

        # Add best data point to our training data
        init_x = np.append(init_x, np.take(unobserved_x, idx, axis=0), axis=0)
        init_y = np.append(init_y, np.take(unobserved_y, idx, axis=0), axis=0)

        # Optimize the model again
        model.init_model()
        model.optimize(init_x, init_y)

        # Remove from unobserved data
        unobserved_x = np.delete(unobserved_x, idx, 0)
        unobserved_y = np.delete(unobserved_y, idx, 0)

        if evaluate:
            evaluate()


def load_data():
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return (x_train, y_train), (x_test, y_test)

@click.command()
@click.option('--initial', default=50, help='Number of rows of initial data to train the neural network')
@click.option('--unobserved', default=3000, help='Total number of unobserved data')
@click.option('--samples', default=10, help='Number of samples to pick in each active learning iteration (active learning batch)')
@click.option('--datasize', default=1000, help='Total rows of data to use for our active learning, excluding the initialization data')
def train(initial, unobserved, samples, datasize):
    print('=============================================')
    print(f'Running Bayesian Neural Network experiment:')
    print(f'Initial data size: {initial}')
    print(f'Unobserved data size: {unobserved}')
    print(f'Active Learning Batch: {samples}')
    print(f'Total datasize used for active learning: {datasize}')
    print('=============================================')

    # Load MNIST datset
    (x_train, y_train), (x_test, y_test) = load_data()

    # Initialize a model
    m = BayesianCNN(keras.losses.kullback_leibler_divergence,
                    keras.optimizers.Adadelta())

    init_x, init_y = x_train[:initial], y_train[:initial]
    m.optimize(init_x, init_y)

    # Let the model actively learn on its own
    unobserved_x, unobserved_y = x_train[initial:initial+unobserved], y_train[initial:initial+unobserved]

    iters = datasize // samples

    # Active learning
    active_learn_functions = {
        'Random': active_learn_random,
        'Max Mean Var': active_learn_mse,
        'Max Var Ratios': active_learn_var_ratio,
        'Max Entropy': active_learn_max_entropy,
    }

    for name, f in active_learn_functions.items():
        print('==============================')
        print(f'Running experiments for {name}')
        print('==============================')

        m.init_model()

        def evaluate():
            _, accuracy = m.evaluate(x_test, y_test)
            print(f'Accuracy: {accuracy}')

        f(m, init_x, init_y, unobserved_x, unobserved_y, iters=iters, k=samples, evaluate=evaluate)

        # Evaluate our model against test set!
        loss, accuracy = m.evaluate(x_test, y_test)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    train()
