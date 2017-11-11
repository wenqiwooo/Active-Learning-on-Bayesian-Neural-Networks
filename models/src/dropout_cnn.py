import os
import sys
import datetime

import numpy as np
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

        self._model.fit(x, y, epochs=epochs, batch_size=batch_size)

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
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
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

    return np.apply_along_axis(lambda x: np.trace(np.outer(x, x)), 1, var)


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


def active_learn_random(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10):
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


def active_learn_mse(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10):
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


def active_learn_max_entropy(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10):
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


def active_learn_mutual_information(model, init_x, init_y, unobserved_x, unobserved_y, iters=100, k=10):
    """
    Starts an active learning process to train the model using the maximum
    mutual information criterion.
    """
    for i in range(iters):
        print(f'Running active learning iteration {i+1}')

        max_mi = -np.inf
        idx = None

        # Reduce the sampling of points to just 10 samples
        for _ in range(10):
            # Naive selection: just choosing ONE data point
            j = np.random.randint(low=0, high=len(unobserved_x))
            pred = model.predict(np.take(unobserved_x, [0], axis=0))

            # Add prediction to observed data
            o_y = np.append(init_y, pred, axis=0)

            # Remove from unobserved data
            u_x = np.delete(unobserved_x, j, 0)

            # Calculate mutual information
            unobserved_preds = model.predict(u_x)

            print(o_y.shape)
            print(unobserved_preds.shape)

            # TODO: This part is not done yet.
            mi = entropy(unobserved_preds)

            if mi > max_mi:
                idx = j
                max_mi = mi

        print(f'Total data used so far: {init_x.shape[0]}')

        # Add best data point to our training data
        init_x = np.append(init_x, np.take(unobserved_x, [idx], axis=0), axis=0)
        init_y = np.append(init_y, np.take(unobserved_y, [idx], axis=0), axis=0)

        # Optimize the model again
        model.optimize(init_x, init_y)

        # Remove from unobserved data
        unobserved_x = np.delete(unobserved_x, idx, 0)
        unobserved_y = np.delete(unobserved_y, idx, 0)

def main():
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

    # Initialize a model
    m = BayesianCNN(keras.losses.kullback_leibler_divergence,
                    keras.optimizers.Adadelta())

    # We initially train the model with only 10 inputs
    init_x, init_y = x_train[:50], y_train[:50]
    m.optimize(init_x, init_y)

    # Let the model actively learn on its own
    unobserved_x, unobserved_y = x_train[50:3000], y_train[50:3000]

    iters = 10
    k = 50

    # active_learn_random(m, init_x, init_y, unobserved_x, unobserved_y, iters=iters, k=k)
    # active_learn_mse(m, init_x, init_y, unobserved_x, unobserved_y, iters=iters, k=k)
    active_learn_max_entropy(m, init_x, init_y, unobserved_x, unobserved_y, iters=iters, k=k)
    # active_learn_mutual_information(m, init_x, init_y, unobserved_x, unobserved_y, iters=iters, k=k)

    # Evaluate our model against test set!
    loss, accuracy = m.evaluate(x_test, y_test)
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main()
