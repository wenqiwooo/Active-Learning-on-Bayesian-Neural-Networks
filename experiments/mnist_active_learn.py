import click
import numpy as np
import keras
from keras.datasets import mnist
from keras import backend as K

from mnist_dropout import BayesianCNN


np.random.seed(42)

num_classes = 10
img_rows, img_cols = 28, 28
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)


def mean_var(var):
    _, classes = var.shape

    return 1 / classes * np.sum(var, axis=1)


def active_learn_random(model,
                        init_x, init_y,
                        unobserved_x, unobserved_y,
                        iters=100, k=10):
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


def active_learn_mean_var(model,
                          init_x, init_y,
                          unobserved_x, unobserved_y,
                          iters=100, k=10):
    """
    Starts an active learning process to train the model using the sum of mean
    variance criterion.
    """
    for i in range(iters):
        print(f'Running active learning iteration {i+1}')

        _, var = model.sample(unobserved_x)

        # Get the data points with top k variance values
        top_k = np.argpartition(mean_var(var), -k)[-k:]

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


def active_learn_max_entropy(model,
                             init_x, init_y,
                             unobserved_x, unobserved_y,
                             iters=100, k=10):
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


def active_learn_var_ratio(model,
                           init_x, init_y,
                           unobserved_x, unobserved_y,
                           iters=100, k=10):
    """
    Starts an active learning process to train the model using the maximum
    variation ratio criterion
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
@click.option('--initial',
              default=50,
              help='Number of rows of initial data to train the neural network')
@click.option('--unobserved',
              default=3000,
              help='Total number of unobserved data')
@click.option('--samples',
              default=10,
              help='Number of samples to pick in each active learning \
                    iteration (active learning batch)')
@click.option('--datasize',
              default=1000,
              help='Total rows of data to use for our active learning, \
                excluding the initialization data')
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
    unobserved_x = x_train[initial:initial+unobserved]
    unobserved_y = y_train[initial:initial+unobserved]

    iters = datasize // samples

    # Active learning
    active_learn_functions = {
        'Random': active_learn_random,
        'Maximum Mean Variance': active_learn_mean_var,
        'Maximum Variation Ratio': active_learn_var_ratio,
        'Maximum Entropy': active_learn_max_entropy,
    }

    for name, f in active_learn_functions.items():
        print('==============================')
        print(f'Running experiments for {name}')
        print('==============================')

        m.init_model()

        f(m, init_x, init_y, unobserved_x, unobserved_y, iters=iters, k=samples)

        # Evaluate our model against test set!
        loss, accuracy = m.evaluate(x_test, y_test)
        print(f'Loss: {loss}')
        print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    train()
