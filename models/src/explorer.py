from functools import reduce
from bayes_opt import BayesianOptimization
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from ner_model_train import Trainer

# plt.style.use(['dark_background'])


def get_obj(trainer):
  """
  Args:
    trainer (Trainer)

  Returns:
    function: objective function
  """
  def obj(lr):
    """
    Args:
      lr (float): learning rate
      hs (int): hidden layer size
      bs (int): batch size
    """
    losses = trainer.train(lr=lr, hidden_size=50, batch_size=50)
    return -reduce(lambda x, y: x + y, losses) / len(losses)
  return obj


def posterior(bo, x):
  bo.gp.fit(bo.X, bo.Y)
  mu, sigma = bo.gp.predict(x, return_std=True)
  return mu, sigma


def _draw_axes_2d(g, x_label, y_label, x_limit, y_limit):
  g.set_xlim(x_limit)
  g.set_ylim(y_limit)
  g.set_xlabel(x_label, fontdict={'size': 14})
  g.set_ylabel(y_label, fontdict={'size': 14})


def plot_2d(bo, x_lower, x_upper, x_label='x'):
  """
  Args:
    bo (BayesianOptimization)
    x_lower (float): lower bound of x-axis
    x_upper (float): upper bound of x-axis
    x_label (str): label of x-axis
  """
  x = np.linspace(x_lower, x_upper, 1000).reshape(-1, 1)
  mu, sigma = posterior(bo, x)

  fig = plt.figure(figsize=(16, 10))
  fig.suptitle('Gaussian process and utility function after {} steps'.format(
      len(bo.X)), 
      fontdict={'size': 30})
  gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
  gp = plt.subplot(gs[0])
  acq = plt.subplot(gs[1])
  
  gp.plot(
      bo.X.flatten(), bo.Y, 'D', markersize=8, label='Observations',
      color='r')
  gp.plot(x, mu, '--', color='k', label='Prediction')
  gp.fill(
      np.concatenate([x, x[::-1]]), 
      np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
      alpha=.6,
      fc='c',
      ec='None',
      label='95% confidence interval')
  _draw_axes_2d(gp, x_label, 'f(x)', (x_lower, x_upper), (None, None))

  utility = bo.util.utility(x, bo.gp, 0)
  acq.plot(x, utility, label='Utility function', color='purple')
  acq.plot(
      x[np.argmax(utility)], np.max(utility), '*', markersize=15, 
      label='Next best guess', markerfacecolor='gold',
      markeredgecolor='k', markeredgewidth=1)
  _draw_axes_2d(
      acq, x_label, 'Utility', (x_lower, x_upper), (np.min(utility), 0))

  gp.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
  acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)


if __name__ == '__main__':
  trainer = Trainer()
  target = get_obj(trainer)
  bo = BayesianOptimization(
      target,
      {
        'lr': (0.001, 0.1),
      })
  bo.maximize(init_points=3, n_iter=0, acq='ucb', kappa=5)
  plot_2d(bo, 0.001, 0.1)
  plt.show()








