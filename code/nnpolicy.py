import numpy as np
import tensorflow as tf
from collections import deque

from models.RegressionNetwork import RegressionNetwork
from models.BoundedRegressionNetwork import BoundedRegressionNetwork


class NNPolicy(object):
  """
  Neural network policy.
  """

  def __init__(self, options={}):
    """
    Initialize a neural network policy.
    """

    super(NNPolicy, self).__init__()

    _options = {
      'x_dim': None,
      'u_dim': None,
      'u_bounds': None,
      'hidden_layers': [
        (64, 'relu'),
        (64, 'relu'),
        (32, 'relu')
      ],
      'use_memory': False,
      'memory_size': 10000,
      'batch_size': 64
    }

    for o in _options:
      if _options[o] is None and o not in options:
        raise Error('Missing required option %s' % (o))
      if o in options:
        _options[o] = options[o]

    self.options = _options

    # initialize memory
    self.memory_X = deque(maxlen=self.options['memory_size'])
    self.memory_U = deque(maxlen=self.options['memory_size'])

    # initialize policy network
    self.policy_net = BoundedRegressionNetwork({
       'x_dim': self.options['x_dim'],
       'y_dim': self.options['u_dim'],
       'y_bounds': self.options['u_bounds'],
       'hidden_layers': self.options['hidden_layers'],
       'use_minibatch': False
    })

    # initialize variance network
    self.variance_net = RegressionNetwork({
       'x_dim': self.options['x_dim'],
       'y_dim': self.options['u_dim'],
       'hidden_layers': self.options['hidden_layers'],
       'use_minibatch': False
    })


  def train(self, X, U, epochs=3, verbose=False):
    """
    Train the policy.
    """

    use_memory = self.options['use_memory']

    # append to the memory
    if use_memory:
      for i in range(len(X)):
        self.memory_X.append(X[i])
        self.memory_U.append(U[i])

    # train the policy network
    hist_policy = self.policy_net.train(X, U, epochs=epochs, verbose=False)

    # predict the mean output values and calculate the sample variance
    U_mean = self.policy_net.predict(X)
    var_U = np.sqrt((U - U_mean) * (U - U_mean))

    # train the variance network
    hist_variance = self.variance_net.train(X, var_U, epochs=epochs, verbose=False)

    policy_loss = hist_policy[-1]['metrics_train']['loss']
    variance_loss = hist_variance[-1]['metrics_train']['loss']

    # log
    if verbose:
      print('nnpolicy trained (policy_loss: %.4f, variance_loss: %.4f)' % (
        policy_loss, variance_loss
      ))

    return policy_loss, variance_loss


  def predict(self, X):
    """
    Predict.
    """

    mu_pred = self.policy_net.predict(X)
    sigma_pred = self.variance_net.predict(X)

    return mu_pred, sigma_pred


  def evaluate(self, X, U):
    """
    Evaluate the model.
    """

    return self.policy_net.evaluate(X, U)


  def test(self, X_test, U_test, verbose=False):
    # test
    metrics_test = self.evaluate(X_test, U_test)

    # log
    if verbose:
      print('test: test_loss=%.4f' % (metrics_test['loss']))

    return metrics_test


def _test():
  """
  Test for the neural network policy.
  """

  # generate data
  N = 50000
  p = (0.8, 0.2)
  X = np.random.random((N, 3))
  U = np.hstack([
    ((X[:,0] + X[:,1]) * X[:,2]).reshape(N, 1),
    ((X[:,0] ** 2 + X[:,1]) - X[:,2] * 5).reshape([N, 1])
  ])
  u_bounds = [
    (-2.0, 2.0),
    None
  ]

  # split data
  N_test = int(p[1] * N)
  X_test, U_test = X[:N_test,:], U[:N_test,:]
  X_train, U_train = X[N_test:,:], U[N_test:,:]

  # inject noise to the training data
  U_train = U_train + np.random.normal(np.zeros(U_train.shape), scale=0.05)

  # initialize a regression network
  options = {
    'x_dim': 3,
    'u_dim': 2,
    'u_bounds': u_bounds
  }

  net = NNPolicy(options)

  # train the network
  batch_size = 128
  N_batches = len(X_train) // batch_size
  for i in range(N_batches):
    policy_loss, variance_loss = net.train(X_train[i*batch_size:(i+1)*batch_size], U_train[i*batch_size:(i+1)*batch_size], epochs=3, verbose=True)

  # test the network
  U_pred, Sigma_pred = net.predict(X_test)
  print('U_true = ')
  print(U_test[:10])
  print('U_pred = ')
  print(U_pred[:10])
  print('Sigma_pred = ')
  print(Sigma_pred[:10])
  metrics_test = net.test(X_test, U_test, verbose=True)


if __name__ == '__main__':
  _test()
