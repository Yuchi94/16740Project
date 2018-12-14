import numpy as np
import tensorflow as tf
import random

from tqdm import tqdm

from . import FCNN


class BoundedRegressionNetwork(object):
  """
  Bounded regression network.
  """

  def __init__(self, options={}):
    super(BoundedRegressionNetwork, self).__init__()

    _options = {
      'x_dim': None, # dimension of the input
      'y_dim': None, # dimension of the output
      'y_bounds': None, # bounds for each output dimension
      'slack_ratio': 1e-3, # slack ratio for each side of the bounds
      'hidden_layers': [
        (64, 'relu'),
        (64, 'relu'),
        (32, 'relu')
      ],
      'use_minibatch': True,
      'batch_size': 32,
      'lr': 0.001
    }

    for o in _options:
      if _options[o] is None and o not in options:
        raise Error('Missing required option %s' % (o))
      if o in options:
        _options[o] = options[o]

    self.options = _options

    self.components = {}
    self.interfaces = None

    # define placeholders
    x_dim = self.options['x_dim']
    y_dim = self.options['y_dim']
    X = tf.placeholder(tf.float32, [None, x_dim])
    Y_true = tf.placeholder(tf.float32, [None, y_dim])

    # build model
    Y_pred = self._build_model(X)

    # define metrics
    metrics = self._define_metrics(X, Y_pred, Y_true)

    # define operations
    op_minloss = self._define_operations(X, Y_pred, Y_true)

    # construct interfaces
    self.interfaces = {
      'X': X,
      'Y_pred': Y_pred,
      'Y_true': Y_true,
      'op_minloss': op_minloss,
      'metrics': metrics
    }

    # initialize session
    self.sess = tf.Session()

    # initialize global variables
    self.sess.run(tf.global_variables_initializer())


  def _build_model(self, X):
    """
    Build model.
    """

    y_bounds = self.options['y_bounds']
    slack_ratio = self.options['slack_ratio']

    # define and connect regression network
    regression_net = FCNN.FCNN(options={
      'input_dim': self.options['x_dim'],
      'hidden_layers': self.options['hidden_layers'],
      'output_dim': self.options['y_dim'],
      'output_act': 'linear'
    })
    self.components['regression_net'] = regression_net

    out_raw = regression_net.connect(X)

    # apply bounds
    Y_pred = []
    for d in range(self.options['y_dim']):
      Y_pred_d = out_raw[:,d]
      if y_bounds[d] is not None:
        slack_range = (y_bounds[d][1] - y_bounds[d][0]) * slack_ratio
        bound_range = y_bounds[d][1] - y_bounds[d][0] + 2 * slack_range
        lower_bound = y_bounds[d][0] - slack_range
        Y_pred_d = bound_range * Y_pred_d + lower_bound
      Y_pred.append(Y_pred_d)

    Y_pred = tf.transpose(tf.convert_to_tensor(Y_pred))

    return Y_pred


  def _define_loss(self, X, Y_pred, Y_true):
    """
    Define loss.
    """

    diff_Y = Y_true - Y_pred

    l = tf.reduce_mean(diff_Y * diff_Y) # MSE loss

    return l


  def _define_metrics(self, X, Y_pred, Y_true):
    """
    Define metrics.
    """

    # define loss metric
    metric_loss = self._define_loss(X, Y_pred, Y_true)

    # constant metrics
    metrics = {
      'loss': metric_loss
    }

    return metrics


  def _define_operations(self, X, Y_pred, Y_true):
    """
    Define operations.
    """

    lr = self.options['lr']

    # define loss minimization operation
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    self.components['optimizer'] = optimizer
    loss = self._define_loss(X, Y_pred, Y_true)
    op_minloss = optimizer.minimize(loss)

    return op_minloss


  def _train_once(self, X_val, Y_val):
    """
    Train once.
    """

    self.sess.run(self.interfaces['op_minloss'], {
      self.interfaces['X']: X_val,
      self.interfaces['Y_true']: Y_val
    })


  def predict(self, X_val):
    """
    Predict.
    """

    y_dim = self.options['y_dim']
    y_bounds = self.options['y_bounds']

    # predict
    Y_pred_val = self.sess.run(self.interfaces['Y_pred'], {
      self.interfaces['X']: X_val
    })

    # enforce bounds
    for d in range(y_dim):
      if y_bounds[d] is not None:
        Y_pred_val[:,d] = np.clip(Y_pred_val[:,d], y_bounds[d][0], y_bounds[d][1])

    return Y_pred_val


  def evaluate(self, X_val, Y_val):
    """
    Evaluate the network.
    """

    metrics_val = {}
    for o in self.interfaces['metrics']:
      metrics_val[o] = self.sess.run(self.interfaces['metrics'][o], {
        self.interfaces['X']: X_val,
        self.interfaces['Y_true']: Y_val
      })

    return metrics_val


  def train(self, X_train, Y_train, X_valid=None, Y_valid=None, epochs=10, verbose=False):
    """
    Train the network.
    """

    use_minibatch = self.options['use_minibatch']
    batch_size = len(X_train)
    if use_minibatch:
      batch_size = self.options['batch_size']

    history = []

    N_batches = len(X_train) // batch_size
    indices_train = [ i for i in range(len(X_train)) ]

    # train for multiple epochs
    for ep in range(epochs):
      # shuffle training samples
      random.shuffle(indices_train)

      # train through all training samples
      enumerator = tqdm(range(N_batches)) if verbose else range(N_batches)
      for i_batch in enumerator:
        # determine sample indices for the training batch
        ptr_lower = i_batch * batch_size
        ptr_upper = ptr_lower + batch_size
        indices_batch = indices_train[ptr_lower:ptr_upper]

        # construct training batch
        X_batch = []
        Y_batch = []
        for i_sample in indices_batch:
          X_batch.append(X_train[i_sample])
          Y_batch.append(Y_train[i_sample])

        # train once
        self._train_once(X_batch, Y_batch)

      # evaluate
      metrics_train = self.evaluate(X_train, Y_train)
      metrics_valid = None
      if (X_valid is not None) and (Y_valid is not None):
        metrics_valid = self.evaluate(X_valid, Y_valid)

      history.append({
        'metrics_train': metrics_train,
        'metrics_valid': metrics_valid
      })

      # log
      if verbose:
        if (X_valid is not None) and (Y_valid is not None):
          print('epoch #%04d: train_loss=%.4f, valid_loss=%.4f' % (
            ep+1, metrics_train['loss'], metrics_valid['loss']
          ))
        else:
          print('epoch #%04d: train_loss=%.4f' % (
            ep+1, metrics_train['loss']
          ))

    return history


  def test(self, X_test, Y_test, verbose=False):
    # test
    metrics_test = self.evaluate(X_test, Y_test)

    # log
    if verbose:
      print('test: test_loss=%.4f' % (metrics_test['loss']))

    return metrics_test


def _test():
  """
  Test for the regression network.
  """

  # generate data
  N = 50000
  p = (0.7, 0.15, 0.15)
  X = np.random.random((N, 3))
  Y = np.hstack([
    ((X[:,0] + X[:,1]) * X[:,2]).reshape(N, 1),
    ((X[:,0] ** 2 + X[:,1]) - X[:,2] * 5).reshape([N, 1])
  ])
  y_bounds = [
    (-2.0, 2.0),
    None
  ]

  # split data
  N_valid, N_test = int(p[1] * N), int(p[2] * N)
  X_valid, Y_valid = X[:N_valid,:], Y[:N_valid,:]
  X_test, Y_test = X[N_valid:N_valid+N_test,:], Y[N_valid:N_valid+N_test,:]
  X_train, Y_train = X[N_valid+N_test:,:], Y[N_valid+N_test:,:]

  # initialize a regression network
  options = {
    'x_dim': 3,
    'y_dim': 2,
    'y_bounds': y_bounds,
    'use_minibatch': True,
    'batch_size': 128
  }

  net = BoundedRegressionNetwork(options)

  # train the network
  history = net.train(X_train, Y_train, X_valid, Y_valid, epochs=20, verbose=True)

  # test the network
  Y_test_pred = net.predict(X_test)
  print('Y_test = ')
  print(Y_test[:10])
  print('Y_test_pred = ')
  print(Y_test_pred[:10])
  metrics_test = net.test(X_test, Y_test, verbose=True)


if __name__ == '__main__':
  _test()
