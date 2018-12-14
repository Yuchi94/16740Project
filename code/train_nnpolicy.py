import numpy as np
import pickle
from nnpolicy import NNPolicy


def train():
  """
  Train neural network policy.
  """

  dataset_file_name = 'dataset.pkl'
  p = (0.8, 0.2) # data splitting
  scalar = 100
  batch_size = 128
  epochs = 50

  # load data
  dataset = None
  with open(dataset_file_name, 'rb') as f:
    dataset = pickle.load(f)

  N = len(dataset)
  X, U = [], []
  for i in range(N):
    X.append(np.array(dataset[i][0]))
    U.append(np.array(dataset[i][1]))
  X, U = np.array(X), np.array(U) * scalar

  # split data
  N_test = int(p[1] * N)
  X_test, U_test = X[:N_test,:], U[:N_test,:]
  X_train, U_train = X[N_test:,:], U[N_test:,:]

  # inject noise to the training data
  U_train = U_train + np.random.normal(np.zeros(U_train.shape), scale=0.05)

  # initialize a regression network
  epsilon = 0.10
  u_bounds = [ (- epsilon * scalar / 4.0, epsilon * scalar / 4.0) for d in range(len(U[0])) ]
  options = {
    'x_dim': len(X[0]),
    'u_dim': len(U[0]),
    'u_bounds': u_bounds,
    'batch_size': batch_size,
    'lr': 1e-4
  }

  net = NNPolicy(options)

  N_batches = len(X_train) // batch_size
  indices_train = [ i for i in range(len(X_train)) ]

  # train for multiple epochs
  for ep in range(epochs):
    # shuffle training samples
    np.random.shuffle(indices_train)

    # construct indices_batches
    indices_batches = []

    for i_batch in range(N_batches):
      ptr_lower = i_batch * batch_size
      ptr_upper = ptr_lower + batch_size
      indices_batch = indices_train[ptr_lower:ptr_upper]
      indices_batches.append(indices_batch)

    if N_batches * batch_size < len(X_train):
      indices_batches.append(indices_train[N_batches * batch_size:])

    # train through all training samples
    policy_loss_sum, variance_loss_sum = 0.0, 0.0
    for i_batch in range(len(indices_batches)):
      # determine sample indices for the training batch
      indices_batch = indices_batches[i_batch]

      # construct training batch
      X_batch = []
      U_batch = []
      for i_sample in indices_batch:
        X_batch.append(X_train[i_sample])
        U_batch.append(U_train[i_sample])

      # train once
      policy_loss, variance_loss = net.train(X_batch, U_batch, epochs=1, verbose=False)
      policy_loss_sum   += policy_loss   * len(indices_batch)
      variance_loss_sum += variance_loss * len(indices_batch)

    print('epoch #%d: policy_loss: %.6f, variance_loss: %.6f' % (
      ep + 1,
      policy_loss_sum / len(indices_train),
      variance_loss_sum / len(indices_train),
    ))

  # test the network
  U_pred, Sigma_pred = net.predict(X_test)
  print('U_true = ')
  print(U_test[:10])
  print('U_pred = ')
  print(U_pred[:10])
  print('Sigma_pred = ')
  print(Sigma_pred[:10])
  metrics_test = net.test(X_test, U_test, verbose=True)

  return net


if __name__ == '__main__':
  train()
