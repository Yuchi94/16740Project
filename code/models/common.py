import tensorflow as tf

activations = {
  'linear':  lambda x: x,
  'sigmoid': tf.nn.sigmoid,
  'relu':    tf.nn.relu,
  'tanh':    tf.tanh
}

rnn_cells = {
  'rnn':  tf.nn.rnn_cell.RNNCell,
  'lstm': tf.nn.rnn_cell.LSTMCell,
  'gru':  tf.nn.rnn_cell.GRUCell
}
