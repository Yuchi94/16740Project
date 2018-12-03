import tensorflow as tf
from . import common


class FCNN(object):
  """
  Fully-connected neural network.
  """

  def __init__(self, options={}):
    super(FCNN, self).__init__()

    _options = {
      'input_dim': None,
      'hidden_layers': [
        (64, 'relu') # (dim, act)
      ],
      'output_dim': None,
      'output_act': 'linear'
    }

    # load options
    for o in _options:
      if _options[o] is None and o not in options:
        raise Error('Missing required option %s' % (o))
      if o in options:
        _options[o] = options[o]

    self.options = _options

    # construct architecture summary
    arch = []
    arch.append((_options['input_dim'], None))
    for i in range(len(_options['hidden_layers'])):
      arch.append(_options['hidden_layers'][i])
    arch.append((_options['output_dim'], _options['output_act']))

    self.arch = arch

    # initialize variables
    self.variables = {}


  def connect(self, X):
    """
    Connect the network.
    """

    self.states = {}

    H = X
    layer_outputs = []
    for i in range(len(self.arch) - 1):
      H = tf.contrib.layers.fully_connected(
        inputs=H,
        num_outputs=self.arch[i+1][0],
        activation_fn=common.activations[self.arch[i+1][1]]
      )
      layer_outputs.append(H)

    self.states['layer_outputs'] = layer_outputs

    return layer_outputs[-1]
