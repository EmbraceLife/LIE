
"""
The `Sequential` model is a linear stack of layers.

You can create a `Sequential` model by passing a list of layer instances to the constructor:
"""

"""
Sequential

(['class Sequential(Model):\n',
  Linear stack of layers.\n',
  '\n',
  '  Arguments:\n',
  '      layers: list of layers to add to the model.\n',
  '\n',
  '  # Note\n',
  '      The first layer passed to a Sequential model\n',
  '      should have a defined input shape. What that\n',
  '      means is that it should have received an `input_shape`\n',
  '      or `batch_input_shape` argument,\n',
  '      or for some type of layers (recurrent, Dense...)\n',
  '      an `input_dim` argument.\n',

  '  def __init__(self, layers=None, name=None):\n',
  '    self.layers = []  # Stack of layers.\n',
  '    self.model = None  # Internal Model instance.\n',
  '    self.inputs = []  # List of input tensors\n',
  '    self.outputs = []  # List of length 1: the output tensor (unique).\n',
  '    self._trainable = True\n',
  '    self._initial_weights = None\n',
  '\n',
  '    # Model attributes.\n',
  '    self.inbound_nodes = []\n',
  '    self.outbound_nodes = []\n',
  '    self.built = False\n',
  '\n',
  '    # Set model name.\n',
  '    if not name:\n',
  "      prefix = 'sequential_'\n",
  '      name = prefix + str(K.get_uid(prefix))\n',
  '    self.name = name\n',
  '\n',
  '    # The following properties are not actually used by Keras;\n',
  "    # they exist for compatibility with TF's variable scoping mechanism.\n",
  '    self._updates = []\n',
  '    self._losses = []\n',
  '    self._scope = None\n',
  '    self._reuse = None\n',
  '    self._base_name = name\n',
  '    self._graph = ops.get_default_graph()\n',
  '\n',
  '    # Add to the model any layers passed to the constructor.\n',
  '    if layers:\n',
  '      for layer in layers:\n',
  '        self.add(layer)\n',
  '\n',
"""

"""
Dense

(['class Dense(tf_core_layers.Dense, Layer):\n',
  Just your regular densely-connected NN layer.\n',
  '\n',
  '  `Dense` implements the operation:\n',
  '  `output = activation(dot(input, kernel) + bias)`\n',
  '  where `activation` is the element-wise activation function\n',
  '  passed as the `activation` argument, `kernel` is a weights matrix\n',
  '  created by the layer, and `bias` is a bias vector created by the layer\n',
  '  (only applicable if `use_bias` is `True`).\n',
  '\n',
  '  Note: if the input to the layer has a rank greater than 2, then\n',
  '  it is flattened prior to the initial dot product with `kernel`.\n',

    Arguments:\n',
  '      units: Positive integer, dimensionality of the output space.\n',
  '      activation: Activation function to use.\n',
  "          If you don't specify anything, no activation is applied\n",
  '          (ie. "linear" activation: `a(x) = x`).\n',
  '      use_bias: Boolean, whether the layer uses a bias vector.\n',
  '      kernel_initializer: Initializer for the `kernel` weights matrix.\n',
  '      bias_initializer: Initializer for the bias vector.\n',
  '      kernel_regularizer: Regularizer function applied to\n',
  '          the `kernel` weights matrix.\n',
  '      bias_regularizer: Regularizer function applied to the bias vector.\n',
  '      activity_regularizer: Regularizer function applied to\n',
  '          the output of the layer (its "activation")..\n',
  '      kernel_constraint: Constraint function applied to\n',
  '          the `kernel` weights matrix.\n',
  '      bias_constraint: Constraint function applied to the bias vector.\n',
  '\n',
  '  Input shape:\n',
  '      nD tensor with shape: `(batch_size, ..., input_dim)`.\n',
  '      The most common situation would be\n',
  '      a 2D input with shape `(batch_size, input_dim)`.\n',
  '\n',
  '  Output shape:\n',
  '      nD tensor with shape: `(batch_size, ..., units)`.\n',
  '      For instance, for a 2D input with shape `(batch_size, input_dim)`,\n',
  '      the output would have shape `(batch_size, units)`.\n',

"""

"""
Activation

(['class Activation(Layer):\n',
  Applies an activation function to an output.\n',
  '\n',
  '  Arguments:\n',
  '      activation: name of activation function to use\n',
  '          or alternatively, a Theano or TensorFlow operation.\n',
  '\n',
  '  Input shape:\n',
  '      Arbitrary. Use the keyword argument `input_shape`\n',
  '      (tuple of integers, does not include the samples axis)\n',
  '      when using this layer as the first layer in a model.\n',
  '\n',
  '  Output shape:\n',
  '      Same shape as input.\n',

  '\n',
  '  def __init__(self, activation, **kwargs):\n',
"""

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.summary()

"""
You can also simply add layers via the `.add()` method:
"""

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


"""
## Specifying the input shape

The first layer needs to receive information about its input shape.
- input_shape:
	- batch_dim is not included,
	- input_shape=(interger,) or
	- input_shape= None, but it seems not working

"""


model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.summary()

model = Sequential()
model.add(Dense(32, input_dim=784))
model.summary()


model = Sequential()
model.add(Dense(32, input_shape=(784,10))) # return tensor (?, 784, 32)
model.add(Dense(1)) # return tensor shape (?, 784, 1)
model.summary()

import numpy as np
x = np.random.random((100, 784, 10))
y = np.random.random((100, 784, 1))


"""
## Compilation
- optimzier: 'rmsprop', 'SGD', ...
- loss: 'mse', 'categorical_crossentropy', 'binary_crossentropy', ...
- metrics: 'accuracy', ... or user_made_func
"""
# For custom metrics
from tensorflow.contrib.keras.python.keras import backend as K

def mean_pred(y_true, y_pred): # score_array = fn(y_true, y_pred) must 2 args
    return K.mean(y_pred)

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy', mean_pred])


hist = model.fit(x, y, validation_split=0.3, epochs=2)

print(hist.history)
