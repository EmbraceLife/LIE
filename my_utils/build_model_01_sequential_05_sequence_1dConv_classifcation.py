
"""
### Sequence classification with 1D convolutions:
"""
print("Sequence classification with 1D convolutions:")
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import Embedding
from tensorflow.contrib.keras.python.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
seq_length=10
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(seq_length, 100)))
model.input # check shape of input tensor (?, 10, 100)
model.output # check shape of first hidden layer output tensor (?, 10, 64)

model.add(Conv1D(64, 3, padding='same',activation='relu')) # (?, 10, 64)
model.add(MaxPooling1D(3)) # (?, 3, 64) see source for output
model.add(Conv1D(128, 3, padding='same',activation='relu'))# (?, 3, 128)
model.add(Conv1D(128, 3, padding='same',activation='relu'))# (?, 3, 128)
model.add(GlobalAveragePooling1D()) # (?, 128)
model.add(Dropout(0.5))# (?, 128)
model.add(Dense(1, activation='sigmoid')) # (?, 1)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

import numpy as np
x_train = np.random.random((1000, seq_length, 100))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, seq_length, 100))
y_test = np.random.randint(2, size=(100, 1))


hist=model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=1)
hist.history

score = model.evaluate(x_test, y_test, batch_size=16) # one batch at a time

"""
(['class Conv1D(tf_convolutional_layers.Conv1D, Layer):\n',
    1D convolution layer (e.g. temporal convolution).\n',
  '\n',
  '  This layer creates a convolution kernel that is convolved\n',
  '  with the layer input over a single spatial (or temporal) dimension\n',
  '  to produce a tensor of outputs.\n',
  '  If `use_bias` is True, a bias vector is created and added to the '
  'outputs.\n',
  '  Finally, if `activation` is not `None`,\n',
  '  it is applied to the outputs as well.\n',
  '\n',
  '  When using this layer as the first layer in a model,\n',
  '  provide an `input_shape` argument\n',
  '  (tuple of integers or `None`, e.g.\n',
  '  `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,\n',
  '  or `(None, 128)` for variable-length sequences of 128-dimensional '
  'vectors.\n',
  '\n',
  '  Arguments:\n',
  '      filters: Integer, the dimensionality of the output space\n',
  '          (i.e. the number output of filters in the convolution).\n',
  '      kernel_size: An integer or tuple/list of a single integer,\n',
  '          specifying the length of the 1D convolution window.\n',
  '      strides: An integer or tuple/list of a single integer,\n',
  '          specifying the stride length of the convolution.\n',
  '          Specifying any stride value != 1 is incompatible with '
  'specifying\n',
  '          any `dilation_rate` value != 1.\n',
  '      padding: One of `"valid"`, `"causal"` or `"same"` '
  '(case-insensitive).\n',
  '          `"causal"` results in causal (dilated) convolutions, e.g. '
  'output[t]\n',
  '          does not depend on input[t+1:]. Useful when modeling temporal '
  'data\n',
  '          where the model should not violate the temporal order.\n',
  '          See [WaveNet: A Generative Model for Raw Audio, section\n',
  '            2.1](https://arxiv.org/abs/1609.03499).\n',
  '      dilation_rate: an integer or tuple/list of a single integer, '
  'specifying\n',
  '          the dilation rate to use for dilated convolution.\n',
  '          Currently, specifying any `dilation_rate` value != 1 is\n',
  '          incompatible with specifying any `strides` value != 1.\n',
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
  '      kernel_constraint: Constraint function applied to the kernel '
  'matrix.\n',
  '      bias_constraint: Constraint function applied to the bias vector.\n',
  '\n',
  '  Input shape:\n',
  '      3D tensor with shape: `(batch_size, steps, input_dim)`\n',
  '\n',
  '  Output shape:\n',
  '      3D tensor with shape: `(batch_size, new_steps, filters)`\n',
  '      `steps` value might have changed due to padding or strides.\n',

  '\n',
  '  def __init__(self,\n',
  '               filters,\n',
  '               kernel_size,\n',
  '               strides=1,\n',
  "               padding='valid',\n",
  '               dilation_rate=1,\n',
  '               activation=None,\n',
  '               use_bias=True,\n',
  "               kernel_initializer='glorot_uniform',\n",
  "               bias_initializer='zeros',\n",
  '               kernel_regularizer=None,\n',
  '               bias_regularizer=None,\n',
  '               activity_regularizer=None,\n',
  '               kernel_constraint=None,\n',
  '               bias_constraint=None,\n',
  '               **kwargs):\n',
"""


"""
(['class MaxPooling1D(tf_pooling_layers.MaxPooling1D, Layer):\n',
    Max pooling operation for temporal data.\n',
  '\n',
  '  Arguments:\n',
  '      pool_size: Integer, size of the max pooling windows.\n',
  '      strides: Integer, or None. Factor by which to downscale.\n',
  '          E.g. 2 will halve the input.\n',
  '          If None, it will default to `pool_size`.\n',
  '      padding: One of `"valid"` or `"same"` (case-insensitive).\n',
  '\n',
  '  Input shape:\n',
  '      3D tensor with shape: `(batch_size, steps, features)`.\n',
  '\n',
  '  Output shape:\n',
  '      3D tensor with shape: `(batch_size, downsampled_steps, features)`.\n',

  '\n',
  "  def __init__(self, pool_size=2, strides=None, padding='valid', "
  '**kwargs):\n',
"""


"""
(['  def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):\n',
  '    Computes the loss on some input data, batch by batch.\n',
  '\n',
  '    Arguments:\n',
  '        x: input data, as a Numpy array or list of Numpy arrays\n',
  '            (if the model has multiple inputs).\n',
  '        y: labels, as a Numpy array.\n',
  '        batch_size: integer. Number of samples per gradient update.\n',
  '        verbose: verbosity mode, 0 or 1.\n',
  '        sample_weight: sample weights, as a Numpy array.\n',
  '\n',
  '    Returns:\n',
  '        Scalar test loss (if the model has no metrics)\n',
  '        or list of scalars (if the model computes other metrics).\n',
  '        The attribute `model.metrics_names` will give you\n',
  '        the display labels for the scalar outputs.\n',
  '\n',
  '    Raises:\n',
  '        RuntimeError: if the model was never compiled.\n',


"""
