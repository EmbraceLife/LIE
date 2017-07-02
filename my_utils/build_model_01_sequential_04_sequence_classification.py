

"""
### Sequence classification with LSTM:
"""

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import Embedding
from tensorflow.contrib.keras.python.keras.layers import LSTM
from tensorflow.contrib.keras.python.keras import backend as K

model = Sequential()
model.add(Embedding(input_dim=64, output_dim=256, input_length=10))
# input_dim: Size of the vocabulary
# input_length: Length of input sequences, length of each sentences
# output_dim: Dimension of the dense embedding
model.input # (?, 10),
model.output # (?, 10, 256)

model.add(LSTM(128)) # unit=128, dimensionality of the output space
model.output

model.add(Dropout(0.5)) # percent to drop out
# model.ouput # will cause error on Dropout
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

import numpy as np
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 10))
y_test = np.random.randint(2, size=(100, 1))

hist=model.fit(x_train, y_train, validation_split=0.2, batch_size=16, epochs=1)
hist.history

score = model.evaluate(x_test, y_test, batch_size=16)

"""
Embedding

(['class Embedding(Layer):\n',
      '  def __init__(self,\n',
  '               input_dim,\n',
  '               output_dim,\n',
  "               embeddings_initializer='uniform',\n",
  '               embeddings_regularizer=None,\n',
  '               activity_regularizer=None,\n',
  '               embeddings_constraint=None,\n',
  '               mask_zero=False,\n',
  '               input_length=None,\n',
  '               **kwargs):\n',
    Turns positive integers (indexes) into dense vectors of fixed size.\n',
  '\n',
  '  eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]\n',
  '\n',
  '  This layer can only be used as the first layer in a model.\n',
  '\n',
  '  Example:\n',
  '\n',
  '  ```python\n',
  '    model = Sequential()\n',
  '    model.add(Embedding(1000, 64, input_length=10))\n',
  '    # the model will take as input an integer matrix of size (batch,\n',
  '    input_length).\n',
  '    # the largest integer (i.e. word index) in the input should be no '
  'larger\n',
  '    than 999 (vocabulary size).\n',
  '    # now model.output_shape == (None, 10, 64), where None is the batch\n',
  '    dimension.\n',
  '\n',
  '    input_array = np.random.randint(1000, size=(32, 10))\n',
  '\n',
  "    model.compile('rmsprop', 'mse')\n",
  '    output_array = model.predict(input_array)\n',
  '    assert output_array.shape == (32, 10, 64)\n',
  '  ```\n',
  '\n',
  '  Arguments:\n',
  '    input_dim: int > 0. Size of the vocabulary,\n',
  '        i.e. maximum integer index + 1.\n',
  '    output_dim: int >= 0. Dimension of the dense embedding.\n',
  '    embeddings_initializer: Initializer for the `embeddings` matrix.\n',
  '    embeddings_regularizer: Regularizer function applied to\n',
  '          the `embeddings` matrix.\n',
  '    embeddings_constraint: Constraint function applied to\n',
  '          the `embeddings` matrix.\n',
  '    mask_zero: Whether or not the input value 0 is a special "padding"\n',
  '        value that should be masked out.\n',
  '        This is useful when using recurrent layers,\n',
  '        which may take variable length inputs.\n',
  '        If this is `True` then all subsequent layers\n',
  '        in the model need to support masking or an exception will be '
  'raised.\n',
  '        If mask_zero is set to True, as a consequence, index 0 cannot be\n',
  '        used in the vocabulary (input_dim should equal size of\n',
  '        vocabulary + 1).\n',
  '    input_length: Length of input sequences, when it is constant.\n',
  '        This argument is required if you are going to connect\n',
  '        `Flatten` then `Dense` layers upstream\n',
  '        (without it, the shape of the dense outputs cannot be computed).\n',
  '\n',
  '  Input shape:\n',
  '      2D tensor with shape: `(batch_size, sequence_length)`.\n',
  '\n',
  '  Output shape:\n',
  '      3D tensor with shape: `(batch_size, sequence_length, output_dim)`.\n',
  '\n',
  '  References:\n',
  '      - [A Theoretically Grounded Application of Dropout in Recurrent '
  'Neural\n',
  '        Networks](http://arxiv.org/abs/1512.05287)\n',
"""


"""
LSTM

(['class LSTM(Recurrent):\n',
  ' Long-Short Term Memory unit - Hochreiter 1997.\n',
  '\n',
  '  For a step-by-step description of the algorithm, see\n',
  '  [this tutorial](http://deeplearning.net/tutorial/lstm.html).\n',
  '\n',
  '  Arguments:\n',
  '      units: Positive integer, dimensionality of the output space.\n',
  '      activation: Activation function to use.\n',
  '          If you pass None, no activation is applied\n',
  '          (ie. "linear" activation: `a(x) = x`).\n',
  '      recurrent_activation: Activation function to use\n',
  '          for the recurrent step.\n',
  '      use_bias: Boolean, whether the layer uses a bias vector.\n',
  '      kernel_initializer: Initializer for the `kernel` weights matrix,\n',
  '          used for the linear transformation of the inputs..\n',
  '      recurrent_initializer: Initializer for the `recurrent_kernel`\n',
  '          weights matrix,\n',
  '          used for the linear transformation of the recurrent state..\n',
  '      bias_initializer: Initializer for the bias vector.\n',
  '      unit_forget_bias: Boolean.\n',
  '          If True, add 1 to the bias of the forget gate at '
  'initialization.\n',
  '          Setting it to true will also force `bias_initializer="zeros"`.\n',
  '          This is recommended in [Jozefowicz et\n',
  '            '
  'al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)\n',
  '      kernel_regularizer: Regularizer function applied to\n',
  '          the `kernel` weights matrix.\n',
  '      recurrent_regularizer: Regularizer function applied to\n',
  '          the `recurrent_kernel` weights matrix.\n',
  '      bias_regularizer: Regularizer function applied to the bias vector.\n',
  '      activity_regularizer: Regularizer function applied to\n',
  '          the output of the layer (its "activation")..\n',
  '      kernel_constraint: Constraint function applied to\n',
  '          the `kernel` weights matrix.\n',
  '      recurrent_constraint: Constraint function applied to\n',
  '          the `recurrent_kernel` weights matrix.\n',
  '      bias_constraint: Constraint function applied to the bias vector.\n',
  '      dropout: Float between 0 and 1.\n',
  '          Fraction of the units to drop for\n',
  '          the linear transformation of the inputs.\n',
  '      recurrent_dropout: Float between 0 and 1.\n',
  '          Fraction of the units to drop for\n',
  '          the linear transformation of the recurrent state.\n',
  '\n',
  '  References:\n',
  '      - [Long short-term\n',
  '        memory](http://www.bioinf.jku.at/publications/older/2604.pdf)\n',
  '        (original 1997 paper)\n',
  '      - [Supervised sequence labeling with recurrent neural\n',
  '        networks](http://www.cs.toronto.edu/~graves/preprint.pdf)\n',
  '      - [A Theoretically Grounded Application of Dropout in Recurrent '
  'Neural\n',
  '        Networks](http://arxiv.org/abs/1512.05287)\n',
  '
  '\n',
  '  def __init__(self,\n',
  '               units,\n',
  "               activation='tanh',\n",
  "               recurrent_activation='hard_sigmoid',\n",
  '               use_bias=True,\n',
  "               kernel_initializer='glorot_uniform',\n",
  "               recurrent_initializer='orthogonal',\n",
  "               bias_initializer='zeros',\n",
  '               unit_forget_bias=True,\n',
  '               kernel_regularizer=None,\n',
  '               recurrent_regularizer=None,\n',
  '               bias_regularizer=None,\n',
  '               activity_regularizer=None,\n',
  '               kernel_constraint=None,\n',
  '               recurrent_constraint=None,\n',
  '               bias_constraint=None,\n',
  '               dropout=0.,\n',
  '               recurrent_dropout=0.,\n',
  '               **kwargs):\n',

"""

"""
(['class Dropout(tf_core_layers.Dropout, Layer):\n',
  '  Applies Dropout to the input.\n',
  '\n',
  '  Dropout consists in randomly setting\n',
  '  a fraction `rate` of input units to 0 at each update during training '
  'time,\n',
  '  which helps prevent overfitting.\n',
  '\n',
  '  Arguments:\n',
  '      rate: float between 0 and 1. Fraction of the input units to drop.\n',
  '      noise_shape: 1D integer tensor representing the shape of the\n',
  '          binary dropout mask that will be multiplied with the input.\n',
  '          For instance, if your inputs have shape\n',
  '          `(batch_size, timesteps, features)` and\n',
  '          you want the dropout mask to be the same for all timesteps,\n',
  '          you can use `noise_shape=(batch_size, 1, features)`.\n',
  '      seed: A Python integer to use as random seed.\n',

  '\n',
  '  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):\n',
"""
