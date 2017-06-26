"""
Goal: learn Relu and Conv2D
"""

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import layers
import numpy as np

# create a toy data
array1 = np.arange(-20,34).reshape(1,3,3,6)

# create an constant tensor
tensor1 = K.constant(value=array1, name='tensor1')

# create a input tensor
input_tensor = layers.Input(tensor=tensor1, name='input_tensor')

# apply relu activation to this input tensor
relu_tensor = K.relu(input_tensor)

"""
def relu(x, alpha=0., max_value=None):

  Rectified linear unit
  With default values, it returns element-wise `max(x, 0)`

  '  Arguments:\n',
  '      x: A tensor or variable.\n',
  '      alpha: A scalar, slope of negative section (default=`0.`).\n',
  '      max_value: Saturation threshold.\n',

  '  Returns:\n',
  '      A tensor.\n',
"""

# apply Conv2D
conv2d_tensor = layers.Conv2D(
	2, (3, 3), activation='relu', padding='same',
	name='block1_conv1')(input_tensor)

import tensorflow as tf
sess = tf.Session()

input_array = sess.run(input_tensor)
relu_array = sess.run(relu_tensor)

sess.run(tf.global_variables_initializer())
conv2d_array = sess.run(conv2d_tensor)

sess.close()

"""
class Conv2D(tf_convolutional_layers.Conv2D, Layer):
  '  def __init__(self,\n',
  '               filters,\n',
  '               kernel_size,\n',
  '               strides=(1, 1),\n',
  "               padding='valid',\n",
  '               data_format=None,\n',
  '               dilation_rate=(1, 1),\n',
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
'  """2D convolution layer (e.g. spatial convolution over images).\n',
  '\n',
  '  This layer creates a convolution kernel that is convolved\n',
  '  with the layer input to produce a tensor of\n',
  '  outputs. If `use_bias` is True,\n',
  '  a bias vector is created and added to the outputs. Finally, if\n',
  '  `activation` is not `None`, it is applied to the outputs as well.\n',
  '\n',
  '  When using this layer as the first layer in a model,\n',
  '  provide the keyword argument `input_shape`\n',
  '  (tuple of integers, does not include the sample axis),\n',
  '  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures\n',
  '  in `data_format="channels_last"`.\n',
  '\n',
  '  Arguments:\n',
  '      filters: Integer, the dimensionality of the output space\n',
  '          (i.e. the number output of filters in the convolution).\n',
  '      kernel_size: An integer or tuple/list of 2 integers, specifying the\n',
  '          width and height of the 2D convolution window.\n',
  '          Can be a single integer to specify the same value for\n',
  '          all spatial dimensions.\n',
  '      strides: An integer or tuple/list of 2 integers,\n',
  '          specifying the strides of the convolution along the width and '
  'height.\n',
  '          Can be a single integer to specify the same value for\n',
  '          all spatial dimensions.\n',
  '          Specifying any stride value != 1 is incompatible with '
  'specifying\n',
  '          any `dilation_rate` value != 1.\n',
  '      padding: one of `"valid"` or `"same"` (case-insensitive).\n',
  '      data_format: A string,\n',
  '          one of `channels_last` (default) or `channels_first`.\n',
  '          The ordering of the dimensions in the inputs.\n',
  '          `channels_last` corresponds to inputs with shape\n',
  '          `(batch, height, width, channels)` while `channels_first`\n',
  '          corresponds to inputs with shape\n',
  '          `(batch, channels, height, width)`.\n',
  '          It defaults to the `image_data_format` value found in your\n',
  '          Keras config file at `~/.keras/keras.json`.\n',
  '          If you never set it, then it will be "channels_last".\n',
  '      dilation_rate: an integer or tuple/list of 2 integers, specifying\n',
  '          the dilation rate to use for dilated convolution.\n',
  '          Can be a single integer to specify the same value for\n',
  '          all spatial dimensions.\n',
  '          Currently, specifying any `dilation_rate` value != 1 is\n',
  '          incompatible with specifying any stride value != 1.\n',
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
  '      4D tensor with shape:\n',
  "      `(samples, channels, rows, cols)` if data_format='channels_first'\n",
  '      or 4D tensor with shape:\n',
  "      `(samples, rows, cols, channels)` if data_format='channels_last'.\n",
  '\n',
  '  Output shape:\n',
  '      4D tensor with shape:\n',
  '      `(samples, filters, new_rows, new_cols)` if '
  "data_format='channels_first'\n",
  '      or 4D tensor with shape:\n',
  '      `(samples, new_rows, new_cols, filters)` if '
  "data_format='channels_last'.\n",
  '      `rows` and `cols` values might have changed due to padding.\n',
