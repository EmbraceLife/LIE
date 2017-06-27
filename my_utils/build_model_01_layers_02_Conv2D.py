"""
Use cases:
- create an input tensor (constant, not placeholder)
- create a Relu tensor
- create a Conv2D tensor
- create a classification model with input tensor and conv2d tensor
- train this model one sample at a time
- after training, access a layer's output with a real sample given as input
"""

from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import layers
import numpy as np
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras.python.keras.optimizers import Adam

# get toy dataset: supervised training require both features and targets to compare predictions with true targets
# use test_img_arrays and test_lab_arrays
from prep_data_utils_01_save_load_large_arrays_bcolz_np_pickle_torch import bz_load_array
# from prep_data_02_vgg16_catsdogs_02_img_folders_2_arrays import test_img_array, train_img_array
val_img_array = bz_load_array("/Users/Natsume/Downloads/data_for_all/dogscats/results/val_img_array")
val_lab_array = bz_load_array("/Users/Natsume/Downloads/data_for_all/dogscats/results/val_lab_array")


# create an constant tensor
tensor1 = K.constant(value=val_img_array, name='tensor1')

# create a input tensor (constant)
input_tensor = layers.Input(tensor=tensor1, name='input_tensor')
# create a input tensor (placeholder) without knowing num of samples
input_tensor = layers.Input(shape=val_img_array.shape[1:], name='input_tensor')



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

# flattened is a must for dense layer to get 2 output only
flattened = layers.Flatten(name='flatten')(conv2d_tensor)

dense_2_cls = layers.Dense(2, activation='softmax', name='dense_2_cls')(flattened)

# create a simple model with just input tensor and a conv2d layer
model = Model(input_tensor, dense_2_cls, name='con2d_simdl')

# see the model
model.summary()

# compile the model
lr = 0.001
model.compile(optimizer=Adam(lr=lr),
		loss='categorical_crossentropy', metrics=['accuracy'])

"""
def compile(self,\n',
  '              optimizer,\n',
  '              loss,\n',
  '              metrics=None,\n',
  '              loss_weights=None,\n',
  '              sample_weight_mode=None,\n',
  '              **kwargs):\n',
  '    Configures the model for training.\n',
  '\n',
  '    Arguments:\n',
  '        optimizer: str (name of optimizer) or optimizer object.\n',
  '            See [optimizers](/optimizers).\n',
  '        loss: str (name of objective function) or objective function.\n',
  '            See [losses](/losses).\n',
  '            If the model has multiple outputs, you can use a different '
  'loss\n',
  '            on each output by passing a dictionary or a list of losses.\n',
  '            The loss value that will be minimized by the model\n',
  '            will then be the sum of all individual losses.\n',
  '        metrics: list of metrics to be evaluated by the model\n',
  '            during training and testing.\n',
  "            Typically you will use `metrics=['accuracy']`.\n",
  '            To specify different metrics for different outputs of a\n',
  '            multi-output model, you could also pass a dictionary,\n',
  "            such as `metrics={'output_a': 'accuracy'}`.\n",
  '        loss_weights: Optional list or dictionary specifying scalar\n',
  '            coefficients (Python floats) to weight the loss contributions\n',
  '            of different model outputs.\n',
  '            The loss value that will be minimized by the model\n',
  '            will then be the *weighted sum* of all individual losses,\n',
  '            weighted by the `loss_weights` coefficients.\n',
  '            If a list, it is expected to have a 1:1 mapping\n',
  "            to the model's outputs. If a tensor, it is expected to map\n",
  '            output names (strings) to scalar coefficients.\n',
  '        sample_weight_mode: if you need to do timestep-wise\n',
  '            sample weighting (2D weights), set this to `"temporal"`.\n',
  '            `None` defaults to sample-wise weights (1D).\n',
  '            If the model has multiple outputs, you can use a different\n',
  '            `sample_weight_mode` on each output by passing a\n',
  '            dictionary or a list of modes.\n',
  '        **kwargs: Additional arguments passed to `tf.Session.run`.\n',
  '\n',
  '    Raises:\n',
  '        ValueError: In case of invalid arguments for\n',
  '            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.\n',
  '        RuntimeError: If the model has no loss to optimize.\n',
"""
## access input layer placeholders, not real tensor values
model.layers[0]
model.layers[0].input
model.layers[0].output
model.layers[0].get_weights()
model.layers[0].weights

# prepare a single sample
x=val_img_array[0].reshape(1, 224, 224, 3)
y=val_lab_array[0].reshape(1,2)

## get conv2d layer output with real values
get_layer_output = K.function([model.layers[0].input],
                                  [model.layers[1].output])
layer_output_0 = get_layer_output([x])[0]

# train 50 samples for once
model.fit(val_img_array, val_lab_array, batch_size=1, epochs=15, verbose=1)

# see whether conv2d layer output be different
layer_output_1 = get_layer_output([x])[0]

# weights are updated after training, so layer output will be different too
(layer_output_0 == layer_output_1).sum()

# how weights of model are saved
model.save_weights("/Users/Natsume/Downloads/data_for_all/dogscats/experiments/weights.h5")

# K.batch_get_value can get weights values
K.batch_get_value(model.layers[1].weights)

"""
def fit(self,\n',
  '          x=None,\n',
  '          y=None,\n',
  '          batch_size=32,\n',
  '          epochs=1,\n',
  '          verbose=1,\n',
  '          callbacks=None,\n',
  '          validation_split=0.,\n',
  '          validation_data=None,\n',
  '          shuffle=True,\n',
  '          class_weight=None,\n',
  '          sample_weight=None,\n',
  '          initial_epoch=0):

('Trains the model for a fixed number of epochs (iterations on a dataset).\n'
 '\n'
 'Arguments:\n'
 '    x: Numpy array of training data,\n'
 '        or list of Numpy arrays if the model has multiple inputs.\n'
 '        If all inputs in the model are named,\n'
 '        you can also pass a dictionary\n'
 '        mapping input names to Numpy arrays.\n'
 '    y: Numpy array of target data,\n'
 '        or list of Numpy arrays if the model has multiple outputs.\n'
 '        If all outputs in the model are named,\n'
 '        you can also pass a dictionary\n'
 '        mapping output names to Numpy arrays.\n'
 '    batch_size: integer. Number of samples per gradient update.\n'
 '    epochs: integer, the number of times to iterate\n'
 '        over the training data arrays.\n'
 '    verbose: 0, 1, or 2. Verbosity mode.\n'
 '        0 = silent, 1 = verbose, 2 = one log line per epoch.\n'
 '    callbacks: list of callbacks to be called during training.\n'
 '        See [callbacks](/callbacks).\n'
 '    validation_split: float between 0 and 1:\n'
 '        fraction of the training data to be used as validation data.\n'
 '        The model will set apart this fraction of the training data,\n'
 '        will not train on it, and will evaluate\n'
 '        the loss and any model metrics\n'
 '        on this data at the end of each epoch.\n'
 '    validation_data: data on which to evaluate\n'
 '        the loss and any model metrics\n'
 '        at the end of each epoch. The model will not\n'
 '        be trained on this data.\n'
 '        This could be a tuple (x_val, y_val)\n'
 '        or a tuple (x_val, y_val, val_sample_weights).\n'
 '    shuffle: boolean, whether to shuffle the training data\n'
 '        before each epoch.\n'
 '    class_weight: optional dictionary mapping\n'
 '        class indices (integers) to\n'
 "        a weight (float) to apply to the model's loss for the samples\n"
 '        from this class during training.\n'
 '        This can be useful to tell the model to "pay more attention" to\n'
 '        samples from an under-represented class.\n'
 '    sample_weight: optional array of the same length as x, containing\n'
 "        weights to apply to the model's loss for each sample.\n"
 '        In the case of temporal data, you can pass a 2D array\n'
 '        with shape (samples, sequence_length),\n'
 '        to apply a different weight to every timestep of every sample.\n'
 '        In this case you should make sure to specify\n'
 '        sample_weight_mode="temporal" in compile().\n'
 '    initial_epoch: epoch at which to start training\n'
 '        (useful for resuming a previous training run)\n'
 '\n'
 'Returns:\n'
 '    A `History` instance. Its `history` attribute contains\n'
 '    all information collected during training.\n'
 '\n'
 'Raises:\n'
 '    ValueError: In case of mismatch between the provided input data\n'
 '        and what the model expects.')
"""




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

' 2D convolution layer (e.g. spatial convolution over images).\n',
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
"""
