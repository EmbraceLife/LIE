"""
### How can I use Keras with datasets that don't fit in memory?

You can do batch training using `model.train_on_batch(x, y)` and `model.test_on_batch(x, y)`. See the [models documentation](/models/sequential).

Alternatively, you can write a generator that yields batches of training data and use the method `model.fit_generator(data_generator, steps_per_epoch, epochs)`.

You can see batch training in action in our [CIFAR10 example](https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py).

`model.fit()` has batch_size can do the same
"""

from tensorflow.contrib.keras.python.keras.layers import Dropout, BatchNormalization, Input, Dense
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K
from tensorflow.contrib.keras.python.keras import losses

x = np.random.random((10,10))*2
y = np.random.randint(2, size=(10,1))

input_tensor = Input(shape=(10,))
bn_tensor = BatchNormalization()(input_tensor)
dp_tensor = Dropout(0.7)(bn_tensor)
final_tensor = Dense(1)(dp_tensor)

model = Model(input_tensor, final_tensor)
model.compile(optimizer='SGD', loss='mse')

# x, y are just a single batch of dataset
model.train_on_batch(x, y)
"""
('Runs a single gradient update on a single batch of data.\n'
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
 '    sample_weight: optional array of the same length as x, containing\n'
 "        weights to apply to the model's loss for each sample.\n"
 '        In the case of temporal data, you can pass a 2D array\n'
 '        with shape (samples, sequence_length),\n'
 '        to apply a different weight to every timestep of every sample.\n'
 '        In this case you should make sure to specify\n'
 '        sample_weight_mode="temporal" in compile().\n'
 '    class_weight: optional dictionary mapping\n'
 '        class indices (integers) to\n'
 "        a weight (float) to apply to the model's loss for the samples\n"
 '        from this class during training.\n'
 '        This can be useful to tell the model to "pay more attention" to\n'
 '        samples from an under-represented class.\n'
 '\n'
 'Returns:\n'
 '    Scalar training loss\n'
 '    (if the model has a single output and no metrics)\n'
 '    or list of scalars (if the model has multiple outputs\n'
 '    and/or metrics). The attribute `model.metrics_names` will give you\n'
 '    the display labels for the scalar outputs.')
"""

model.fit(x, y, batch_size=2)
"""
(['  def fit(self,\n',
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
  '          initial_epoch=0):\n',
Trains the model for a fixed number of epochs (iterations on a '
  'dataset).\n',
  '\n',
  '    Arguments:\n',
  '        x: Numpy array of training data,\n',
  '            or list of Numpy arrays if the model has multiple inputs.\n',
  '            If all inputs in the model are named,\n',
  '            you can also pass a dictionary\n',
  '            mapping input names to Numpy arrays.\n',
  '        y: Numpy array of target data,\n',
  '            or list of Numpy arrays if the model has multiple outputs.\n',
  '            If all outputs in the model are named,\n',
  '            you can also pass a dictionary\n',
  '            mapping output names to Numpy arrays.\n',
  '        batch_size: integer. Number of samples per gradient update.\n',
  '        epochs: integer, the number of times to iterate\n',
  '            over the training data arrays.\n',
  '        verbose: 0, 1, or 2. Verbosity mode.\n',
  '            0 = silent, 1 = verbose, 2 = one log line per epoch.\n',
  '        callbacks: list of callbacks to be called during training.\n',
  '            See [callbacks](/callbacks).\n',
  '        validation_split: float between 0 and 1:\n',
  '            fraction of the training data to be used as validation data.\n',
  '            The model will set apart this fraction of the training data,\n',
  '            will not train on it, and will evaluate\n',
  '            the loss and any model metrics\n',
  '            on this data at the end of each epoch.\n',
  '        validation_data: data on which to evaluate\n',
  '            the loss and any model metrics\n',
  '            at the end of each epoch. The model will not\n',
  '            be trained on this data.\n',
  '            This could be a tuple (x_val, y_val)\n',
  '            or a tuple (x_val, y_val, val_sample_weights).\n',
  '        shuffle: boolean, whether to shuffle the training data\n',
  '            before each epoch.\n',
  '        class_weight: optional dictionary mapping\n',
  '            class indices (integers) to\n',
  "            a weight (float) to apply to the model's loss for the samples\n",
  '            from this class during training.\n',
  '            This can be useful to tell the model to "pay more attention" '
  'to\n',
  '            samples from an under-represented class.\n',
  '        sample_weight: optional array of the same length as x, containing\n',
  "            weights to apply to the model's loss for each sample.\n",
  '            In the case of temporal data, you can pass a 2D array\n',
  '            with shape (samples, sequence_length),\n',
  '            to apply a different weight to every timestep of every '
  'sample.\n',
  '            In this case you should make sure to specify\n',
  '            sample_weight_mode="temporal" in compile().\n',
  '        initial_epoch: epoch at which to start training\n',
  '            (useful for resuming a previous training run)\n',
  '\n',
  '    Returns:\n',
  '        A `History` instance. Its `history` attribute contains\n',
  '        all information collected during training.\n',
  '\n',
  '    Raises:\n',
  '        ValueError: In case of mismatch between the provided input data\n',
  '            and what the model expects.\n',
"""
