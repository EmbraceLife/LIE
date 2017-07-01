"""
Uses for functional api:

1. build a model and train, evaluate and predict
2. model can be used as a layer
3. TimeDistributed a model is to apply a layer on every sequence of inputs
4. build multi-input-losses-output models

"""


"""
The Keras functional API can define complex models
1.
2. directed acyclic graphs
3. models with shared layers.
4. A layer instance is callable (on a tensor), and it returns a tensor
5. Input tensor(s) and output tensor(s) can then be used to define a `Model`
6. Such a model can be trained just like Keras `Sequential` models.
"""


"""
Use_1:

Build a 2-hidden dense model
"""

from tensorflow.contrib.keras.python.keras.layers import Input, Dense
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np

# create an input tensor (placeholder)
inputs = Input(shape=(784,)) # 1-D this case

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Create a model needs an input tensor and an output tensor
model = Model(inputs=inputs, outputs=predictions)

# before training, a model has to have optimizer, loss and metrics
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 784))
labels = np.random.randint(2, size=(1000, 10))

model.fit(data, labels)  # starts training



"""
Use_2: model as layer

## All models are callable, just like layers

1. you can treat any model as if it were a layer, by calling it on a tensor.
2. calling a model is use both its architecture and latest weights
"""

x = Input(shape=(784,))

# This works, and returns the 10-way softmax we defined above.
# like doing forward pass
y = model(x)


"""
quickly create models that can process *sequences* of inputs.

- turn an image classification model into a video classification model, in just one line.
"""

from tensorflow.contrib.keras.python.keras.layers import TimeDistributed


# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences = Input(shape=(20, 784)) # tensor (?, 20, 784)

out = model(input_sequences)
# This applies our previous model to every timestep in the input sequences.
# the output of the previous model was a 10-way softmax,
# so the output of the layer below will be a sequence of 20 vectors of size 10.
processed_sequences = TimeDistributed(model)(input_sequences)
# but what exactly is the difference between out an processed_sequences???

"""
(['class TimeDistributed(Wrapper):\n',

This wrapper allows to apply a layer to every temporal slice of an '
  'input.\n',
  '\n',
  '  The input should be at least 3D, and the dimension of index one\n',
  '  will be considered to be the temporal dimension.\n',
  '\n',
  '  Consider a batch of 32 samples,\n',
  '  where each sample is a sequence of 10 vectors of 16 dimensions.\n',
  '  The batch input shape of the layer is then `(32, 10, 16)`,\n',
  '  and the `input_shape`, not including the samples dimension, is `(10, '
  '16)`.\n',
  '\n',
  '  You can then use `TimeDistributed` to apply a `Dense` layer\n',
  '  to each of the 10 timesteps, independently:\n',
  '\n',
  '  ```python\n',
  '      # as the first layer in a model\n',
  '      model = Sequential()\n',
  '      model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))\n',
  '      # now model.output_shape == (None, 10, 8)\n',
  '  ```\n',
  '\n',
  '  The output will then have shape `(32, 10, 8)`.\n',
  '\n',
  '  In subsequent layers, there is no need for the `input_shape`:\n',
  '\n',
  '  ```python\n',
  '      model.add(TimeDistributed(Dense(32)))\n',
  '      # now model.output_shape == (None, 10, 32)\n',
  '  ```\n',
  '\n',
  '  The output will then have shape `(32, 10, 32)`.\n',
  '\n',
  '  `TimeDistributed` can be used with arbitrary layers, not just `Dense`,\n',
  '  for instance with a `Conv2D` layer:\n',
  '\n',
  '  ```python\n',
  '      model = Sequential()\n',
  '      model.add(TimeDistributed(Conv2D(64, (3, 3)),\n',
  '                                input_shape=(10, 299, 299, 3)))\n',
  '  ```\n',
  '\n',
  '  Arguments:\n',
  '      layer: a layer instance.\n',

"""
