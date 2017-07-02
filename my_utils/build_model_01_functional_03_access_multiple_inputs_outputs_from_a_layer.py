

"""
## Shared layers

inputs:
a dataset of tweets.

Goals:
We want to build a model that can tell whether two tweets are from the same person or not (this can allow us to compare users by the similarity of their tweets, for instance).

One way to achieve this is to build a model that encodes two tweets into two vectors, concatenates the vectors and adds a logistic regression of top, outputting a probability that the two tweets share the same author. The model would then be trained on positive tweet pairs and negative tweet pairs.

Because the problem is symmetric, the mechanism that encodes the first tweet should be reused (weights and all) to encode the second tweet. Here we use a shared LSTM layer to encode the tweets.

We will take as input for a tweet a binary matrix of shape `(140, 256)`, i.e. a sequence of 140 vectors of size 256, where each dimension in the 256-dimensional vector encodes the presence/absence of a character (out of an alphabet of 256 frequent characters).
"""

from tensorflow.contrib.keras.python.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Conv2D
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np

tweet_a = Input(shape=(140, 256)) # (?, 140, 256)
tweet_b = Input(shape=(140, 256))

# create fake data
data_a = np.random.random((1000, 140, 256))
data_b = np.random.random((1000, 140, 256))

# To share a layer across different inputs, simply instantiate the layer once, then call it on as many inputs as you want:


# This layer can take as input a matrix
# and will return a vector of size 64
shared_lstm = LSTM(64, name='encode_a_b')

# When we reuse the same layer instance
# multiple times, the weights of the layer
# are also being reused
# (it is effectively *the same* layer)
encoded_a = shared_lstm(tweet_a) # (?, 64)
encoded_b = shared_lstm(tweet_b) # (?, 64)

# We can then concatenate the two vectors:
merged_vector = concatenate([encoded_a, encoded_b], axis=-1, name='cbind_encode_a_b') # (?, 64+64)

"""
(['def concatenate(inputs, axis=-1, **kwargs):\n',
   Functional interface to the `Concatenate` layer.\n',
  '\n',
  '  Arguments:\n',
  '      inputs: A list of input tensors (at least 2).\n',
  '      axis: Concatenation axis.\n',
  '      **kwargs: Standard layer keyword arguments.\n',
  '\n',
  '  Returns:\n',
  '      A tensor, the concatenation of the inputs alongside axis `axis`.\n',

  '  return Concatenate(axis=axis, **kwargs)(inputs)\n'],
"""

# And add a logistic regression on top
predictions = Dense(1, activation='sigmoid', name='predictions')(merged_vector) # (?, 1)
# create fake data for predictions: labels
labels = np.random.random((1000, 1))

# We define a trainable model linking the
# tweet inputs to the predictions
model = Model(inputs=[tweet_a, tweet_b], outputs=predictions)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit([data_a, data_b], labels, epochs=1)


# Let's pause to take a look at how to read the shared layer's output or output shape.


"""
The concept of layer "node"

Whenever you are calling a layer on some input, you are creating a new tensor (the output of the layer), and you are adding a "node" to the layer, linking the input tensor to the output tensor. When you are calling the same layer multiple times, that layer owns multiple nodes indexed as 0, 1, 2...

In previous versions of Keras, you could obtain the output tensor of a layer instance via `layer.get_output()`, or its output shape via `layer.output_shape`. You still can (except `get_output()` has been replaced by the property `output`). But what if a layer is connected to multiple inputs?

As long as a layer is only connected to one input, there is no confusion, and `.output` will return the one output of the layer:
"""

a = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)

assert lstm.output == encoded_a


# Not so if the layer has multiple inputs:

a = Input(shape=(140, 256))
b = Input(shape=(140, 256))

lstm = LSTM(32)
encoded_a = lstm(a)
encoded_b = lstm(b)

"""
lstm.output

>> AssertionError: Layer lstm_1 has multiple inbound nodes,
hence the notion of "layer output" is ill-defined.
Use `get_output_at(node_index)` instead.

Okay then. The following works:
"""

assert lstm.get_output_at(0) == encoded_a
assert lstm.get_output_at(1) == encoded_b

"""
Simple enough, right?

The same is true for the properties `input_shape` and `output_shape`: as long as the layer has only one node, or as long as all nodes have the same input/output shape, then the notion of "layer output/input shape" is well defined, and that one shape will be returned by `layer.output_shape`/`layer.input_shape`. But if, for instance, you apply a same `Conv2D` layer to an input of shape `(3, 32, 32)`, and then to an input of shape `(3, 64, 64)`, the layer will have multiple input/output shapes, and you will have to fetch them by specifying the index of the node they belong to:
"""

a = Input(shape=(32, 32, 3))
b = Input(shape=(64, 64, 3))

conv = Conv2D(16, (3, 3), padding='same')
conved_a = conv(a)

# Only one input so far, the following will work:
assert conv.input_shape == (None, 32, 32, 3)

conved_b = conv(b)
# now the `.input_shape` property wouldn't work, but this does:
assert conv.get_input_shape_at(0) == (None, 32, 32, 3)
assert conv.get_input_shape_at(1) == (None, 64, 64, 3)
