

"""
## Multi-input and multi-output models

The functional API makes it easy to manipulate a large number of intertwined datastreams.

Goal:
We seek to predict how many retweets and likes a news headline will receive on Twitter.

Inputs:
1. The main input to the model will be the headline itself, as a sequence of words,
2. our model will also have an auxiliary input, receiving extra data such as the time of day when the headline was posted, etc.

Losses:
The model will also be supervised via two loss functions.

Using the main loss function earlier in a model is a good regularization mechanism for deep models.

Here's what our model looks like:
https://s3.amazonaws.com/keras.io/img/multi-input-multi-output-graph.png

Let's implement it with the functional API.

1. The main input will receive the headline, as a sequence of integers (each integer encodes a word, a vocabulary of 10,000 words)
2. the sequences will be 100 words long.

"""
from tensorflow.contrib.keras.python.keras.layers import Input, Embedding, LSTM, Dense, concatenate
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
main_input = Input(shape=(100,), dtype='int32', name='main_input')

# Generate dummy data as main_input
main_input_array = np.random.random((1000, 100))
# main_output_array = np.random.random()


# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = LSTM(32)(x)

# Here we insert the auxiliary loss, allowing the LSTM and Embedding layer to be trained smoothly even though the main loss will be much higher in the model.
auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
# create auxiliary_output_array for training later
auxiliary_output_array = np.random.random((1000, 1))



# At this point, we feed into the model our auxiliary input data by concatenating it with the LSTM output:

auxiliary_input = Input(shape=(5,), name='aux_input')
# create auxiliary_input_array
auxiliary_input_array = np.random.random((1000, 5))

x = concatenate([lstm_out, auxiliary_input])

# We stack a deep densely-connected network on top
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

# And finally we add the main logistic regression layer
main_output = Dense(1, activation='sigmoid', name='main_output')(x)
# create main_output_array for training later
main_output_array = np.random.random((1000, 1))

# This defines a model with two inputs and two outputs:
model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])


# We compile the model and assign a weight of 0.2 to the auxiliary loss.
# To specify different `loss_weights` or `loss` for each different output, you can use a list or a dictionary.
# Here we pass a single loss as the `loss` argument, so the same loss will be used on all outputs.
model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])


# We can train the model by passing it lists of input arrays and target arrays:
model.fit([main_input_array, auxiliary_input_array], [main_output_array, auxiliary_output_array],
          epochs=2, batch_size=32)


# Since our inputs and outputs are named (we passed them a "name" argument),
# We could also have compiled the model via:
model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

# And trained it via:
model.fit({'main_input': main_input_array, 'aux_input': auxiliary_input_array},
          {'main_output': main_output_array, 'aux_output': auxiliary_output_array},
          epochs=2, batch_size=32)
