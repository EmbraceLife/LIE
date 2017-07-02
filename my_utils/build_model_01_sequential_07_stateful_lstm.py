
"""
### Same stacked LSTM model, rendered "stateful"

A stateful recurrent model is one for which the internal states (memories) obtained after processing a batch of samples are reused as initial states for the samples of the next batch. 

This allows to process longer sequences while keeping computational complexity manageable.

[You can read more about stateful RNNs in the FAQ.](/getting-started/faq/#how-can-i-use-stateful-rnns)
"""

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10
batch_size = 32

# Expected input batch shape: (batch_size, timesteps, data_dim)
# Note that we have to provide the full batch_input_shape since the network is stateful.
# the sample of index i in batch k is the follow-up for the sample i in batch k-1.
model = Sequential()
model.add(LSTM(32, return_sequences=True, stateful=True,#input(32,8,16)
               batch_input_shape=(batch_size, timesteps, data_dim)))#output(32,?,32)
model.add(LSTM(32, return_sequences=True, stateful=True))# output(32,?,32)
model.add(LSTM(32, stateful=True))# output (32,32)
model.add(Dense(10, activation='softmax')) # output (32, 10)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((batch_size * 10, timesteps, data_dim))
y_train = np.random.random((batch_size * 10, num_classes))

# Generate dummy validation data
x_val = np.random.random((batch_size * 3, timesteps, data_dim))
y_val = np.random.random((batch_size * 3, num_classes))# lazy? or must? to_categorical?

model.fit(x_train, y_train,
          batch_size=batch_size, epochs=5, shuffle=False,# order is important
          validation_data=(x_val, y_val))
