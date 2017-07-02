
"""
### Stacked LSTM for sequence classification

In this model, we stack 3 LSTM layers on top of each other,
making the model capable of learning higher-level temporal representations.

The first two LSTMs return their full output sequences, but the last one only returns
the last step in its output sequence, thus dropping the temporal dimension
(i.e. converting the input sequence into a single vector).

https://keras.io/img/regular_stacked_lstm.png
"""
print("Stacked LSTM for sequence classification")
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import LSTM, Dense
from tensorflow.contrib.keras.python.keras.utils import to_categorical
import numpy as np

data_dim = 16
timesteps = 8
num_classes = 10

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(timesteps, data_dim)))# input (?, 8, 16), output (?, ?, 32)
model.add(LSTM(32, return_sequences=True))  # output (?, ?, 32)
model.add(LSTM(32))  # output (?, 32)
model.add(Dense(10, activation='softmax')) # (?, 10)

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Generate dummy training data
x_train = np.random.random((1000, timesteps, data_dim))
y_train = np.random.randint(10, size=(1000, 1))
y_train = to_categorical(y_train, num_classes=10)

# Generate dummy validation data
x_val = np.random.random((100, timesteps, data_dim))
y_val = np.random.randint(10, size=(100, 1))
y_val = to_categorical(y_val, num_classes=10)

model.fit(x_train, y_train,
          batch_size=64, epochs=1,
          validation_data=(x_val, y_val))

"""
LSTM

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
