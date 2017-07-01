

"""
### Sequence classification with LSTM:
"""

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import Embedding
from tensorflow.contrib.keras.python.keras.layers import LSTM

model = Sequential()
model.add(Embedding(input_dim=64, output_dim=256, input_length=10))
model.input # (?, 10), but why 64 is not in the shape?
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

import numpy as np
x_train = np.random.random((1000, 10))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, 10))
y_test = np.random.randint(2, size=(100, 1))

model.fit(x_train, y_train, batch_size=16, epochs=1)
score = model.evaluate(x_test, y_test, batch_size=16)
