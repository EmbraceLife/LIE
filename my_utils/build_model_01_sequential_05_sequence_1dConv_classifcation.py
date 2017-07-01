
"""
### Sequence classification with 1D convolutions:
"""
print("Sequence classification with 1D convolutions:")
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout
from tensorflow.contrib.keras.python.keras.layers import Embedding
from tensorflow.contrib.keras.python.keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D
seq_length=10
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=(seq_length, 100)))
model.add(Conv1D(64, 3, padding='same',activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(128, 3, padding='same',activation='relu'))
model.add(Conv1D(128, 3, padding='same',activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

import numpy as np
x_train = np.random.random((1000, seq_length, 100))
y_train = np.random.randint(2, size=(1000, 1))
x_test = np.random.random((100, seq_length, 100))
y_test = np.random.randint(2, size=(100, 1))


model.fit(x_train, y_train, batch_size=16, epochs=1)
score = model.evaluate(x_test, y_test, batch_size=16)
