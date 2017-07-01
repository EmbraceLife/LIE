
"""
The `Sequential` model is a linear stack of layers.

You can create a `Sequential` model by passing a list of layer instances to the constructor:
"""

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

model.summary()

"""
You can also simply add layers via the `.add()` method:
"""

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


"""
## Specifying the input shape

The first layer needs to receive information about its input shape.
- input_shape:
	- batch_dim is not included,
	- input_shape=(interger,) or
	- input_shape= None, but it seems not working

"""


model = Sequential()
model.add(Dense(32, input_shape=(784,)))
model.summary()

model = Sequential()
model.add(Dense(32, input_dim=784))
model.summary()


model = Sequential()
model.add(Dense(32, input_shape=(784,10)))
model.summary() # return tensor shape (?, 784, 32)

"""
## Compilation
- optimzier
- loss
- metrics: accuracy for classification problem
- define your own metrics
"""

# For a multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# For a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

# For custom metrics
from tensorflow.contrib.keras.python.keras import backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])


"""
## Training

# For a single-input model with 2 classes (binary classification):
"""

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=1, batch_size=32)

# For a single-input model with 10 classes (categorical classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

from tensorflow.contrib.keras.python.keras.utils import to_categorical
# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(labels, num_classes=10)

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, one_hot_labels, epochs=1, batch_size=32)
