



"""
## Examples

Here are a few examples to get you started!

In the [examples folder](https://github.com/fchollet/keras/tree/master/examples), you will also find example models for real datasets:

- CIFAR10 small images classification: Convolutional Neural Network (CNN) with realtime data augmentation
- IMDB movie review sentiment classification: LSTM over sequences of words
- Reuters newswires topic classification: Multilayer Perceptron (MLP)
- MNIST handwritten digits classification: MLP & CNN
- Character-level text generation with LSTM

...and more.
"""

"""
### Multilayer Perceptron (MLP) for multi-class softmax classification:
"""

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout, Activation
from tensorflow.contrib.keras.python.keras.optimizers import SGD
from tensorflow.contrib.keras.python.keras.utils import to_categorical

# Generate dummy data
import numpy as np
x_train = np.random.random((1000, 20))
y_train = to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
x_test = np.random.random((100, 20))
y_test = to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=20))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# specify optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) # param adjust
model.compile(loss='categorical_crossentropy', # multi-class
              optimizer=sgd,
              metrics=['accuracy'])

hist = model.fit(x_train, y_train,
          validation_split=0.2,
          epochs=1,
          batch_size=128)

print(hist.history)

score = model.evaluate(x_test, y_test, batch_size=128)
