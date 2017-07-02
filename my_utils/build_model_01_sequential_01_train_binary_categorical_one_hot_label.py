

"""
## Training

# For a single-input model with 2 classes (binary classification):
"""
from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels1 = np.random.randint(2, size=(1000, 1))

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model, iterating on the data in batches of 32 samples
print("\nNo validation_set split")
model.fit(data, labels1, epochs=1, batch_size=32)


#######################################
# For a single-input model with 10 classes (categorical classification):
# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels2 = np.random.randint(10, size=(1000, 1))
from tensorflow.contrib.keras.python.keras.utils import to_categorical
# Convert labels to categorical one-hot encoding
one_hot_labels = to_categorical(labels2, num_classes=10)



model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Train the model, iterating on the data in batches of 32 samples
print("\nintro validation_split")
model.fit(data, one_hot_labels, validation_split = 0.2, epochs=1, batch_size=32)
