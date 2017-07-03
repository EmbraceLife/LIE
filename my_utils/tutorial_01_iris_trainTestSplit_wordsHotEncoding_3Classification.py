"""
Uses:

- load iris dataset with labels from seaborn
- convert words into one-hot-encoding
- train-test split (easy with sklearn)
- train-validation split can be done within fit() (keras, validation_split)
- 3-class classification
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Activation
from tensorflow.contrib.keras.python.keras.utils import to_categorical

"""
## Why use iris dataset:
- very simple and small dataset for machine learning classification
## why use iris dataset from seaborn
- labels are given
- plot with seaborn
"""

iris = sns.load_dataset("iris")
iris.head()
sns.pairplot(iris, hue='species'); # plt.show()

X = iris.values[:, :4]
y = iris.values[:, 4]

train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.5, random_state=0) # random_state is random seed, just integer usually

lr = LogisticRegressionCV()
lr.fit(train_X, train_y) # no need for words into numbers, or one-hot

print("Accuracy = {:.2f}".format(lr.score(test_X, test_y))) # get metrics

def one_hot_encode_object_array(arr):
    '''One hot encode a numpy array of objects (e.g. strings)'''
    uniques, ids = np.unique(arr, return_inverse=True) # convert 3 words into 0, 1, 2
    return to_categorical(ids, len(uniques)) # convert 0, 1, 2 to one-hot

train_y_ohe = one_hot_encode_object_array(train_y)
test_y_ohe = one_hot_encode_object_array(test_y)

model = Sequential()

model.add(Dense(16, input_shape=(4,))) # each sample has 4 features
model.add(Activation('sigmoid')) # add non-linearity to hidden layer 1

model.add(Dense(3)) # add another 3 neuron final layer
model.add(Activation('softmax')) # give it non-linearity as output
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

model.fit(train_X, train_y_ohe, validation_split=0.2, epochs=10, batch_size=1, verbose=1);

loss, accuracy = model.evaluate(test_X, test_y_ohe, batch_size=32, verbose=1)
print("Accuracy = {:.2f}".format(accuracy))
