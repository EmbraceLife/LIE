################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

#######################################
#######################################
# all tutorials in github
# https://github.com/MorvanZhou/tutorials/tree/master/kerasTUT


"""

#######################################
# temporarily switch from theano to tensorflow
import os
os.environ['KERAS_BACKEND']='tensorflow'
import keras

#######################################
# permanently switch from theano to tensorflow
# go to ~/.keras/keras.json
"""
{
    "image_dim_ordering": "th",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
"""

#######################################
# How to build a regression model with 1 single dense layer with just 1 node
X_train, Y_train = X[:160], Y[:160]     # first 160 data points
X_test, Y_test = X[160:], Y[160:]       # last 40 data points

# build a neural network from the 1st layer to the last layer
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# training
print('Training -----------')
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost: ', cost)

# test: a new data sample never seen in training or even validation
# test: see how well model's prediction matches with true y
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)

# access weights of a layer
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# prediction: don't care how well model do, just want to see what model produces given x
Y_pred = model.predict(X_test)

"""

#######################################
#######################################
# How to build 2 dense layers model to train mnist
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.   # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.      # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your neural net
model = Sequential([
    Dense(32, input_dim=784), # first number is output_dim
    Activation('relu'),
    Dense(10), # output_dim, input_dim is taken for granted from above
    Activation('softmax'),
])

# define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')

# Another way to train the model
model.fit(X_train, y_train, epochs=2, batch_size=32)

print('\nEvaluate ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)
