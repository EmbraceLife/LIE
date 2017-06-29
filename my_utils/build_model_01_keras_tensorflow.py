"""
Using keras with tensorflow
"""


import tensorflow as tf
# sess = tf.Session()

from tensorflow.contrib.keras.python.keras import backend as K
# K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

from tensorflow.contrib.keras.python.keras.layers import Dense

x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation
