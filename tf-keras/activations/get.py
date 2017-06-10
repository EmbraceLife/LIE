from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf

###################################
# get
###################################
get == tf.contrib.keras.activations.get

# get('elu') return elu function
elu_try = tf.contrib.keras.activations.get('elu')

# get(layer object) return layer object itself
dense = tf.contrib.keras.layers.Dense(2)
dense_get = get(dense)
dense == dense_get
# source code is written in this module
# deserialize() is used by get()
