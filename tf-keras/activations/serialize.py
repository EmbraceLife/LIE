from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf


###################################
# serialize
###################################
tf.contrib.keras.activations.serialize == serialize

# how serialize is defined
"""
(Pdb++) sources serialize
(['def serialize(activation):\n', '  return activation.__name__\n'], 85)
"""
relu_name = serialize(relu)
