from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf

###################################
# elu
###################################
tf.contrib.keras.activations.elu == tf.contrib.keras.activations.deserialize("elu") == elu

elu_try = elu([1.0,2.0,3.0])
# source code can't be accessed using `s` here
# why: frozen importlib._bootstrap maybe
# source code should be found in K

# x must be float type
"""
(Pdb++) sources elu
(['def elu(x, alpha=1.0):\n', '  return K.elu(x, alpha)\n'], 53)
"""
