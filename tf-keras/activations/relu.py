from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf

###################################
# relu
###################################
tf.contrib.keras.activations.relu == tf.contrib.keras.activations.deserialize("relu") == relu


# sources relu
"""
(Pdb++) sources relu
(['def relu(x, alpha=0., max_value=None):\n',
  '  return K.relu(x, alpha=alpha, max_value=max_value)\n'],
 65)
"""

x_int = [-1,2,3] # int works
x_float = [-1., 2., 3.] # float works too
relu_try = relu(x=x_int, alpha=0., max_value=None)
# source code can't be accessed using `s` here
# why: frozen importlib._bootstrap maybe
# source code should be found in K

with tf.Session() as sess:
	sess.run(relu_try)
