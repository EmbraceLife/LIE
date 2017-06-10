from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf

###################################
# relu
###################################
print(tf.contrib.keras.activations.sigmoid == tf.contrib.keras.activations.deserialize("sigmoid") == sigmoid)


# sources relu
"""
(Pdb++) sources sigmoid
(['def sigmoid(x):\n', '  return K.sigmoid(x)\n'], 73)
"""

x_int = [-1,2,3] # int not allowed
x_float = [-1., 2., 3.] # float works
sigmoid_try = sigmoid(x=x_float)
# source code can't be accessed using `s` here
# why: frozen importlib._bootstrap maybe
# source code should be found in K

# linear return x_int rather than a tensor, so following won't apply

sess = tf.Session()
sess.run(sigmoid_try)
