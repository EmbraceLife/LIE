from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf

###################################
# relu
###################################
print(tf.contrib.keras.activations.linear == tf.contrib.keras.activations.deserialize("linear") == linear)


# sources relu
"""
(Pdb++) sources linear
(['def linear(x):\n', '  return x\n'], 81)
"""

x_int = [-1,2,3] # int works
x_float = [-1., 2., 3.] # float works too
linear_try = linear(x=x_int)
# source code can't be accessed using `s` here
# why: frozen importlib._bootstrap maybe
# source code should be found in K

# linear return x_int rather than a tensor, so following won't apply
"""
with tf.Session() as sess:
	sess.run(linear_try)
"""
