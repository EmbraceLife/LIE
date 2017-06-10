from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf


###################################
# deserialize
###################################
tf.contrib.keras.activations.deserialize == deserialize

# how deserialize is defined
"""
1. from string 'tensorflow.contrib' to import tensorflow.contrib module
2. make tensorflow.contrib module the parents
3. make use of tensorflow.contrib.keras
4. module_objects are from all objects from tf.contrib.keras.activations module
5. any objects or functions can be accessible by deserialize() using string name
6. module_objects extract activation function using its string name
"""
# step into source
serialize_des = deserialize('serialize')

# serialize is to return activation function name
deserialize_name = serialize_des(deserialize)
