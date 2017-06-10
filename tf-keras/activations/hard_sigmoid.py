from tensorflow.contrib.keras.python.keras.activations import deserialize, serialize, get, elu, relu, linear, sigmoid, hard_sigmoid, softmax, softplus, softsign, tanh

import tensorflow as tf
import numpy as np
###################################
# relu
###################################
print(tf.contrib.keras.activations.hard_sigmoid == tf.contrib.keras.activations.deserialize("hard_sigmoid") == hard_sigmoid)


# sources relu
"""
(Pdb++) sources hard_sigmoid
(['def hard_sigmoid(x):\n', '  return K.hard_sigmoid(x)\n'], 77)
"""

# sometimes, step can go inside K.hard_sigmoid
"""
2970  -> def hard_sigmoid(x):
2971       Segment-wise linear approximation of sigmoid.
2972
2973       Faster than sigmoid.
2974       Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
2975       In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.
2976
2977       Arguments:
2978           x: A tensor or variable.
2979
2980       Returns:
2981           A tensor.
2982
2983       x = (0.2 * x) + 0.5
2984       zero = _to_tensor(0., x.dtype.base_dtype) # make 0. same tensor dtype
2985       one = _to_tensor(1., x.dtype.base_dtype)
2986       x = clip_ops.clip_by_value(x, zero, one)
2987       return x
"""

x_int = np.arange(-5, 10).reshape(3,5)# int not allowed
x_float = np.linspace(-5, 9, 15).reshape(3,5).astype(float) # float works
# x_tensor_int = tf.constant(x_int)
x_tensor_float = tf.constant(x_float)

sigmoid_try = hard_sigmoid(x=x_tensor_float)
sigmoid_try = hard_sigmoid(x=x_tensor_int)

sess = tf.Session()
sess.run(sigmoid_try)
