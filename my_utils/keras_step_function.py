"""
keras_step_function

I want to write an Activation function with the idea of step function in keras.

The formula of the activation function in numpy would look like:

def step_func(x, lower_threshold=0.33, higher_threshold=0.66):

	# x is an array, and return an array

	for index in range(len(x)):
		if x[index] < lower_threshold:
			x[index] = 0.0
		elif x[index] > higher_threshold:
			x[index] = 1.0
		else:
			x[index] = 0.5
"""
import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import _to_tensor
import numpy as np

def stepy(x):
	if x < 0.33:
		return 0.0
	elif x > 0.66:
		return 1.0
	else:
		return 0.5

import numpy as np
np_stepy = np.vectorize(stepy)

def d_stepy(x): # derivative
	if x < 0.33:
		return 0.0
	elif x > 0.66:
		return 1.0
	else:
		return 0.5
np_d_stepy = np.vectorize(d_stepy)

import tensorflow as tf
from tensorflow.python.framework import ops

np_d_stepy_32 = lambda x: np_d_stepy(x).astype(np.float32)

def py_func(func, inp, Tout, stateful=True, name=None, grad=None):

    # Need to generate a unique name to avoid duplicates:
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))

    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def tf_d_stepy(x,name=None):
    with ops.op_scope([x], name, "d_stepy") as name:
        y = tf.py_func(np_d_stepy_32,
                        [x],
                        [tf.float32],
                        name=name,
                        stateful=False)
        return y[0]

def stepygrad(op, grad):
    x = op.inputs[0]

    n_gr = tf_d_stepy(x)
    return grad * n_gr

np_stepy_32 = lambda x: np_stepy(x).astype(np.float32)

def tf_stepy(x, name=None):

    with ops.op_scope([x], name, "stepy") as name:
        y = py_func(np_stepy_32,
                        [x],
                        [tf.float32],
                        name=name,
                        grad=stepygrad)  # <-- here's the call to the gradient
        return y[0]

with tf.Session() as sess:

    x = tf.constant([0.2,0.7,0.4,0.6])
    y = tf_stepy(x)
    tf.initialize_all_variables().run()

    print(x.eval(), y.eval(), tf.gradients(y, [x])[0].eval())

# print("use keras")
# x = K.ones((10,1)) * 0.7
# step_tensor = tf_stepy(x)
# sess = tf.Session()
# tf.initialize_all_variables()
# print(sess.run(step_tensor))
