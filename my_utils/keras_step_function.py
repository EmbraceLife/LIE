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
def buy_hold_sell(x, lower_threshold=0.33, higher_threshold=0.66):
	"""
	x: tensor
	return a tensor
	"""
	x_array = K.get_value(x)
	for index in range(len(x_array)):
		if x_array[index] < lower_threshold:
			x_array[index] = 0.0
		elif x_array[index] > higher_threshold:
			x_array[index] = 1.0
		else:
			x_array[index] = 0.5

	return K._to_tensor(x_array, x.dtype.base_dtype)


	# pred_shape = K.get_variable_shape(x)
	# lower_array = np.ones(pred_shape)*0.33
	# higher_array = np.ones(pred_shape)*0.66
	# lower_tensor = _to_tensor(lower_array, x.dtype.base_dtype)
	# higher_tensor = _to_tensor(higher_array, x.dtype.base_dtype)
	#
	# buy_array = np.ones(pred_shape)
	# not_sure_array = np.ones(pred_shape)*0.5
	# sell_array = np.zeros(pred_shape)
	# buy_tensor = _to_tensor(buy_array, x.dtype.base_dtype)
	# not_sure_tensor = _to_tensor(not_sure_array, x.dtype.base_dtype)
	# sell_tensor = _to_tensor(sell_array, x.dtype.base_dtype)
	#
	# r = tf.where(tf.less(higher_tensor, x), buy_tensor,
    # x)
	# r = tf.where(tf.less(x, lower_tensor), sell_tensor,
    # x)
	# r = tf.where(lower_tensor <= x <= higher_tensor, not_sure_tensor, x)
	# if K.less(higher_tensor, x):
	# 	r = buy_tensor
	# return r



x = K.ones((2,5)) * 0.7
buy_hold_sell(x)



#
# def relu(x, alpha=0., max_value=None):
#     """Rectified linear unit.
#     With default values, it returns element-wise `max(x, 0)`.
#     # Arguments
#         x: A tensor or variable.
#         alpha: A scalar, slope of negative section (default=`0.`).
#         max_value: Saturation threshold.
#     # Returns
#         A tensor.
#     """
#     if alpha != 0.:
#         negative_part = tf.nn.relu(-x)
#     x = tf.nn.relu(x)
#     if max_value is not None:
#         max_value = _to_tensor(max_value, x.dtype.base_dtype) # useful
#         zero = _to_tensor(0., x.dtype.base_dtype) # useful
#         x = tf.clip_by_value(x, zero, max_value)
#     if alpha != 0.:
#         alpha = _to_tensor(alpha, x.dtype.base_dtype)
#         x -= alpha * negative_part
#     return x
