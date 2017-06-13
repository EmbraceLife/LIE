#######################################
### Goals
# ways of fill a tensor with 0s and 1s
# args: tf.zeros(shape, dtype=tf.float32, name=None)
# args: tf.zeros_like(tensor=c, dtype=None, name='d', optimize=True)
# tf.ones(shape, dtype=tf.float32, name=None)
# tf.ones_like(tensor, dtype=None, name=None, optimize=True)
# tf.fill(dims=[3,4], value=9, name=None)
## tips:
# dtype: ignore it, only use it when has to specify dtype
## tensorboard:
# Main Graph: operations actually run
# Auxiliary Nodes: operations did not run

import tensorflow as tf
import numpy as np
#########################
g1 = tf.get_default_graph()

with g1.as_default():

	a = tf.zeros(shape=[2, 5],
				# dtype=None, # cause error if set to None
				dtype=tf.int32, # only specify it when necesssary
				name='a')
	b = tf.zeros(shape=[2, 5],
				dtype=tf.float32,
				name='b')

	c = tf.constant(
		value=np.linspace(1,10, 10).reshape(2,5), # can be numpy array
		name='c')
	np_arr = np.linspace(1,10, 10).reshape(2,5)

	d = tf.zeros_like(tensor=c, # tensor = tf.constant
		dtype=None, name='d', optimize=True)
	d_arr = tf.zeros_like(tensor=np_arr,  # tensor = numpy.array
		dtype=None, name='d_arr', optimize=True)


	f = tf.ones(shape=[2,5], name='f') # dtype=None cause error
	# TypeError: Cannot convert value None to a TensorFlow DType.

	## tf.ones_like and tf.fill considered as operations which can be displayed by tensorboard, not tf.ones, tf.zeros, tf.zeros_like
	g = tf.ones_like(tensor=c, dtype=None, name='g', optimize=True)
	h = tf.fill(dims=[2,5], value=2, name='h') # output has same type as value
	op1 = tf.add(a, h, name="add")

with tf.Session(graph=g1) as sess:
	writer = tf.summary.FileWriter("log/01_tensor_fill", sess.graph)
	sess.run(op1)

writer.close()
############################
## during pdb, start witth sess = tf.Session(), then add create and test on tensors and ops
