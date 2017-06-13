#############################################
### Goals:
# how to use tf.constant
# tricks with tf.constant

########################
## import
import tensorflow as tf
import numpy as np

########################
# tf.constant(value, dtype=None, shape=None, name='Const', verify_shape=False)
# value: a constant value or a list of values
# dtype: is to set resulting tensor's type

########################
## use default graph to build nodes
g1 = tf.get_default_graph()

with g1.as_default():
	a = tf.constant(value=[2, 2], name="a")
	b = tf.constant(value=[[0, 1], [2, 3]], name="b")
	x = tf.add(x=a, y=b, name="add")
	y = tf.multiply(x=a, y=b, name="mul")

	## Error situation with verify_shape=True
	# edge1 = tf.constant(
	# 		value=2, dtype=None, shape=[2,2], name="wrong_shape", verify_shape=True)
			# verify_shape=True, error if shape not match

	edge2 = tf.constant(
			value=2, dtype=tf.float64, shape=[2,2], name="edge2", verify_shape=False)
			# verify_shape=False, if shape not match, will add to match

	edge3 = tf.constant(
			value=[1,2,3,4], dtype=None, shape=[4,3], name="edge3", verify_shape=False)
			# repeat the last value by row, from left to right

	# reassign or copy works
	edge2c = edge2
	edge3c = edge3

	edge4 = tf.constant(np.ones((2,2)), dtype=None, shape=None, name="shape22", verify_shape=False)
	# increase row by row, from left to right
	edge5 = tf.constant(np.ones((4,3)), dtype=None, shape=[4,3], name="shape43", verify_shape=False)

	add_all = tf.add(edge4, edge2c, name='add_all')
	# all nodes without linked to ops are not displayed by tensorboard

with tf.Session(graph=g1) as sess:
	writer = tf.summary.FileWriter('./log/01_tf', sess.graph)
	x, y = sess.run([x, y])
	sess.run(edge4)
	sess.run(edge5)
	sess.run(edge2c)
	sess.run(add_all)

writer.close()
