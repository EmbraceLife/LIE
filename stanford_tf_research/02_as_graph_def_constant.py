import tensorflow as tf

my_const = tf.constant([1.0, 2.0], name="my_const")
add_op = tf.add(my_const, 5, name='my_add')

with tf.Session() as sess:
	sess.graph.as_graph_def()
