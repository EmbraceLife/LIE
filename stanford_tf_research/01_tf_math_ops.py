import tensorflow as tf

g = tf.get_default_graph()

with g.as_default():

	a = tf.constant([3, 6], name='a')
	b = tf.constant([2, 2], name='b')
	add = tf.add(a, b) # >> [5 8]
	addn = tf.add_n([a, b, b]) # >> [7 10]. Equivalent to a + b + b
	mul = tf.multiply(a, b) # >> [6 12] because mul is element wise
	# tf.matmul(a, b) # >> ValueError
	mm = tf.matmul(tf.reshape(a, shape=[1, 2]), tf.reshape(b, shape=[2, 1]), name="mm") # >> [[18]]
	div = tf.div(a, b, name='div') # >> [1 3]
	mod = tf.mod(a, b, name='mod') # >> [1 0]

with tf.Session(graph=g) as sess:
	writer = tf.summary.FileWriter("log/01_tf_math", sess.graph)

writer.close()
