#############################
## create random dataset
# tf.random_normal: a normal distribution with specific shape, mean, std, type
# tf.truncated_normal: a normal distribution with specific shape, mean, std(less than 3), type
# tf.random_uniform: a uniform distribution with specific shape, minval, maxval, type
# tf.random_shuffle: given a tensor, shuffle its 1-dim
# tf.random_crop: given a tensor, crop a smaller shape out of it, like a floating filter screening (convolutional filter)
# graph looks cool here


import tensorflow as tf
import numpy as np

# what is Graph, and what does it do
g_user = tf.Graph()

# access the default graph
g = tf.get_default_graph()
c = tf.constant(4.0)
c.graph == g # true
c.graph == g_user # false

# use default graph for adding ops and tensor below
with g.as_default():

	# values can be 3 std away from mean
	rand_norm = tf.random_normal(shape=[2,5], mean=0, stddev=1, dtype=tf.float32, seed=123, name='rand_norm2_5')

	rand_norm.graph == g # true

	# values with more than 2 std away from mean are dropped
	truc_norm = tf.truncated_normal(shape=[2,5], mean=0, stddev=1, dtype=tf.float32, seed=123, name='truc_norm2_5')

	# Note: minval is inclusive, maxval is exclusive
	unif_rand = tf.random_uniform(shape=[2,5], minval=0, maxval=1, dtype=tf.float32, seed=123, name='unif_rand')

	#### value: tensor be shuffled on first dim
	# 1d tensor
	seq = tf.linspace(start=0.0, stop=10.0, num=10, name="seq")
	rand_shuf = tf.random_shuffle(value=seq, seed=123, name='rand_shuf')
	# 2d tensor
	seq2 = tf.linspace(start=0.0, stop=10.0, num=9, name="seq")
	seq2 = tf.reshape(seq2, (3,3))
	rand_shuf2 = tf.random_shuffle(value=seq2, seed=123, name='rand_shuf2')

	# crop a smaller shape out of given tensor
	# seq2.shape = [3,3] > size
	crop_rand = tf.random_crop(value=seq2, size=[2,2], seed=123, name="crop_rand")


	# multinomial = tf.multinomial(logits, num_samples, seed=123, name="multinomial")
	# tf.random_gamma(shape, alpha, beta=None, dtype=tf.float32, seed=None, name=None)

with tf.Session(graph=g) as sess:
	writer = tf.summary.FileWriter("log/01_create_random", sess.graph)
	sess.run(crop_rand)

writer.close()


# tensorboard --logdir log/01_create_random
