import tensorflow as tf

# create variable a with scalar value
a = tf.Variable(2, name="scalar")
# create variable b as a vector
b = tf.Variable([2, 3], name="vector")
# create variable c as a 2x2 matrix
c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# create variable W as 784 x 10 tensor, filled with zeros
W = tf.Variable(tf.zeros([784,10]), name='tfzeros')


############################################
## tf.Variable is class with many op, tf.constant is just an op
# tf.Variable holds several ops:
# x = tf.Variable(...)
# x.initializer # init op
# x.value() # read op
# x.assign(...) # write op
# x.assign_add(...) # and more


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)


init_ab = tf.variables_initializer([a, b], name="init_ab")
with tf.Session() as sess:
	sess.run(init_ab) # must run it before the next line
	sess.run(a)
	sess.run(b)


# Initialize a single variable
W = tf.Variable(tf.zeros([784,10]))
with tf.Session() as sess:
	sess.run(W.initializer)
	sess.run(W)
	W.eval()


W = tf.Variable(tf.truncated_normal([700, 10]))
with tf.Session() as sess:
	sess.run(W.initializer) # this must be run first before anything else below
	W.eval()

W = tf.Variable(10)
W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	W.eval() # >> 10

W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
	sess.run(W.initializer)
	sess.run(assign_op)
	W.eval() # >> 100


# You don’t need to initialize variable
# because assign_op does it for you
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
sess.run(assign_op)
print W.eval() # >> 100


# In fact, initializer op is the assign op that
# assigns the variable’s initial value to the
# variable itself.
W = tf.Variable(10)
assign_op = W.assign(100)
with tf.Session() as sess:
sess.run(assign_op)
print W.eval() # >> 100
