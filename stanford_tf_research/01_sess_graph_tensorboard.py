#############################################
### Goals:
# how to use graph?
# how to use session?
# how to display nodes in tensorboard?

#######################
## import
import tensorflow as tf

#######################
## create a graph
g1 = tf.get_default_graph() # get official default graph
g2 = tf.Graph() # create user graph

#######################
## create nodes in a graph
with g1.as_default(): # specify which graph to use

	with tf.variable_scope('ops1'): # create a box inside graph
		a = tf.constant(3, name="a") # create a node
		c = tf.constant(5, name="c")
		op1 = tf.add(x=a, y=c, name="add1") # create an operation on nodes

	with tf.variable_scope('ops3-5'):
		x = 2 # even a scalar int is a node tensor too
		y = 3
		op3 = tf.add(x, y, name="add2")
		op4 = tf.multiply(x, y, name="multiply1")
		useless = tf.multiply(x, op1, name="multiply2")
		op5 = tf.pow(op3, op4, name="power")

with g2.as_default(): # specify a different graph
	b = tf.constant(5, name="b")
	d = tf.constant(6, name="d")
	op2 = tf.add(b, d, name="add3")



#######################
## create a session based on a graph
sess = tf.Session(graph=g1)
writer = tf.summary.FileWriter('./log/01_default_op1',
											sess.graph) # plot graph
sess.run(op1) # as long as nodes are used by an op, they will be displayed, even though the ops are not run in session
sess.close()
writer.close()
# tensorboard --logdir log/01_default_op1

## create a new session on a user graph
sess = tf.Session(graph=g2)
writer = tf.summary.FileWriter('./log/01_user', sess.graph)
sess.run(op2)
sess.close()
writer.close()
# tensorboard --logdir log/01_user
