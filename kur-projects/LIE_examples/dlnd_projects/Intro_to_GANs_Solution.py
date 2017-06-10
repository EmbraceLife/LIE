#################################
# tools for examine 
import matplotlib.pyplot as plt
import numpy as np
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues



import pickle as pkl
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')



def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    return inputs_real, inputs_z


def generator(z, out_dim, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('generator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(z, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        # Logits and tanh output
        logits = tf.layers.dense(h1, out_dim, activation=None)
        out = tf.tanh(logits)

        return out


def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits


# Size of input image to discriminator
input_size = 784
# Size of latent vector to generator
z_size = 100
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
# Smoothing
smooth = 0.1


# reset everything back from start
tf.reset_default_graph()

# Create our input placeholders
# input_real (?, 784)
# input_z (?, 100)
input_real, input_z = model_inputs(input_size, z_size)


# Build the model
g_model = generator(input_z, input_size)
# generator produces both logits and outputs
# logits + tahn = outputs
# g_model is the generator output
# output shape (?, 784)

# discriminator throw out both output and logit
# output = logit + sigmoid

# with real images input: get logits and output toward measuring how well d_model can be certain the real images are real
d_model_real, d_logits_real = discriminator(input_real)

# with fake images input: get logits and output toward measuring how well d_model can be certain the fake(generated) images are fake
d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)



# Calculate losses

# d_loss_real: how well can d_model be certain of real images from real images
# inputs:
# 1. logits produced by d_model with real image inputs
# 2. labels: all true or real, all 1s
# 3. 1s has to be smoothened
d_loss_real = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                          labels=tf.ones_like(d_logits_real) * (1 - smooth)))

# d_loss_fake: how well d_model can be certain of fake images from generated images
# inputs:
# 1. logits produced by d_model feeded with generated images
# 2. labels: all fake, all false, all 0s
# 3. shape of logits and labels are always the same
d_loss_fake = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                          labels=tf.zeros_like(d_logits_real)))
# discriminator loss = how much d_model is good at identify real from real + how much d_model is good at identify fake from generated
d_loss = d_loss_real + d_loss_fake

# get loss by comparing logits and labels
# generator loss measures how good g_model is at generating images as close as real ones
# output of g_model is to making images, neight its logits are not for loss

# d_model (weights) are the same to both real images and generated images, logits_real and logits_fake are produced by the same weights
# g_loss is compared between logtis_fake and labels_true
g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                     labels=tf.ones_like(d_logits_fake)))



# Optimizers
learning_rate = 0.002


# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
# generator weights objects (2 weights, 2 biases, 2 dense layers)
g_vars = [var for var in t_vars if var.name.startswith('generator')]
# discriminator weights objects (2 weights, 2 biases, 2 dense layers)
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]


# optimizer for discriminator is to update discriminator weights and biases
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)
# optimizer for generator is to update generator weights and biases
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)




batch_size = 100
epochs = 1
samples = []
losses = []

# Only save generator variables
saver = tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for ii in range(mnist.train.num_examples//batch_size):

			# get a single batch (100 samples a batch)
			# each batch is a tuple of two arrays
			# array1 shape: (100, 784); array2 shape: (100,)
            batch = mnist.train.next_batch(batch_size)

            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))

			# rescale it from 0,1 to (-1, 1)
            batch_images = batch_images*2 - 1

            # Sample random noise for G
			# shape: 100, 100
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            # Run both discriminator and generator optimizers
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})

            set_trace()
        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})

        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))
        # Save losses to view after training
        losses.append((train_loss_d, train_loss_g))

        # Sample from generator as we're training for viewing afterwards
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, reuse=True),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')


# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)


# ## Training loss
#
# Here we'll check out the training losses for the generator and discriminator.

# In[12]:

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()


# ## Generator samples from training
#
# Here we can view samples of images from the generator. First we'll look at images taken while training.

# In[13]:

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

    return fig, axes


# In[14]:

# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)


# These are samples from the final training epoch. You can see the generator is able to reproduce numbers like 1, 7, 3, 2. Since this is just a sample, it isn't representative of the full range of images this generator can make.

# In[15]:

_ = view_samples(-1, samples)


# Below I'm showing the generated images as the network was training, every 10 epochs. With bonus optical illusion!

# In[16]:

rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)



saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(
                   generator(input_z, input_size, reuse=True),
                   feed_dict={input_z: sample_z})
_ = view_samples(0, [gen_samples])


# In[ ]:
