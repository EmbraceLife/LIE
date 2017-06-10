# Write gan_mnist in kur

## write LeakyReLU in keras_kur (done)
## pytorch_kur (done, but `alpha` or `negative_slope` is not added, but issue has been made, waiting for answers)


## dive into each layer's output in mnist1.yml (working)
	- where to access `output: labels` ???




## Understanding dlnd_mnist_gan in tensorflow (done)
- read codes
- experiment on codes like doing in kur
- write each block in kur

what does the model look like? (ctrl+shift+m)
![GAN diagram](../../LIE_examples/gan_mnist/assets/gan_diagram.png)
![GAN Network](../../LIE_examples/gan_mnist/assets/gan_network.png)

### Gan_mnist in kur
- get input_z by learning from demo from scratch
- images or inputs_real is standard from any mnist.yml
- when build, there will be lots of errors to indicate how to improve

```yaml
model:  # see model code in tensorflow below
  generator:
    - input: input_z # shape (?, 100)
    - dense: 128 # g_hidden_size = 128
    - activation:
        name: leakyrelu
        alpha: 0.01
    - dense: 784 # out_dim (generator) = input_size (real image) = 784
    - activation:
        name: tanh
    - output: # output of the latest layer
        name: g_out # shape (?, 784)

  discriminator_real:
    - input: input_real # or images # shape (?, 784)
    - dense: 128 # d_hidden_size
    - activation:
        name: leakyrelu
        alpha: 0.01
    - dense: 1 # shrink nodes from 128 to 1, for 2_labels classification with sigmoid (non softmax)
        logits: d_logits_real # can I output logits here
# do I need to output logits
    - activation:
        name: sigmoid
    - output: # output of the latest layer
# can logits in the layer before the latest layer be accessed from here?
        name: d_out_real # not used at all ?

  discriminator_fake:
    - input: g_out # shape (?, 784)
  - dense: 128 # d_hidden_size
  - activation:
      name: leakyrelu
      alpha: 0.01
  - dense: 1 # shrink nodes from 128 to 1, for 2_labels classification with sigmoid (non softmax)
      logits: d_logits_fake # can I output logits here
# do I need to output logits
  - activation:
      name: sigmoid
  - output: # output of the latest layer
# can logits in the layer before the latest layer be accessed from here?
      name: d_out_fake # not used at all?

# https://kur.deepgram.com/specification.html?highlight=loss#loss
loss:  # see loss code in tensorflow below
  generator:
    - target: labels_g # labels=tf.ones_like(d_logits_fake)
    - logits: d_logits_fake # when to use `-`, when not????
      name: categorical_crossentropy
      g_loss: g_loss
  discriminator_real:
    - target: labels_d_real # labels=tf.ones_like(d_logits_real) * (1 - smooth)
    - logits: d_logits_real
      name: categorical_crossentropy
      d_loss_real: d_loss_real
  discriminator_fake:
    - target: labels_d_fake # labels=tf.zeros_like(d_logits_fake)
    - logits: d_logits_fake
      name: categorical_crossentropy
      d_loss_fake: d_loss_fake

train:
  optimizer: # see the optimizers tensorflow code below
    - opt_discriminator:
        name: adam
        learning_rate: 0.001
        d_loss: d_loss #  d_loss = d_loss_real + d_loss_fake
        d_trainable: d_vars
    - opt_generator:
        name: adam
        learning_rate: 0.001
        g_loss: g_loss
        g_trainable: g_vars
```

```yaml
##########################################
# examples of loss from kur documentation
  loss:
    - target: MODEL_OUTPUT_1
      name: LOSS_FUNCTION
      weight: WEIGHT
      param_1: value_1
      param_2: value_2
    # how to get a second model output
    - target: MODEL_OUTPUT_2
      # ... etc
```


**Inputs for generator and discriminator**
```python
def model_inputs(real_dim, z_dim):
	# real_dim is 784 for sure
    inputs_real = tf.placeholder(tf.float32, (None, real_dim), name='input_real')

	# z_dim is set 100, but can be almost any number
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')

    return inputs_real, inputs_z
```

**Generator model**
```python
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
```

**Discriminator model**
```python
# but why just x, one input data? the chart shows 2 inputs data???
def discriminator(x, n_units=128, reuse=False, alpha=0.01):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Hidden layer
        h1 = tf.layers.dense(x, n_units, activation=None)
        # Leaky ReLU
        h1 = tf.maximum(alpha * h1, h1)

        logits = tf.layers.dense(h1, 1, activation=None)
        out = tf.sigmoid(logits)

        return out, logits
```


**Hyperparameters**
```python
# Size of input image to discriminator
input_size = 784
# Size of latent vector to generator
# The latent sample is a random vector the generator uses to construct it's fake images. As the generator learns through training, it figures out how to map these random vectors to recognizable images that can fool the discriminator
z_size = 100 # not 784! so it can be any number?
# Sizes of hidden layers in generator and discriminator
g_hidden_size = 128
d_hidden_size = 128
# Leak factor for leaky ReLU
alpha = 0.01
# Smoothing
smooth = 0.1
```

**Build network**
```python
tf.reset_default_graph()

# Create our input placeholders
input_real, input_z = model_inputs(input_size, z_size)

# Build the model
g_out = generator(input_z, input_size)
# g_out is the generator output, not model object

# discriminate on real images, get output and logits
d_out_real, d_logits_real = discriminator(input_real)
# discriminate on generated images, get output and logits
d_out_fake, d_logits_fake = discriminator(g_out, reuse=True)

```

**Calculate losses**
```python
# get loss on how good discriminator work on real images
d_loss_real = tf.reduce_mean(
                  tf.nn.sigmoid_cross_entropy_with_logits(
				  			logits=d_logits_real,
							# labels are all true as 1s
							# label smoothing *(1-smooth)
                            labels=tf.ones_like(d_logits_real) * (1 - smooth)))

# get loss on how good discriminator work on generated images 							
d_loss_fake = tf.reduce_mean(# get the mean for all the images in the batch
                  tf.nn.sigmoid_cross_entropy_with_logits(
				  			logits=d_logits_fake,
							# labels are all false, as 0s
                            labels=tf.zeros_like(d_logits_real)))

# get total loss by adding up 							
d_loss = d_loss_real + d_loss_fake

# get loss on how well generator work for generating images as real as possible
g_loss = tf.reduce_mean(
             tf.nn.sigmoid_cross_entropy_with_logits(
			 			logits=d_logits_fake,
						# generator wants images all be real as possible, so set True, 1s
                        labels=tf.ones_like(d_logits_fake)))
```


**Optimizers**
```python
# Optimizers
learning_rate = 0.002

# Get the trainable_variables, split into G and D parts
t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

# update the selected weights, or the discriminator weights
d_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(d_loss, var_list=d_vars)

# update the selected weights, or the generator weights
g_train_opt = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)
```

**Training**
```python
batch_size = 100
epochs = 100
samples = []
losses = []

# Only save generator variables
saver = tf.train.Saver(var_list=g_vars)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for e in range(epochs):
		# for every batch idx of an epoch
        for ii in range(mnist.train.num_examples//batch_size):

			# get all
            batch = mnist.train.next_batch(batch_size)

			# get real image batch input
            # Get images, reshape and rescale to pass to D
            batch_images = batch[0].reshape((batch_size, 784))
            batch_images = batch_images*2 - 1

			# get fake image batch input
            # Sample random noise for G
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))

            # Run optimizers for g and d
			# for d_train_op, we need both inputs
            _ = sess.run(d_train_opt, feed_dict={input_real: batch_images, input_z: batch_z})
			# for g_train_opt, we only need the fake image input
            _ = sess.run(g_train_opt, feed_dict={input_z: batch_z})

        # At the end of each epoch, get the losses and print them out
        train_loss_d = sess.run(d_loss, {input_z: batch_z, input_real: batch_images})
        train_loss_g = g_loss.eval({input_z: batch_z})

        print("Epoch {}/{}...".format(e+1, epochs),
              "Discriminator Loss: {:.4f}...".format(train_loss_d),
              "Generator Loss: {:.4f}".format(train_loss_g))  

        # Save each epoch's d_loss and g_loss for displaying after entire training
        losses.append((train_loss_d, train_loss_g))

        # at the end of each epoch training
		# create a random sample and feed to the trained generator model to produce outputs, which will be reshaped to images for viewing after entire training
        sample_z = np.random.uniform(-1, 1, size=(16, z_size))
        gen_samples = sess.run(
                       generator(input_z, input_size, reuse=True),
                       feed_dict={input_z: sample_z})
        samples.append(gen_samples)
        saver.save(sess, './checkpoints/generator.ckpt')

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)
```

**Plot loss**
```python
# Now the entire training is finished, let's plot losses
fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
```
How to plot loss nicely in kur?
![Losses plotted](../../LIE_examples/gan_mnist/assets/loss_figure.png)


**Plot generated sample outputs for images**
```python
# samples: array for storing all generated sample outputs
# samples[epoch]: select an array of a particular epoch's generated sample
def view_samples(epoch, samples):
	# 4x4=16, as samples[epoch].shape: (16, ...)
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)

    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')

    return fig, axes

# Load samples from generator taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# plot the last epoch's generated sample images
_ = view_samples(-1, samples)
```
![Last epoch's generated sample plotted](../../LIE_examples/gan_mnist/assets/last_epoch_trained_samples.png)


**Display generated samples every 10 epochs of training**
```python
rows, cols = 10, 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
```
![generated_samples_every_10epochs](../../LIE_examples/gan_mnist/assets/generated_samples_every_10epochs.png)


**Restore a saved model and generate images**
```python
#
saver = tf.train.Saver(var_list=g_vars)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(
                   generator(input_z, input_size, reuse=True),
                   feed_dict={input_z: sample_z})
_ = view_samples(0, [gen_samples])
```


write the model in kurfile (just keras now, pytorch later)
