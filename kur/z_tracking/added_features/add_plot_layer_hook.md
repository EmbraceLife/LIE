# How to plot convolutional layer


Add `layer_index` to `mnist-defaults.yml` inside 'LIE_examples/deepgram demo'**

```yml
hooks:
- plot_weights:
  layer_index: [1,2] # convol1, relu1
  plot_every_n_epochs: 1
  plot_directory: mnist/mnist_plot_weights
  weight_file: mnist/mnist.best.valid.w
  with_weights:
  #   - ["kernel", "dense"]
    - ["kernel", "convol"]
  # pytorch use weight, keras use kernel
  # plot_weights are for arrays can be well splitted as images
  #   - ["weight", "dense"]
    - ["weight", "convol"]
```


Add additional arguments to `__init__` in `plot_weights_hooks.py`

```python
def __init__(self, layer_index, plot_directory, weight_file, with_weights, plot_every_n_epochs, *args, **kwargs):
	...
	# added self.layer_idx
	self.layer_idx = layer_index
	...
```

Add convolutional layer plotting function to `notify()`:

```python
# image is an image sampe data
def plot_conv_layer(layer_out, layer_name):


	values = layer_out
	# Number of filters used in the conv. layer.
	num_filters = values.shape[3]

	# Number of grids to plot.
	# Rounded-up, square-root of the number of filters.
	num_grids = math.ceil(math.sqrt(num_filters))

	# Create figure with a grid of sub-plots.
	fig, axes = plt.subplots(num_grids, num_grids)

	# Plot the output images of all the filters.
	for i, ax in enumerate(axes.flat):
		# Only plot the images for valid filters.
		if i<num_filters:
			# Get the output image of using the i'th filter.
			# See new_conv_layer() for details on the format
			# of this 4-dim tensor.
			img = values[0, :, :, i]
			# 0 cos there is only one image
			# i refers to index of output channels/images of this convol layer

			# Plot image.
			ax.imshow(img, interpolation='nearest', cmap='binary')

		# Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])

	# if we plot while training, we can't save it
	# plt.show()
	# save figure with a nicer name
	plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, layer_name, info['epoch']))
```

Add conditions to plot convol layers after plotting weights in `notify()`:
```python
# sample data comes from info['sample']
model_keras = model.compiled['raw']

from keras import backend as K

for index in self.layer_idx:

	layer_output = K.function([model_keras.layers[0].input],
									  [model_keras.layers[index].output])
	input_dim = model_keras.layers[0].input._keras_shape
	img_dim = (1,) + input_dim[1:]
	sample_img = info['sample'].reshape(img_dim)
	# layer is a numpy.array

	layer_out = layer_output([sample_img])[0]

	plot_conv_layer(layer_out, layer_name)

```

Add codes to create a sample data inside `wrapped_train()` of `executor.py`: (see code in between ########)

```python
if training_hooks:
	for hook in training_hooks:
		hook.notify(
			TrainingHook.TRAINING_START,
			log=log
		)

all_done = False

#############################################################
# prepare a single image sample data
sample = None
for batch in provider:
	values = [v for v in batch.values()]
	if len(values[0].shape) > len(values[1].shape):
		sample = values[0][0]#.reshape(1,28,28,1)


#############################################################
# Main training loop.
timers['all'].resume()
while not all_done:
```

Add a single line of code inside `run_training_hooks()` of `executor.py`: (see code in between)

```python
info = {
	'epoch' : epoch+1,
	'total_epochs' : epochs,
	'Training loss' : cur_train_loss, ## don't forget , here
	#################
	# added a single image data sample
	'sample': sample
	################
}

```
