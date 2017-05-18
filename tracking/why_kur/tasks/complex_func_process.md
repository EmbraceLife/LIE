1. write simple lines of code to do something
2. add comment to find values needs to become variable
3. try to replace those values with variables or expressions
4. add func decorations to make a func

**Step 1**

```python
ax2 = plt.subplot2grid((5,1), (1,0), rowspan=1, colspan=1)
img=mpimg.imread('/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights/convolution_0+weight.kur_epoch_1.png')
ax2.imshow(img)
```

**Step 2, 3**

```python
plt.figure(figsize=(5, 10))

# dirpath is a global variable from above
dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"
# make an ordered list of layers to plot
layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

# define how many rows of subplots we need
num_rows = len(layer_order)

# i from animation(), start from 0, repetition
for i in range(10):
	epoch_idx = i
	# empty file name
	img_file = None

	# plot layer one by one
	for row in range(num_rows):

		# access all filenames in the dirpath
		for dirpath, _, filenames in os.walk(dirpath):
			# looping through every filename
			for filename in filenames:

				# search all filenames with a row's keywords
				# "epoch"+str(i+1): make sure all layers in one figure share the same epoch
				if filename.find(layer_order[row][0]) > -1 and filename.find(str(str(i+1)+".png"))> -1:
					img_file = dirpath + '/' + filename
					img=mpimg.imread(img_file)

					ax1 = plt.subplot2grid((num_rows,1), (row,0), rowspan=1, colspan=1)
					ax1.set_title(layer_order[row][0] + "_epoch_" + str(i+1))
					ax1.imshow(img)
			set_trace()

```

**Step 4: make animate(i) function for animation**

```python

def animate(i):
	# dirpath is a global variable from above
	dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"

	# make an ordered list of layers to plot
	layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

	# define how many rows of subplots we need
	num_cols = len(layer_order)

	# i from animation(), start from 0, repetition

	# empty file name
	img_file = None

	# plt.figure(figsize=(5, 10))
	# plot layer one by one
	for col in range(num_cols):

		# access all filenames in the dirpath
		for dirpath, _, filenames in os.walk(dirpath):
			# looping through every filename
			for filename in filenames:

				# search all filenames with a row's keywords
				# "epoch"+str(i+1): make sure all layers in one figure share the same epoch
				if filename.find(layer_order[col][0]) > -1 and filename.find(str(i+1)+".png")> -1:
					img_file = dirpath + '/' + filename
					img=mpimg.imread(img_file)

					ax = plt.subplot2grid((2, num_cols), (0, col), rowspan=2, colspan=1)
					ax.set_title(layer_order[col][0] + "_epoch_" + str(i+1))
					ax.imshow(img)
	# plt.show() can plot all subplots, meaning all subplots are stored inside plt
	return plt


```
**Step 4.2: make init() function for animation**

```python

def init():
	# dirpath is a global variable from above
	dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"

	# make an ordered list of layers to plot
	layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

	# define how many rows of subplots we need
	num_cols = len(layer_order)

	# init() only plot on the first epoch
	epoch_idx = 1
	# empty file name
	img_file = None

	# plt.figure(figsize=(5, 10))
	# plot layer one by one
	for col in range(num_cols):

		# access all filenames in the dirpath
		for dirpath, _, filenames in os.walk(dirpath):
			# looping through every filename
			for filename in filenames:

				# search all filenames with a col's keywords
				# "epoch"+str(i+1): make sure all layers in one figure share the same epoch
				if filename.find(layer_order[col][0]) > -1 and filename.find(str(epoch_idx)+".png")> -1:
					img_file = dirpath + '/' + filename
					img=mpimg.imread(img_file)

					ax = plt.subplot2grid((2, num_cols), (0, col), rowspan=2, colspan=1)
					ax.autoscale(True)
					ax.set_title(layer_order[col][0] + "_epoch_" + str(epoch_idx))
					ax.imshow(img)

	# plt.subplots_adjust(left=0.25, bottom=0.25)
	axframe = plt.axes([0, 0, 0.5, 0.3])
	sframe = Slider(axframe, 'Frame', 0, 99, valinit=0,valfmt='%d')
	# plt.show() can plot all subplots, meaning all subplots are stored inside plt
	return plt


```

**Step 5: make animation**

```python

fig = plt.figure(figsize=(20, 3))
# call the animator.  blit=True means only re-draw the parts that have changed.
# blit=True dose not work on Mac, set blit=False
# interval= update frequency
ani = animation.FuncAnimation(fig=fig, func=animate, frames=1000, init_func=init,
                              interval=5000, blit=False)

plt.show()
```
