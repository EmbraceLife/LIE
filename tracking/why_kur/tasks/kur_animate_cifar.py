################################
# inside __main__.animate()

def animate_layers(args):

	""" Builds a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)

	layer_order = spec.data['train']['hooks'][0]['plot_weights']['animate_layers']


	def animate(i):
		# dirpath is a global variable from above
		dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"

		# # make an ordered list of layers to plot
		# layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

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
					if filename.find(layer_order[col]) > -1 and filename.find(str(i+1)+".png")> -1:
						img_file = dirpath + '/' + filename
						img=mpimg.imread(img_file)

						ax = plt.subplot2grid((3, num_cols), (0, col), rowspan=3, colspan=1)
						ax.set_title(layer_order[col] + "_epoch_" + str(i+1))
						ax.imshow(img)
		# plt.show() can plot all subplots, meaning all subplots are stored inside plt
		return plt


	def init():
		# dirpath is a global variable from above
		dirpath = "/Users/Natsume/Downloads/temp_folders/demo_cifar/cifar_plot_weights"

		# # make an ordered list of layers to plot
		# layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]

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
					if filename.find(layer_order[col]) > -1 and filename.find(str(epoch_idx)+".png")> -1:
						img_file = dirpath + '/' + filename
						img=mpimg.imread(img_file)

						# set num_rows and num_cols
						ax = plt.subplot2grid((3, num_cols), (0, col), rowspan=3, colspan=1)
						ax.set_xmargin(0.1)
						ax.set_title(layer_order[col] + "_epoch_" + str(epoch_idx))
						ax.imshow(img)


		# plt.show() can plot all subplots, meaning all subplots are stored inside plt
		return plt


	# make an ordered list of layers to plot
	# layer_order = [['images'], ['convolution_0'], ['conv_layer1'], ['convolution_1'], ['conv_layer2']]
	# layer_order = [['images'],  ['conv_layer1'], ['conv_layer2']]
	# layer_order = ['images', 'conv_layer1', 'conv_layer2']
	# create the figure
	fig = plt.figure(figsize=(20, 6))

	ani = animation.FuncAnimation(fig=fig, func=animate, frames=10000, init_func=init,
	                              interval=5000, blit=False)

	plt.show()

################################
# Inside __main__.build_parser(), right below prepare_data block, add the following lines 
subparser = subparsers.add_parser('animate',
	help='Animate plots of selected layers')
subparser.set_defaults(func=animate_layers)
