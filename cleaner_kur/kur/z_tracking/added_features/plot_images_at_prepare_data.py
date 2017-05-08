################################
# inside __main__.prepare_data(), at the end of this function

# plotting a few images
		if args.plot_images:
			logger.critical("\n\nPlot 9 images\n\n")
			for k, v in spec.data['train']['data'][0].items():
				name1 = k

			if name1 == 'mnist':
				images_p = batch['images'][0:9]
				image_dim = images_p.shape[1:-1]

			elif name1 == 'cifar':
				images_p = batch['images'][0:9]
				image_dim = images_p.shape[1:]

			else:
				return None # no plotting

			labels_p = [np.argmax(label) for label in batch['labels'][0:9]]

			plot_images(images=images_p, cls_true=labels_p, image_dim=image_dim)

		


################################
# outside __main__.prepare_data(), right after this function
# borrowed from https://hyp.is/9KkYnCyIEeeeUt-8LW6lMw/nbviewer.jupyter.org/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb
def plot_images(images, cls_true, image_dim, cls_pred=None):

	assert len(images) == len(cls_true) == 9

	# Create figure with 3x3 sub-plots.
	fig, axes = plt.subplots(3, 3)
	fig.subplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		# Plot image.
		ax.imshow(images[i].reshape(image_dim), cmap='binary')


		# Show true and predicted classes.
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
		ax.set_xticks([])
		ax.set_yticks([])
	plt.show()

################################
# Inside __main__.build_parser(), within the block of "subparser = subparsers.add_parser('data', ...."
# near the end of this block, before "subparser.set_defaults(func=prepare_data)", insert the following code
subparser.add_argument('-plot', '--plot_images', action='store_true', help='plot 9 images from a batch of data provider.')
