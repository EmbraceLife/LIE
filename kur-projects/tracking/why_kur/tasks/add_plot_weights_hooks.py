#######################################
# inside kur/model/hooks/ add a new file plot_weights_hooks.py below


import colorsys
import os

import itertools
from collections import OrderedDict

import numpy
import tempfile

from . import TrainingHook
from ...loggers import PersistentLogger, Statistic

import logging
import matplotlib.pyplot as plt
import numpy as np
import math
logger = logging.getLogger(__name__)
from ...utils import DisableLogging, idx

###############################################################################
class PlotWeightsHook(TrainingHook):
	""" Hook for creating plots of weights.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the hook.
		"""
		return 'plot_weights'

	###########################################################################
	def __init__(self, plot_directory, weight_file, with_weights, plot_every_n_epochs, *args, **kwargs):
		""" Creates a new plot_weights hook, get weights filenames, path for saving plots, keywords for selecting layer-weights, num_epochs before plotting, and matplotlib ready.
		"""

		super().__init__(*args, **kwargs)

		self.directory = plot_directory
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		# plot weights every n number of epochs training
		self.plot_every_n_epochs = plot_every_n_epochs

		# use tempfile or user specified folders
		if weight_file is None:
			self.weight_file = None
		else:
			self.weight_file = weight_file

		self.with_weights = with_weights

		try:
			import matplotlib					# pylint: disable=import-error
		except:
			logger.exception('Failed to import "matplotlib". Make sure it is '
				'installed, and if you have continued trouble, please check '
				'out our troubleshooting page: https://kur.deepgram.com/'
				'troubleshooting.html#plotting')
			raise

		# Set the matplotlib backend to one of the known backends.
		matplotlib.use('Agg')

	###########################################################################
	# added model=None to bring in model for saving weights
	def notify(self, status, log=None, info=None, model=None):
		""" Creates the plot.
		"""

		from matplotlib import pyplot as plt	# pylint: disable=import-error

		if status not in (
			# TrainingHook.TRAINING_END,
			# TrainingHook.VALIDATION_END,
			TrainingHook.EPOCH_END, # , is a must here
		):

			return

		# create a tempfolder for the current model weights
		weight_path = None
		tempdir = tempfile.mkdtemp()
		weight_path = os.path.join(tempdir, 'current_epoch_model')

		# save this model weights to this tempfolder
		model.save(weight_path)


		# borrowed from https://hyp.is/MKzd7C4eEeeWlPvso_EWdg/nbviewer.jupyter.org/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb
		def plot_weights(kernel_filename):
			# designed to plot weights with dim (n_samples, img_height*img_width), given height == width

			# load weights from weight files in idx format
			w = idx.load(kernel_filename)

			# Get the lowest and highest values for the weights.
			# This is used to correct the colour intensity across
			# the images so they can be compared with each other.
			w_min = np.min(w)
			w_max = np.max(w)

			# make sure weight array has the following shape (flattend_pixels, num_classes)
			s1, s2 = w.shape
			if s1 < s2:
				w = w.reshape((s2, s1))


			flattend_pixels, num_classes = w.shape
			# Number of grids to plot.
			# Rounded-up, square-root of the number of filters.
			num_grids = math.ceil(math.sqrt(num_classes))
			width_pixels = math.ceil(math.sqrt(flattend_pixels))

			# Create figure with a grid of sub-plots.
			fig, axes = plt.subplots(num_grids, num_grids)
			# Create figure with 3x4 sub-plots,
			# where the last 2 sub-plots are unused.
			# fig, axes = plt.subplots(3, 4)
			fig.subplots_adjust(hspace=0.3, wspace=0.3)

			# plot weights into images one by one (according to the shape above)
			for i, ax in enumerate(axes.flat):

				if i<num_classes:

					image = w[:, i].reshape((width_pixels, width_pixels))


					# Set the label for the sub-plot.
					ax.set_xlabel("Weights: {0}".format(i))


					# Plot the image.
					ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

				if i == 0:
					# how to make a title for plotting
					ax.set_title("validation_loss: {}".format(round(info['Validation loss'][None]['labels'], 3)))

				# Remove ticks from each sub-plot.
				ax.set_xticks([])
				ax.set_yticks([])
			# if we plot while training, we can't save it
			# plt.show()


			# cut the filename part before 'dense': this is working for both with and without a given weight folder, like mnist.best.valid.w
			filename_cut_dir = kernel_filename[kernel_filename.find("dense") :]
			# save figure with a nicer name
			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))

		# borrowed from  https://hyp.is/4mtFzjBSEeeNikfkfV9o4w/nbviewer.jupyter.org/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/02_Convolutional_Neural_Network.ipynb
		def plot_conv_weights(kernel_filename, input_channel=0):

			# load weights from weight files in idx format
			w = idx.load(kernel_filename)

			# Get the lowest and highest values for the weights.
			# This is used to correct the colour intensity across
			# the images so they can be compared with each other.
			w_min = np.min(w)
			w_max = np.max(w)

			# add this block, because in pytorch, convolution for cifar dataset, dimension order is different from keras
			# this way below can handle both pytorch and keras when plotting cifar images
			s1, s2, s3, s4 = w.shape
			if s1 > s4:
				w = w.reshape((s3, s4, s2, s1))

			# set_trace()
			# Number of filters used in the conv. layer.
			num_filters = w.shape[3]

			# Number of grids to plot.
			# Rounded-up, square-root of the number of filters.
			num_grids = math.ceil(math.sqrt(num_filters))

			# Create figure with a grid of sub-plots.
			fig, axes = plt.subplots(num_grids, num_grids)

			# Plot all the filter-weights.
			for i, ax in enumerate(axes.flat):
				# Only plot the valid filter-weights.
				if i<num_filters:

					img = w[:, :, input_channel, i]

					# Plot image.
					ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')

				if i == 0:
					# plot loss on the first image
					ax.set_title("validation_loss: {}".format(round(info['Validation loss'][None]['labels'], 3)))
				# Remove ticks from the plot.
				ax.set_xticks([])
				ax.set_yticks([])


		    # if we plot while training, we can't save it
			# plt.show()

			# cut the dirname part before "convolution"
			filename_cut_dir = kernel_filename[kernel_filename.find("convol") :]
			# save figure with a nicer name
			plt.savefig('{}/{}_epoch_{}.png'.format(self.directory, filename_cut_dir, info['epoch']))



		if info['epoch'] == 1 or info['epoch'] % self.plot_every_n_epochs == 0:
			# save weights plots
			# logger.critical("\n\nLet's print weights at epoch idx 1 or every %s epochs\n\n", self.plot_every_n_epochs)


			# get all the validation weights names
			valid_weights_filenames = []

			if self.weight_file is None:
				self.weight_file = weight_path

			for dirpath, _, filenames in os.walk(self.weight_file): # mnist or cifar

				for this_file in filenames:
					valid_weights_filenames.append(dirpath+"/"+this_file)

			# find two layers-weights with selected keywords, and plot their weights, either single dense layer model or covolutional layer weights
			for this_file in valid_weights_filenames:
				for weight_keywords in self.with_weights:

					if this_file.find(weight_keywords[0]) > -1 and this_file.find(weight_keywords[1]) > -1:

						if weight_keywords[0].find("convol") > -1 or weight_keywords[1].find("convol") > -1:

							plot_conv_weights(this_file)

						else:
							plot_weights(this_file)




#######################################
# add `model = None` to all Training hooks `notify()` function
# for example, go inside `plot_hook.py` and its `notify()`
# add `model = None` into
def notify(self, status, log=None, info=None, model=None):



#######################################
# Inside `kur/model/hooks/__init__.py`, add to the end the following code
from .plot_weights_hook import PlotWeightsHook


#######################################
# There seems nothing to be added in `executor.py`
