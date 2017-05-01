"""
Copyright 2017 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import colorsys
import os

import itertools
from collections import OrderedDict

import numpy

from . import TrainingHook
from ...loggers import PersistentLogger, Statistic

import logging
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)
from ...utils import DisableLogging, idx
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

###############################################################################
class PlotWeightsHook(TrainingHook):
	""" Hook for creating plots of loss.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the hook.
		"""
		return 'plot_weights'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new plotting hook, get plot filenames and matplotlib ready.
		"""

		super().__init__(*args, **kwargs)

		# put plot filenames int of dict
		# plots = dict(zip(
		# 	('loss_per_batch', 'loss_per_time', 'throughput_per_time'),
		# 	(loss_per_batch, loss_per_time, throughput_per_time)
		# ))

		# get plot filenames as dict into self.plots
		# self.plots = plots
		# for k, v in self.plots.items():
		# 	if v is not None:
		# 		self.plots[k] = os.path.expanduser(os.path.expandvars(v))

		# import matplotlib and use
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
	def notify(self, status, log=None, info=None):
		""" Creates the plot.
		"""

		from matplotlib import pyplot as plt	# pylint: disable=import-error

		logger.critical('Plotting hook received training message.')

		if status not in (
			TrainingHook.TRAINING_END,
			TrainingHook.VALIDATION_END,
			TrainingHook.EPOCH_END
		):
			logger.debug('Plotting hook does not handle this status.')
			return



		def plot_weights(kernel_filename):

			# load weights from weight files in idx format
			w = idx.load(kernel_filename)

			# Get the lowest and highest values for the weights.
			# This is used to correct the colour intensity across
			# the images so they can be compared with each other.
			w_min = np.min(w)
			w_max = np.max(w)


			# Create figure with 3x4 sub-plots,
			# where the last 2 sub-plots are unused.
			fig, axes = plt.subplots(3, 4)
			fig.subplots_adjust(hspace=0.3, wspace=0.3)


			for i, ax in enumerate(axes.flat):
				# Only use the weights for the first 10 sub-plots.
				if i<10:
				# if i<64:
					# Get the weights for the i'th digit and reshape it.
					# Note that w.shape == (img_size_flat, 10)
					# mnist (28, 28)
					# cifar (32,32,3)
					image = w[:, i].reshape((28, 28))


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

			# get filename without "dir/.."
			filename_cut_dir = kernel_filename[kernel_filename.find("/..")+3 :]
			# save figure with a nicer name
			plt.savefig('plot_weights/{}_epoch_{}.png'.format(filename_cut_dir, info['epoch']))




		if info['epoch'] == 1 or info['epoch'] % 2 == 0:
			# save weights plots
			logger.critical("\n\nLet's print weights every 100 epochs\n\n")


			# get all the validation weights names
			valid_weights_filenames = []
			# how to give a path name to plot_weights???
			for dirpath, _, filenames in os.walk("mnist.best.valid.w"): # mnist or cifar
				for this_file in filenames:
					valid_weights_filenames.append(dirpath+"/"+this_file)


			for this_file in valid_weights_filenames:
				if this_file.find("kernel") > -1:
					plot_weights(this_file)

			# save validation_loss on the plotting