"""
Copyright 2016 Deepgram

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
import matplotlib.pyplot as plt
import numpy as np
import numpy
import logging
from ..utils import idx, package
from . import Supplier
from ..sources import VanillaSource

logger = logging.getLogger(__name__)
from ..utils import DisableLogging
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())
###############################################################################
class MnistSupplier(Supplier):
	""" A supplier which supplies MNIST image/label pairs. These are downloaded
		from the internet, verified, and parsed as IDX files.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'mnist'

	###########################################################################
	def __init__(self, labels, images, *args, **kwargs):
		""" Creates a new MNIST supplier.

			# Arguments

			labels: str or dict. If str, the path to the IDX file containing
				the image labels. If a dict, it should follow one of these
				formats:

				1. {"url" : URL, "checksum" : SHA256, "path" : PATH}, where URL
				   is the source URL (if the image file needs downloading),
				   SHA256 is the SHA-256 hash of the file (optional, and can be
				   missing or None to skip the verification), and PATH is the
				   path to save the file in (if missing or None it defaults to
				   the system temporary directory).

			images: str or dict. Specifies where the MNIST images can be found.
				Accepts the same values as `labels`.
		"""

		logger.critical("(self, images, labels, *args, **kwargs): \n\nInstantiate MnistSupplier object with spec.data[section]['data'] \n\n1. get args from super().__init__(*args, **kwargs); \n\n2. ensure data file exist locally and return the file path, (download them if not avaialbe locally); \n\n3. load the idx file into numpy arrays; \n\n4. make the VanillaSource object from numpy array; \n\n5. normalize & onehot (data processing) the data in the form of VanillaSource; \n\n6. save it as a dict element inside MnistSupplier.data dict \n\n")

		if logger.isEnabledFor(logging.CRITICAL):
			print("""
super().__init__(*args, **kwargs)

self.data = {
	'images' : MnistSupplier._normalize(
		VanillaSource(idx.load(MnistSupplier._get_filename(images)))
	),
	'labels' : MnistSupplier._onehot(
		VanillaSource(idx.load(MnistSupplier._get_filename(labels)))
	)
}
			""")


		super().__init__(*args, **kwargs)

		self.data = {
			'images' : MnistSupplier._normalize(
				VanillaSource(idx.load(MnistSupplier._get_filename(images)))
			),
			'labels' : MnistSupplier._onehot(
				VanillaSource(idx.load(MnistSupplier._get_filename(labels)))
			)
		}

		# plot 9 images with labels
		images_p = idx.load(MnistSupplier._get_filename(images))[0:9]
		labels_p = idx.load(MnistSupplier._get_filename(labels))[0:9]
		image_dim = images_p[0].shape

		MnistSupplier.plot_images(images=images_p, image_dim = image_dim, cls_true=labels_p)


	@staticmethod
	def plot_images(images, image_dim, cls_true, cls_pred=None):

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

	###########################################################################
	@staticmethod
	def _onehot(source):
		onehot = numpy.zeros((len(source), 10))
		for i, row in enumerate(source.data):
			onehot[i][row] = 1
		source.data = onehot

		return source

	###########################################################################
	@staticmethod
	def _normalize(source):
		# Numpy won't automatically promote the uint8 fields to float32.
		source.data = source.data.astype(numpy.float32)

		# Normalize
		source.data /= 255
		source.data -= source.data.mean()

		source.data = numpy.expand_dims(source.data, axis=-1)

		return source

	###########################################################################
	@staticmethod
	def _get_filename(target):
		""" Returns the filename associated with a particular target.

			# Arguments

			target: str or dict. The target specification. For locally-stored
				files, it can be a string (path to file) or a dictionary with
				key 'local' that contains the file path. For network files,
				it is a dictionary with 'url' (source URL); it may also
				optionally contain 'sha256' (SHA256 checksum) and 'path' (local
				storage directory for the file).

			# Return value

			String to the file's locally stored path. May not exist.
		"""

		if isinstance(target, str):
			target = {'path' : target}
		path, _ = package.install(
			url=target.get('url'),
			path=target.get('path'),
			checksum=target.get('checksum')
		)
		return path

	###########################################################################
	def get_sources(self, sources=None):
		""" Returns all sources from this provider.
		"""

		if sources is None:
			sources = list(self.data.keys())
		elif not isinstance(sources, (list, tuple)):
			sources = [sources]

		for source in sources:
			if source not in self.data:
				raise KeyError(
					'Invalid data key: {}. Valid keys are: {}'.format(
						source, ', '.join(str(k) for k in self.data.keys())
				))

		return {k : self.data[k] for k in sources}

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
