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

import numpy

from ..utils import idx, package
from . import Supplier
from ..sources import VanillaSource

import logging
logger = logging.getLogger(__name__)
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule
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
		""" Creates a new MNIST supplier by download or access data file based on dicts (store url, path, checksum)

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
		logger.debug("(self, images, labels, *args, **kwargs): start \nInstantiate MnistSupplier object with spec object \n1. get args from super().__init__(*args, **kwargs); \n2. ensure data file exist locally and return the file path; \n3. load the idx file into numpy arrays; \n4. make the VanillaSource object from numpy array; \n5. normalize the data in the form of VanillaSource; \n6. save it as a dict element inside MnistSupplier.data dict \n\n")

		super().__init__(*args, **kwargs)

		# MnistSupplier._get_filename: download a gz file and store the dataset in a path, and return this path
		# idx.load: load the gz file into numpy arrays
		# VanillaSource(): make numpy array a data source for creating a data supplier
		images_path = MnistSupplier._get_filename(images)
		idx_array_img = idx.load(images_path)
		VanillaSource_object_img = VanillaSource(idx_array_img)
		image_norm_MnistSupplier = MnistSupplier._normalize(VanillaSource_object_img)

		labels_path = MnistSupplier._get_filename(labels)
		idx_array_label = idx.load(labels_path)
		VanillaSource_object_label = VanillaSource(idx_array_label)
		labels_onehot_MnistSupplier = MnistSupplier._normalize(VanillaSource_object_label)
		self.data = {
			'images' : image_norm_MnistSupplier,
			'labels' : labels_onehot_MnistSupplier
		}

		logger.debug("(self, images, labels, *args, **kwargs): start \nInstantiate MnistSupplier object with spec object \n1. get args from super().__init__(*args, **kwargs); \n2. ensure data file exist locally and return the file path; \n3. load the idx file into numpy arrays; \n4. make the VanillaSource object from numpy array; \n5. normalize the data in the form of VanillaSource; \n6. save it as a dict element inside MnistSupplier.data dict \n\n Processed_Inputs: \n1. images: \n%s \n2. image idx file path: \n%s \n3. load idx file into numpy array: \n%s \n4. save numpy array into VanillaSource: \n%s \n5. normalize(onehot) VanillaSource object \n%s \n\n", images, images_path, idx_array_img.shape, VanillaSource_object_img.__dict__.keys(), image_norm_MnistSupplier.__dict__.keys())
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

		# package.install: ensure the path exist locally
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
