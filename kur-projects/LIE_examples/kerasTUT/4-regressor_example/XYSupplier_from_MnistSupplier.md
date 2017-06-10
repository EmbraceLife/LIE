# Create a dataset in format of MnistSupplier

1. prepare dataset as a large numpy array

```python
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
import kur.utils.idx as idx

X = np.linspace(-1, 1, 200)
np.random.shuffle(X)    # randomize the data
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200, ))

# plot data
# plt.scatter(X, Y)
# plt.show()

# split dataset into Train, Validate, Test 3 parts
X_train, Y_train = X[:100], Y[:100]
X_test, Y_test = X[100:150], Y[100:150]
X_valid, Y_valid = X[150:], Y[150:]

```
2. save this array using `kur.utils.idx.save` into idx format file

```python

# Store the 3 datasets into 6 idx files
idx.save("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/x_train", X_train)
idx.save("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/y_train", Y_train)
idx.save("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/x_valid", X_valid)
idx.save("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/y_valid", Y_valid)
idx.save("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/x_test", X_test)
idx.save("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/y_test", Y_test)

# load the idx file into numpy array
valid_x = idx.load("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/x_valid")
valid_y = idx.load("/Users/Natsume/Documents/kur_experiment/LIE_examples/Hvass_Mnist_tutorials/tutorial1/y_valid")
```

3. save this file online, providing a url; store this file locally, providing a path; create a shasum, providing a shasum number

```yml
train:
  # Let's include checksums for all of the data we download.
  data:
    - xy:
        x:
          checksum:
          path: "/Users/Natsume/Documents/kur_experiment/LIE_examples/Morvan_keras_tutorials/tutorial1/x_train"
        y:
          checksum:
          path: "/Users/Natsume/Documents/kur_experiment/LIE_examples/Morvan_keras_tutorials/tutorial1/y_train"
```

## create xy_replace_mnist.py for kur/suppliers/
4. `kur.utils.package.install` will either install the dataset from url if not available, then provide the full file path given it is stored locally and shasum is available and correct
5. idx.load(path) into numpy.array, then save array into VanillaSource, then preprocess on this VanillaSource, finally store this preprocessed VanillaSource inside MnistSupplier.data
6. Provider class takes VanillaSource out of MnistSupplier.data, and make self.lens, self.keys and self.sources for provider class and BatchProvider class to use

```python
# added_features
################################
# create temporal folders
import tempfile

################################
# prepare logger functionality
# import logging
import matplotlib.pyplot as plt
import numpy as np
# logger = logging.getLogger(__name__)
# from .utils import DisableLogging
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)

################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

################################
# use pytorch to do tensor ops, layers, activations
import torch
from torch.autograd import Variable
import torch.nn.functional as F

################################################################
################################################################

import numpy

from ..utils import idx, package
from . import Supplier
from ..sources import VanillaSource

###############################################################################
class XYSupplier(Supplier):
	""" A supplier which supplies MNIST image/label pairs. These are downloaded
		from the internet, verified, and parsed as IDX files.
	"""

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the supplier.
		"""
		return 'xy'

	###########################################################################
	# x and y are dicts extracted from {xy: {x: {}, y: {}}}
	def __init__(self, x, y, *args, **kwargs):
		""" Load dataset into VanillaSource
		"""

		super().__init__(*args, **kwargs)

		self.data = {
						# MnistSupplier._get_filename is staticmethod can be used in __init__
			'x' :
				# VanillaSource.data == numpy.array
				VanillaSource(idx.load(XYSupplier._get_filename(x))
				# dataset.shape = (60000, 28, 28)
			),
			'y' :
				VanillaSource(idx.load(XYSupplier._get_filename(y))
			)
		}

	###########################################################################
	@staticmethod
	def _onehot(source):
		""" Not used in here
		"""
		onehot = numpy.zeros((len(source), 10))
		for i, row in enumerate(source.data):
			onehot[i][row] = 1
		source.data = onehot

		return source

	###########################################################################
	@staticmethod
	def _normalize(source):
		""" Not used in here
		"""
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

```

## Then do the following

7. BatchProvider class makes a lot more attributes like self.batch_size, self.num_batches, self.force (force_batch_size: true or false)...

8. More important BatchProvider.__iter__() can split data samples into a batch one at a time, when doing:

```python
for batch in provider:
	batch
```

9. input arguments are sent to BatchProvider.__init__() from kurfile.yml Provider: specs
  provider:
    batch_size:
    num_batches: 1

10. add `from .xy_replace_mnist import XYSupplier` into `kur/supplier/__init__.py`

11. create yml file to fit this idx files

12. see `xy_mnist_default.yml` and `xy_mnist1.yml` in 'kerasTUT/4-regressor_example/'
