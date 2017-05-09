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


"""
The workflow of making MnistSupplier Style dataset
1. prepare dataset as a large numpy array
2. save this array using `kur.utils.idx.save` into idx format file
3. save this file online, providing a url; store this file locally, providing a path; create a shasum, providing a shasum number
3. then we can get the dataset using kurfile to set dataset as the following
train:
  data:
    - mnist:
        images:
          checksum: 440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
          path: "~/kur"
        labels:
          checksum: 3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
          path: "~/kur"
4. `kur.utils.package.install` will either install the dataset from url if not available, then provide the full file path given it is stored locally and shasum is available and correct
5. idx.load(path) into numpy.array, then save array into VanillaSource, then preprocess on this VanillaSource, finally store this preprocessed VanillaSource inside MnistSupplier.data
6. Provider class takes VanillaSource out of MnistSupplier.data, and make self.lens, self.keys and self.sources for provider class and BatchProvider class to use
7. BatchProvider class makes a lot more attributes like self.batch_size, self.num_batches, self.force (force_batch_size: true or false)...
8. More important BatchProvider.__iter__() can split data samples into a batch one at a time, when doing:
```
for batch in provider:
	batch
```
9. input arguments are sent to BatchProvider.__init__() from kurfile.yml Provider: specs
  provider:
    batch_size:
    num_batches: 1
"""

# Transfer MnistSupplier to XYSupplier
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

"""
0. create `xy_replace_mnist.py` into kur/supplier/
1. add `from .xy_replace_mnist import XYSupplier` into `kur/supplier/__init__.py`
1. create yml file to fit this idx files
2. see `xy_mnist_default.yml` and `xy_mnist1.yml`
"""
