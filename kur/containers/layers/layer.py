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

from .. import Container

import logging
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)
from ...utils import DisableLogging
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

###############################################################################
class Layer(Container):						# pylint: disable=abstract-method
	""" Base class for layers, which are containers that produce
		backend-specific layers, or which wrap backend-specific layers with
		additional backend-specific operations.
	"""

	###########################################################################
	def terminal(self):
		""" The identifying feature of a Layer is that it is terminal, meaning
			that it actually produces layers.
		"""
		return True

	###########################################################################
	def shape(self, input_shapes):
		""" Returns the output shape of this layer for a given input shape.

			# Arguments

			input_shape: list of tuples. A list of tuples, corresponding to the
				shapes of the input layers, one for each input layer, excluding
				batch size.

			# Return value

			A tuple specifying the shape of each output from the layer,
			excluding batch size.
		"""
		raise NotImplementedError

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
