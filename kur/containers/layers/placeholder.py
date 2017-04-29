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


import warnings
from . import Layer, ParsingError

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
class Placeholder(Layer):				# pylint: disable=too-few-public-methods
	""" A shape placeholder which serves as a size hint, typically used as the
		first layer for each input to the model.

		Placeholders are also convenient places to name an input stream.
	"""

	###########################################################################
	@classmethod
	def get_container_name(cls):
		""" Returns the name of the container class.
		"""
		return 'input'

	###########################################################################
	def __init__(self, *args, **kwargs):
		""" Creates a new placeholder.
		"""
		super().__init__(*args, **kwargs)
		self._shape = None
		self.type = None

	###########################################################################
	def _parse_pre(self, engine):
		""" Pre-parsing hook.
		"""

		super()._parse_pre(engine)

		# get container_name: input
		container_name = self.get_container_name()

		if container_name in self.data:
			# evaluate Placeholder.data's value: images
			data = engine.evaluate(self.data[container_name])
			if isinstance(data, str):
				if 'name' in self.data:
					name = engine.evaluate(self.data['name'])
					logger.warning('Conflicting naming schemes for '
						'placeholder: "%s" and "%s". Using: "%s".',
						data, name, name)
				else:
					logger.trace('Using short-hand name for placeholder: %s',
						data)
					self.name = data
		else:
			warnings.warn('Parsing oddity in placeholder: {}'.format(
				self.data
			))

	###########################################################################
	def set_shape(self, shape):
		""" Sets a shape.
		"""
		if self._shape is not None:
			logger.warning('Modifying the shape of Placeholder "%s".',
				self.name)

		if not isinstance(shape, (list, tuple)):
			shape = (shape, )
		shape = tuple(x if x != 'None' else None for x in shape)
		for x in shape:
			if not isinstance(x, (int, type(None))):
				raise ParsingError(
					'All entries in "shape" must be integers, or in special '
					'cases None. Shape is: {}'.format(shape)
				)
		self._shape = shape

	###########################################################################
	def _parse(self, engine):
		""" Parse the placeholder.
		"""
		if 'shape' not in self.args:
			logger.trace('Placeholder "%s" has a deferred shape.', self.name)
		else:
			self.set_shape(engine.evaluate(self.args['shape'], recursive=True))

		if 'type' in self.args:
			self.type = engine.evaluate(self.args['type'], recursive=True)

	###########################################################################

	def _infer_shape(self, model):

		# the shape is (28,28,1) for mnist dataset
		inferred_shape = model.get_inferred_shape(self.name)

		if inferred_shape is None:
			if self._shape is None:
				raise ParsingError(
					'Placeholder "{}" requires a shape.'.format(self.name))
		else:
			# if placeholder._shape is still None
			if self._shape is None:
				# then give it a new shape
				self._shape = inferred_shape
				logger.trace('Inferred shape: %s', self._shape)
			else:
				if len(self._shape) != len(inferred_shape):
					raise ValueError('Placeholder "{}" does not have the '
						'same dimensionality as the data source it is '
						'connected to. Placeholder: {}, data source: {}.'
						.format(self.name, self._shape, inferred_shape))

				merged_shape = ()
				for user_shape, data_shape in \
					zip(self._shape, inferred_shape):
					if user_shape is None or data_shape is None:
						if user_shape is None:
							merged_shape += (data_shape, )
						else:
							merged_shape += (user_shape, )
					elif user_shape != data_shape:
						logger.warning('Placeholder "%s" specified a '
							'along a dimension that disagrees with the '
							'data source. We will defer to the data '
							'source, but this may cause an unexpected '
							'model to be built. Placeholder: %s, data: '
							'%s.', self.name, self._shape, inferred_shape)
						merged_shape += (data_shape, )
					else:
						merged_shape += (data_shape, )

				if merged_shape != self._shape:
					logger.trace('Using inferred shape for data %s: %s '
						'instead of %s', self.name, merged_shape,
						self._shape)
					self._shape = merged_shape

		# if the new shape is empty, then set it to be a scalar
		if self._shape == ():
			logger.trace('Promoting an empty shape to a scalar.')
			self._shape = (1,)

	###########################################################################
	def _build(self, model):
		""" Create the backend-specific placeholder.
		"""

		backend = model.get_backend()
		if backend.get_name() == 'keras':

			# import keras.backend
			import keras.backend as K			# pylint: disable=import-error

			# set up type for placehold layer
			if self.type is None:
				dtype = K.floatx()
			else:
				dtype = self.type
			logger.trace('Creating placeholder for "%s" with data type "%s".',
				self.name, dtype)

			# set up shape for placeholder, using shape info stored inside model object
			self._infer_shape(model)

			# import keras.layers
			import keras.layers as L			# pylint: disable=import-error

			# yield keras.layers.Input
			# to instantiate placeholder layer, we need name, shape, type
			# there is no **kwargs inside L.Input(), so no trainable arg input 
			yield L.Input(
				shape=self._shape,
				name=self.name,
				dtype=dtype
			)

		elif backend.get_name() == 'pytorch':

			self._infer_shape(model)
			yield {
				'shape' : self._shape,
				'layer' : model.data.placeholder(self.name)
			}

		else:
			raise ValueError(
				'Unknown or unsupported backend: {}'.format(backend))

	###########################################################################
	def shape(self, input_shapes=None):
		""" Returns the output shape of this layer for a given input shape.
		"""
		if input_shapes is not None:
			raise ValueError('Input placeholders do not take inputs.')
		return self._shape

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
