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

import contextlib
import io
import re
import os
import sys
import tempfile
import shutil
import logging
import functools
import warnings
from collections import OrderedDict
import numpy
from . import Backend
from .. import __homepage__
from ..loss import Loss
from ..utils import can_import, EnvironmentalVariable, redirect_stderr, \
	idx, DisableLogging
from ..providers import BatchProvider

logger = logging.getLogger(__name__)

# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule
import numpy as np
###############################################################################
class KerasBackend(Backend):
	""" A Keras backend.

		# Dependencies

		- keras
		- theano OR tensorflow
		- h5py
	"""

	###########################################################################
	@classmethod
	def is_supported(cls):
		""" Returns True if this backend can be used.
		"""
		return can_import('keras') and (
			can_import('theano') or can_import('tensorflow')
		)

	###########################################################################
	def __init__(self, backend=None, optimizer=None, theano_flags=None,
		*args, **kwargs):
		""" Creates a new Keras backend.

			As per the base class documentation, we should do all necessary
			Keras-related initialization here, including checking obvious
			things like "Is Keras installed?" or "Is a backend installed?"

			# Arguments

			backend: str or None (default: None). The Keras backend to use
				(either "theano" or "tensorflow"). None uses the system
				default.
			optimizer: None or False (default: None). If False, Theano is told
				to disable optimizations. If None, Theano is not told to do
				anything special with optimization. This is supplied as a
				workaround to installing a BLAS library when training on the
				CPU. This is ignored for the TensorFlow backend.
		"""

		super().__init__(*args, **kwargs)

		if backend is not None:
			logger.info('The %s backend for Keras has been requested.',
				backend)

			if 'keras' in sys.modules:
				import keras.backend as K		# pylint: disable=import-error
				if K.backend() != backend:
					logger.warning('Keras was already imported by the time '
						'the Kur backend was instantiated. Kur was asked to '
						'use Keras %s backend, but Keras is already using %s. '
						'We cannot change the Keras backend at this point, so '
						'we will try to work with the currently loaded '
						'backend. In the future, try to let Kur manage '
						'importing Keras.', backend, K.backend())

			deps = {
				'theano' : ['theano'],
				'tensorflow' : ['tensorflow']
			}[backend]
			for dep in deps:
				if can_import(dep):
					continue
				if backend == 'tensorflow':
					logger.error('Your Kurfile is trying to use TensorFlow.')
					logger.error('However, we cannot find TensorFlow '
						'installed.')
					logger.error('At least it is easy to install!')
					logger.error('To install TensorFlow for CPU: pip install '
						'tensorflow')
					logger.error('To install TensorFlow for GPU: pip install '
						'tensorflow-gpu')
					logger.error('See our troubleshooting page for more '
						'information: %s', os.path.join(__homepage__,
						'troubleshooting.html'))
					raise ValueError('Need to install TensorFlow for this '
						'Kurfile to work.')
				else:
					logger.warning('The Keras backend was asked to use the %s '
						'backend, but %s does not appear to be installed. You '
						'will likely get an error about this soon.',
						backend, dep)

		else:
			logger.debug('No particular backend for Keras has been requested.')
			if can_import('theano') and can_import('tensorflow'):
				logger.trace('Using the system-default Keras backend.')
			elif can_import('theano'):
				backend = 'theano'
				logger.trace('Only the Theano backend for Keras is installed, '
					'so we will try to use it.')
			elif can_import('tensorflow'):
				backend = 'tensorflow'
				logger.trace('Only the TensorFlow backend for Keras is '
					'installed, so we will try to use it.')
			else:
				logger.warning('No supported Keras backend seems to be '
					'installed. You will probably get an error about this '
					'shortly.')

		# Make sure Keras is loaded.
		# Now, Keras always prints out a "Using {Theano|TensorFlow} backend."
		# statement that is frankly unbecoming. So we'll just gobble it up here.
		x = io.StringIO()
		with redirect_stderr(x):

			env = {
				'KERAS_BACKEND' : backend,
				'THEANO_FLAGS' : os.environ.get('THEANO_FLAGS')
			}

			def replace_theano_flag(key, value):
				""" Updates the Theano flag variable.
				"""
				if env['THEANO_FLAGS']:
					parts = [i for i in env['THEANO_FLAGS'].split(',') \
						if not i.startswith('{}='.format(key))]
				else:
					parts = []
				parts.append('{}={}'.format(key, value))
				env['THEANO_FLAGS'] = ','.join(parts)

			if optimizer is False:
				logger.trace('Disabling the Theano optimizer.')
				replace_theano_flag('optimizer', 'None')

			if theano_flags is not None:
				for k, v in theano_flags.items():
					logger.trace('Setting Theano flag %s = %s', k, v)
					replace_theano_flag(k, v)

			replace_theano_flag('force_device', 'true')
			if not self.devices:
				replace_theano_flag('device', 'cpu')
				env['CUDA_VISIBLE_DEVICES'] = '100'
				logger.info('Requesting CPU')
			else:
				replace_theano_flag('device', 'gpu')
				env['CUDA_VISIBLE_DEVICES'] = ','.join(
					str(x) for x in self.devices)
				logger.info('Requesting GPUs: %s', self.devices)

			# Supress the deluge of TensorFlow messages that we aren't
			# interested in.
			env['TF_CPP_MIN_LOG_LEVEL'] = '1'

			logger.trace('Overriding environmental variables: %s', env)
			EnvironmentalVariable(**env).push()

			import keras	# pylint: disable=import-error,unused-variable
			import keras.backend as K		# pylint: disable=import-error
			logger.info('Keras is loaded. The backend is: %s',
				K.backend())
			self.toolchain = K.backend()

		# And now we can set the dimension ordering.
		keras.backend.set_image_dim_ordering('tf')

		# The Keras `Wrapper` class accesses `Layer`'s `regularizers`
		# property (see `Wrapper.build()`), which triggers Keras' own
		# deprecation warning. Let's suppress this for now so that we don't
		# confuse our users.
		logging.getLogger('py.warnings').addFilter(
			type('theano_filter', (), {
				'filter' : lambda record: not (
					record.module == 'topology' and
					record.levelname == 'WARNING' and
					record.funcName == 'regularizers'
				)
			})
		)

		if self.parallel > 1 and self.get_toolchain() == 'theano':
			logger.warning('Multiple GPUs were requested, but are not '
				'supported with Keras\' Theano backend. Try the PyTorch '
				'backend or Keras\' TensorFlow backend instead. Falling back '
				'to a single device.')
			self.devices = self.devices[:1]

	###########################################################################
	def get_toolchain(self):
		""" Returns a string describing the Keras backend being used, either
			'theano' or 'tensorflow'.
		"""
		return self.toolchain

	###########################################################################
	@classmethod
	def get_name(cls):
		""" Returns the name of the backend class.

			This is used by the base class's factory method.
		"""
		return 'keras'

	###########################################################################
	def connect(self, inputs, target, data):
		""" Use the Keras functional API to connect to layers

			# Notes:

			- You will need input placeholders in place before doing this,
			  otherwise Keras's shape-checking will fail.
		"""
		if self.keras_version() == 1:
			if not isinstance(inputs, list):
				inputs = [inputs]
		else:
			if isinstance(inputs, (list, tuple)):
				if len(inputs) == 1:
					inputs = inputs[0]

		pool_2d = None
		if self.get_toolchain() == 'theano':
			import theano						# pylint: disable=import-error
			if theano.__version__ < '0.9':
				# We need to patch Theano
				from theano.tensor.signal import pool # pylint: disable=import-error
				original_pool = pool.pool_2d
				def pool_2d(input, ws=None, ignore_border=None, stride=None,
						pad=(0, 0), mode='max', ds=None, st=None,
						padding=None):
					return original_pool(
						input=input,
						ds=ds if ds is not None else ws,
						ignore_border=ignore_border,
						st=st if st is not None else stride,
						padding=padding if padding is not None else pad,
						mode=mode
					)

		logger.trace('Connecting: %s(%s)',
			target,
			', '.join(
				str(x) for x in (
					inputs if isinstance(inputs, (list, tuple)) else [inputs]
				)
			)
		)
		with warnings.catch_warnings():
			warnings.filterwarnings(
				'ignore',
				message='.*tensor.nnet.abstract_conv.conv2d.*',
				module='.*theano_backend.*'
			)
			if pool_2d is None:
				return target(inputs)
			else:
				from unittest.mock import patch
				with patch('theano.tensor.signal.pool.pool_2d', pool_2d):
					return target(inputs)

	###########################################################################
	@staticmethod
	def keras_version():
		""" Retrieves the Keras major version.
		"""
		from keras import __version__			# pylint: disable=import-error
		return int(__version__.split('.')[0])

	###########################################################################
	@staticmethod
	def make_model(inputs, outputs):
		""" Compiles a Keras model in a version-agnostic way.
		"""
		import keras.models as M				# pylint: disable=import-error
		if KerasBackend.keras_version() == 1:
			return M.Model(input=inputs, output=outputs)
		return M.Model(inputs=inputs, outputs=outputs)

	###########################################################################
	def save(self, model, filename):
		""" Saves the model weights to the given filename.
		"""
		keras_model = self.make_model(
			inputs=[node.value for node in model.inputs.values()],
			outputs=[node.value for node in model.outputs.values()]
		)

		self._save_keras(keras_model, filename)

	###########################################################################
	def _save_keras(self, keras_model, filename):
		""" Saves a native Keras model.
		"""
		logger.critical("(self, keras_model, filename): \n\nSave weights arrays in temporal files \n\n1. weights of layers can be found in spec.model.compiled['raw'].flattened_layers, \n2. save weights of each layer into separate files of temporal dir \n\n")

		path = os.path.expanduser(os.path.expandvars(filename))
		logger.info("The path for saving weights: %s \n\n", path)
		if os.path.exists(path):
			if not os.path.isdir(path):
				raise ValueError('Target weight exists, but it is not a '
					'directory. Kur expected a directory that it can work '
					'with. Please move or delete the existing path: {}'
					.format(path))

			for dirpath, _, filenames in os.walk(path):
				for this_file in filenames:
					if this_file.endswith('.kur'):
						os.unlink(os.path.join(dirpath, this_file))
		else:
			os.makedirs(path, exist_ok=True)

		# get layers from model.compiled['raw']
		layers = keras_model.flattened_layers \
			if hasattr(keras_model, 'flattened_layers') else keras_model.layers


		# look into each layer
		for layer in layers:
			# get layer name
			layer_name = layer.name
			# get layer weights
			symbolic_weights = layer.weights
			# get weight_names, weight_values of this layer
			weight_names, weight_values = \
				self._get_weight_names_and_values_from_symbolic(
					symbolic_weights
				)
			#
			for name, val in zip(weight_names, weight_values):
				name = name.replace('/', '_')
				# add temperal folder name with weight_names
				target = os.path.join(
					path,
					'{}+{}.kur'.format(layer_name, name)
				)
				# save weights arrays with file path in format of idx
				idx.save(target, val)
				logger.info("weights are saved inside idx file in the form of (name: values) see below: \n")
				print(target, ": array's shape", val.shape, "\n\n")


	###########################################################################
	def _get_weight_names_and_values_from_symbolic(self, symbolic_weights):
		import keras.backend as K				# pylint: disable=import-error
		weight_values = K.batch_get_value(symbolic_weights)
		weight_names = [
			(
				str(w.name) if hasattr(w, 'name') and w.name \
					else 'param_{}'.format(i)
			)
			for i, (w, val) in enumerate(
				zip(symbolic_weights, weight_values)
			)
		]
		return weight_names, weight_values

	###########################################################################
	def restore(self, model, filename):
		""" Load the model weights from the given filename.
		"""
		keras_model = self.make_model(
			inputs=[node.value for node in model.inputs.values()],
			outputs=[node.value for node in model.outputs.values()]
		)

		try:
			self._restore_keras(keras_model, filename)
		except:
			logger.exception('Failed to load previously-saved Keras model. '
				'Are you accidentally loading pre-existing weights from an '
				'incompatible model? Make sure that any weights on disk are '
				'actually associated with the model you are trying to load '
				'them into.')
			raise

	###########################################################################
	def _restore_keras(self, keras_model, filename):
		""" Loads a native Keras model.
		"""
		logger.critical("(self, keras_model, filename): \n\nRestore weights of layers from saved weights files or idx files\n\n1. get the path name ready; \n\n2. map layer_weight_name with layer objects, then map layer_weight_name with numpy array loaded from idx files;\n\n3. find appropriate layer_weight_names; \n\n4. save weights arrays back to theano.tensor.sharedvar.TensorSharedVariables \n\n")


		import keras.backend as K				# pylint: disable=import-error

		path = os.path.expanduser(os.path.expandvars(filename))
		if os.path.exists(path):
			if not os.path.isdir(path):
				raise ValueError('Target weight exists, but it is not a '
					'directory. Kur expected a directory that it can work '
					'with. Please move or delete the existing path: {}'
					.format(path))
		else:
			raise ValueError('Target weight directory does not exist: {}'
				.format(path))

		print("input: filename or path: \n{}\n\n".format(path))
		print("input: keras_model: \n{}\n\n".format(keras_model))

		# get the same temporal path ready and get all keras layers objects ready
		layers = keras_model.flattened_layers \
			if hasattr(keras_model, 'flattened_layers') else keras_model.layers

		# inside index:
		# Get a map from "layer name" to "layer instance" in the current model, like {'..dense.0': [<keras.layers.core.Dense object at 0x1495a1208>], 'images': [<keras.engine.topology.InputLayer object at 0x117f5e908>], ...}
		index = {}
		for layer in layers:
			if layer.name:
				index.setdefault(layer.name, []).append(layer)
		logger.warning("index (variable): dict of `layer_name: layer_instance`\n")
		pprint(index)
		print("\n\n")


		# inside tensors:
		# build upon index, map names to idx weights files, like
		# {'..dense.0': {'..dense.0_bias': '/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/tmpllhby5wa/weights/..dense.0+..dense.0_bias.kur', '..dense.0_kernel': '/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/tmpllhby5wa/weights/..dense.0+..dense.0_kernel.kur'}, '..convolution.1': {'..convolution.1_bias': '/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/tmpllhby5wa/weights/..convolution.1+..convolution.1_bias.kur', '..convolution.1_kernel': '/var/folders/gz/ch3n2mp51m9386sytqf97s6w0000gn/T/tmpllhby5wa/weights/..convolution.1+..convolution.1_kernel.kur'},...
		tensors = self.enumerate_saved_tensors(path)
		logger.warning("tensors (variable): dict 'weight_name: weights_kur_file' \n")
		pprint(tensors)
		print("\n\n")

		# We want to put (symbolic_weights, weight_values) tuples in this.
		weight_value_tuples = []

		# Loop over the available weights.
		for layer_name, weights in tensors.items():


			# layer by layer, for each layer,
			# Load the weights from saved idx files
			# This maps weight names to numpy arrays, like below
			# (Pdb) weights
			# {'..dense.0_bias': array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.], dtype=float32), '..dense.0_kernel': array([[ 0.00429164, -0.01248324, -0.0103093 , ...,  0.00017912,
			#         -0.00616769,  0.00401405],
			#        [ 0.01165415,  0.00312235, -0.00928085, ..., -0.00362225,
			#         -0.001397  , -0.00422575],
			#        [ 0.01059127,  0.00223762,  0.00044865, ..., -0.01010111,
			#          0.00584908,  0.00210532],
			#        ...,
			#        [-0.00725042, -0.00071572,  0.01256828, ...,  0.00209043,
			#          0.00866257,  0.00212232],
			#        [-0.00259544,  0.00826614, -0.00888701, ...,  0.00044817,
			#          0.0022627 , -0.00267635],
			#        [-0.0025061 ,  0.00577335,  0.01267747, ..., -0.00301542,
			#          0.00812706, -0.00881728]], dtype=float32)}
			weights = {k : idx.load(v) for k, v in weights.items()}

			# Now assign all of the weights to their corresponding symbolic
			# weights. Loop over all layers which use this name.
			for layer in index.get(layer_name, []):

				# Get the symbolic weights.
				symbolic_weights = layer.weights

				# Get the associated names (so we know what order to assign the
				# weights in.
				weight_names, _ = \
					self._get_weight_names_and_values_from_symbolic(
						symbolic_weights
					)

				# get weight names from saved weights
				available = set(weights.keys())
				needed = set(name.replace('/', '_') for name in weight_names)
				if available ^ needed:
					logger.error('Weight discrepancy in the weights we are '
						'supposed to load.')
					logger.error('These weights are on-disk, but not '
						'requested: %s', ', '.join(available - needed))
					logger.error('These weights were requested, but not '
						'available: %s', ', '.join(needed - available))
					raise ValueError('Layer "{}" expected {} weights, but we '
						'found {} on disk.'.format(layer_name,
						len(needed), len(available)))

				# tuple layer_weight_name with idx.weights under the same name
				# same all weights and names as tuples into a list
				for i, name in enumerate(weight_names):
					name = name.replace('/', '_')
					weight_value_tuples.append((symbolic_weights[i], weights[name]))

		logger.warning("weight_value_tuples (variable): 'weight_object : numpy.arrays' \n")
		print("the weight object: {}\n".format(type(weight_value_tuples[0][0])))
		pprint(weight_value_tuples)
		print("\n\n")

		logger.warning("before K.batch_set_value(weight_value_tuples), weight_value_tuples[0][0]'s inner value is the same to\n weight_value_tuples[0][0].container (array inside): '\n")
		pprint(weight_value_tuples[0][0].container)
		print("\n\npprint(weight_value_tuples[0][0].get_value() is the same to weight_value_tuples[0][0].container\n")
		pprint(weight_value_tuples[0][0].get_value())
		# Assign all the weights (arrays) to ..dense.0/kernel, which is theano.tensor.sharedvar.TensorSharedVariable
		K.batch_set_value(weight_value_tuples)
		logger.warning("\n\nAfter K.batch_set_value(weight_value_tuples), weight_value_tuples[0][0].get_value() has filled with new arrays: '\n")
		pprint(weight_value_tuples[0][0].get_value())
		print("\n\nHow about pprint(weight_value_tuples[0][0].container? yes, it changed as above\n")
		pprint(weight_value_tuples[0][0].container)

		logger.warning("\n\nNow, all the weights tensors have been restored into the following objects: \n")
		pprint([weight_tensor[0] for weight_tensor in weight_value_tuples])

		print("\n\n")


	###########################################################################
	def enumerate_saved_tensors(self, path):
		""" Enumerates saved tensors (weights).
		"""
		result = {}

		regex = re.compile(r'^(?P<layer>.*?)\+(?P<weight>.*?)\.kur$')
		for dirpath, dirnames, filenames in os.walk(path): # pylint: disable=unused-variable
			for filename in filenames:
				match = regex.match(filename)
				if match is None:
					continue
				filename = os.path.join(dirpath, filename)
				layer, weight = match.groups()
				if layer not in result:
					result[layer] = {}
				if weight in result[layer]:
					logger.warning('Tensor weights have already been loaded '
						'for layer="%s", tensor="%s" from file: %s. We will '
						'skip this new file we just found: %s.', layer, weight,
						result[layer][weight], filename)
					continue
				result[layer][weight] = filename

		return result

	###########################################################################
	@staticmethod
	def find_compiled_layer_by_name(model, layer_name):
		""" Returns the Keras tensor associated with a given name.

			# Arguments

			model: Model instance. The Kur model. It must be compiled.
			layer_name: str. The name of the layer to find, or one of its
				aliases.

			# Return value

			The Keras tensor
		"""
		if model.compiled is None or 'raw' not in model.compiled:
			raise ValueError('The model must be compiled first.')
		if layer_name in model.output_aliases:
			target = model.output_aliases[layer_name]
		elif layer_name in model.input_aliases:
			target = model.input_aliases[layer_name]
		else:
			raise ValueError('Failed to find a layer named "{}"'
				.format(layer_name))
		for keras_layer, kur_layer in \
			zip(model.compiled['raw'].outputs, model.outputs):
			if kur_layer == target:
				return keras_layer
		raise ValueError('Did not find expected layer. This is a bug.')

	###########################################################################
	def process_loss(self, model, loss):
		""" Process the loss functions.

			# Arguments

			model: The Kur model. It must be compiled.
			loss: Loss instance, list/tuple of Loss instances, or a dictionary
				of model layer names mapped to Loss instances.
		"""
		import keras.backend as K				# pylint: disable=import-error

		if loss is None:
			num_outputs = len(model.outputs)
			logger.error('You are trying to construct a training/validation'
				'/testing model, but you haven\'t specified any loss '
				'functions. Your model has %d outputs: %s. You need to '
				'specify %d loss functions, one for each output.',
				num_outputs, ', '.join(model.outputs), num_outputs)
			raise ValueError('No loss functions were specified, but are '
				'required for training, testing, and validation.')

		if isinstance(loss, Loss):
			loss = [loss]

		if len(loss) != len(model.outputs):
			raise ValueError('Model has {} outputs, but only {} loss '
				'functions were specified.'
				.format(len(model.outputs), len(loss)))

		if isinstance(loss, (list, tuple)):
			loss = dict(zip(model.outputs, loss))

		if not isinstance(loss, (dict, OrderedDict)):
			raise ValueError('Loss functions given to "compile" should be '
				'a list/tuple, a dictionary, or a single Loss instance. '
				'Instead we received this: {} (type={})'
				.format(loss, type(loss)))

		loss_inputs = OrderedDict()
		loss_outputs = OrderedDict()
		for target, this_loss in loss.items():
			ins, out = this_loss.get_loss(
				model,
				target,
				self.find_compiled_layer_by_name(model, target)
			)
			# FIXME: Re-using a network output in different loss functions will
			# probably break, since each loss function is creating its own
			# placeholder inputs, but then we are throwing some away using
			# 'update'.
			loss_inputs.update(ins)
			loss_outputs[target] = K.mean(out)
			logger.trace('Adding additional inputs: %s',
				', '.join(x[0] for x in ins))

		total_loss = functools.reduce(
			lambda x, y: x + y,
			loss_outputs.values()
		)
		return loss_inputs, loss_outputs, total_loss

	###########################################################################
	def compile(self, model, loss=None, optimizer=None, blocking=True,
		assemble_only=False):
		""" Returns the Keras model instance.
		"""
		logger.critical("(self, model, loss=None, optimizer=None, blocking=True,assemble_only=False): \nCreate a keras model instance \n\n1. set model.compiled from None to {}; \n2. If 'raw' is not a key to model.compiled, create a keras model object; \n3. print model with num of params nicely when -vv; \n4. if self.parallel>1, run make_parallel(...); \n5. import keras.backend as K, get loss_inputs, loss_outputs as 2 ordered dicts, total_loss as a mean value; \n6. get updates as list of tuples full of ops??? using args such as compiled.trainable_weights and total_loss; \n7. create a theano.compile.function_module.Func object; \n8. get input_shapes, input_names and output names; \n9. store func, input_names, output_names, input_shapes, optimizer into a dict named result; \n10. store result in model.compiled['train'] \n11. Compiling to build a model in Keras\n\n")

		# set model.compiled from None to {}
		if model.compiled is None:
			model.compiled = {}

		# If 'raw' is not a key to model.compiled,
		if 'raw' not in model.compiled:

			logger.info("Using model.inputs and model.outpus to create a Keras model and save it in model.compiled['raw'].\n\nThis is one big step toward a model with a specific backend, trainable and runnable \n\nLet's see model.compiled['raw'].__dict__")

			compiled = self.make_model(
				inputs=[node.value for node in model.inputs.values()],
				outputs=[node.value for node in model.outputs.values()]
			)

			# print model with num of params nicely when -vv
			# if logger.isEnabledFor(logging.INFO):
			# 	x = io.StringIO()
			# 	with contextlib.redirect_stdout(x):
			# 		compiled.summary()
			# 	for line in x.getvalue().split('\n'):
			# 		# logger.info("\n%s",line)
			# 		print(line, "\n")

			# if self.parallel>1, run make_parallel(...)
			if self.parallel > 1:
				from ..utils.parallelism import make_parallel
				compiled = make_parallel(compiled, self.parallel)

			model.compiled['raw'] = compiled

		else:
			logger.info('Reusing an existing model.')
			compiled = model.compiled['raw']

		logger.critical("\n\nlet's see model.compiled['raw'].__dict__\n")
		pprint(model.compiled['raw'].__dict__)
		print("\n\n")
		print("model.compiled['raw'].inputs[0].__dict__: == /images.__dict__ as below")
		pprint(model.compiled['raw'].inputs[0].__dict__)
		print('\n\n')
		print("model.compiled['raw'].input_layers[0].__dict__: ")
		pprint(model.compiled['raw'].input_layers[0].__dict__)
		print("\n\nmodel.compiled['raw'].input_layers[0].inbound_nodes[0].__dict__: ")
		pprint(model.compiled['raw'].input_layers[0].inbound_nodes[0].__dict__)
		print("\n\n")

		logger.info('See the summary of this compiled/keras model:\n')
		pprint(model.compiled['raw'].summary())
		print("\n\n")


		import keras.backend as K				# pylint: disable=import-error
		if loss is None and optimizer is None:
			logger.info('Assembling an evaluation function from the model.\n\n')

			loss_inputs = loss_outputs = {}
			if not assemble_only:
				func = K.function(
					compiled.inputs + \
						[K.learning_phase()],
					compiled.outputs
				)
			key = 'evaluate'

		elif optimizer is None:
			logger.info('Assembling a testing function from the model.\n\n')

			loss_inputs, loss_outputs, _ = \
				self.process_loss(model, loss)

			if not assemble_only:
				func = K.function(
					compiled.inputs + \
						list(loss_inputs.values()) + \
						[K.learning_phase()],
					compiled.outputs + \
						list(loss_outputs.values())
				)
			key = 'test'

		else:
			logger.critical("\n\nAssembling a training function from the model.\n\n1. get loss_inputs, loss_outputs, total_loss ready using `loss_inputs, loss_outputs, total_loss = self.process_loss(model, loss)`; \n2. get updates in terms of flow of operations using optimizer from Executor_trainer.optimizer and optimizer.get_optimizer(self)(compiled.trainable_weights, total_loss); \n3. use compiled.inputs, compiled.outputs, loss_inputs, loss_outputs, updates to create a specific backend(Theano, TF, Pytorch?) training function \n\n")

			# Loss inputs: additional inputs needed by the loss function.
			# Loss outputs: output of the loss function

			# get loss_inputs, loss_outputs as 2 ordered dicts, total_loss as a mean value
			loss_inputs, loss_outputs, total_loss = \
				self.process_loss(model, loss)
			print("loss_inputs: {}\n".format(loss_inputs))
			for k, v in loss_inputs.items():
				pprint(v.__dict__)
			print("\n\nloss_outputs: {} \n".format(loss_outputs))
			for k, v in loss_outputs.items():
				pprint(v.__dict__)
			print("\n\ntotal_loss: has interesting object inside {}\n".format(total_loss))
			pprint(total_loss.__dict__)
			print("\n\n")


			# get update or the specific optimzier for training
			updates = optimizer.get_optimizer(self)(
				compiled.trainable_weights, total_loss
			)

			print("updates from optimizer.get_optimizer(self)(compiled.trainable_weights, total_loss): \n")
			pprint(updates)

			print("\n\npprint(updates[0][0]) is interesting object: {}\n".format(updates[0][0]))
			pprint(updates[0][0].__dict__)
			print("\n\npprint(updates[0][1].__dict__) is interesting too: {}\n".format(updates[0][1]))
			pprint(updates[0][1].__dict__)
			print("\n\n")


			# create a theano.compile.function_module.Func object
			if not assemble_only:
				func = K.function(
					compiled.inputs + \
						list(loss_inputs.values()) + \
						[K.learning_phase()],
					compiled.outputs + \
						list(loss_outputs.values()),
					updates=updates
				)
				print("create a specific-backend func using keras: \nHowever, I can't pprint(getsourcelines(func)), is there a way to print out the source code of such a func?\n")
				pprint(func.__dict__)
				print("\n\n")
			key = 'train'


		# logger.trace('Additional inputs for log functions: %s\n\n',
		# 	', '.join(loss_inputs.keys()))

		logger.critical("\n\nCreate model.compiled['train'] to store input_names, output_names, input_shapes, optimizer (Executor_trainer.optimizer), backend-specific-func\n\n1. get input_names, output_names from spec.model.compiled['raw'].input_names|output_names and loss_inputs|outputs (from immediate above)\n2. get input_shapes from spec.model.compiled['raw']['inputs'] and loss_inputs.values() \n3. store all above, Executor.optimizer, and the specific-backend function into a dict called result \n4. and then store result inside model.compiled['train'] \n\n")
		# get input_names and output names
		input_names = compiled.input_names + \
			list(loss_inputs.keys())
		output_names = compiled.output_names + \
			list(loss_outputs.keys())


		# get input_shapes
		input_shapes = [
			layer._keras_shape
			# layer is a theano or backend-specific tensor
			for layer in compiled.inputs
		] + [
			layer._keras_shape
			for layer in loss_inputs.values()
		]
		print("create input_names out of compiled model: \n")
		pprint(input_names)
		print("\ncreate output_names out of compiled model: \n")
		pprint(output_names)
		print("\ncreate input_shapes out of compiled model: \n")
		pprint(input_shapes)
		print("\n\n")
		# logger.trace('Expected input shapes: %s /n/n',
			# ', '.join('{}={}'.format(k, v) for k, v in \
			# 	zip(input_names, input_shapes)
			# ))

		if assemble_only:
			func = None

		# store compiled model into a dict named result
		result = {
			'func' : func,
			'names' : {
				'input' : input_names,
				'output' : output_names
			},
			'shapes' : {
				'input' : input_shapes
			},
			'kur_optimizer' : optimizer
		}

		print("puts func, names, shapes, optimizer into a single dict: named result \n And save result inside model.compiled['train']: \n")
		pprint(result)
		print("\n\n")
		# if logger.isEnabledFor(logging.INFO):
		# 	logger.info('Compiled model: \n')
		# 	pprint(result)
		# 	print("\n\n")

		# store result into model.compiled[key]
		if not assemble_only:
			model.compiled[key] = result

			print("model.compiled[key]: \n")
			pprint(model.compiled[key])
			print("\n\n")

			if blocking:
				logger.info("\n\nWith spec.model.compiled['raw'] and spec.model.compiled[key] ready, Now use self.wait_for_compile(model, key) to save, test and restore the specific-backend model \n\n")
				self.wait_for_compile(model, key)


		return result

	###########################################################################
	def wait_for_compile(self, model, key):
		""" Waits for the model to finish compiling.
		"""
		logger.critical("(self, model, key): \n\nCompiling to build a model in Keras \n\n1. get provider ready \n2. get temporal dir ready; \n3. save weights arrays of layers into temporal files; \n4. test weights and model by using 2 sample data and weights to make predictions and calc loss; \n5. finall restore the weights to the model using self._restore_keras(model.compiled['raw'], weight_path) \n\n")

		if model.provider is None:
			logger.warning('No data provider available, so we cannot reliably '
				'wait for compiling to finish.')
			return

		# get batch provider for this section of data
		with DisableLogging():
			provider = BatchProvider(
				sources=dict(zip(model.provider.keys, model.provider.sources)),
				batch_size=2*max(1, self.parallel),
				num_batches=1,
				randomize=False
			)

		# add more data sources to the provider above
		model.supplement_provider(provider)

		logger.warning("get provider ready, if available add additional sources to provider \n")
		pprint(provider.__dict__)
		print("\n\n")

		# create an empty directory
		weight_path = None
		tempdir = tempfile.mkdtemp()
		try:
			# create a weights dir
			weight_path = os.path.join(tempdir, 'weights')

			# save weights of each layer in idx format in a temporal dir
			self._save_keras(model.compiled['raw'], weight_path)

			logger.info('Waiting for model to finish compiling.../n/n')

			# Internal StopIteration will allow only one loop
			# use 2 data samples to test out the working of model by produce predictions and a loss
			for batch in provider:
				self.run_batch(model, batch, key, False)
			logger.info('Model is tested by 2 data samples and ready for use.')

		finally:
			if weight_path and os.path.isdir(weight_path):
				try:

					# load a native keras model
					self._restore_keras(model.compiled['raw'], weight_path)
				except:
					logger.error('We were waiting for the model to finish '
						'compiling, but failed to restore the model weights. '
						'The weights may be in a bad state.')
					raise

			shutil.rmtree(tempdir, ignore_errors=True)

	###########################################################################
	def run_batch(self, model, batch, key, is_train):
		""" Test out the model with 2 data samples: make predictions and calc loss for them
		"""
		logger.critical("(self, model, batch, key, is_train): \n\nTest the model by predict and calc loss out of 2 sample data or a batch of data samples\n\n1. use spec.model.compiled[key]['shape']|['names'] to get 2 data samples for testing; \n2. use spec.model.compiled[key]['func'](inputs) to get 2 arrays \n3. array1: 2 arrays of predictions; \n4. array2: a single loss \n\n")
		print("input model: \n{}\n".format(model))
		print("input batch: \n{}\n".format(batch))
		print("input key: \n{}\n\n".format(key))

		if model.compiled is None or key not in model.compiled:
			raise ValueError('A model has not been compiled to: {}'
				.format(key))

		compiled = model.compiled[key]
		raw = model.compiled['raw']

		assert isinstance(is_train, bool)

		def coerce_shape(data, shape, name):
			if data.ndim < len(shape):
				return numpy.expand_dims(data, -1)
			else:
				return data

		# what inside this inputs?
		# list of 3 items(2 arrays, 1 boolean):
		# Array1: 2 samples of data source 1,
		# Array2: 2 samples of data source 2,
		# and false boolean
		inputs = [
			coerce_shape(
				batch[model.get_data_name_by_layer_name(batch, name)],
				shape, name
			)
			for shape, name in zip(
				compiled['shapes']['input'],
				compiled['names']['input']
			)
		] + [is_train]


		# spec.model.compiled[key]['func'](inputs) => a list of 2 arrays:
		# array1: predictions
		# array2: loss
		outputs = compiled['func'](inputs)

		num_outputs = len(raw.outputs)
		# metrics is probably the first batch's loss
		metrics = {
			k : v for k, v in zip(
				compiled['names']['output'][num_outputs:],
				outputs[num_outputs:]
			)
		}
		# this is 2 data samples predictions
		predictions = {name : data for name, data in zip(model.outputs, outputs[:num_outputs])}
		return predictions, metrics

	###########################################################################
	def train(self, model, data):
		""" Fits the given model on a batch of data.
		"""

		# get optimizer
		kur_optimizer = model.compiled['train']['kur_optimizer']

		if kur_optimizer.scale_rate:
			if kur_optimizer.scale_rate in data:
				import keras.backend as K		# pylint: disable=import-error
				factor = data[kur_optimizer.scale_rate].mean()
				keras_optimizer = kur_optimizer.optimizer
				K.set_value(
					keras_optimizer.lr,
					K.get_value(keras_optimizer.lr) * factor
				)
				result = self.run_batch(model, data, 'train', True)
				K.set_value(
					keras_optimizer.lr,
					K.get_value(keras_optimizer.lr) / factor
				)
				return result
			else:
				logger.warning('The optimizer "scale_rate" was specified, but '
					'no such data column was found: %s. Ignoring this.',
					kur_optimizer.scale_rate)
		return self.run_batch(model, data, 'train', True)

	###########################################################################
	def test(self, model, data):
		""" Calculates the model loss on a batch of data.
		"""
		return self.run_batch(model, data, 'test', False)

	###########################################################################
	def evaluate(self, model, data):
		""" Evaluates the model on a batch of data.
		"""
		return self.run_batch(model, data, 'evaluate', False)

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
