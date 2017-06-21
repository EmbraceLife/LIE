
## __init__.py
class __init___py:
	"""The Keras API.

		with the following codes, we can access for example `activations` directly from `tensorflow.contrib.keras.activations` rather than from `tensorflow.contrib.keras.python.keras.activations`
	"""
	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function

	from tensorflow.contrib.keras.python.keras import activations
	from tensorflow.contrib.keras.python.keras import applications
	from tensorflow.contrib.keras.python.keras import backend
	from tensorflow.contrib.keras.python.keras import callbacks
	from tensorflow.contrib.keras.python.keras import constraints
	from tensorflow.contrib.keras.python.keras import datasets
	from tensorflow.contrib.keras.python.keras import engine
	from tensorflow.contrib.keras.python.keras import initializers
	from tensorflow.contrib.keras.python.keras import layers
	from tensorflow.contrib.keras.python.keras import losses
	from tensorflow.contrib.keras.python.keras import metrics
	from tensorflow.contrib.keras.python.keras import models
	from tensorflow.contrib.keras.python.keras import optimizers
	from tensorflow.contrib.keras.python.keras import preprocessing
	from tensorflow.contrib.keras.python.keras import regularizers
	from tensorflow.contrib.keras.python.keras import utils
	from tensorflow.contrib.keras.python.keras import wrappers
	from tensorflow.contrib.keras.python.keras.layers import Input

	__version__ = '2.0.4-tf'

## activations.py
class activations_py:
	"""Keras built-in activation functions.
	"""
	def import_libs():
		# so that python 2.7 and python 3 can both use the following functions
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import six

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras.engine import Layer
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
		from tensorflow.python.platform import tf_logging as logging


	def softmax(x, axis=-1):
	  """Softmax activation function.

	  Arguments:
	      x : Tensor.
	      axis: Integer, axis along which the softmax normalization is applied.

	  Returns:
	      Tensor, output of softmax transformation.

	  Raises:
	      ValueError: In case `dim(x) == 1`.
	  """
	  ndim = K.ndim(x)
	  if ndim == 2:
	    return K.softmax(x)
	  elif ndim > 2:
	    e = K.exp(x - K.max(x, axis=axis, keepdims=True))
	    s = K.sum(e, axis=axis, keepdims=True)
	    return e / s
	  else:
	    raise ValueError('Cannot apply softmax to a tensor that is 1D')


	def elu(x, alpha=1.0):
	  return K.elu(x, alpha)


	def softplus(x):
	  return K.softplus(x)


	def softsign(x):
	  return K.softsign(x)


	def relu(x, alpha=0., max_value=None):
	  return K.relu(x, alpha=alpha, max_value=max_value)


	def tanh(x):
	  return K.tanh(x)


	def sigmoid(x):
	  return K.sigmoid(x)


	def hard_sigmoid(x):
	  return K.hard_sigmoid(x)


	def linear(x):
	  return x


	def serialize(activation):
	  return activation.__name__


	def deserialize(name, custom_objects=None):
	  return deserialize_keras_object(
	      name,
	      module_objects=globals(),
	      custom_objects=custom_objects,
	      printable_module_name='activation function')


	def get(identifier):
	  if identifier is None:
	    return linear
	  if isinstance(identifier, six.string_types):
	    identifier = str(identifier)
	    return deserialize(identifier)
	  elif callable(identifier):
	    if isinstance(identifier, Layer):
	      logging.warning(
	          'Do not pass a layer instance (such as {identifier}) as the '
	          'activation argument of another layer. Instead, advanced '
	          'activation layers should be used just like any other '
	          'layer in a model.'.format(identifier=identifier.__class__.__name__))
	    return identifier
	  else:
	    raise ValueError('Could not interpret '
	                     'activation function identifier:', identifier)

## backend.py
class backend_py:

	"""Keras backend API.
	"""

	def import_libs():
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import json
		import os

		import numpy as np

		from tensorflow.core.protobuf import config_pb2
		from tensorflow.python.client import session as session_module
		from tensorflow.python.framework import constant_op
		from tensorflow.python.framework import dtypes as dtypes_module
		from tensorflow.python.framework import ops
		from tensorflow.python.framework import sparse_tensor
		from tensorflow.python.layers import base as tf_base_layers
		from tensorflow.python.ops import array_ops
		from tensorflow.python.ops import clip_ops
		from tensorflow.python.ops import control_flow_ops
		from tensorflow.python.ops import ctc_ops as ctc
		from tensorflow.python.ops import functional_ops
		from tensorflow.python.ops import gradients as gradients_module
		from tensorflow.python.ops import image_ops
		from tensorflow.python.ops import init_ops
		from tensorflow.python.ops import logging_ops
		from tensorflow.python.ops import math_ops
		from tensorflow.python.ops import nn
		from tensorflow.python.ops import random_ops
		from tensorflow.python.ops import sparse_ops
		from tensorflow.python.ops import state_ops
		from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
		from tensorflow.python.ops import tensor_array_ops
		from tensorflow.python.ops import variables as variables_module
		from tensorflow.python.training import moving_averages
		from tensorflow.python.util import tf_inspect

	def hyper_parameters():
		py_all = all
		py_sum = sum

		# INTERNAL UTILS

		# This is the default internal TF session used by Keras.
		# It can be set manually via `set_session(sess)`.
		_SESSION = None

		# This dictionary holds a mapping {graph: learning_phase}.
		# A learning phase is a bool tensor used to run Keras models in
		# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
		_GRAPH_LEARNING_PHASES = {}

		# This dictionary holds a mapping {graph: UID_DICT}.
		# each UID_DICT is a dictionary mapping name prefixes to a current index,
		# used for generatic graph-specific string UIDs
		# for various names (e.g. layer names).
		_GRAPH_UID_DICTS = {}

		# This boolean flag can be set to True to leave variable initialization
		# up to the user.
		# Change its value via `manual_variable_initialization(value)`.
		_MANUAL_VAR_INIT = False

		# The type of float to use throughout a session.
		_FLOATX = 'float32'

		# Epsilon fuzz factor used throughout the codebase.
		_EPSILON = 10e-8

		# Default image data format, one of "channels_last", "channels_first".
		_IMAGE_DATA_FORMAT = 'channels_last'


	def backend():
	  """Publicly accessible method for determining the current backend.

	  Only exists for API compatibility with multi-backend Keras.

	  Returns:
	      The string "tensorflow".
	  """
	  return 'tensorflow'


	def epsilon():
	  """Returns the value of the fuzz factor used in numeric expressions.

	  Returns:
	      A float.

	  Example:
	  ```python
	      >>> keras.backend.epsilon()
	      1e-08
	  ```
	  """
	  return _EPSILON


	def set_epsilon(value):
	  """Sets the value of the fuzz factor used in numeric expressions.

	  Arguments:
	      value: float. New value of epsilon.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> K.epsilon()
	      1e-08
	      >>> K.set_epsilon(1e-05)
	      >>> K.epsilon()
	      1e-05
	  ```
	  """
	  global _EPSILON
	  _EPSILON = value


	def floatx():
	  """Returns the default float type, as a string.

	  E.g. 'float16', 'float32', 'float64'.

	  Returns:
	      String, the current default float type.

	  Example:
	  ```python
	      >>> keras.backend.floatx()
	      'float32'
	  ```
	  """
	  return _FLOATX


	def set_floatx(value):
	  """Sets the default float type.

	  Arguments:
	      value: String; 'float16', 'float32', or 'float64'.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> K.floatx()
	      'float32'
	      >>> K.set_floatx('float16')
	      >>> K.floatx()
	      'float16'
	  ```

	  Raises:
	      ValueError: In case of invalid value.
	  """
	  global _FLOATX
	  if value not in {'float16', 'float32', 'float64'}:
	    raise ValueError('Unknown floatx type: ' + str(value))
	  _FLOATX = str(value)


	def cast_to_floatx(x):
	  """Cast a Numpy array to the default Keras float type.

	  Arguments:
	      x: Numpy array.

	  Returns:
	      The same Numpy array, cast to its new type.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> K.floatx()
	      'float32'
	      >>> arr = numpy.array([1.0, 2.0], dtype='float64')
	      >>> arr.dtype
	      dtype('float64')
	      >>> new_arr = K.cast_to_floatx(arr)
	      >>> new_arr
	      array([ 1.,  2.], dtype=float32)
	      >>> new_arr.dtype
	      dtype('float32')
	  ```
	  """
	  return np.asarray(x, dtype=_FLOATX)


	def image_data_format():
	  """Returns the default image data format convention.

	  Returns:
	      A string, either `'channels_first'` or `'channels_last'`

	  Example:
	  ```python
	      >>> keras.backend.image_data_format()
	      'channels_first'
	  ```
	  """
	  return _IMAGE_DATA_FORMAT


	def set_image_data_format(data_format):
	  """Sets the value of the image data format convention.

	  Arguments:
	      data_format: string. `'channels_first'` or `'channels_last'`.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> K.image_data_format()
	      'channels_first'
	      >>> K.set_image_data_format('channels_last')
	      >>> K.image_data_format()
	      'channels_last'
	  ```

	  Raises:
	      ValueError: In case of invalid `data_format` value.
	  """
	  global _IMAGE_DATA_FORMAT
	  if data_format not in {'channels_last', 'channels_first'}:
	    raise ValueError('Unknown data_format:', data_format)
	  _IMAGE_DATA_FORMAT = str(data_format)


	def get_uid(prefix=''):
	  """Associates a string prefix with an integer counter in a TensorFlow graph.

	  Arguments:
	    prefix: String prefix to index.

	  Returns:
	    Unique integer ID.

	  Example:

	  ```
	    >>> get_uid('dense')
	    1
	    >>> get_uid('dense')
	    2
	  ```
	  """
	  graph = ops.get_default_graph()
	  layer_name_uids = tf_base_layers.PER_GRAPH_LAYER_NAME_UIDS[graph]
	  layer_name_uids[prefix] += 1
	  return layer_name_uids[prefix]


	def reset_uids():
	  layer_name_uids_collection = ops.get_collection_ref('LAYER_NAME_UIDS')
	  if layer_name_uids_collection:
	    layer_name_uids_collection.pop()


	def clear_session():
	  """Destroys the current TF graph and creates a new one.

	  Useful to avoid clutter from old models / layers.
	  """
	  global _SESSION
	  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
	  ops.reset_default_graph()
	  reset_uids()
	  _SESSION = None
	  phase = array_ops.placeholder(dtype='bool', name='keras_learning_phase')
	  _GRAPH_LEARNING_PHASES = {}
	  _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = phase


	def manual_variable_initialization(value):
	  """Sets the manual variable initialization flag.

	  This boolean flag determines whether
	  variables should be initialized
	  as they are instantiated (default), or if
	  the user should handle the initialization
	  (e.g. via `tf.initialize_all_variables()`).

	  Arguments:
	      value: Python boolean.
	  """
	  global _MANUAL_VAR_INIT
	  _MANUAL_VAR_INIT = value


	def learning_phase():
	  """Returns the learning phase flag.

	  The learning phase flag is a bool tensor (0 = test, 1 = train)
	  to be passed as input to any Keras function
	  that uses a different behavior at train time and test time.

	  Returns:
	      Learning phase (scalar integer tensor or Python integer).
	  """
	  graph = ops.get_default_graph()
	  if graph not in _GRAPH_LEARNING_PHASES:
	    phase = array_ops.placeholder(dtype='bool', name='keras_learning_phase')
	    _GRAPH_LEARNING_PHASES[graph] = phase
	  return _GRAPH_LEARNING_PHASES[graph]


	def set_learning_phase(value):
	  """Sets the learning phase to a fixed value.

	  Arguments:
	      value: Learning phase value, either 0 or 1 (integers).

	  Raises:
	      ValueError: if `value` is neither `0` nor `1`.
	  """
	  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
	  if value not in {0, 1}:
	    raise ValueError('Expected learning phase to be ' '0 or 1.')
	  _GRAPH_LEARNING_PHASES[ops.get_default_graph()] = value


	def get_session():
	  """Returns the TF session to be used by the backend.

	  If a default TensorFlow session is available, we will return it.

	  Else, we will return the global Keras session.

	  If no global Keras session exists at this point:
	  we will create a new global session.

	  Note that you can manually set the global session
	  via `K.set_session(sess)`.

	  Returns:
	      A TensorFlow session.
	  """
	  global _SESSION
	  if ops.get_default_session() is not None:
	    session = ops.get_default_session()
	  else:
	    if _SESSION is None:
	      if not os.environ.get('OMP_NUM_THREADS'):
	        config = config_pb2.ConfigProto(allow_soft_placement=True)
	      else:
	        num_thread = int(os.environ.get('OMP_NUM_THREADS'))
	        config = config_pb2.ConfigProto(
	            intra_op_parallelism_threads=num_thread, allow_soft_placement=True)
	      _SESSION = session_module.Session(config=config)
	    session = _SESSION
	  if not _MANUAL_VAR_INIT:
	    with session.graph.as_default():
	      _initialize_variables()
	  return session


	def set_session(session):
	  """Sets the global TensorFlow session.

	  Arguments:
	      session: A TF Session.
	  """
	  global _SESSION
	  _SESSION = session


	# VARIABLE MANIPULATION


	def _convert_string_dtype(dtype):
	  """Get the type from a string.

	  Arguments:
	      dtype: A string representation of a type.

	  Returns:
	      The type requested.

	  Raises:
	      ValueError: if `dtype` is not supported.
	  """
	  if dtype == 'float16':
	    return dtypes_module.float16
	  if dtype == 'float32':
	    return dtypes_module.float32
	  elif dtype == 'float64':
	    return dtypes_module.float64
	  elif dtype == 'int16':
	    return dtypes_module.int16
	  elif dtype == 'int32':
	    return dtypes_module.int32
	  elif dtype == 'int64':
	    return dtypes_module.int64
	  elif dtype == 'uint8':
	    return dtypes_module.int8
	  elif dtype == 'uint16':
	    return dtypes_module.uint16
	  else:
	    raise ValueError('Unsupported dtype:', dtype)


	def _to_tensor(x, dtype):
	  """Convert the input `x` to a tensor of type `dtype`.

	  Arguments:
	      x: An object to be converted (numpy array, list, tensors).
	      dtype: The destination type.

	  Returns:
	      A tensor.
	  """
	  x = ops.convert_to_tensor(x)
	  if x.dtype != dtype:
	    x = math_ops.cast(x, dtype)
	  return x


	def is_sparse(tensor):
	  """Returns whether a tensor is a sparse tensor.

	  Arguments:
	      tensor: A tensor instance.

	  Returns:
	      A boolean.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> a = K.placeholder((2, 2), sparse=False)
	      >>> print(K.is_sparse(a))
	      False
	      >>> b = K.placeholder((2, 2), sparse=True)
	      >>> print(K.is_sparse(b))
	      True
	  ```
	  """
	  return isinstance(tensor, sparse_tensor.SparseTensor)


	def to_dense(tensor):
	  """Converts a sparse tensor into a dense tensor and returns it.

	  Arguments:
	      tensor: A tensor instance (potentially sparse).

	  Returns:
	      A dense tensor.

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> b = K.placeholder((2, 2), sparse=True)
	      >>> print(K.is_sparse(b))
	      True
	      >>> c = K.to_dense(b)
	      >>> print(K.is_sparse(c))
	      False
	  ```
	  """
	  if is_sparse(tensor):
	    return sparse_ops.sparse_tensor_to_dense(tensor)
	  else:
	    return tensor


	name_scope = ops.name_scope


	def variable(value, dtype=None, name=None):
	  """Instantiates a variable and returns it.

	  Arguments:
	      value: Numpy array, initial value of the tensor.
	      dtype: Tensor type.
	      name: Optional name string for the tensor.

	  Returns:
	      A variable instance (with Keras metadata included).

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> val = np.array([[1, 2], [3, 4]])
	      >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
	      >>> K.dtype(kvar)
	      'float64'
	      >>> print(kvar)
	      example_var
	      >>> kvar.eval()
	      array([[ 1.,  2.],
	             [ 3.,  4.]])
	  ```
	  """
	  if dtype is None:
	    dtype = floatx()
	  if hasattr(value, 'tocoo'):
	    sparse_coo = value.tocoo()
	    indices = np.concatenate((np.expand_dims(sparse_coo.row, 1), np.expand_dims(
	        sparse_coo.col, 1)), 1)
	    v = sparse_tensor.SparseTensor(
	        indices=indices, values=sparse_coo.data, dense_shape=sparse_coo.shape)
	    v._uses_learning_phase = False
	    return v
	  v = variables_module.Variable(
	      value, dtype=_convert_string_dtype(dtype), name=name)
	  v._uses_learning_phase = False
	  return v


	def _initialize_variables():
	  """Utility to initialize uninitialized variables on the fly.
	  """
	  variables = variables_module.global_variables()
	  uninitialized_variables = []
	  for v in variables:
	    if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
	      uninitialized_variables.append(v)
	      v._keras_initialized = True
	  if uninitialized_variables:
	    sess = get_session()
	    sess.run(variables_module.variables_initializer(uninitialized_variables))


	def constant(value, dtype=None, shape=None, name=None):
	  """Creates a constant tensor.

	  Arguments:
	      value: A constant value (or list)
	      dtype: The type of the elements of the resulting tensor.
	      shape: Optional dimensions of resulting tensor.
	      name: Optional name for the tensor.

	  Returns:
	      A Constant Tensor.
	  """
	  if dtype is None:
	    dtype = floatx()
	  return constant_op.constant(value, dtype=dtype, shape=shape, name=name)


	def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
	  """Instantiates a placeholder tensor and returns it.

	  Arguments:
	      shape: Shape of the placeholder
	          (integer tuple, may include `None` entries).
	      ndim: Number of axes of the tensor.
	          At least one of {`shape`, `ndim`} must be specified.
	          If both are specified, `shape` is used.
	      dtype: Placeholder type.
	      sparse: Boolean, whether the placeholder should have a sparse type.
	      name: Optional name string for the placeholder.

	  Returns:
	      Tensor instance (with Keras metadata included).

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> input_ph = K.placeholder(shape=(2, 4, 5))
	      >>> input_ph
	      <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
	  ```
	  """
	  if dtype is None:
	    dtype = floatx()
	  if not shape:
	    if ndim:
	      shape = tuple([None for _ in range(ndim)])
	  if sparse:
	    x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
	  else:
	    x = array_ops.placeholder(dtype, shape=shape, name=name)
	  x._uses_learning_phase = False
	  return x


	def shape(x):
	  """Returns the symbolic shape of a tensor or variable.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A symbolic shape (which is itself a tensor).

	  Examples:
	  ```
	      # TensorFlow example
	      >>> from keras import backend as K
	      >>> tf_session = K.get_session()
	      >>> val = np.array([[1, 2], [3, 4]])
	      >>> kvar = K.variable(value=val)
	      >>> input = keras.backend.placeholder(shape=(2, 4, 5))
	      >>> K.shape(kvar)
	      <tf.Tensor 'Shape_8:0' shape=(2,) dtype=int32>
	      >>> K.shape(input)
	      <tf.Tensor 'Shape_9:0' shape=(3,) dtype=int32>
	      # To get integer shape (Instead, you can use K.int_shape(x))
	      >>> K.shape(kvar).eval(session=tf_session)
	      array([2, 2], dtype=int32)
	      >>> K.shape(input).eval(session=tf_session)
	      array([2, 4, 5], dtype=int32)
	  ```
	  """
	  return array_ops.shape(x)


	def int_shape(x):
	  """Returns the shape tensor or variable as a tuple of int or None entries.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tuple of integers (or None entries).

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> input = K.placeholder(shape=(2, 4, 5))
	      >>> K.int_shape(input)
	      (2, 4, 5)
	      >>> val = np.array([[1, 2], [3, 4]])
	      >>> kvar = K.variable(value=val)
	      >>> K.int_shape(kvar)
	      (2, 2)
	  ```
	  """
	  shape = x.get_shape()
	  try:
	    return tuple([i.__int__() for i in shape])
	  except ValueError:
	    return None


	def ndim(x):
	  """Returns the number of axes in a tensor, as an integer.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      Integer (scalar), number of axes.

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> input = K.placeholder(shape=(2, 4, 5))
	      >>> val = np.array([[1, 2], [3, 4]])
	      >>> kvar = K.variable(value=val)
	      >>> K.ndim(input)
	      3
	      >>> K.ndim(kvar)
	      2
	  ```
	  """
	  dims = x.get_shape()._dims
	  if dims is not None:
	    return len(dims)
	  return None


	def dtype(x):
	  """Returns the dtype of a Keras tensor or variable, as a string.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      String, dtype of `x`.

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> K.dtype(K.placeholder(shape=(2,4,5)))
	      'float32'
	      >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
	      'float32'
	      >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
	      'float64'
	      # Keras variable
	      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
	      >>> K.dtype(kvar)
	      'float32_ref'
	      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
	      >>> K.dtype(kvar)
	      'float32_ref'
	  ```
	  """
	  return x.dtype.name


	def eval(x):
	  """Evaluates the value of a variable.

	  Arguments:
	      x: A variable.

	  Returns:
	      A Numpy array.

	  Examples:
	  ```python
	      >>> from keras import backend as K
	      >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
	      >>> K.eval(kvar)
	      array([[ 1.,  2.],
	             [ 3.,  4.]], dtype=float32)
	  ```
	  """
	  return to_dense(x).eval(session=get_session())


	def zeros(shape, dtype=None, name=None):
	  """Instantiates an all-zeros variable and returns it.

	  Arguments:
	      shape: Tuple of integers, shape of returned Keras variable
	      dtype: String, data type of returned Keras variable
	      name: String, name of returned Keras variable

	  Returns:
	      A variable (including Keras metadata), filled with `0.0`.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> kvar = K.zeros((3,4))
	      >>> K.eval(kvar)
	      array([[ 0.,  0.,  0.,  0.],
	             [ 0.,  0.,  0.,  0.],
	             [ 0.,  0.,  0.,  0.]], dtype=float32)
	  ```
	  """
	  if dtype is None:
	    dtype = floatx()
	  shape = tuple(map(int, shape))
	  tf_dtype = _convert_string_dtype(dtype)
	  return variable(
	      init_ops.constant_initializer(0., dtype=tf_dtype)(shape), dtype, name)


	def ones(shape, dtype=None, name=None):
	  """Instantiates an all-ones tensor variable and returns it.

	  Arguments:
	      shape: Tuple of integers, shape of returned Keras variable.
	      dtype: String, data type of returned Keras variable.
	      name: String, name of returned Keras variable.

	  Returns:
	      A Keras variable, filled with `1.0`.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> kvar = K.ones((3,4))
	      >>> K.eval(kvar)
	      array([[ 1.,  1.,  1.,  1.],
	             [ 1.,  1.,  1.,  1.],
	             [ 1.,  1.,  1.,  1.]], dtype=float32)
	  ```
	  """
	  if dtype is None:
	    dtype = floatx()
	  shape = tuple(map(int, shape))
	  tf_dtype = _convert_string_dtype(dtype)
	  return variable(
	      init_ops.constant_initializer(1., dtype=tf_dtype)(shape), dtype, name)


	def eye(size, dtype=None, name=None):
	  """Instantiate an identity matrix and returns it.

	  Arguments:
	      size: Integer, number of rows/columns.
	      dtype: String, data type of returned Keras variable.
	      name: String, name of returned Keras variable.

	  Returns:
	      A Keras variable, an identity matrix.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> kvar = K.eye(3)
	      >>> K.eval(kvar)
	      array([[ 1.,  0.,  0.],
	             [ 0.,  1.,  0.],
	             [ 0.,  0.,  1.]], dtype=float32)
	  ```

	  """
	  return variable(np.eye(size), dtype, name)


	def zeros_like(x, dtype=None, name=None):
	  """Instantiates an all-zeros variable of the same shape as another tensor.

	  Arguments:
	      x: Keras variable or Keras tensor.
	      dtype: String, dtype of returned Keras variable.
	           None uses the dtype of x.
	      name: String, name for the variable to create.

	  Returns:
	      A Keras variable with the shape of x filled with zeros.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> kvar = K.variable(np.random.random((2,3)))
	      >>> kvar_zeros = K.zeros_like(kvar)
	      >>> K.eval(kvar_zeros)
	      array([[ 0.,  0.,  0.],
	             [ 0.,  0.,  0.]], dtype=float32)
	  ```
	  """
	  return array_ops.zeros_like(x, dtype=dtype, name=name)


	def ones_like(x, dtype=None, name=None):
	  """Instantiates an all-ones variable of the same shape as another tensor.

	  Arguments:
	      x: Keras variable or tensor.
	      dtype: String, dtype of returned Keras variable.
	           None uses the dtype of x.
	      name: String, name for the variable to create.

	  Returns:
	      A Keras variable with the shape of x filled with ones.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> kvar = K.variable(np.random.random((2,3)))
	      >>> kvar_ones = K.ones_like(kvar)
	      >>> K.eval(kvar_ones)
	      array([[ 1.,  1.,  1.],
	             [ 1.,  1.,  1.]], dtype=float32)
	  ```
	  """
	  return array_ops.ones_like(x, dtype=dtype, name=name)


	def identity(x):
	  """Returns a tensor with the same content as the input tensor.

	  Arguments:
	      x: The input tensor.

	  Returns:
	      A tensor of the same shape, type and content.
	  """
	  return array_ops.identity(x)


	def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
	  """Instantiates a variable with values drawn from a uniform distribution.

	  Arguments:
	      shape: Tuple of integers, shape of returned Keras variable.
	      low: Float, lower boundary of the output interval.
	      high: Float, upper boundary of the output interval.
	      dtype: String, dtype of returned Keras variable.
	      name: String, name of returned Keras variable.
	      seed: Integer, random seed.

	  Returns:
	      A Keras variable, filled with drawn samples.

	  Example:
	  ```python
	      # TensorFlow example
	      >>> kvar = K.random_uniform_variable((2,3), 0, 1)
	      >>> kvar
	      <tensorflow.python.ops.variables.Variable object at 0x10ab40b10>
	      >>> K.eval(kvar)
	      array([[ 0.10940075,  0.10047495,  0.476143  ],
	             [ 0.66137183,  0.00869417,  0.89220798]], dtype=float32)
	  ```
	  """
	  if dtype is None:
	    dtype = floatx()
	  shape = tuple(map(int, shape))
	  tf_dtype = _convert_string_dtype(dtype)
	  if seed is None:
	    # ensure that randomness is conditioned by the Numpy RNG
	    seed = np.random.randint(10e8)
	  value = init_ops.random_uniform_initializer(
	      low, high, dtype=tf_dtype, seed=seed)(shape)
	  return variable(value, dtype=dtype, name=name)


	def random_normal_variable(shape, mean, scale, dtype=None, name=None,
	                           seed=None):
	  """Instantiates a variable with values drawn from a normal distribution.

	  Arguments:
	      shape: Tuple of integers, shape of returned Keras variable.
	      mean: Float, mean of the normal distribution.
	      scale: Float, standard deviation of the normal distribution.
	      dtype: String, dtype of returned Keras variable.
	      name: String, name of returned Keras variable.
	      seed: Integer, random seed.

	  Returns:
	      A Keras variable, filled with drawn samples.

	  Example:
	  ```python
	      # TensorFlow example
	      >>> kvar = K.random_normal_variable((2,3), 0, 1)
	      >>> kvar
	      <tensorflow.python.ops.variables.Variable object at 0x10ab12dd0>
	      >>> K.eval(kvar)
	      array([[ 1.19591331,  0.68685907, -0.63814116],
	             [ 0.92629528,  0.28055015,  1.70484698]], dtype=float32)
	  ```
	  """
	  if dtype is None:
	    dtype = floatx()
	  shape = tuple(map(int, shape))
	  tf_dtype = _convert_string_dtype(dtype)
	  if seed is None:
	    # ensure that randomness is conditioned by the Numpy RNG
	    seed = np.random.randint(10e8)
	  value = init_ops.random_normal_initializer(
	      mean, scale, dtype=tf_dtype, seed=seed)(shape)
	  return variable(value, dtype=dtype, name=name)


	def count_params(x):
	  """Returns the number of scalars in a Keras variable.

	  Arguments:
	      x: Keras variable.

	  Returns:
	      Integer, the number of scalars in `x`.

	  Example:
	  ```python
	      >>> kvar = K.zeros((2,3))
	      >>> K.count_params(kvar)
	      6
	      >>> K.eval(kvar)
	      array([[ 0.,  0.,  0.],
	             [ 0.,  0.,  0.]], dtype=float32)
	  ```
	  """
	  shape = x.get_shape()
	  return np.prod([shape[i]._value for i in range(len(shape))])


	def cast(x, dtype):
	  """Casts a tensor to a different dtype and returns it.

	  You can cast a Keras variable but it still returns a Keras tensor.

	  Arguments:
	      x: Keras tensor (or variable).
	      dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

	  Returns:
	      Keras tensor with dtype `dtype`.

	  Example:
	  ```python
	      >>> from keras import backend as K
	      >>> input = K.placeholder((2, 3), dtype='float32')
	      >>> input
	      <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
	      # It doesn't work in-place as below.
	      >>> K.cast(input, dtype='float16')
	      <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
	      >>> input
	      <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
	      # you need to assign it.
	      >>> input = K.cast(input, dtype='float16')
	      >>> input
	      <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
	  ```
	  """
	  return math_ops.cast(x, dtype)


	# UPDATES OPS


	def update(x, new_x):
	  return state_ops.assign(x, new_x)


	def update_add(x, increment):
	  """Update the value of `x` by adding `increment`.

	  Arguments:
	      x: A Variable.
	      increment: A tensor of same shape as `x`.

	  Returns:
	      The variable `x` updated.
	  """
	  return state_ops.assign_add(x, increment)


	def update_sub(x, decrement):
	  """Update the value of `x` by subtracting `decrement`.

	  Arguments:
	      x: A Variable.
	      decrement: A tensor of same shape as `x`.

	  Returns:
	      The variable `x` updated.
	  """
	  return state_ops.assign_sub(x, decrement)


	def moving_average_update(x, value, momentum):
	  """Compute the moving average of a variable.

	  Arguments:
	      x: A Variable.
	      value: A tensor with the same shape as `variable`.
	      momentum: The moving average momentum.

	  Returns:
	      An Operation to update the variable.
	  """
	  return moving_averages.assign_moving_average(
	      x, value, momentum, zero_debias=False)


	# LINEAR ALGEBRA


	def dot(x, y):
	  """Multiplies 2 tensors (and/or variables) and returns a *tensor*.

	  When attempting to multiply a nD tensor
	  with a nD tensor, it reproduces the Theano behavior.
	  (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A tensor, dot product of `x` and `y`.

	  Examples:
	  ```python
	      # dot product between tensors
	      >>> x = K.placeholder(shape=(2, 3))
	      >>> y = K.placeholder(shape=(3, 4))
	      >>> xy = K.dot(x, y)
	      >>> xy
	      <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
	  ```

	  ```python
	      # dot product between tensors
	      >>> x = K.placeholder(shape=(32, 28, 3))
	      >>> y = K.placeholder(shape=(3, 4))
	      >>> xy = K.dot(x, y)
	      >>> xy
	      <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
	  ```

	  ```python
	      # Theano-like behavior example
	      >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
	      >>> y = K.ones((4, 3, 5))
	      >>> xy = K.dot(x, y)
	      >>> K.int_shape(xy)
	      (2, 4, 5)
	  ```
	  """
	  if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
	    x_shape = []
	    for i, s in zip(int_shape(x), array_ops.unstack(array_ops.shape(x))):
	      if i is not None:
	        x_shape.append(i)
	      else:
	        x_shape.append(s)
	    x_shape = tuple(x_shape)
	    y_shape = []
	    for i, s in zip(int_shape(y), array_ops.unstack(array_ops.shape(y))):
	      if i is not None:
	        y_shape.append(i)
	      else:
	        y_shape.append(s)
	    y_shape = tuple(y_shape)
	    y_permute_dim = list(range(ndim(y)))
	    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
	    xt = array_ops.reshape(x, [-1, x_shape[-1]])
	    yt = array_ops.reshape(
	        array_ops.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
	    return array_ops.reshape(
	        math_ops.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
	  if is_sparse(x):
	    out = sparse_ops.sparse_tensor_dense_matmul(x, y)
	  else:
	    out = math_ops.matmul(x, y)
	  return out


	def batch_dot(x, y, axes=None):
	  """Batchwise dot product.

	  `batch_dot` is used to compute dot product of `x` and `y` when
	  `x` and `y` are data in batch, i.e. in a shape of
	  `(batch_size, :)`.
	  `batch_dot` results in a tensor or variable with less dimensions
	  than the input. If the number of dimensions is reduced to 1,
	  we use `expand_dims` to make sure that ndim is at least 2.

	  Arguments:
	      x: Keras tensor or variable with `ndim >= 2`.
	      y: Keras tensor or variable with `ndim >= 2`.
	      axes: list of (or single) int with target dimensions.
	          The lengths of `axes[0]` and `axes[1]` should be the same.

	  Returns:
	      A tensor with shape equal to the concatenation of `x`'s shape
	      (less the dimension that was summed over) and `y`'s shape
	      (less the batch dimension and the dimension that was summed over).
	      If the final rank is 1, we reshape it to `(batch_size, 1)`.

	  Examples:
	      Assume `x = [[1, 2], [3, 4]]` and `y = [[5, 6], [7, 8]]`
	      `batch_dot(x, y, axes=1) = [[17, 53]]` which is the main diagonal
	      of `x.dot(y.T)`, although we never have to calculate the off-diagonal
	      elements.

	      Shape inference:
	      Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
	      If `axes` is (1, 2), to find the output shape of resultant tensor,
	          loop through each dimension in `x`'s shape and `y`'s shape:

	      * `x.shape[0]` : 100 : append to output shape
	      * `x.shape[1]` : 20 : do not append to output shape,
	          dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
	      * `y.shape[0]` : 100 : do not append to output shape,
	          always ignore first dimension of `y`
	      * `y.shape[1]` : 30 : append to output shape
	      * `y.shape[2]` : 20 : do not append to output shape,
	          dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
	      `output_shape` = `(100, 30)`

	  ```python
	      >>> x_batch = K.ones(shape=(32, 20, 1))
	      >>> y_batch = K.ones(shape=(32, 30, 20))
	      >>> xy_batch_dot = K.batch_dot(x_batch, y_batch, axes=[1, 2])
	      >>> K.int_shape(xy_batch_dot)
	      (32, 1, 30)
	  ```
	  """
	  if isinstance(axes, int):
	    axes = (axes, axes)
	  x_ndim = ndim(x)
	  y_ndim = ndim(y)
	  if x_ndim > y_ndim:
	    diff = x_ndim - y_ndim
	    y = array_ops.reshape(y,
	                          array_ops.concat(
	                              [array_ops.shape(y), [1] * (diff)], axis=0))
	  elif y_ndim > x_ndim:
	    diff = y_ndim - x_ndim
	    x = array_ops.reshape(x,
	                          array_ops.concat(
	                              [array_ops.shape(x), [1] * (diff)], axis=0))
	  else:
	    diff = 0
	  if ndim(x) == 2 and ndim(y) == 2:
	    if axes[0] == axes[1]:
	      out = math_ops.reduce_sum(math_ops.multiply(x, y), axes[0])
	    else:
	      out = math_ops.reduce_sum(
	          math_ops.multiply(array_ops.transpose(x, [1, 0]), y), axes[1])
	  else:
	    if axes is not None:
	      adj_x = None if axes[0] == ndim(x) - 1 else True
	      adj_y = True if axes[1] == ndim(y) - 1 else None
	    else:
	      adj_x = None
	      adj_y = None
	    out = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
	  if diff:
	    if x_ndim > y_ndim:
	      idx = x_ndim + y_ndim - 3
	    else:
	      idx = x_ndim - 1
	    out = array_ops.squeeze(out, list(range(idx, idx + diff)))
	  if ndim(out) == 1:
	    out = expand_dims(out, 1)
	  return out


	def transpose(x):
	  """Transposes a tensor and returns it.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.

	  Examples:
	  ```python
	      >>> var = K.variable([[1, 2, 3], [4, 5, 6]])
	      >>> K.eval(var)
	      array([[ 1.,  2.,  3.],
	             [ 4.,  5.,  6.]], dtype=float32)
	      >>> var_transposed = K.transpose(var)
	      >>> K.eval(var_transposed)
	      array([[ 1.,  4.],
	             [ 2.,  5.],
	             [ 3.,  6.]], dtype=float32)
	  ```

	  ```python
	      >>> input = K.placeholder((2, 3))
	      >>> input
	      <tf.Tensor 'Placeholder_11:0' shape=(2, 3) dtype=float32>
	      >>> input_transposed = K.transpose(input)
	      >>> input_transposed
	      <tf.Tensor 'transpose_4:0' shape=(3, 2) dtype=float32>

	  ```
	  """
	  return array_ops.transpose(x)


	def gather(reference, indices):
	  """Retrieves the elements of indices `indices` in the tensor `reference`.

	  Arguments:
	      reference: A tensor.
	      indices: An integer tensor of indices.

	  Returns:
	      A tensor of same type as `reference`.
	  """
	  return array_ops.gather(reference, indices)


	# ELEMENT-WISE OPERATIONS


	def _normalize_axis(axis, ndim):
	  """Converts negative axes to positive values.

	  Arguments:
	      axis: Integer axis (possibly negative).
	      ndim: Rank of the tensor considered.

	  Returns:
	      Positive integer axis.
	  """
	  if isinstance(axis, tuple):
	    axis = list(axis)
	  if isinstance(axis, list):
	    for i, a in enumerate(axis):
	      if a is not None and a < 0:
	        axis[i] = a % ndim
	  else:
	    if axis is not None and axis < 0:
	      axis %= ndim
	  return axis


	def max(x, axis=None, keepdims=False):
	  """Maximum value in a tensor.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to find maximum values.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`,
	          the reduced dimension is retained with length 1.

	  Returns:
	      A tensor with maximum values of `x`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.reduce_max(x, reduction_indices=axis, keep_dims=keepdims)


	def min(x, axis=None, keepdims=False):
	  """Minimum value in a tensor.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to find minimum values.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`,
	          the reduced dimension is retained with length 1.

	  Returns:
	      A tensor with miminum values of `x`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.reduce_min(x, reduction_indices=axis, keep_dims=keepdims)


	def sum(x, axis=None, keepdims=False):
	  """Sum of the values in a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to sum over.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`,
	          the reduced dimension is retained with length 1.

	  Returns:
	      A tensor with sum of `x`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.reduce_sum(x, reduction_indices=axis, keep_dims=keepdims)


	def prod(x, axis=None, keepdims=False):
	  """Multiplies the values in a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to compute the product.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`,
	          the reduced dimension is retained with length 1.

	  Returns:
	      A tensor with the product of elements of `x`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.reduce_prod(x, reduction_indices=axis, keep_dims=keepdims)


	def cumsum(x, axis=0):
	  """Cumulative sum of the values in a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to compute the sum.

	  Returns:
	      A tensor of the cumulative sum of values of `x` along `axis`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.cumsum(x, axis=axis)


	def cumprod(x, axis=0):
	  """Cumulative product of the values in a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to compute the product.

	  Returns:
	      A tensor of the cumulative product of values of `x` along `axis`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.cumprod(x, axis=axis)


	def var(x, axis=None, keepdims=False):
	  """Variance of a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to compute the variance.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`,
	          the reduced dimension is retained with length 1.

	  Returns:
	      A tensor with the variance of elements of `x`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  if x.dtype.base_dtype == dtypes_module.bool:
	    x = math_ops.cast(x, floatx())
	  m = math_ops.reduce_mean(x, reduction_indices=axis, keep_dims=True)
	  devs_squared = math_ops.square(x - m)
	  return math_ops.reduce_mean(
	      devs_squared, reduction_indices=axis, keep_dims=keepdims)


	def std(x, axis=None, keepdims=False):
	  """Standard deviation of a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to compute the standard deviation.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`,
	          the reduced dimension is retained with length 1.

	  Returns:
	      A tensor with the standard deviation of elements of `x`.
	  """
	  return math_ops.sqrt(var(x, axis=axis, keepdims=keepdims))


	def mean(x, axis=None, keepdims=False):
	  """Mean of a tensor, alongside the specified axis.

	  Arguments:
	      x: A tensor or variable.
	      axis: A list of integer. Axes to compute the mean.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1 for each entry in `axis`. If `keep_dims` is `True`,
	          the reduced dimensions are retained with length 1.

	  Returns:
	      A tensor with the mean of elements of `x`.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  if x.dtype.base_dtype == dtypes_module.bool:
	    x = math_ops.cast(x, floatx())
	  return math_ops.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)


	def any(x, axis=None, keepdims=False):
	  """Bitwise reduction (logical OR).

	  Arguments:
	      x: Tensor or variable.
	      axis: axis along which to perform the reduction.
	      keepdims: whether the drop or broadcast the reduction axes.

	  Returns:
	      A uint8 tensor (0s and 1s).
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  x = math_ops.cast(x, dtypes_module.bool)
	  return math_ops.reduce_any(x, reduction_indices=axis, keep_dims=keepdims)


	def all(x, axis=None, keepdims=False):
	  """Bitwise reduction (logical AND).

	  Arguments:
	      x: Tensor or variable.
	      axis: axis along which to perform the reduction.
	      keepdims: whether the drop or broadcast the reduction axes.

	  Returns:
	      A uint8 tensor (0s and 1s).
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  x = math_ops.cast(x, dtypes_module.bool)
	  return math_ops.reduce_all(x, reduction_indices=axis, keep_dims=keepdims)


	def argmax(x, axis=-1):
	  """Returns the index of the maximum value along an axis.

	  Arguments:
	      x: Tensor or variable.
	      axis: axis along which to perform the reduction.

	  Returns:
	      A tensor.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.argmax(x, axis)


	def argmin(x, axis=-1):
	  """Returns the index of the minimum value along an axis.

	  Arguments:
	      x: Tensor or variable.
	      axis: axis along which to perform the reduction.

	  Returns:
	      A tensor.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.argmin(x, axis)


	def square(x):
	  """Element-wise square.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.square(x)


	def abs(x):
	  """Element-wise absolute value.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.abs(x)


	def sqrt(x):
	  """Element-wise square root.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  zero = _to_tensor(0., x.dtype.base_dtype)
	  inf = _to_tensor(np.inf, x.dtype.base_dtype)
	  x = clip_ops.clip_by_value(x, zero, inf)
	  return math_ops.sqrt(x)


	def exp(x):
	  """Element-wise exponential.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.exp(x)


	def log(x):
	  """Element-wise log.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.log(x)


	def logsumexp(x, axis=None, keepdims=False):
	  """Computes log(sum(exp(elements across dimensions of a tensor))).

	  This function is more numerically stable than log(sum(exp(x))).
	  It avoids overflows caused by taking the exp of large inputs and
	  underflows caused by taking the log of small inputs.

	  Arguments:
	      x: A tensor or variable.
	      axis: An integer, the axis to reduce over.
	      keepdims: A boolean, whether to keep the dimensions or not.
	          If `keepdims` is `False`, the rank of the tensor is reduced
	          by 1. If `keepdims` is `True`, the reduced dimension is
	          retained with length 1.

	  Returns:
	      The reduced tensor.
	  """
	  axis = _normalize_axis(axis, ndim(x))
	  return math_ops.reduce_logsumexp(x, axis=axis, keep_dims=keepdims)


	def round(x):
	  """Element-wise rounding to the closest integer.

	  In case of tie, the rounding mode used is "half to even".

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.round(x)


	def sign(x):
	  """Element-wise sign.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.sign(x)


	def pow(x, a):
	  """Element-wise exponentiation.

	  Arguments:
	      x: Tensor or variable.
	      a: Python integer.

	  Returns:
	      A tensor.
	  """
	  return math_ops.pow(x, a)


	def clip(x, min_value, max_value):
	  """Element-wise value clipping.

	  Arguments:
	      x: Tensor or variable.
	      min_value: Python float or integer.
	      max_value: Python float or integer.

	  Returns:
	      A tensor.
	  """
	  if max_value is not None and max_value < min_value:
	    max_value = min_value
	  if max_value is None:
	    max_value = np.inf
	  min_value = _to_tensor(min_value, x.dtype.base_dtype)
	  max_value = _to_tensor(max_value, x.dtype.base_dtype)
	  return clip_ops.clip_by_value(x, min_value, max_value)


	def equal(x, y):
	  """Element-wise equality between two tensors.

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A bool tensor.
	  """
	  return math_ops.equal(x, y)


	def not_equal(x, y):
	  """Element-wise inequality between two tensors.

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A bool tensor.
	  """
	  return math_ops.not_equal(x, y)


	def greater(x, y):
	  """Element-wise truth value of (x > y).

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A bool tensor.
	  """
	  return math_ops.greater(x, y)


	def greater_equal(x, y):
	  """Element-wise truth value of (x >= y).

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A bool tensor.
	  """
	  return math_ops.greater_equal(x, y)


	def less(x, y):
	  """Element-wise truth value of (x < y).

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A bool tensor.
	  """
	  return math_ops.less(x, y)


	def less_equal(x, y):
	  """Element-wise truth value of (x <= y).

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A bool tensor.
	  """
	  return math_ops.less_equal(x, y)


	def maximum(x, y):
	  """Element-wise maximum of two tensors.

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.maximum(x, y)


	def minimum(x, y):
	  """Element-wise minimum of two tensors.

	  Arguments:
	      x: Tensor or variable.
	      y: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.minimum(x, y)


	def sin(x):
	  """Computes sin of x element-wise.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.sin(x)


	def cos(x):
	  """Computes cos of x element-wise.

	  Arguments:
	      x: Tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return math_ops.cos(x)


	def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
	  """Computes mean and std for batch then apply batch_normalization on batch.

	  Arguments:
	      x: Input tensor or variable.
	      gamma: Tensor by which to scale the input.
	      beta: Tensor with which to center the input.
	      reduction_axes: iterable of integers,
	          axes over which to normalize.
	      epsilon: Fuzz factor.

	  Returns:
	      A tuple length of 3, `(normalized_tensor, mean, variance)`.
	  """
	  mean, var = nn.moments(
	      x, reduction_axes, shift=None, name=None, keep_dims=False)
	  if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
	    normed = nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
	  else:
	    # need broadcasting
	    target_shape = []
	    for axis in range(ndim(x)):
	      if axis in reduction_axes:
	        target_shape.append(1)
	      else:
	        target_shape.append(array_ops.shape(x)[axis])
	    target_shape = array_ops.stack(target_shape)

	    broadcast_mean = array_ops.reshape(mean, target_shape)
	    broadcast_var = array_ops.reshape(var, target_shape)
	    if gamma is None:
	      broadcast_gamma = None
	    else:
	      broadcast_gamma = array_ops.reshape(gamma, target_shape)
	    if beta is None:
	      broadcast_beta = None
	    else:
	      broadcast_beta = array_ops.reshape(beta, target_shape)
	    normed = nn.batch_normalization(x, broadcast_mean, broadcast_var,
	                                    broadcast_beta, broadcast_gamma, epsilon)
	  return normed, mean, var


	def batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3):
	  """Applies batch normalization on x given mean, var, beta and gamma.

	  I.e. returns:
	  `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

	  Arguments:
	      x: Input tensor or variable.
	      mean: Mean of batch.
	      var: Variance of batch.
	      beta: Tensor with which to center the input.
	      gamma: Tensor by which to scale the input.
	      epsilon: Fuzz factor.

	  Returns:
	      A tensor.
	  """
	  return nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


	# SHAPE OPERATIONS


	def concatenate(tensors, axis=-1):
	  """Concatenates a list of tensors alongside the specified axis.

	  Arguments:
	      tensors: list of tensors to concatenate.
	      axis: concatenation axis.

	  Returns:
	      A tensor.
	  """
	  if axis < 0:
	    rank = ndim(tensors[0])
	    if rank:
	      axis %= rank
	    else:
	      axis = 0

	  if py_all([is_sparse(x) for x in tensors]):
	    return sparse_ops.sparse_concat(axis, tensors)
	  else:
	    return array_ops.concat([to_dense(x) for x in tensors], axis)


	def reshape(x, shape):
	  """Reshapes a tensor to the specified shape.

	  Arguments:
	      x: Tensor or variable.
	      shape: Target shape tuple.

	  Returns:
	      A tensor.
	  """
	  return array_ops.reshape(x, shape)


	def permute_dimensions(x, pattern):
	  """Permutes axes in a tensor.

	  Arguments:
	      x: Tensor or variable.
	      pattern: A tuple of
	          dimension indices, e.g. `(0, 2, 1)`.

	  Returns:
	      A tensor.
	  """
	  return array_ops.transpose(x, perm=pattern)


	def resize_images(x, height_factor, width_factor, data_format):
	  """Resizes the images contained in a 4D tensor.

	  Arguments:
	      x: Tensor or variable to resize.
	      height_factor: Positive integer.
	      width_factor: Positive integer.
	      data_format: One of `"channels_first"`, `"channels_last"`.

	  Returns:
	      A tensor.

	  Raises:
	      ValueError: if `data_format` is neither
	          `channels_last` or `channels_first`.
	  """
	  if data_format == 'channels_first':
	    original_shape = int_shape(x)
	    new_shape = array_ops.shape(x)[2:]
	    new_shape *= constant_op.constant(
	        np.array([height_factor, width_factor]).astype('int32'))
	    x = permute_dimensions(x, [0, 2, 3, 1])
	    x = image_ops.resize_nearest_neighbor(x, new_shape)
	    x = permute_dimensions(x, [0, 3, 1, 2])
	    x.set_shape((None, None, original_shape[2] * height_factor
	                 if original_shape[2] is not None else None,
	                 original_shape[3] * width_factor
	                 if original_shape[3] is not None else None))
	    return x
	  elif data_format == 'channels_last':
	    original_shape = int_shape(x)
	    new_shape = array_ops.shape(x)[1:3]
	    new_shape *= constant_op.constant(
	        np.array([height_factor, width_factor]).astype('int32'))
	    x = image_ops.resize_nearest_neighbor(x, new_shape)
	    x.set_shape((None, original_shape[1] * height_factor
	                 if original_shape[1] is not None else None,
	                 original_shape[2] * width_factor
	                 if original_shape[2] is not None else None, None))
	    return x
	  else:
	    raise ValueError('Invalid data_format:', data_format)


	def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
	  """Resizes the volume contained in a 5D tensor.

	  Arguments:
	      x: Tensor or variable to resize.
	      depth_factor: Positive integer.
	      height_factor: Positive integer.
	      width_factor: Positive integer.
	      data_format: One of `"channels_first"`, `"channels_last"`.

	  Returns:
	      A tensor.

	  Raises:
	      ValueError: if `data_format` is neither
	          `channels_last` or `channels_first`.
	  """
	  if data_format == 'channels_first':
	    output = repeat_elements(x, depth_factor, axis=2)
	    output = repeat_elements(output, height_factor, axis=3)
	    output = repeat_elements(output, width_factor, axis=4)
	    return output
	  elif data_format == 'channels_last':
	    output = repeat_elements(x, depth_factor, axis=1)
	    output = repeat_elements(output, height_factor, axis=2)
	    output = repeat_elements(output, width_factor, axis=3)
	    return output
	  else:
	    raise ValueError('Invalid data_format:', data_format)


	def repeat_elements(x, rep, axis):
	  """Repeats the elements of a tensor along an axis, like `np.repeat`.

	  If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
	  will have shape `(s1, s2 * rep, s3)`.

	  Arguments:
	      x: Tensor or variable.
	      rep: Python integer, number of times to repeat.
	      axis: Axis along which to repeat.

	  Raises:
	      ValueError: In case `x.shape[axis]` is undefined.

	  Returns:
	      A tensor.
	  """
	  x_shape = x.get_shape().as_list()
	  if x_shape[axis] is None:
	    raise ValueError('Axis ' + str(axis) + ' of input tensor '
	                     'should have a defined dimension, but is None. '
	                     'Full tensor shape: ' + str(tuple(x_shape)) + '. '
	                     'Typically you need to pass a fully-defined '
	                     '`input_shape` argument to your first layer.')
	  # slices along the repeat axis
	  splits = array_ops.split(value=x, num_or_size_splits=x_shape[axis], axis=axis)
	  # repeat each slice the given number of reps
	  x_rep = [s for s in splits for _ in range(rep)]
	  return concatenate(x_rep, axis)


	def repeat(x, n):
	  """Repeats a 2D tensor.

	  if `x` has shape (samples, dim) and `n` is `2`,
	  the output will have shape `(samples, 2, dim)`.

	  Arguments:
	      x: Tensor or variable.
	      n: Python integer, number of times to repeat.

	  Returns:
	      A tensor.
	  """
	  assert ndim(x) == 2
	  x = array_ops.expand_dims(x, 1)
	  pattern = array_ops.stack([1, n, 1])
	  return array_ops.tile(x, pattern)


	def arange(start, stop=None, step=1, dtype='int32'):
	  """Creates a 1D tensor containing a sequence of integers.

	  The function arguments use the same convention as
	  Theano's arange: if only one argument is provided,
	  it is in fact the "stop" argument.

	  The default type of the returned tensor is `'int32'` to
	  match TensorFlow's default.

	  Arguments:
	      start: Start value.
	      stop: Stop value.
	      step: Difference between two successive values.
	      dtype: Integer dtype to use.

	  Returns:
	      An integer tensor.

	  """
	  # Match the behavior of numpy and Theano by returning an empty seqence.
	  if stop is None and start < 0:
	    start = 0
	  result = math_ops.range(start, limit=stop, delta=step, name='arange')
	  if dtype != 'int32':
	    result = cast(result, dtype)
	  return result


	def tile(x, n):
	  """Creates a tensor by tiling `x` by `n`.

	  Arguments:
	      x: A tensor or variable
	      n: A list of integer. The length must be the same as the number of
	          dimensions in `x`.

	  Returns:
	      A tiled tensor.
	  """
	  if isinstance(n, int):
	    n = [n]
	  return array_ops.tile(x, n)


	def flatten(x):
	  """Flatten a tensor.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor, reshaped into 1-D
	  """
	  return array_ops.reshape(x, [-1])


	def batch_flatten(x):
	  """Turn a nD tensor into a 2D tensor with same 0th dimension.

	  In other words, it flattens each data samples of a batch.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  x = array_ops.reshape(x, array_ops.stack([-1, prod(shape(x)[1:])]))
	  return x


	def expand_dims(x, axis=-1):
	  """Adds a 1-sized dimension at index "axis".

	  Arguments:
	      x: A tensor or variable.
	      axis: Position where to add a new axis.

	  Returns:
	      A tensor with expanded dimensions.
	  """
	  return array_ops.expand_dims(x, axis)


	def squeeze(x, axis):
	  """Removes a 1-dimension from the tensor at index "axis".

	  Arguments:
	      x: A tensor or variable.
	      axis: Axis to drop.

	  Returns:
	      A tensor with the same data as `x` but reduced dimensions.
	  """
	  return array_ops.squeeze(x, [axis])


	def temporal_padding(x, padding=(1, 1)):
	  """Pads the middle dimension of a 3D tensor.

	  Arguments:
	      x: Tensor or variable.
	      padding: Tuple of 2 integers, how many zeros to
	          add at the start and end of dim 1.

	  Returns:
	      A padded 3D tensor.
	  """
	  assert len(padding) == 2
	  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
	  return array_ops.pad(x, pattern)


	def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
	  """Pads the 2nd and 3rd dimensions of a 4D tensor.

	  Arguments:
	      x: Tensor or variable.
	      padding: Tuple of 2 tuples, padding pattern.
	      data_format: One of `channels_last` or `channels_first`.

	  Returns:
	      A padded 4D tensor.

	  Raises:
	      ValueError: if `data_format` is neither
	          `channels_last` or `channels_first`.
	  """
	  assert len(padding) == 2
	  assert len(padding[0]) == 2
	  assert len(padding[1]) == 2
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  if data_format == 'channels_first':
	    pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
	  else:
	    pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
	  return array_ops.pad(x, pattern)


	def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
	  """Pads 5D tensor with zeros along the depth, height, width dimensions.

	  Pads these dimensions with respectively
	  "padding[0]", "padding[1]" and "padding[2]" zeros left and right.

	  For 'channels_last' data_format,
	  the 2nd, 3rd and 4th dimension will be padded.
	  For 'channels_first' data_format,
	  the 3rd, 4th and 5th dimension will be padded.

	  Arguments:
	      x: Tensor or variable.
	      padding: Tuple of 3 tuples, padding pattern.
	      data_format: One of `channels_last` or `channels_first`.

	  Returns:
	      A padded 5D tensor.

	  Raises:
	      ValueError: if `data_format` is neither
	          `channels_last` or `channels_first`.

	  """
	  assert len(padding) == 3
	  assert len(padding[0]) == 2
	  assert len(padding[1]) == 2
	  assert len(padding[2]) == 2
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  if data_format == 'channels_first':
	    pattern = [[0, 0], [0, 0], [padding[0][0], padding[0][1]],
	               [padding[1][0], padding[1][1]], [padding[2][0], padding[2][1]]]
	  else:
	    pattern = [[0, 0], [padding[0][0], padding[0][1]],
	               [padding[1][0], padding[1][1]], [padding[2][0],
	                                                padding[2][1]], [0, 0]]
	  return array_ops.pad(x, pattern)


	def stack(x, axis=0):
	  """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

	  Arguments:
	      x: List of tensors.
	      axis: Axis along which to perform stacking.

	  Returns:
	      A tensor.
	  """
	  return array_ops.stack(x, axis=axis)


	def one_hot(indices, num_classes):
	  """Computes the one-hot representation of an integer tensor.

	  Arguments:
	      indices: nD integer tensor of shape
	          `(batch_size, dim1, dim2, ... dim(n-1))`
	      num_classes: Integer, number of classes to consider.

	  Returns:
	      (n + 1)D one hot representation of the input
	      with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`

	  Returns:
	      The one-hot tensor.
	  """
	  return array_ops.one_hot(indices, depth=num_classes, axis=-1)


	def reverse(x, axes):
	  """Reverse a tensor along the specified axes.

	  Arguments:
	      x: Tensor to reverse.
	      axes: Integer or iterable of integers.
	          Axes to reverse.

	  Returns:
	      A tensor.
	  """
	  if isinstance(axes, int):
	    axes = [axes]
	  return array_ops.reverse(x, axes)


	# VALUE MANIPULATION


	def get_value(x):
	  """Returns the value of a variable.

	  Arguments:
	      x: input variable.

	  Returns:
	      A Numpy array.
	  """
	  return x.eval(session=get_session())


	def batch_get_value(tensors):
	  """Returns the value of more than one tensor variable.

	  Arguments:
	      tensors: list of ops to run.

	  Returns:
	      A list of Numpy arrays.
	  """
	  if tensors:
	    return get_session().run(tensors)
	  else:
	    return []


	def set_value(x, value):
	  """Sets the value of a variable, from a Numpy array.

	  Arguments:
	      x: Tensor to set to a new value.
	      value: Value to set the tensor to, as a Numpy array
	          (of the same shape).
	  """
	  value = np.asarray(value)
	  tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
	  if hasattr(x, '_assign_placeholder'):
	    assign_placeholder = x._assign_placeholder
	    assign_op = x._assign_op
	  else:
	    assign_placeholder = array_ops.placeholder(tf_dtype, shape=value.shape)
	    assign_op = x.assign(assign_placeholder)
	    x._assign_placeholder = assign_placeholder
	    x._assign_op = assign_op
	  get_session().run(assign_op, feed_dict={assign_placeholder: value})


	def batch_set_value(tuples):
	  """Sets the values of many tensor variables at once.

	  Arguments:
	      tuples: a list of tuples `(tensor, value)`.
	          `value` should be a Numpy array.
	  """
	  if tuples:
	    assign_ops = []
	    feed_dict = {}
	    for x, value in tuples:
	      value = np.asarray(value)
	      tf_dtype = _convert_string_dtype(x.dtype.name.split('_')[0])
	      if hasattr(x, '_assign_placeholder'):
	        assign_placeholder = x._assign_placeholder
	        assign_op = x._assign_op
	      else:
	        assign_placeholder = array_ops.placeholder(tf_dtype, shape=value.shape)
	        assign_op = x.assign(assign_placeholder)
	        x._assign_placeholder = assign_placeholder
	        x._assign_op = assign_op
	      assign_ops.append(assign_op)
	      feed_dict[assign_placeholder] = value
	    get_session().run(assign_ops, feed_dict=feed_dict)


	def print_tensor(x, message=''):
	  """Prints `message` and the tensor value when evaluated.

	  Arguments:
	      x: Tensor to print.
	      message: Message to print jointly with the tensor.

	  Returns:
	      The same tensor `x`, unchanged.
	  """
	  return logging_ops.Print(x, [x], message)


	# GRAPH MANIPULATION


	class Function(object):
	  """Runs a computation graph.

	  Arguments:
	      inputs: Feed placeholders to the computation graph.
	      outputs: Output tensors to fetch.
	      updates: Additional update ops to be run at function call.
	      name: a name to help users identify what this function does.
	  """

	  def __init__(self, inputs, outputs, updates=None, name=None,
	               **session_kwargs):
	    updates = updates or []
	    if not isinstance(inputs, (list, tuple)):
	      raise TypeError('`inputs` to a TensorFlow backend function '
	                      'should be a list or tuple.')
	    if not isinstance(outputs, (list, tuple)):
	      raise TypeError('`outputs` of a TensorFlow backend function '
	                      'should be a list or tuple.')
	    if not isinstance(updates, (list, tuple)):
	      raise TypeError('`updates` in a TensorFlow backend function '
	                      'should be a list or tuple.')
	    self.inputs = list(inputs)
	    self.outputs = list(outputs)
	    with ops.control_dependencies(self.outputs):
	      updates_ops = []
	      for update in updates:
	        if isinstance(update, tuple):
	          p, new_p = update
	          updates_ops.append(state_ops.assign(p, new_p))
	        else:
	          # assumed already an op
	          updates_ops.append(update)
	      self.updates_op = control_flow_ops.group(*updates_ops)
	    self.name = name
	    self.session_kwargs = session_kwargs

	  def __call__(self, inputs):
	    if not isinstance(inputs, (list, tuple)):
	      raise TypeError('`inputs` should be a list or tuple.')
	    feed_dict = {}
	    for tensor, value in zip(self.inputs, inputs):
	      if is_sparse(tensor):
	        sparse_coo = value.tocoo()
	        indices = np.concatenate((np.expand_dims(sparse_coo.row, 1),
	                                  np.expand_dims(sparse_coo.col, 1)), 1)
	        value = (indices, sparse_coo.data, sparse_coo.shape)
	      feed_dict[tensor] = value
	    session = get_session()
	    updated = session.run(
	        self.outputs + [self.updates_op],
	        feed_dict=feed_dict,
	        **self.session_kwargs)
	    return updated[:len(self.outputs)]


	def function(inputs, outputs, updates=None, **kwargs):
	  """Instantiates a Keras function.

	  Arguments:
	      inputs: List of placeholder tensors.
	      outputs: List of output tensors.
	      updates: List of update ops.
	      **kwargs: Passed to `tf.Session.run`.

	  Returns:
	      Output values as Numpy arrays.

	  Raises:
	      ValueError: if invalid kwargs are passed in.
	  """
	  if kwargs:
	    for key in kwargs:
	      if (key not in tf_inspect.getargspec(session_module.Session.run)[0] and
	          key not in tf_inspect.getargspec(Function.__init__)[0]):
	        msg = ('Invalid argument "%s" passed to K.function with Tensorflow '
	               'backend') % key
	        raise ValueError(msg)
	  return Function(inputs, outputs, updates=updates, **kwargs)


	def gradients(loss, variables):
	  """Returns the gradients of `variables` w.r.t. `loss`.

	  Arguments:
	      loss: Scalar tensor to minimize.
	      variables: List of variables.

	  Returns:
	      A gradients tensor.
	  """
	  return gradients_module.gradients(
	      loss, variables, colocate_gradients_with_ops=True)


	def stop_gradient(variables):
	  """Returns `variables` but with zero gradient w.r.t. every other variable.

	  Arguments:
	      variables: List of variables.

	  Returns:
	      The same list of variables.
	  """
	  return array_ops.stop_gradient(variables)


	# CONTROL FLOW


	def rnn(step_function,
	        inputs,
	        initial_states,
	        go_backwards=False,
	        mask=None,
	        constants=None,
	        unroll=False):
	  """Iterates over the time dimension of a tensor.

	  Arguments:
	      step_function: RNN step function.
	          Parameters;
	              input; tensor with shape `(samples, ...)` (no time dimension),
	                  representing input for the batch of samples at a certain
	                  time step.
	              states; list of tensors.
	          Returns;
	              output; tensor with shape `(samples, output_dim)`
	                  (no time dimension).
	              new_states; list of tensors, same length and shapes
	                  as 'states'. The first state in the list must be the
	                  output tensor at the previous timestep.
	      inputs: tensor of temporal data of shape `(samples, time, ...)`
	          (at least 3D).
	      initial_states: tensor with shape (samples, output_dim)
	          (no time dimension),
	          containing the initial values for the states used in
	          the step function.
	      go_backwards: boolean. If True, do the iteration over the time
	          dimension in reverse order and return the reversed sequence.
	      mask: binary tensor with shape `(samples, time, 1)`,
	          with a zero for every element that is masked.
	      constants: a list of constant values passed at each step.
	      unroll: whether to unroll the RNN or to use a symbolic loop
	          (`while_loop` or `scan` depending on backend).

	  Returns:
	      A tuple, `(last_output, outputs, new_states)`.
	          last_output: the latest output of the rnn, of shape `(samples, ...)`
	          outputs: tensor with shape `(samples, time, ...)` where each
	              entry `outputs[s, t]` is the output of the step function
	              at time `t` for sample `s`.
	          new_states: list of tensors, latest states returned by
	              the step function, of shape `(samples, ...)`.

	  Raises:
	      ValueError: if input dimension is less than 3.
	      ValueError: if `unroll` is `True` but input timestep is not a fixed
	      number.
	      ValueError: if `mask` is provided (not `None`) but states is not provided
	          (`len(states)` == 0).
	  """
	  ndim = len(inputs.get_shape())
	  if ndim < 3:
	    raise ValueError('Input should be at least 3D.')
	  axes = [1, 0] + list(range(2, ndim))
	  inputs = array_ops.transpose(inputs, (axes))

	  if mask is not None:
	    if mask.dtype != dtypes_module.bool:
	      mask = math_ops.cast(mask, dtypes_module.bool)
	    if len(mask.get_shape()) == ndim - 1:
	      mask = expand_dims(mask)
	    mask = array_ops.transpose(mask, axes)

	  if constants is None:
	    constants = []

	  if unroll:
	    if not inputs.get_shape()[0]:
	      raise ValueError('Unrolling requires a ' 'fixed number of timesteps.')
	    states = initial_states
	    successive_states = []
	    successive_outputs = []

	    input_list = array_ops.unstack(inputs)
	    if go_backwards:
	      input_list.reverse()

	    if mask is not None:
	      mask_list = array_ops.unstack(mask)
	      if go_backwards:
	        mask_list.reverse()

	      for inp, mask_t in zip(input_list, mask_list):
	        output, new_states = step_function(inp, states + constants)

	        # tf.where needs its condition tensor
	        # to be the same shape as its two
	        # result tensors, but in our case
	        # the condition (mask) tensor is
	        # (nsamples, 1), and A and B are (nsamples, ndimensions).
	        # So we need to
	        # broadcast the mask to match the shape of A and B.
	        # That's what the tile call does,
	        # it just repeats the mask along its second dimension
	        # n times.
	        tiled_mask_t = array_ops.tile(mask_t,
	                                      array_ops.stack(
	                                          [1, array_ops.shape(output)[1]]))

	        if not successive_outputs:
	          prev_output = zeros_like(output)
	        else:
	          prev_output = successive_outputs[-1]

	        output = array_ops.where(tiled_mask_t, output, prev_output)

	        return_states = []
	        for state, new_state in zip(states, new_states):
	          # (see earlier comment for tile explanation)
	          tiled_mask_t = array_ops.tile(mask_t,
	                                        array_ops.stack(
	                                            [1,
	                                             array_ops.shape(new_state)[1]]))
	          return_states.append(array_ops.where(tiled_mask_t, new_state, state))
	        states = return_states
	        successive_outputs.append(output)
	        successive_states.append(states)
	      last_output = successive_outputs[-1]
	      new_states = successive_states[-1]
	      outputs = array_ops.stack(successive_outputs)
	    else:
	      for inp in input_list:
	        output, states = step_function(inp, states + constants)
	        successive_outputs.append(output)
	        successive_states.append(states)
	      last_output = successive_outputs[-1]
	      new_states = successive_states[-1]
	      outputs = array_ops.stack(successive_outputs)

	  else:
	    if go_backwards:
	      inputs = reverse(inputs, 0)

	    states = tuple(initial_states)

	    time_steps = array_ops.shape(inputs)[0]
	    outputs, _ = step_function(inputs[0], initial_states + constants)
	    output_ta = tensor_array_ops.TensorArray(
	        dtype=outputs.dtype, size=time_steps, tensor_array_name='output_ta')
	    input_ta = tensor_array_ops.TensorArray(
	        dtype=inputs.dtype, size=time_steps, tensor_array_name='input_ta')
	    input_ta = input_ta.unstack(inputs)
	    time = constant_op.constant(0, dtype='int32', name='time')

	    if mask is not None:
	      if not states:
	        raise ValueError('No initial states provided! '
	                         'When using masking in an RNN, you should '
	                         'provide initial states '
	                         '(and your step function should return '
	                         'as its first state at time `t` '
	                         'the output at time `t-1`).')
	      if go_backwards:
	        mask = reverse(mask, 0)

	      mask_ta = tensor_array_ops.TensorArray(
	          dtype=dtypes_module.bool,
	          size=time_steps,
	          tensor_array_name='mask_ta')
	      mask_ta = mask_ta.unstack(mask)

	      def _step(time, output_ta_t, *states):
	        """RNN step function.

	        Arguments:
	            time: Current timestep value.
	            output_ta_t: TensorArray.
	            *states: List of states.

	        Returns:
	            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
	        """
	        current_input = input_ta.read(time)
	        mask_t = mask_ta.read(time)
	        output, new_states = step_function(current_input,
	                                           tuple(states) + tuple(constants))
	        for state, new_state in zip(states, new_states):
	          new_state.set_shape(state.get_shape())
	        tiled_mask_t = array_ops.tile(mask_t,
	                                      array_ops.stack(
	                                          [1, array_ops.shape(output)[1]]))
	        output = array_ops.where(tiled_mask_t, output, states[0])
	        new_states = [
	            array_ops.where(tiled_mask_t, new_states[i], states[i])
	            for i in range(len(states))
	        ]
	        output_ta_t = output_ta_t.write(time, output)
	        return (time + 1, output_ta_t) + tuple(new_states)
	    else:

	      def _step(time, output_ta_t, *states):
	        """RNN step function.

	        Arguments:
	            time: Current timestep value.
	            output_ta_t: TensorArray.
	            *states: List of states.

	        Returns:
	            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
	        """
	        current_input = input_ta.read(time)
	        output, new_states = step_function(current_input,
	                                           tuple(states) + tuple(constants))
	        for state, new_state in zip(states, new_states):
	          new_state.set_shape(state.get_shape())
	        output_ta_t = output_ta_t.write(time, output)
	        return (time + 1, output_ta_t) + tuple(new_states)

	    final_outputs = control_flow_ops.while_loop(
	        cond=lambda time, *_: time < time_steps,
	        body=_step,
	        loop_vars=(time, output_ta) + states,
	        parallel_iterations=32,
	        swap_memory=True)
	    last_time = final_outputs[0]
	    output_ta = final_outputs[1]
	    new_states = final_outputs[2:]

	    outputs = output_ta.stack()
	    last_output = output_ta.read(last_time - 1)

	  axes = [1, 0] + list(range(2, len(outputs.get_shape())))
	  outputs = array_ops.transpose(outputs, axes)
	  return last_output, outputs, new_states


	def switch(condition, then_expression, else_expression):
	  """Switches between two operations depending on a scalar value.

	  Note that both `then_expression` and `else_expression`
	  should be symbolic tensors of the *same shape*.

	  Arguments:
	      condition: scalar tensor (`int` or `bool`).
	      then_expression: either a tensor, or a callable that returns a tensor.
	      else_expression: either a tensor, or a callable that returns a tensor.

	  Returns:
	      The selected tensor.
	  """
	  if condition.dtype != dtypes_module.bool:
	    condition = math_ops.cast(condition, 'bool')
	  if not callable(then_expression):

	    def then_expression_fn():
	      return then_expression
	  else:
	    then_expression_fn = then_expression
	  if not callable(else_expression):

	    def else_expression_fn():
	      return else_expression
	  else:
	    else_expression_fn = else_expression
	  x = control_flow_ops.cond(condition, then_expression_fn, else_expression_fn)
	  return x


	def in_train_phase(x, alt, training=None):
	  """Selects `x` in train phase, and `alt` otherwise.

	  Note that `alt` should have the *same shape* as `x`.

	  Arguments:
	      x: What to return in train phase
	          (tensor or callable that returns a tensor).
	      alt: What to return otherwise
	          (tensor or callable that returns a tensor).
	      training: Optional scalar tensor
	          (or Python boolean, or Python integer)
	          specifying the learning phase.

	  Returns:
	      Either `x` or `alt` based on the `training` flag.
	      the `training` flag defaults to `K.learning_phase()`.
	  """
	  if training is None:
	    training = learning_phase()
	    uses_learning_phase = True
	  else:
	    uses_learning_phase = False

	  if training is 1 or training is True:
	    if callable(x):
	      return x()
	    else:
	      return x

	  elif training is 0 or training is False:
	    if callable(alt):
	      return alt()
	    else:
	      return alt

	  # else: assume learning phase is a placeholder tensor.
	  x = switch(training, x, alt)
	  if uses_learning_phase:
	    x._uses_learning_phase = True
	  return x


	def in_test_phase(x, alt, training=None):
	  """Selects `x` in test phase, and `alt` otherwise.

	  Note that `alt` should have the *same shape* as `x`.

	  Arguments:
	      x: What to return in test phase
	          (tensor or callable that returns a tensor).
	      alt: What to return otherwise
	          (tensor or callable that returns a tensor).
	      training: Optional scalar tensor
	          (or Python boolean, or Python integer)
	          specifying the learning phase.

	  Returns:
	      Either `x` or `alt` based on `K.learning_phase`.
	  """
	  return in_train_phase(alt, x, training=training)


	# NN OPERATIONS


	def relu(x, alpha=0., max_value=None):
	  """Rectified linear unit.

	  With default values, it returns element-wise `max(x, 0)`.

	  Arguments:
	      x: A tensor or variable.
	      alpha: A scalar, slope of negative section (default=`0.`).
	      max_value: Saturation threshold.

	  Returns:
	      A tensor.
	  """
	  if alpha != 0.:
	    negative_part = nn.relu(-x)
	  x = nn.relu(x)
	  if max_value is not None:
	    max_value = _to_tensor(max_value, x.dtype.base_dtype)
	    zero = _to_tensor(0., x.dtype.base_dtype)
	    x = clip_ops.clip_by_value(x, zero, max_value)
	  if alpha != 0.:
	    alpha = _to_tensor(alpha, x.dtype.base_dtype)
	    x -= alpha * negative_part
	  return x


	def elu(x, alpha=1.):
	  """Exponential linear unit.

	  Arguments:
	      x: A tenor or variable to compute the activation function for.
	      alpha: A scalar, slope of positive section.

	  Returns:
	      A tensor.
	  """
	  res = nn.elu(x)
	  if alpha == 1:
	    return res
	  else:
	    return array_ops.where(x > 0, res, alpha * res)


	def softmax(x):
	  """Softmax of a tensor.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return nn.softmax(x)


	def softplus(x):
	  """Softplus of a tensor.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return nn.softplus(x)


	def softsign(x):
	  """Softsign of a tensor.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return nn.softsign(x)


	def categorical_crossentropy(output, target, from_logits=False):
	  """Categorical crossentropy between an output tensor and a target tensor.

	  Arguments:
	      output: A tensor resulting from a softmax
	          (unless `from_logits` is True, in which
	          case `output` is expected to be the logits).
	      target: A tensor of the same shape as `output`.
	      from_logits: Boolean, whether `output` is the
	          result of a softmax, or is a tensor of logits.

	  Returns:
	      Output tensor.
	  """
	  # Note: nn.softmax_cross_entropy_with_logits
	  # expects logits, Keras expects probabilities.
	  if not from_logits:
	    # scale preds so that the class probas of each sample sum to 1
	    output /= math_ops.reduce_sum(
	        output, reduction_indices=len(output.get_shape()) - 1, keep_dims=True)
	    # manual computation of crossentropy
	    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
	    output = clip_ops.clip_by_value(output, epsilon, 1. - epsilon)
	    return -math_ops.reduce_sum(
	        target * math_ops.log(output),
	        reduction_indices=len(output.get_shape()) - 1)
	  else:
	    return nn.softmax_cross_entropy_with_logits(labels=target, logits=output)


	def sparse_categorical_crossentropy(output, target, from_logits=False):
	  """Categorical crossentropy with integer targets.

	  Arguments:
	      output: A tensor resulting from a softmax
	          (unless `from_logits` is True, in which
	          case `output` is expected to be the logits).
	      target: An integer tensor.
	      from_logits: Boolean, whether `output` is the
	          result of a softmax, or is a tensor of logits.

	  Returns:
	      Output tensor.
	  """
	  # Note: nn.softmax_cross_entropy_with_logits
	  # expects logits, Keras expects probabilities.
	  if not from_logits:
	    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
	    output = clip_ops.clip_by_value(output, epsilon, 1 - epsilon)
	    output = math_ops.log(output)

	  output_shape = output.get_shape()
	  targets = cast(flatten(target), 'int64')
	  logits = array_ops.reshape(output, [-1, int(output_shape[-1])])
	  res = nn.sparse_softmax_cross_entropy_with_logits(
	      labels=targets, logits=logits)
	  if len(output_shape) == 3:
	    # if our output includes timesteps we need to reshape
	    return array_ops.reshape(res, array_ops.shape(output)[:-1])
	  else:
	    return res


	def binary_crossentropy(output, target, from_logits=False):
	  """Binary crossentropy between an output tensor and a target tensor.

	  Arguments:
	      output: A tensor.
	      target: A tensor with the same shape as `output`.
	      from_logits: Whether `output` is expected to be a logits tensor.
	          By default, we consider that `output`
	          encodes a probability distribution.

	  Returns:
	      A tensor.
	  """
	  # Note: nn.softmax_cross_entropy_with_logits
	  # expects logits, Keras expects probabilities.
	  if not from_logits:
	    # transform back to logits
	    epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
	    output = clip_ops.clip_by_value(output, epsilon, 1 - epsilon)
	    output = math_ops.log(output / (1 - output))
	  return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)


	def sigmoid(x):
	  """Element-wise sigmoid.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return nn.sigmoid(x)


	def hard_sigmoid(x):
	  """Segment-wise linear approximation of sigmoid.

	  Faster than sigmoid.
	  Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
	  In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  x = (0.2 * x) + 0.5
	  zero = _to_tensor(0., x.dtype.base_dtype)
	  one = _to_tensor(1., x.dtype.base_dtype)
	  x = clip_ops.clip_by_value(x, zero, one)
	  return x


	def tanh(x):
	  """Element-wise tanh.

	  Arguments:
	      x: A tensor or variable.

	  Returns:
	      A tensor.
	  """
	  return nn.tanh(x)


	def dropout(x, level, noise_shape=None, seed=None):
	  """Sets entries in `x` to zero at random, while scaling the entire tensor.

	  Arguments:
	      x: tensor
	      level: fraction of the entries in the tensor
	          that will be set to 0.
	      noise_shape: shape for randomly generated keep/drop flags,
	          must be broadcastable to the shape of `x`
	      seed: random seed to ensure determinism.

	  Returns:
	      A tensor.
	  """
	  retain_prob = 1. - level
	  if seed is None:
	    seed = np.random.randint(10e6)
	  # the dummy 1. works around a TF bug
	  # (float32_ref vs. float32 incomptability)
	  return nn.dropout(x * 1., retain_prob, noise_shape, seed=seed)


	def l2_normalize(x, axis):
	  """Normalizes a tensor wrt the L2 norm alongside the specified axis.

	  Arguments:
	      x: Tensor or variable.
	      axis: axis along which to perform normalization.

	  Returns:
	      A tensor.
	  """
	  if axis < 0:
	    axis %= len(x.get_shape())
	  return nn.l2_normalize(x, dim=axis)


	def in_top_k(predictions, targets, k):
	  """Returns whether the `targets` are in the top `k` `predictions`.

	  Arguments:
	      predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
	      targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
	      k: An `int`, number of top elements to consider.

	  Returns:
	      A 1D tensor of length `batch_size` and type `bool`.
	      `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
	      values of `predictions[i]`.
	  """
	  return nn.in_top_k(predictions, targets, k)


	# CONVOLUTIONS


	def _preprocess_deconv_output_shape(x, shape, data_format):
	  """Get the output_shape for the deconvolution.

	  Arguments:
	      x: input tensor.
	      shape: output shape.
	      data_format: string, one of 'channels_last', 'channels_first'.

	  Returns:
	      The output shape.
	  """
	  if data_format == 'channels_first':
	    shape = (shape[0], shape[2], shape[3], shape[1])

	  if shape[0] is None:
	    shape = (array_ops.shape(x)[0],) + tuple(shape[1:])
	    shape = array_ops.stack(list(shape))
	  return shape


	def _preprocess_conv2d_input(x, data_format):
	  """Transpose and cast the input before the conv2d.

	  Arguments:
	      x: input tensor.
	      data_format: string, one of 'channels_last', 'channels_first'.

	  Returns:
	      A tensor.
	  """
	  if dtype(x) == 'float64':
	    x = math_ops.cast(x, 'float32')
	  if data_format == 'channels_first':
	    # TF uses the last dimension as channel dimension,
	    # instead of the 2nd one.
	    # TH input shape: (samples, input_depth, rows, cols)
	    # TF input shape: (samples, rows, cols, input_depth)
	    x = array_ops.transpose(x, (0, 2, 3, 1))
	  return x


	def _preprocess_conv3d_input(x, data_format):
	  """Transpose and cast the input before the conv3d.

	  Arguments:
	      x: input tensor.
	      data_format: string, one of 'channels_last', 'channels_first'.

	  Returns:
	      A tensor.
	  """
	  if dtype(x) == 'float64':
	    x = math_ops.cast(x, 'float32')
	  if data_format == 'channels_first':
	    x = array_ops.transpose(x, (0, 2, 3, 4, 1))
	  return x


	def _preprocess_conv2d_kernel(kernel, data_format):
	  """Transpose and cast the kernel before the conv2d.

	  Arguments:
	      kernel: kernel tensor.
	      data_format: string, one of 'channels_last', 'channels_first'.

	  Returns:
	      A tensor.
	  """
	  if dtype(kernel) == 'float64':
	    kernel = math_ops.cast(kernel, 'float32')
	  if data_format == 'channels_first':
	    kernel = array_ops.transpose(kernel, (2, 3, 1, 0))
	  return kernel


	def _preprocess_conv3d_kernel(kernel, data_format):
	  """Transpose and cast the kernel before the conv3d.

	  Arguments:
	      kernel: kernel tensor.
	      data_format: string, one of 'channels_last', 'channels_first'.

	  Returns:
	      A tensor.
	  """
	  if dtype(kernel) == 'float64':
	    kernel = math_ops.cast(kernel, 'float32')
	  if data_format == 'channels_first':
	    kernel = array_ops.transpose(kernel, (2, 3, 4, 1, 0))
	  return kernel


	def _preprocess_padding(padding):
	  """Convert keras' padding to tensorflow's padding.

	  Arguments:
	      padding: string, one of 'same' , 'valid'

	  Returns:
	      a string, one of 'SAME', 'VALID'.

	  Raises:
	      ValueError: if invalid `padding'`
	  """
	  if padding == 'same':
	    padding = 'SAME'
	  elif padding == 'valid':
	    padding = 'VALID'
	  else:
	    raise ValueError('Invalid padding:', padding)
	  return padding


	def _postprocess_conv2d_output(x, data_format):
	  """Transpose and cast the output from conv2d if needed.

	  Arguments:
	      x: A tensor.
	      data_format: string, one of "channels_last", "channels_first".

	  Returns:
	      A tensor.
	  """

	  if data_format == 'channels_first':
	    x = array_ops.transpose(x, (0, 3, 1, 2))

	  if floatx() == 'float64':
	    x = math_ops.cast(x, 'float64')
	  return x


	def _postprocess_conv3d_output(x, data_format):
	  """Transpose and cast the output from conv3d if needed.

	  Arguments:
	      x: A tensor.
	      data_format: string, one of "channels_last", "channels_first".

	  Returns:
	      A tensor.
	  """
	  if data_format == 'channels_first':
	    x = array_ops.transpose(x, (0, 4, 1, 2, 3))

	  if floatx() == 'float64':
	    x = math_ops.cast(x, 'float64')
	  return x


	def conv1d(x,
	           kernel,
	           strides=1,
	           padding='valid',
	           data_format=None,
	           dilation_rate=1):
	  """1D convolution.

	  Arguments:
	      x: Tensor or variable.
	      kernel: kernel tensor.
	      strides: stride integer.
	      padding: string, `"same"`, `"causal"` or `"valid"`.
	      data_format: string, one of "channels_last", "channels_first".
	      dilation_rate: integer dilate rate.

	  Returns:
	      A tensor, result of 1D convolution.
	  """
	  kernel_shape = kernel.get_shape().as_list()
	  if padding == 'causal':
	    # causal (dilated) convolution:
	    left_pad = dilation_rate * (kernel_shape[0] - 1)
	    x = temporal_padding(x, (left_pad, 0))
	    padding = 'valid'
	  padding = _preprocess_padding(padding)
	  if data_format == 'channels_last':
	    tf_data_format = 'NWC'
	  else:
	    tf_data_format = 'NCW'
	  x = nn.convolution(
	      input=x,
	      filter=kernel,
	      dilation_rate=(dilation_rate,),
	      strides=(strides,),
	      padding=padding,
	      data_format=tf_data_format)
	  return x


	def conv2d(x,
	           kernel,
	           strides=(1, 1),
	           padding='valid',
	           data_format=None,
	           dilation_rate=(1, 1)):
	  """2D convolution.

	  Arguments:
	      x: Tensor or variable.
	      kernel: kernel tensor.
	      strides: strides tuple.
	      padding: string, `"same"` or `"valid"`.
	      data_format: `"channels_last"` or `"channels_first"`.
	          Whether to use Theano or TensorFlow data format
	          for inputs/kernels/ouputs.
	      dilation_rate: tuple of 2 integers.

	  Returns:
	      A tensor, result of 2D convolution.

	  Raises:
	      ValueError: if `data_format` is neither `channels_last` or
	      `channels_first`.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  # With 4d inputs, nn.convolution only supports
	  # data_format NHWC, so we transpose the inputs
	  # in case we are in data_format channels_first.
	  x = _preprocess_conv2d_input(x, data_format)
	  padding = _preprocess_padding(padding)
	  x = nn.convolution(
	      input=x,
	      filter=kernel,
	      dilation_rate=dilation_rate,
	      strides=strides,
	      padding=padding,
	      data_format='NHWC')
	  return _postprocess_conv2d_output(x, data_format)


	def conv2d_transpose(x,
	                     kernel,
	                     output_shape,
	                     strides=(1, 1),
	                     padding='valid',
	                     data_format=None):
	  """2D deconvolution (i.e.

	  transposed convolution).

	  Arguments:
	      x: Tensor or variable.
	      kernel: kernel tensor.
	      output_shape: 1D int tensor for the output shape.
	      strides: strides tuple.
	      padding: string, `"same"` or `"valid"`.
	      data_format: `"channels_last"` or `"channels_first"`.
	          Whether to use Theano or TensorFlow data format
	          for inputs/kernels/ouputs.

	  Returns:
	      A tensor, result of transposed 2D convolution.

	  Raises:
	      ValueError: if `data_format` is neither `channels_last` or
	      `channels_first`.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))
	  if isinstance(output_shape, (tuple, list)):
	    output_shape = array_ops.stack(output_shape)

	  x = _preprocess_conv2d_input(x, data_format)
	  output_shape = _preprocess_deconv_output_shape(x, output_shape, data_format)
	  padding = _preprocess_padding(padding)
	  strides = (1,) + strides + (1,)

	  x = nn.conv2d_transpose(x, kernel, output_shape, strides, padding=padding)
	  x = _postprocess_conv2d_output(x, data_format)
	  return x


	def separable_conv2d(x,
	                     depthwise_kernel,
	                     pointwise_kernel,
	                     strides=(1, 1),
	                     padding='valid',
	                     data_format=None,
	                     dilation_rate=(1, 1)):
	  """2D convolution with separable filters.

	  Arguments:
	      x: input tensor
	      depthwise_kernel: convolution kernel for the depthwise convolution.
	      pointwise_kernel: kernel for the 1x1 convolution.
	      strides: strides tuple (length 2).
	      padding: padding mode, "valid" or "same".
	      data_format: data format, "channels_first" or "channels_last".
	      dilation_rate: tuple of integers,
	          dilation rates for the separable convolution.

	  Returns:
	      Output tensor.

	  Raises:
	      ValueError: if `data_format` is neither `channels_last` or
	      `channels_first`.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  x = _preprocess_conv2d_input(x, data_format)
	  padding = _preprocess_padding(padding)
	  strides = (1,) + strides + (1,)

	  x = nn.separable_conv2d(
	      x,
	      depthwise_kernel,
	      pointwise_kernel,
	      strides=strides,
	      padding=padding,
	      rate=dilation_rate)
	  return _postprocess_conv2d_output(x, data_format)


	def conv3d(x,
	           kernel,
	           strides=(1, 1, 1),
	           padding='valid',
	           data_format=None,
	           dilation_rate=(1, 1, 1)):
	  """3D convolution.

	  Arguments:
	      x: Tensor or variable.
	      kernel: kernel tensor.
	      strides: strides tuple.
	      padding: string, `"same"` or `"valid"`.
	      data_format: `"channels_last"` or `"channels_first"`.
	          Whether to use Theano or TensorFlow data format
	          for inputs/kernels/ouputs.
	      dilation_rate: tuple of 3 integers.

	  Returns:
	      A tensor, result of 3D convolution.

	  Raises:
	      ValueError: if `data_format` is neither `channels_last` or
	      `channels_first`.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  # With 5d inputs, nn.convolution only supports
	  # data_format NDHWC, so we transpose the inputs
	  # in case we are in data_format channels_first.
	  x = _preprocess_conv3d_input(x, data_format)
	  padding = _preprocess_padding(padding)
	  x = nn.convolution(
	      input=x,
	      filter=kernel,
	      dilation_rate=dilation_rate,
	      strides=strides,
	      padding=padding,
	      data_format='NDHWC')
	  return _postprocess_conv3d_output(x, data_format)


	def pool2d(x,
	           pool_size,
	           strides=(1, 1),
	           padding='valid',
	           data_format=None,
	           pool_mode='max'):
	  """2D Pooling.

	  Arguments:
	      x: Tensor or variable.
	      pool_size: tuple of 2 integers.
	      strides: tuple of 2 integers.
	      padding: one of `"valid"`, `"same"`.
	      data_format: one of `"channels_first"`, `"channels_last"`.
	      pool_mode: one of `"max"`, `"avg"`.

	  Returns:
	      A tensor, result of 2D pooling.

	  Raises:
	      ValueError: if `data_format` is neither `channels_last` or
	      `channels_first`.
	      ValueError: if `pool_mode` is neither `max` or `avg`.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  padding = _preprocess_padding(padding)
	  strides = (1,) + strides + (1,)
	  pool_size = (1,) + pool_size + (1,)

	  x = _preprocess_conv2d_input(x, data_format)

	  if pool_mode == 'max':
	    x = nn.max_pool(x, pool_size, strides, padding=padding)
	  elif pool_mode == 'avg':
	    x = nn.avg_pool(x, pool_size, strides, padding=padding)
	  else:
	    raise ValueError('Invalid pooling mode:', pool_mode)

	  return _postprocess_conv2d_output(x, data_format)


	def pool3d(x,
	           pool_size,
	           strides=(1, 1, 1),
	           padding='valid',
	           data_format=None,
	           pool_mode='max'):
	  """3D Pooling.

	  Arguments:
	      x: Tensor or variable.
	      pool_size: tuple of 3 integers.
	      strides: tuple of 3 integers.
	      padding: one of `"valid"`, `"same"`.
	      data_format: one of `"channels_first"`, `"channels_last"`.
	      pool_mode: one of `"max"`, `"avg"`.

	  Returns:
	      A tensor, result of 3D pooling.

	  Raises:
	      ValueError: if `data_format` is neither
	          `channels_last` or `channels_first`.
	      ValueError: if `pool_mode` is neither `max` or `avg`.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))

	  padding = _preprocess_padding(padding)
	  strides = (1,) + strides + (1,)
	  pool_size = (1,) + pool_size + (1,)

	  x = _preprocess_conv3d_input(x, data_format)

	  if pool_mode == 'max':
	    x = nn.max_pool3d(x, pool_size, strides, padding=padding)
	  elif pool_mode == 'avg':
	    x = nn.avg_pool3d(x, pool_size, strides, padding=padding)
	  else:
	    raise ValueError('Invalid pooling mode:', pool_mode)

	  return _postprocess_conv3d_output(x, data_format)


	def bias_add(x, bias, data_format=None):
	  """Adds a bias vector to a tensor.

	  Arguments:
	      x: Tensor or variable.
	      bias: Bias tensor to add.
	      data_format: Data format for 3D, 4D or 5D tensors:
	          one of "channels_first", "channels_last".

	  Returns:
	      Output tensor.

	  Raises:
	      ValueError: In case of invalid `data_format` argument.
	  """
	  if data_format is None:
	    data_format = image_data_format()
	  if data_format not in {'channels_first', 'channels_last'}:
	    raise ValueError('Unknown data_format ' + str(data_format))
	  if ndim(x) == 5:
	    if data_format == 'channels_first':
	      x += reshape(bias, (1, int_shape(bias)[0], 1, 1, 1))
	    elif data_format == 'channels_last':
	      x += reshape(bias, (1, 1, 1, 1, int_shape(bias)[0]))
	  elif ndim(x) == 4:
	    if data_format == 'channels_first':
	      # No support yet for NCHW in bias_add.
	      x += reshape(bias, (1, int_shape(bias)[0], 1, 1))
	    elif data_format == 'channels_last':
	      x = nn.bias_add(x, bias, data_format='NHWC')
	  elif ndim(x) == 3:
	    if data_format == 'channels_first':
	      x += reshape(bias, (1, int_shape(bias)[0], 1))
	    elif data_format == 'channels_last':
	      x += reshape(bias, (1, 1, int_shape(bias)[0]))
	  else:
	    x = nn.bias_add(x, bias)
	  return x


	# RANDOMNESS


	def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
	  """Returns a tensor with normal distribution of values.

	  Arguments:
	      shape: A tuple of integers, the shape of tensor to create.
	      mean: A float, mean of the normal distribution to draw samples.
	      stddev: A float, standard deviation of the normal distribution
	          to draw samples.
	      dtype: String, dtype of returned tensor.
	      seed: Integer, random seed.

	  Returns:
	      A tensor.
	  """
	  if dtype is None:
	    dtype = floatx()
	  if seed is None:
	    seed = np.random.randint(10e6)
	  return random_ops.random_normal(
	      shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


	def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
	  """Returns a tensor with uniform distribution of values.

	  Arguments:
	      shape: A tuple of integers, the shape of tensor to create.
	      minval: A float, lower boundary of the uniform distribution
	          to draw samples.
	      maxval: A float, upper boundary of the uniform distribution
	          to draw samples.
	      dtype: String, dtype of returned tensor.
	      seed: Integer, random seed.

	  Returns:
	      A tensor.
	  """
	  if dtype is None:
	    dtype = floatx()
	  if seed is None:
	    seed = np.random.randint(10e6)
	  return random_ops.random_uniform(
	      shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


	def random_binomial(shape, p=0.0, dtype=None, seed=None):
	  """Returns a tensor with random binomial distribution of values.

	  Arguments:
	      shape: A tuple of integers, the shape of tensor to create.
	      p: A float, `0. <= p <= 1`, probability of binomial distribution.
	      dtype: String, dtype of returned tensor.
	      seed: Integer, random seed.

	  Returns:
	      A tensor.
	  """
	  if dtype is None:
	    dtype = floatx()
	  if seed is None:
	    seed = np.random.randint(10e6)
	  return array_ops.where(
	      random_ops.random_uniform(shape, dtype=dtype, seed=seed) <= p,
	      array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))


	def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
	  """Returns a tensor with truncated random normal distribution of values.

	  The generated values follow a normal distribution
	  with specified mean and standard deviation,
	  except that values whose magnitude is more than
	  two standard deviations from the mean are dropped and re-picked.

	  Arguments:
	      shape: A tuple of integers, the shape of tensor to create.
	      mean: Mean of the values.
	      stddev: Standard deviation of the values.
	      dtype: String, dtype of returned tensor.
	      seed: Integer, random seed.

	  Returns:
	      A tensor.
	  """
	  if dtype is None:
	    dtype = floatx()
	  if seed is None:
	    seed = np.random.randint(10e6)
	  return random_ops.truncated_normal(
	      shape, mean, stddev, dtype=dtype, seed=seed)


	# CTC
	# tensorflow has a native implemenation, but it uses sparse tensors
	# and therefore requires a wrapper for Keras. The functions below convert
	# dense to sparse tensors and also wraps up the beam search code that is
	# in tensorflow's CTC implementation


	def ctc_label_dense_to_sparse(labels, label_lengths):
	  """Converts CTC labels from dense to sparse.

	  Arguments:
	      labels: dense CTC labels.
	      label_lengths: length of the labels.

	  Returns:
	      A sparse tensor representation of the lablels.
	  """
	  label_shape = array_ops.shape(labels)
	  num_batches_tns = array_ops.stack([label_shape[0]])
	  max_num_labels_tns = array_ops.stack([label_shape[1]])

	  def range_less_than(_, current_input):
	    return array_ops.expand_dims(
	        math_ops.range(label_shape[1]), 0) < array_ops.fill(
	            max_num_labels_tns, current_input)

	  init = math_ops.cast(
	      array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
	  dense_mask = functional_ops.scan(
	      range_less_than, label_lengths, initializer=init, parallel_iterations=1)
	  dense_mask = dense_mask[:, 0, :]

	  label_array = array_ops.reshape(
	      array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns),
	      label_shape)
	  label_ind = array_ops.boolean_mask(label_array, dense_mask)

	  batch_array = array_ops.transpose(
	      array_ops.reshape(
	          array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns),
	          reverse(label_shape, 0)))
	  batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
	  indices = array_ops.transpose(
	      array_ops.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))

	  vals_sparse = array_ops.gather_nd(labels, indices)

	  return sparse_tensor.SparseTensor(
	      math_ops.to_int64(indices), vals_sparse, math_ops.to_int64(label_shape))


	def ctc_batch_cost(y_true, y_pred, input_length, label_length):
	  """Runs CTC loss algorithm on each batch element.

	  Arguments:
	      y_true: tensor `(samples, max_string_length)`
	          containing the truth labels.
	      y_pred: tensor `(samples, time_steps, num_categories)`
	          containing the prediction, or output of the softmax.
	      input_length: tensor `(samples, 1)` containing the sequence length for
	          each batch item in `y_pred`.
	      label_length: tensor `(samples, 1)` containing the sequence length for
	          each batch item in `y_true`.

	  Returns:
	      Tensor with shape (samples,1) containing the
	          CTC loss of each element.
	  """
	  label_length = math_ops.to_int32(array_ops.squeeze(label_length))
	  input_length = math_ops.to_int32(array_ops.squeeze(input_length))
	  sparse_labels = math_ops.to_int32(
	      ctc_label_dense_to_sparse(y_true, label_length))

	  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)

	  return array_ops.expand_dims(
	      ctc.ctc_loss(
	          inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)


	def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
	  """Decodes the output of a softmax.

	  Can use either greedy search (also known as best path)
	  or a constrained dictionary search.

	  Arguments:
	      y_pred: tensor `(samples, time_steps, num_categories)`
	          containing the prediction, or output of the softmax.
	      input_length: tensor `(samples, )` containing the sequence length for
	          each batch item in `y_pred`.
	      greedy: perform much faster best-path search if `true`.
	          This does not use a dictionary.
	      beam_width: if `greedy` is `false`: a beam search decoder will be used
	          with a beam of this width.
	      top_paths: if `greedy` is `false`,
	          how many of the most probable paths will be returned.

	  Returns:
	      Tuple:
	          List: if `greedy` is `true`, returns a list of one element that
	              contains the decoded sequence.
	              If `false`, returns the `top_paths` most probable
	              decoded sequences.
	              Important: blank labels are returned as `-1`.
	          Tensor `(top_paths, )` that contains
	              the log probability of each decoded sequence.
	  """
	  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
	  input_length = math_ops.to_int32(input_length)

	  if greedy:
	    (decoded, log_prob) = ctc.ctc_greedy_decoder(
	        inputs=y_pred, sequence_length=input_length)
	  else:
	    (decoded, log_prob) = ctc.ctc_beam_search_decoder(
	        inputs=y_pred,
	        sequence_length=input_length,
	        beam_width=beam_width,
	        top_paths=top_paths)
	  decoded_dense = [
	      sparse_ops.sparse_to_dense(
	          st.indices, st.dense_shape, st.values, default_value=-1)
	      for st in decoded
	  ]
	  return (decoded_dense, log_prob)


	# HIGH ORDER FUNCTIONS


	def map_fn(fn, elems, name=None, dtype=None):
	  """Map the function fn over the elements elems and return the outputs.

	  Arguments:
	      fn: Callable that will be called upon each element in elems
	      elems: tensor
	      name: A string name for the map node in the graph
	      dtype: Output data type.

	  Returns:
	      Tensor with dtype `dtype`.
	  """
	  return functional_ops.map_fn(fn, elems, name=name, dtype=dtype)


	def foldl(fn, elems, initializer=None, name=None):
	  """Reduce elems using fn to combine them from left to right.

	  Arguments:
	      fn: Callable that will be called upon each element in elems and an
	          accumulator, for instance `lambda acc, x: acc + x`
	      elems: tensor
	      initializer: The first value used (`elems[0]` in case of None)
	      name: A string name for the foldl node in the graph

	  Returns:
	      Tensor with same type and shape as `initializer`.
	  """
	  return functional_ops.foldl(fn, elems, initializer=initializer, name=name)


	def foldr(fn, elems, initializer=None, name=None):
	  """Reduce elems using fn to combine them from right to left.

	  Arguments:
	      fn: Callable that will be called upon each element in elems and an
	          accumulator, for instance `lambda acc, x: acc + x`
	      elems: tensor
	      initializer: The first value used (`elems[-1]` in case of None)
	      name: A string name for the foldr node in the graph

	  Returns:
	      Same type and shape as initializer
	  """
	  return functional_ops.foldr(fn, elems, initializer=initializer, name=name)


	# Load Keras default configuration from config file if present.
	_keras_base_dir = os.path.expanduser('~')
	_keras_dir = os.path.join(_keras_base_dir, '.keras')
	_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
	if os.path.exists(_config_path):
	  try:
	    _config = json.load(open(_config_path))
	  except ValueError:
	    _config = {}
	  _floatx = _config.get('floatx', floatx())
	  assert _floatx in {'float16', 'float32', 'float64'}
	  _epsilon = _config.get('epsilon', epsilon())
	  assert isinstance(_epsilon, float)
	  _image_data_format = _config.get('image_data_format', image_data_format())
	  assert _image_data_format in {'channels_last', 'channels_first'}
	  set_floatx(_floatx)
	  set_epsilon(_epsilon)
	  set_image_data_format(_image_data_format)

	# Save config file.
	if not os.path.exists(_keras_dir):
	  try:
	    os.makedirs(_keras_dir)
	  except OSError:
	    # Except permission denied and potential race conditions
	    # in multi-threaded environments.
	    pass

	if not os.path.exists(_config_path):
	  _config = {
	      'floatx': floatx(),
	      'epsilon': epsilon(),
	      'backend': 'tensorflow',
	      'image_data_format': image_data_format()
	  }
	  try:
	    with open(_config_path, 'w') as f:
	      f.write(json.dumps(_config, indent=4))
	  except IOError:
	    # Except permission denied.
	    pass

## callbacks.py
class callbacks_py:

	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

	"""Keras callbacks: utilities called at certain points during model training.
	"""

	def import_libs():
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from collections import deque
		from collections import Iterable
		from collections import OrderedDict
		import csv
		import json
		import os
		import time

		import numpy as np
		import six

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar
		from tensorflow.contrib.tensorboard.plugins import projector
		from tensorflow.python.ops import array_ops
		from tensorflow.python.platform import tf_logging as logging
		from tensorflow.python.summary import summary as tf_summary
		from tensorflow.python.training import saver as saver_lib


	# pylint: disable=g-import-not-at-top
	try:
	  import requests
	except ImportError:
	  requests = None
	# pylint: enable=g-import-not-at-top


	class CallbackList(object):
	  """Container abstracting a list of callbacks.

	  Arguments:
	      callbacks: List of `Callback` instances.
	      queue_length: Queue length for keeping
	          running statistics over callback execution time.
	  """

	  def __init__(self, callbacks=None, queue_length=10):
	    callbacks = callbacks or []
	    self.callbacks = [c for c in callbacks]
	    self.queue_length = queue_length

	  def append(self, callback):
	    self.callbacks.append(callback)

	  def set_params(self, params):
	    for callback in self.callbacks:
	      callback.set_params(params)

	  def set_model(self, model):
	    for callback in self.callbacks:
	      callback.set_model(model)

	  def on_epoch_begin(self, epoch, logs=None):
	    """Called at the start of an epoch.

	    Arguments:
	        epoch: integer, index of epoch.
	        logs: dictionary of logs.
	    """
	    logs = logs or {}
	    for callback in self.callbacks:
	      callback.on_epoch_begin(epoch, logs)
	    self._delta_t_batch = 0.
	    self._delta_ts_batch_begin = deque([], maxlen=self.queue_length)
	    self._delta_ts_batch_end = deque([], maxlen=self.queue_length)

	  def on_epoch_end(self, epoch, logs=None):
	    """Called at the end of an epoch.

	    Arguments:
	        epoch: integer, index of epoch.
	        logs: dictionary of logs.
	    """
	    logs = logs or {}
	    for callback in self.callbacks:
	      callback.on_epoch_end(epoch, logs)

	  def on_batch_begin(self, batch, logs=None):
	    """Called right before processing a batch.

	    Arguments:
	        batch: integer, index of batch within the current epoch.
	        logs: dictionary of logs.
	    """
	    logs = logs or {}
	    t_before_callbacks = time.time()
	    for callback in self.callbacks:
	      callback.on_batch_begin(batch, logs)
	    self._delta_ts_batch_begin.append(time.time() - t_before_callbacks)
	    delta_t_median = np.median(self._delta_ts_batch_begin)
	    if (self._delta_t_batch > 0. and
	        delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1):
	      logging.warning(
	          'Method on_batch_begin() is slow compared '
	          'to the batch update (%f). Check your callbacks.' % delta_t_median)
	    self._t_enter_batch = time.time()

	  def on_batch_end(self, batch, logs=None):
	    """Called at the end of a batch.

	    Arguments:
	        batch: integer, index of batch within the current epoch.
	        logs: dictionary of logs.
	    """
	    logs = logs or {}
	    if not hasattr(self, '_t_enter_batch'):
	      self._t_enter_batch = time.time()
	    self._delta_t_batch = time.time() - self._t_enter_batch
	    t_before_callbacks = time.time()
	    for callback in self.callbacks:
	      callback.on_batch_end(batch, logs)
	    self._delta_ts_batch_end.append(time.time() - t_before_callbacks)
	    delta_t_median = np.median(self._delta_ts_batch_end)
	    if (self._delta_t_batch > 0. and
	        (delta_t_median > 0.95 * self._delta_t_batch and delta_t_median > 0.1)):
	      logging.warning(
	          'Method on_batch_end() is slow compared '
	          'to the batch update (%f). Check your callbacks.' % delta_t_median)

	  def on_train_begin(self, logs=None):
	    """Called at the beginning of training.

	    Arguments:
	        logs: dictionary of logs.
	    """
	    logs = logs or {}
	    for callback in self.callbacks:
	      callback.on_train_begin(logs)

	  def on_train_end(self, logs=None):
	    """Called at the end of training.

	    Arguments:
	        logs: dictionary of logs.
	    """
	    logs = logs or {}
	    for callback in self.callbacks:
	      callback.on_train_end(logs)

	  def __iter__(self):
	    return iter(self.callbacks)


	class Callback(object):
	  """Abstract base class used to build new callbacks.

	  # Properties
	      params: dict. Training parameters
	          (eg. verbosity, batch size, number of epochs...).
	      model: instance of `keras.models.Model`.
	          Reference of the model being trained.

	  The `logs` dictionary that callback methods
	  take as argument will contain keys for quantities relevant to
	  the current batch or epoch.

	  Currently, the `.fit()` method of the `Sequential` model class
	  will include the following quantities in the `logs` that
	  it passes to its callbacks:

	      on_epoch_end: logs include `acc` and `loss`, and
	          optionally include `val_loss`
	          (if validation is enabled in `fit`), and `val_acc`
	          (if validation and accuracy monitoring are enabled).
	      on_batch_begin: logs include `size`,
	          the number of samples in the current batch.
	      on_batch_end: logs include `loss`, and optionally `acc`
	          (if accuracy monitoring is enabled).
	  """

	  def __init__(self):
	    self.validation_data = None

	  def set_params(self, params):
	    self.params = params

	  def set_model(self, model):
	    self.model = model

	  def on_epoch_begin(self, epoch, logs=None):
	    pass

	  def on_epoch_end(self, epoch, logs=None):
	    pass

	  def on_batch_begin(self, batch, logs=None):
	    pass

	  def on_batch_end(self, batch, logs=None):
	    pass

	  def on_train_begin(self, logs=None):
	    pass

	  def on_train_end(self, logs=None):
	    pass


	class BaseLogger(Callback):
	  """Callback that accumulates epoch averages of metrics.

	  This callback is automatically applied to every Keras model.
	  """

	  def on_epoch_begin(self, epoch, logs=None):
	    self.seen = 0
	    self.totals = {}

	  def on_batch_end(self, batch, logs=None):
	    logs = logs or {}
	    batch_size = logs.get('size', 0)
	    self.seen += batch_size

	    for k, v in logs.items():
	      if k in self.totals:
	        self.totals[k] += v * batch_size
	      else:
	        self.totals[k] = v * batch_size

	  def on_epoch_end(self, epoch, logs=None):
	    if logs is not None:
	      for k in self.params['metrics']:
	        if k in self.totals:
	          # Make value available to next callbacks.
	          logs[k] = self.totals[k] / self.seen


	class TerminateOnNaN(Callback):
	  """Callback that terminates training when a NaN loss is encountered."""

	  def __init__(self):
	    super(TerminateOnNaN, self).__init__()

	  def on_batch_end(self, batch, logs=None):
	    logs = logs or {}
	    loss = logs.get('loss')
	    if loss is not None:
	      if np.isnan(loss) or np.isinf(loss):
	        print('Batch %d: Invalid loss, terminating training' % (batch))
	        self.model.stop_training = True


	class ProgbarLogger(Callback):
	  """Callback that prints metrics to stdout.

	  Arguments:
	      count_mode: One of "steps" or "samples".
	          Whether the progress bar should
	          count samples seens or steps (batches) seen.

	  Raises:
	      ValueError: In case of invalid `count_mode`.
	  """

	  def __init__(self, count_mode='samples'):
	    super(ProgbarLogger, self).__init__()
	    if count_mode == 'samples':
	      self.use_steps = False
	    elif count_mode == 'steps':
	      self.use_steps = True
	    else:
	      raise ValueError('Unknown `count_mode`: ' + str(count_mode))

	  def on_train_begin(self, logs=None):
	    self.verbose = self.params['verbose']
	    self.epochs = self.params['epochs']

	  def on_epoch_begin(self, epoch, logs=None):
	    if self.verbose:
	      print('Epoch %d/%d' % (epoch + 1, self.epochs))
	      if self.use_steps:
	        target = self.params['steps']
	      else:
	        target = self.params['samples']
	      self.target = target
	      self.progbar = Progbar(target=self.target, verbose=self.verbose)
	    self.seen = 0

	  def on_batch_begin(self, batch, logs=None):
	    if self.seen < self.target:
	      self.log_values = []

	  def on_batch_end(self, batch, logs=None):
	    logs = logs or {}
	    batch_size = logs.get('size', 0)
	    if self.use_steps:
	      self.seen += 1
	    else:
	      self.seen += batch_size

	    for k in self.params['metrics']:
	      if k in logs:
	        self.log_values.append((k, logs[k]))

	    # Skip progbar update for the last batch;
	    # will be handled by on_epoch_end.
	    if self.verbose and self.seen < self.target:
	      self.progbar.update(self.seen, self.log_values)

	  def on_epoch_end(self, epoch, logs=None):
	    logs = logs or {}
	    for k in self.params['metrics']:
	      if k in logs:
	        self.log_values.append((k, logs[k]))
	    if self.verbose:
	      self.progbar.update(self.seen, self.log_values, force=True)


	class History(Callback):
	  """Callback that records events into a `History` object.

	  This callback is automatically applied to
	  every Keras model. The `History` object
	  gets returned by the `fit` method of models.
	  """

	  def on_train_begin(self, logs=None):
	    self.epoch = []
	    self.history = {}

	  def on_epoch_end(self, epoch, logs=None):
	    logs = logs or {}
	    self.epoch.append(epoch)
	    for k, v in logs.items():
	      self.history.setdefault(k, []).append(v)


	class ModelCheckpoint(Callback):
	  """Save the model after every epoch.

	  `filepath` can contain named formatting options,
	  which will be filled the value of `epoch` and
	  keys in `logs` (passed in `on_epoch_end`).

	  For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
	  then the model checkpoints will be saved with the epoch number and
	  the validation loss in the filename.

	  Arguments:
	      filepath: string, path to save the model file.
	      monitor: quantity to monitor.
	      verbose: verbosity mode, 0 or 1.
	      save_best_only: if `save_best_only=True`,
	          the latest best model according to
	          the quantity monitored will not be overwritten.
	      mode: one of {auto, min, max}.
	          If `save_best_only=True`, the decision
	          to overwrite the current save file is made
	          based on either the maximization or the
	          minimization of the monitored quantity. For `val_acc`,
	          this should be `max`, for `val_loss` this should
	          be `min`, etc. In `auto` mode, the direction is
	          automatically inferred from the name of the monitored quantity.
	      save_weights_only: if True, then only the model's weights will be
	          saved (`model.save_weights(filepath)`), else the full model
	          is saved (`model.save(filepath)`).
	      period: Interval (number of epochs) between checkpoints.
	  """

	  def __init__(self,
	               filepath,
	               monitor='val_loss',
	               verbose=0,
	               save_best_only=False,
	               save_weights_only=False,
	               mode='auto',
	               period=1):
	    super(ModelCheckpoint, self).__init__()
	    self.monitor = monitor
	    self.verbose = verbose
	    self.filepath = filepath
	    self.save_best_only = save_best_only
	    self.save_weights_only = save_weights_only
	    self.period = period
	    self.epochs_since_last_save = 0

	    if mode not in ['auto', 'min', 'max']:
	      logging.warning('ModelCheckpoint mode %s is unknown, '
	                      'fallback to auto mode.' % (mode))
	      mode = 'auto'

	    if mode == 'min':
	      self.monitor_op = np.less
	      self.best = np.Inf
	    elif mode == 'max':
	      self.monitor_op = np.greater
	      self.best = -np.Inf
	    else:
	      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
	        self.monitor_op = np.greater
	        self.best = -np.Inf
	      else:
	        self.monitor_op = np.less
	        self.best = np.Inf

	  def on_epoch_end(self, epoch, logs=None):
	    logs = logs or {}
	    self.epochs_since_last_save += 1
	    if self.epochs_since_last_save >= self.period:
	      self.epochs_since_last_save = 0
	      filepath = self.filepath.format(epoch=epoch, **logs)
	      if self.save_best_only:
	        current = logs.get(self.monitor)
	        if current is None:
	          logging.warning('Can save best model only with %s available, '
	                          'skipping.' % (self.monitor))
	        else:
	          if self.monitor_op(current, self.best):
	            if self.verbose > 0:
	              print('Epoch %05d: %s improved from %0.5f to %0.5f,'
	                    ' saving model to %s' % (epoch, self.monitor, self.best,
	                                             current, filepath))
	            self.best = current
	            if self.save_weights_only:
	              self.model.save_weights(filepath, overwrite=True)
	            else:
	              self.model.save(filepath, overwrite=True)
	          else:
	            if self.verbose > 0:
	              print('Epoch %05d: %s did not improve' % (epoch, self.monitor))
	      else:
	        if self.verbose > 0:
	          print('Epoch %05d: saving model to %s' % (epoch, filepath))
	        if self.save_weights_only:
	          self.model.save_weights(filepath, overwrite=True)
	        else:
	          self.model.save(filepath, overwrite=True)


	class EarlyStopping(Callback):
	  """Stop training when a monitored quantity has stopped improving.

	  Arguments:
	      monitor: quantity to be monitored.
	      min_delta: minimum change in the monitored quantity
	          to qualify as an improvement, i.e. an absolute
	          change of less than min_delta, will count as no
	          improvement.
	      patience: number of epochs with no improvement
	          after which training will be stopped.
	      verbose: verbosity mode.
	      mode: one of {auto, min, max}. In `min` mode,
	          training will stop when the quantity
	          monitored has stopped decreasing; in `max`
	          mode it will stop when the quantity
	          monitored has stopped increasing; in `auto`
	          mode, the direction is automatically inferred
	          from the name of the monitored quantity.
	  """

	  def __init__(self,
	               monitor='val_loss',
	               min_delta=0,
	               patience=0,
	               verbose=0,
	               mode='auto'):
	    super(EarlyStopping, self).__init__()

	    self.monitor = monitor
	    self.patience = patience
	    self.verbose = verbose
	    self.min_delta = min_delta
	    self.wait = 0
	    self.stopped_epoch = 0

	    if mode not in ['auto', 'min', 'max']:
	      logging.warning('EarlyStopping mode %s is unknown, '
	                      'fallback to auto mode.' % (self.mode))
	      mode = 'auto'

	    if mode == 'min':
	      self.monitor_op = np.less
	    elif mode == 'max':
	      self.monitor_op = np.greater
	    else:
	      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
	        self.monitor_op = np.greater
	      else:
	        self.monitor_op = np.less

	    if self.monitor_op == np.greater:
	      self.min_delta *= 1
	    else:
	      self.min_delta *= -1

	  def on_train_begin(self, logs=None):
	    # Allow instances to be re-used
	    self.wait = 0
	    self.stopped_epoch = 0
	    self.best = np.Inf if self.monitor_op == np.less else -np.Inf

	  def on_epoch_end(self, epoch, logs=None):
	    current = logs.get(self.monitor)
	    if current is None:
	      logging.warning('Early stopping requires %s available!' % (self.monitor))

	    if self.monitor_op(current - self.min_delta, self.best):
	      self.best = current
	      self.wait = 0
	    else:
	      if self.wait >= self.patience:
	        self.stopped_epoch = epoch
	        self.model.stop_training = True
	      self.wait += 1

	  def on_train_end(self, logs=None):
	    if self.stopped_epoch > 0 and self.verbose > 0:
	      print('Epoch %05d: early stopping' % (self.stopped_epoch))


	class RemoteMonitor(Callback):
	  """Callback used to stream events to a server.

	  Requires the `requests` library.
	  Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
	  HTTP POST, with a `data` argument which is a
	  JSON-encoded dictionary of event data.

	  Arguments:
	      root: String; root url of the target server.
	      path: String; path relative to `root` to which the events will be sent.
	      field: String; JSON field under which the data will be stored.
	      headers: Dictionary; optional custom HTTP headers.
	          Defaults to:
	          `{'Accept': 'application/json', 'Content-Type': 'application/json'}`
	  """

	  def __init__(self,
	               root='http://localhost:9000',
	               path='/publish/epoch/end/',
	               field='data',
	               headers=None):
	    super(RemoteMonitor, self).__init__()
	    if headers is None:
	      headers = {
	          'Accept': 'application/json',
	          'Content-Type': 'application/json'
	      }
	    self.root = root
	    self.path = path
	    self.field = field
	    self.headers = headers

	  def on_epoch_end(self, epoch, logs=None):
	    if requests is None:
	      raise ImportError('RemoteMonitor requires ' 'the `requests` library.')
	    logs = logs or {}
	    send = {}
	    send['epoch'] = epoch
	    for k, v in logs.items():
	      send[k] = v
	    try:
	      requests.post(
	          self.root + self.path, {self.field: json.dumps(send)},
	          headers=self.headers)
	    except requests.exceptions.RequestException:
	      logging.warning('Warning: could not reach RemoteMonitor '
	                      'root server at ' + str(self.root))


	class LearningRateScheduler(Callback):
	  """Learning rate scheduler.

	  Arguments:
	      schedule: a function that takes an epoch index as input
	          (integer, indexed from 0) and returns a new
	          learning rate as output (float).
	  """

	  def __init__(self, schedule):
	    super(LearningRateScheduler, self).__init__()
	    self.schedule = schedule

	  def on_epoch_begin(self, epoch, logs=None):
	    if not hasattr(self.model.optimizer, 'lr'):
	      raise ValueError('Optimizer must have a "lr" attribute.')
	    lr = self.schedule(epoch)
	    if not isinstance(lr, (float, np.float32, np.float64)):
	      raise ValueError('The output of the "schedule" function '
	                       'should be float.')
	    K.set_value(self.model.optimizer.lr, lr)


	class TensorBoard(Callback):
	  # pylint: disable=line-too-long
	  """Tensorboard basic visualizations.

	  This callback writes a log for TensorBoard, which allows
	  you to visualize dynamic graphs of your training and test
	  metrics, as well as activation histograms for the different
	  layers in your model.

	  TensorBoard is a visualization tool provided with TensorFlow.

	  If you have installed TensorFlow with pip, you should be able
	  to launch TensorBoard from the command line:

	  ```
	  tensorboard --logdir=/full_path_to_your_logs
	  ```

	  You can find more information about TensorBoard
	  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

	  Arguments:
	      log_dir: the path of the directory where to save the log
	          files to be parsed by TensorBoard.
	      histogram_freq: frequency (in epochs) at which to compute activation
	          and weight histograms for the layers of the model. If set to 0,
	          histograms won't be computed. Validation data (or split) must be
	          specified for histogram visualizations.
	      write_graph: whether to visualize the graph in TensorBoard.
	          The log file can become quite large when
	          write_graph is set to True.
	      write_grads: whether to visualize gradient histograms in TensorBoard.
	          `histogram_freq` must be greater than 0.
	      batch_size: size of batch of inputs to feed to the network
	          for histograms computation.
	      write_images: whether to write model weights to visualize as
	          image in TensorBoard.
	      embeddings_freq: frequency (in epochs) at which selected embedding
	          layers will be saved.
	      embeddings_layer_names: a list of names of layers to keep eye on. If
	          None or empty list all the embedding layer will be watched.
	      embeddings_metadata: a dictionary which maps layer name to a file name
	          in which metadata for this embedding layer is saved. See the
	          [details](https://www.tensorflow.org/how_tos/embedding_viz/#metadata_optional)
	          about metadata files format. In case if the same metadata file is
	          used for all embedding layers, string can be passed.
	  """

	  # pylint: enable=line-too-long

	  def __init__(self,
	               log_dir='./logs',
	               histogram_freq=0,
	               batch_size=32,
	               write_graph=True,
	               write_grads=False,
	               write_images=False,
	               embeddings_freq=0,
	               embeddings_layer_names=None,
	               embeddings_metadata=None):
	    super(TensorBoard, self).__init__()
	    self.log_dir = log_dir
	    self.histogram_freq = histogram_freq
	    self.merged = None
	    self.write_graph = write_graph
	    self.write_grads = write_grads
	    self.write_images = write_images
	    self.embeddings_freq = embeddings_freq
	    self.embeddings_layer_names = embeddings_layer_names
	    self.embeddings_metadata = embeddings_metadata or {}
	    self.batch_size = batch_size

	  def set_model(self, model):
	    self.model = model
	    self.sess = K.get_session()
	    if self.histogram_freq and self.merged is None:
	      for layer in self.model.layers:
	        for weight in layer.weights:
	          tf_summary.histogram(weight.name, weight)
	          if self.write_grads:
	            grads = model.optimizer.get_gradients(model.total_loss, weight)
	            tf_summary.histogram('{}_grad'.format(weight.name), grads)
	          if self.write_images:
	            w_img = array_ops.squeeze(weight)
	            shape = K.int_shape(w_img)
	            if len(shape) == 2:  # dense layer kernel case
	              if shape[0] > shape[1]:
	                w_img = array_ops.transpose(w_img)
	                shape = K.int_shape(w_img)
	              w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
	            elif len(shape) == 3:  # convnet case
	              if K.image_data_format() == 'channels_last':
	                # switch to channels_first to display
	                # every kernel as a separate image
	                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
	                shape = K.int_shape(w_img)
	              w_img = array_ops.reshape(w_img,
	                                        [shape[0], shape[1], shape[2], 1])
	            elif len(shape) == 1:  # bias case
	              w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
	            else:
	              # not possible to handle 3D convnets etc.
	              continue

	            shape = K.int_shape(w_img)
	            assert len(shape) == 4 and shape[-1] in [1, 3, 4]
	            tf_summary.image(weight.name, w_img)

	        if hasattr(layer, 'output'):
	          tf_summary.histogram('{}_out'.format(layer.name), layer.output)
	    self.merged = tf_summary.merge_all()

	    if self.write_graph:
	      self.writer = tf_summary.FileWriter(self.log_dir, self.sess.graph)
	    else:
	      self.writer = tf_summary.FileWriter(self.log_dir)

	    if self.embeddings_freq:
	      embeddings_layer_names = self.embeddings_layer_names

	      if not embeddings_layer_names:
	        embeddings_layer_names = [
	            layer.name for layer in self.model.layers
	            if type(layer).__name__ == 'Embedding'
	        ]

	      embeddings = {
	          layer.name: layer.weights[0]
	          for layer in self.model.layers if layer.name in embeddings_layer_names
	      }

	      self.saver = saver_lib.Saver(list(embeddings.values()))

	      embeddings_metadata = {}

	      if not isinstance(self.embeddings_metadata, str):
	        embeddings_metadata = self.embeddings_metadata
	      else:
	        embeddings_metadata = {
	            layer_name: self.embeddings_metadata
	            for layer_name in embeddings.keys()
	        }

	      config = projector.ProjectorConfig()
	      self.embeddings_ckpt_path = os.path.join(self.log_dir,
	                                               'keras_embedding.ckpt')

	      for layer_name, tensor in embeddings.items():
	        embedding = config.embeddings.add()
	        embedding.tensor_name = tensor.name

	        if layer_name in embeddings_metadata:
	          embedding.metadata_path = embeddings_metadata[layer_name]

	      projector.visualize_embeddings(self.writer, config)

	  def on_epoch_end(self, epoch, logs=None):
	    logs = logs or {}

	    if self.validation_data and self.histogram_freq:
	      if epoch % self.histogram_freq == 0:

	        val_data = self.validation_data
	        tensors = (
	            self.model.inputs + self.model.targets + self.model.sample_weights)

	        if self.model.uses_learning_phase:
	          tensors += [K.learning_phase()]

	        assert len(val_data) == len(tensors)
	        val_size = val_data[0].shape[0]
	        i = 0
	        while i < val_size:
	          step = min(self.batch_size, val_size - i)
	          batch_val = []
	          batch_val.append(val_data[0][i:i + step])
	          batch_val.append(val_data[1][i:i + step])
	          batch_val.append(val_data[2][i:i + step])
	          if self.model.uses_learning_phase:
	            batch_val.append(val_data[3])
	          feed_dict = dict(zip(tensors, batch_val))
	          result = self.sess.run([self.merged], feed_dict=feed_dict)
	          summary_str = result[0]
	          self.writer.add_summary(summary_str, epoch)
	          i += self.batch_size

	    if self.embeddings_freq and self.embeddings_ckpt_path:
	      if epoch % self.embeddings_freq == 0:
	        self.saver.save(self.sess, self.embeddings_ckpt_path, epoch)

	    for name, value in logs.items():
	      if name in ['batch', 'size']:
	        continue
	      summary = tf_summary.Summary()
	      summary_value = summary.value.add()
	      summary_value.simple_value = value.item()
	      summary_value.tag = name
	      self.writer.add_summary(summary, epoch)
	    self.writer.flush()

	  def on_train_end(self, _):
	    self.writer.close()


	class ReduceLROnPlateau(Callback):
	  """Reduce learning rate when a metric has stopped improving.

	  Models often benefit from reducing the learning rate by a factor
	  of 2-10 once learning stagnates. This callback monitors a
	  quantity and if no improvement is seen for a 'patience' number
	  of epochs, the learning rate is reduced.

	  Example:

	  ```python
	  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
	                                patience=5, min_lr=0.001)
	  model.fit(X_train, Y_train, callbacks=[reduce_lr])
	  ```

	  Arguments:
	      monitor: quantity to be monitored.
	      factor: factor by which the learning rate will
	          be reduced. new_lr = lr * factor
	      patience: number of epochs with no improvement
	          after which learning rate will be reduced.
	      verbose: int. 0: quiet, 1: update messages.
	      mode: one of {auto, min, max}. In `min` mode,
	          lr will be reduced when the quantity
	          monitored has stopped decreasing; in `max`
	          mode it will be reduced when the quantity
	          monitored has stopped increasing; in `auto`
	          mode, the direction is automatically inferred
	          from the name of the monitored quantity.
	      epsilon: threshold for measuring the new optimum,
	          to only focus on significant changes.
	      cooldown: number of epochs to wait before resuming
	          normal operation after lr has been reduced.
	      min_lr: lower bound on the learning rate.
	  """

	  def __init__(self,
	               monitor='val_loss',
	               factor=0.1,
	               patience=10,
	               verbose=0,
	               mode='auto',
	               epsilon=1e-4,
	               cooldown=0,
	               min_lr=0):
	    super(ReduceLROnPlateau, self).__init__()

	    self.monitor = monitor
	    if factor >= 1.0:
	      raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
	    self.factor = factor
	    self.min_lr = min_lr
	    self.epsilon = epsilon
	    self.patience = patience
	    self.verbose = verbose
	    self.cooldown = cooldown
	    self.cooldown_counter = 0  # Cooldown counter.
	    self.wait = 0
	    self.best = 0
	    self.mode = mode
	    self.monitor_op = None
	    self._reset()

	  def _reset(self):
	    """Resets wait counter and cooldown counter.
	    """
	    if self.mode not in ['auto', 'min', 'max']:
	      logging.warning('Learning Rate Plateau Reducing mode %s is unknown, '
	                      'fallback to auto mode.' % (self.mode))
	      self.mode = 'auto'
	    if (self.mode == 'min' or
	        (self.mode == 'auto' and 'acc' not in self.monitor)):
	      self.monitor_op = lambda a, b: np.less(a, b - self.epsilon)
	      self.best = np.Inf
	    else:
	      self.monitor_op = lambda a, b: np.greater(a, b + self.epsilon)
	      self.best = -np.Inf
	    self.cooldown_counter = 0
	    self.wait = 0
	    self.lr_epsilon = self.min_lr * 1e-4

	  def on_train_begin(self, logs=None):
	    self._reset()

	  def on_epoch_end(self, epoch, logs=None):
	    logs = logs or {}
	    logs['lr'] = K.get_value(self.model.optimizer.lr)
	    current = logs.get(self.monitor)
	    if current is None:
	      logging.warning('Learning Rate Plateau Reducing requires %s available!' %
	                      self.monitor)
	    else:
	      if self.in_cooldown():
	        self.cooldown_counter -= 1
	        self.wait = 0

	      if self.monitor_op(current, self.best):
	        self.best = current
	        self.wait = 0
	      elif not self.in_cooldown():
	        if self.wait >= self.patience:
	          old_lr = float(K.get_value(self.model.optimizer.lr))
	          if old_lr > self.min_lr + self.lr_epsilon:
	            new_lr = old_lr * self.factor
	            new_lr = max(new_lr, self.min_lr)
	            K.set_value(self.model.optimizer.lr, new_lr)
	            if self.verbose > 0:
	              print('\nEpoch %05d: reducing learning rate to %s.' % (epoch,
	                                                                     new_lr))
	            self.cooldown_counter = self.cooldown
	            self.wait = 0
	        self.wait += 1

	  def in_cooldown(self):
	    return self.cooldown_counter > 0


	class CSVLogger(Callback):
	  """Callback that streams epoch results to a csv file.

	  Supports all values that can be represented as a string,
	  including 1D iterables such as np.ndarray.

	  Example:
	      ```python
	      csv_logger = CSVLogger('training.log')
	      model.fit(X_train, Y_train, callbacks=[csv_logger])
	      ```

	  Arguments:
	      filename: filename of the csv file, e.g. 'run/log.csv'.
	      separator: string used to separate elements in the csv file.
	      append: True: append if file exists (useful for continuing
	          training). False: overwrite existing file,
	  """

	  def __init__(self, filename, separator=',', append=False):
	    self.sep = separator
	    self.filename = filename
	    self.append = append
	    self.writer = None
	    self.keys = None
	    self.append_header = True
	    self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
	    super(CSVLogger, self).__init__()

	  def on_train_begin(self, logs=None):
	    if self.append:
	      if os.path.exists(self.filename):
	        with open(self.filename, 'r' + self.file_flags) as f:
	          self.append_header = not bool(len(f.readline()))
	      self.csv_file = open(self.filename, 'a' + self.file_flags)
	    else:
	      self.csv_file = open(self.filename, 'w' + self.file_flags)

	  def on_epoch_end(self, epoch, logs=None):
	    logs = logs or {}

	    def handle_value(k):
	      is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
	      if isinstance(k, six.string_types):
	        return k
	      elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
	        return '"[%s]"' % (', '.join(map(str, k)))
	      else:
	        return k

	    if not self.writer:
	      self.keys = sorted(logs.keys())

	      class CustomDialect(csv.excel):
	        delimiter = self.sep

	      self.writer = csv.DictWriter(
	          self.csv_file,
	          fieldnames=['epoch'] + self.keys,
	          dialect=CustomDialect)
	      if self.append_header:
	        self.writer.writeheader()

	    row_dict = OrderedDict({'epoch': epoch})
	    row_dict.update((key, handle_value(logs[key])) for key in self.keys)
	    self.writer.writerow(row_dict)
	    self.csv_file.flush()

	  def on_train_end(self, logs=None):
	    self.csv_file.close()
	    self.writer = None


	class LambdaCallback(Callback):
	  """Callback for creating simple, custom callbacks on-the-fly.

	  This callback is constructed with anonymous functions that will be called
	  at the appropriate time. Note that the callbacks expects positional
	  arguments, as:

	   - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
	      `epoch`, `logs`
	   - `on_batch_begin` and `on_batch_end` expect two positional arguments:
	      `batch`, `logs`
	   - `on_train_begin` and `on_train_end` expect one positional argument:
	      `logs`

	  Arguments:
	      on_epoch_begin: called at the beginning of every epoch.
	      on_epoch_end: called at the end of every epoch.
	      on_batch_begin: called at the beginning of every batch.
	      on_batch_end: called at the end of every batch.
	      on_train_begin: called at the beginning of model training.
	      on_train_end: called at the end of model training.

	  Example:
	      ```python
	      # Print the batch number at the beginning of every batch.
	      batch_print_callback = LambdaCallback(
	          on_batch_begin=lambda batch,logs: print(batch))

	      # Plot the loss after every epoch.
	      import numpy as np
	      import matplotlib.pyplot as plt
	      plot_loss_callback = LambdaCallback(
	          on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch),
	                                                    logs['loss']))

	      # Terminate some processes after having finished model training.
	      processes = ...
	      cleanup_callback = LambdaCallback(
	          on_train_end=lambda logs: [
	              p.terminate() for p in processes if p.is_alive()])

	      model.fit(...,
	                callbacks=[batch_print_callback,
	                           plot_loss_callback,
	                           cleanup_callback])
	      ```
	  """

	  def __init__(self,
	               on_epoch_begin=None,
	               on_epoch_end=None,
	               on_batch_begin=None,
	               on_batch_end=None,
	               on_train_begin=None,
	               on_train_end=None,
	               **kwargs):
	    super(LambdaCallback, self).__init__()
	    self.__dict__.update(kwargs)
	    if on_epoch_begin is not None:
	      self.on_epoch_begin = on_epoch_begin
	    else:
	      self.on_epoch_begin = lambda epoch, logs: None
	    if on_epoch_end is not None:
	      self.on_epoch_end = on_epoch_end
	    else:
	      self.on_epoch_end = lambda epoch, logs: None
	    if on_batch_begin is not None:
	      self.on_batch_begin = on_batch_begin
	    else:
	      self.on_batch_begin = lambda batch, logs: None
	    if on_batch_end is not None:
	      self.on_batch_end = on_batch_end
	    else:
	      self.on_batch_end = lambda batch, logs: None
	    if on_train_begin is not None:
	      self.on_train_begin = on_train_begin
	    else:
	      self.on_train_begin = lambda logs: None
	    if on_train_end is not None:
	      self.on_train_end = on_train_end
	    else:
	      self.on_train_end = lambda logs: None

## constraints.py
class constraints_py:
	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

	"""Constraints: functions that impose constraints on weights values.
	"""

	def import_libs():
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import six

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object


	class Constraint(object):

	  def __call__(self, w):
	    return w

	  def get_config(self):
	    return {}


	class MaxNorm(Constraint):
	  """MaxNorm weight constraint.

	  Constrains the weights incident to each hidden unit
	  to have a norm less than or equal to a desired value.

	  Arguments:
	      m: the maximum norm for the incoming weights.
	      axis: integer, axis along which to calculate weight norms.
	          For instance, in a `Dense` layer the weight matrix
	          has shape `(input_dim, output_dim)`,
	          set `axis` to `0` to constrain each weight vector
	          of length `(input_dim,)`.
	          In a `Convolution2D` layer with `data_format="channels_last"`,
	          the weight tensor has shape
	          `(rows, cols, input_depth, output_depth)`,
	          set `axis` to `[0, 1, 2]`
	          to constrain the weights of each filter tensor of size
	          `(rows, cols, input_depth)`.

	  References:
	      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
	        Srivastava, Hinton, et al.
	        2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
	  """

	  def __init__(self, max_value=2, axis=0):
	    self.max_value = max_value
	    self.axis = axis

	  def __call__(self, w):
	    norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
	    desired = K.clip(norms, 0, self.max_value)
	    w *= (desired / (K.epsilon() + norms))
	    return w

	  def get_config(self):
	    return {'max_value': self.max_value, 'axis': self.axis}


	class NonNeg(Constraint):
	  """Constrains the weights to be non-negative.
	  """

	  def __call__(self, w):
	    w *= K.cast(w >= 0., K.floatx())
	    return w


	class UnitNorm(Constraint):
	  """Constrains the weights incident to each hidden unit to have unit norm.

	  Arguments:
	      axis: integer, axis along which to calculate weight norms.
	          For instance, in a `Dense` layer the weight matrix
	          has shape `(input_dim, output_dim)`,
	          set `axis` to `0` to constrain each weight vector
	          of length `(input_dim,)`.
	          In a `Convolution2D` layer with `data_format="channels_last"`,
	          the weight tensor has shape
	          `(rows, cols, input_depth, output_depth)`,
	          set `axis` to `[0, 1, 2]`
	          to constrain the weights of each filter tensor of size
	          `(rows, cols, input_depth)`.
	  """

	  def __init__(self, axis=0):
	    self.axis = axis

	  def __call__(self, w):
	    return w / (
	        K.epsilon() + K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True)))

	  def get_config(self):
	    return {'axis': self.axis}


	class MinMaxNorm(Constraint):
	  """MinMaxNorm weight constraint.

	  Constrains the weights incident to each hidden unit
	  to have the norm between a lower bound and an upper bound.

	  Arguments:
	      min_value: the minimum norm for the incoming weights.
	      max_value: the maximum norm for the incoming weights.
	      rate: rate for enforcing the constraint: weights will be
	          rescaled to yield
	          `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
	          Effectively, this means that rate=1.0 stands for strict
	          enforcement of the constraint, while rate<1.0 means that
	          weights will be rescaled at each step to slowly move
	          towards a value inside the desired interval.
	      axis: integer, axis along which to calculate weight norms.
	          For instance, in a `Dense` layer the weight matrix
	          has shape `(input_dim, output_dim)`,
	          set `axis` to `0` to constrain each weight vector
	          of length `(input_dim,)`.
	          In a `Convolution2D` layer with `dim_ordering="tf"`,
	          the weight tensor has shape
	          `(rows, cols, input_depth, output_depth)`,
	          set `axis` to `[0, 1, 2]`
	          to constrain the weights of each filter tensor of size
	          `(rows, cols, input_depth)`.
	  """

	  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
	    self.min_value = min_value
	    self.max_value = max_value
	    self.rate = rate
	    self.axis = axis

	  def __call__(self, w):
	    norms = K.sqrt(K.sum(K.square(w), axis=self.axis, keepdims=True))
	    desired = (self.rate * K.clip(norms, self.min_value, self.max_value) +
	               (1 - self.rate) * norms)
	    w *= (desired / (K.epsilon() + norms))
	    return w

	  def get_config(self):
	    return {
	        'min_value': self.min_value,
	        'max_value': self.max_value,
	        'rate': self.rate,
	        'axis': self.axis
	    }


	# Aliases.

	# pylint: disable=invalid-name
	max_norm = MaxNorm
	non_neg = NonNeg
	unit_norm = UnitNorm
	min_max_norm = MinMaxNorm

	# pylint: enable=invalid-name


	def serialize(constraint):
	  return serialize_keras_object(constraint)


	def deserialize(config, custom_objects=None):
	  return deserialize_keras_object(
	      config,
	      module_objects=globals(),
	      custom_objects=custom_objects,
	      printable_module_name='constraint')


	def get(identifier):
	  if identifier is None:
	    return None
	  if isinstance(identifier, dict):
	    return deserialize(identifier)
	  elif isinstance(identifier, six.string_types):
	    config = {'class_name': str(identifier), 'config': {}}
	    return deserialize(config)
	  elif callable(identifier):
	    return identifier
	  else:
	    raise ValueError('Could not interpret constraint identifier:', identifier)

## initializers.py
class initializers_py:
	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.

	"""Keras initializer classes (soon to be replaced with core TF initializers).
	"""

	def import_libs():
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import numpy as np
		import six

		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object
		from tensorflow.python.ops.init_ops import Constant
		from tensorflow.python.ops.init_ops import Initializer
		from tensorflow.python.ops.init_ops import Ones
		from tensorflow.python.ops.init_ops import Orthogonal
		from tensorflow.python.ops.init_ops import RandomNormal
		from tensorflow.python.ops.init_ops import RandomUniform
		from tensorflow.python.ops.init_ops import TruncatedNormal
		from tensorflow.python.ops.init_ops import VarianceScaling
		from tensorflow.python.ops.init_ops import Zeros


	class Identity(Initializer):
	  """Initializer that generates the identity matrix.

	  Only use for square 2D matrices.

	  Arguments:
	      gain: Multiplicative factor to apply to the identity matrix.
	  """

	  def __init__(self, gain=1.):
	    self.gain = gain

	  def __call__(self, shape, dtype=None):
	    if len(shape) != 2 or shape[0] != shape[1]:
	      raise ValueError('Identity matrix initializer can only be used '
	                       'for 2D square matrices.')
	    else:
	      return self.gain * np.identity(shape[0])

	  def get_config(self):
	    return {'gain': self.gain}


	def lecun_uniform(seed=None):
	  """LeCun uniform initializer.

	  It draws samples from a uniform distribution within [-limit, limit]
	  where `limit` is `sqrt(3 / fan_in)`
	  where `fan_in` is the number of input units in the weight tensor.

	  Arguments:
	      seed: A Python integer. Used to seed the random generator.

	  Returns:
	      An initializer.

	  References:
	      LeCun 98, Efficient Backprop,
	      http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
	  """
	  return VarianceScaling(
	      scale=1., mode='fan_in', distribution='uniform', seed=seed)


	def glorot_normal(seed=None):
	  """Glorot normal initializer, also called Xavier normal initializer.

	  It draws samples from a truncated normal distribution centered on 0
	  with `stddev = sqrt(2 / (fan_in + fan_out))`
	  where `fan_in` is the number of input units in the weight tensor
	  and `fan_out` is the number of output units in the weight tensor.

	  Arguments:
	      seed: A Python integer. Used to seed the random generator.

	  Returns:
	      An initializer.

	  References:
	      Glorot & Bengio, AISTATS 2010
	      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
	  """
	  return VarianceScaling(
	      scale=1., mode='fan_avg', distribution='normal', seed=seed)


	def glorot_uniform(seed=None):
	  """Glorot uniform initializer, also called Xavier uniform initializer.

	  It draws samples from a uniform distribution within [-limit, limit]
	  where `limit` is `sqrt(6 / (fan_in + fan_out))`
	  where `fan_in` is the number of input units in the weight tensor
	  and `fan_out` is the number of output units in the weight tensor.

	  Arguments:
	      seed: A Python integer. Used to seed the random generator.

	  Returns:
	      An initializer.

	  References:
	      Glorot & Bengio, AISTATS 2010
	      http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
	  """
	  return VarianceScaling(
	      scale=1., mode='fan_avg', distribution='uniform', seed=seed)


	def he_normal(seed=None):
	  """He normal initializer.

	  It draws samples from a truncated normal distribution centered on 0
	  with `stddev = sqrt(2 / fan_in)`
	  where `fan_in` is the number of input units in the weight tensor.

	  Arguments:
	      seed: A Python integer. Used to seed the random generator.

	  Returns:
	      An initializer.

	  References:
	      He et al., http://arxiv.org/abs/1502.01852
	  """
	  return VarianceScaling(
	      scale=2., mode='fan_in', distribution='normal', seed=seed)


	def he_uniform(seed=None):
	  """He uniform variance scaling initializer.

	  It draws samples from a uniform distribution within [-limit, limit]
	  where `limit` is `sqrt(6 / fan_in)`
	  where `fan_in` is the number of input units in the weight tensor.

	  Arguments:
	      seed: A Python integer. Used to seed the random generator.

	  Returns:
	      An initializer.

	  References:
	      He et al., http://arxiv.org/abs/1502.01852
	  """
	  return VarianceScaling(
	      scale=2., mode='fan_in', distribution='uniform', seed=seed)


	# Compatibility aliases

	# pylint: disable=invalid-name
	zero = zeros = Zeros
	one = ones = Ones
	constant = Constant
	uniform = random_uniform = RandomUniform
	normal = random_normal = RandomNormal
	truncated_normal = TruncatedNormal
	identity = Identity
	orthogonal = Orthogonal

	# pylint: enable=invalid-name

	# Utility functions


	def serialize(initializer):
	  return serialize_keras_object(initializer)


	def deserialize(config, custom_objects=None):
	  return deserialize_keras_object(
	      config,
	      module_objects=globals(),
	      custom_objects=custom_objects,
	      printable_module_name='initializer')


	def get(identifier):
	  if isinstance(identifier, dict):
	    return deserialize(identifier)
	  elif isinstance(identifier, six.string_types):
	    config = {'class_name': str(identifier), 'config': {}}
	    return deserialize(config)
	  elif callable(identifier):
	    return identifier
	  else:
	    raise ValueError('Could not interpret initializer identifier:', identifier)


## losses.py
class losses_py:
	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
	"""Built-in Keras loss functions.
	"""
	def import_libs():

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import six

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object


	def mean_squared_error(y_true, y_pred):
	  return K.mean(K.square(y_pred - y_true), axis=-1)


	def mean_absolute_error(y_true, y_pred):
	  return K.mean(K.abs(y_pred - y_true), axis=-1)


	def mean_absolute_percentage_error(y_true, y_pred):
	  # Equivalent to MAE, but sometimes easier to interpret.
	  diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true), K.epsilon(), None))
	  return 100. * K.mean(diff, axis=-1)


	def mean_squared_logarithmic_error(y_true, y_pred):
	  first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
	  second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
	  return K.mean(K.square(first_log - second_log), axis=-1)


	def squared_hinge(y_true, y_pred):
	  return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


	def hinge(y_true, y_pred):
	  return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


	def categorical_hinge(y_true, y_pred):
	  pos = K.sum(y_true * y_pred, axis=-1)
	  neg = K.max((1. - y_true) * y_pred, axis=-1)
	  return K.maximum(neg - pos + 1., 0.)


	def logcosh(y_true, y_pred):

	  def cosh(x):
	    return (K.exp(x) + K.exp(-x)) / 2

	  return K.mean(K.log(cosh(y_pred - y_true)), axis=-1)


	def categorical_crossentropy(y_true, y_pred):
	  return K.categorical_crossentropy(y_pred, y_true)


	def sparse_categorical_crossentropy(y_true, y_pred):
	  return K.sparse_categorical_crossentropy(y_pred, y_true)


	def binary_crossentropy(y_true, y_pred):
	  return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


	def kullback_leibler_divergence(y_true, y_pred):
	  y_true = K.clip(y_true, K.epsilon(), 1)
	  y_pred = K.clip(y_pred, K.epsilon(), 1)
	  return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


	def poisson(y_true, y_pred):
	  return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


	def cosine_proximity(y_true, y_pred):
	  y_true = K.l2_normalize(y_true, axis=-1)
	  y_pred = K.l2_normalize(y_pred, axis=-1)
	  return -K.mean(y_true * y_pred, axis=-1)


	# Aliases.

	mse = MSE = mean_squared_error
	mae = MAE = mean_absolute_error
	mape = MAPE = mean_absolute_percentage_error
	msle = MSLE = mean_squared_logarithmic_error
	kld = KLD = kullback_leibler_divergence
	cosine = cosine_proximity


	def serialize(loss):
	  return loss.__name__


	def deserialize(name, custom_objects=None):
	  return deserialize_keras_object(
	      name,
	      module_objects=globals(),
	      custom_objects=custom_objects,
	      printable_module_name='loss function')


	def get(identifier):
	  if identifier is None:
	    return None
	  if isinstance(identifier, six.string_types):
	    identifier = str(identifier)
	    return deserialize(identifier)
	  elif callable(identifier):
	    return identifier
	  else:
	    raise ValueError('Could not interpret '
	                     'loss function identifier:', identifier)


## metrics.py
class metrics_py:

	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
	"""Built-in Keras metrics functions.
	"""
	def import_libs():

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import six

		from tensorflow.contrib.keras.python.keras import backend as K
		# pylint: disable=unused-import
		from tensorflow.contrib.keras.python.keras.losses import binary_crossentropy
		from tensorflow.contrib.keras.python.keras.losses import categorical_crossentropy
		from tensorflow.contrib.keras.python.keras.losses import cosine_proximity
		from tensorflow.contrib.keras.python.keras.losses import hinge
		from tensorflow.contrib.keras.python.keras.losses import kullback_leibler_divergence
		from tensorflow.contrib.keras.python.keras.losses import logcosh
		from tensorflow.contrib.keras.python.keras.losses import mean_absolute_error
		from tensorflow.contrib.keras.python.keras.losses import mean_absolute_percentage_error
		from tensorflow.contrib.keras.python.keras.losses import mean_squared_error
		from tensorflow.contrib.keras.python.keras.losses import mean_squared_logarithmic_error
		from tensorflow.contrib.keras.python.keras.losses import poisson
		from tensorflow.contrib.keras.python.keras.losses import sparse_categorical_crossentropy
		from tensorflow.contrib.keras.python.keras.losses import squared_hinge
		# pylint: disable=unused-import
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object


	def binary_accuracy(y_true, y_pred):
	  return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)


	def categorical_accuracy(y_true, y_pred):
	  return K.cast(
	      K.equal(K.argmax(y_true, axis=-1), K.argmax(y_pred, axis=-1)), K.floatx())


	def sparse_categorical_accuracy(y_true, y_pred):
	  return K.cast(
	      K.equal(
	          K.max(y_true, axis=-1), K.cast(K.argmax(y_pred, axis=-1),
	                                         K.floatx())), K.floatx())


	def top_k_categorical_accuracy(y_true, y_pred, k=5):
	  return K.mean(K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k), axis=-1)


	# Aliases

	mse = MSE = mean_squared_error
	mae = MAE = mean_absolute_error
	mape = MAPE = mean_absolute_percentage_error
	msle = MSLE = mean_squared_logarithmic_error
	cosine = cosine_proximity


	def serialize(metric):
	  return metric.__name__


	def deserialize(name, custom_objects=None):
	  return deserialize_keras_object(
	      name,
	      module_objects=globals(),
	      custom_objects=custom_objects,
	      printable_module_name='metric function')


	def get(identifier):
	  if isinstance(identifier, six.string_types):
	    identifier = str(identifier)
	    return deserialize(identifier)
	  elif callable(identifier):
	    return identifier
	  else:
	    raise ValueError('Could not interpret '
	                     'metric function identifier:', identifier)


## models.py
class models_py:
	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
	"""Home of the Sequential model, and the `save_model`/`load_model` functions.
	"""
	def import_libs():

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import copy
		import json
		import os

		import numpy as np

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras import layers as layer_module
		from tensorflow.contrib.keras.python.keras import optimizers
		from tensorflow.contrib.keras.python.keras.engine import topology
		from tensorflow.contrib.keras.python.keras.engine.topology import Input
		from tensorflow.contrib.keras.python.keras.engine.topology import Layer
		from tensorflow.contrib.keras.python.keras.engine.training import Model
		from tensorflow.contrib.keras.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
		from tensorflow.python.framework import ops
		from tensorflow.python.platform import tf_logging as logging


	# pylint: disable=g-import-not-at-top
	try:
	  import h5py
	except ImportError:
	  h5py = None

	try:
	  import yaml
	except ImportError:
	  yaml = None
	# pylint: enable=g-import-not-at-top


	def save_model(model, filepath, overwrite=True, include_optimizer=True):
	  """Save a model to a HDF5 file.

	  The saved model contains:
	      - the model's configuration (topology)
	      - the model's weights
	      - the model's optimizer's state (if any)

	  Thus the saved model can be reinstantiated in
	  the exact same state, without any of the code
	  used for model definition or training.

	  Arguments:
	      model: Keras model instance to be saved.
	      filepath: String, path where to save the model.
	      overwrite: Whether we should overwrite any existing
	          model at the target location, or instead
	          ask the user with a manual prompt.
	      include_optimizer: If True, save optimizer's state together.

	  Raises:
	      ImportError: if h5py is not available.
	  """

	  if h5py is None:
	    raise ImportError('`save_model` requires h5py.')

	  def get_json_type(obj):
	    """Serialize any object to a JSON-serializable structure.

	    Arguments:
	        obj: the object to serialize

	    Returns:
	        JSON-serializable structure representing `obj`.

	    Raises:
	        TypeError: if `obj` cannot be serialized.
	    """
	    # if obj is a serializable Keras class instance
	    # e.g. optimizer, layer
	    if hasattr(obj, 'get_config'):
	      return {'class_name': obj.__class__.__name__, 'config': obj.get_config()}

	    # if obj is any numpy type
	    if type(obj).__module__ == np.__name__:
	      return obj.item()

	    # misc functions (e.g. loss function)
	    if callable(obj):
	      return obj.__name__

	    # if obj is a python 'type'
	    if type(obj).__name__ == type.__name__:
	      return obj.__name__

	    raise TypeError('Not JSON Serializable:', obj)

	  from tensorflow.contrib.keras.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

	  # If file exists and should not be overwritten.
	  if not overwrite and os.path.isfile(filepath):
	    proceed = ask_to_proceed_with_overwrite(filepath)
	    if not proceed:
	      return

	  f = h5py.File(filepath, 'w')
	  f.attrs['keras_version'] = str(keras_version).encode('utf8')
	  f.attrs['backend'] = K.backend().encode('utf8')
	  f.attrs['model_config'] = json.dumps(
	      {
	          'class_name': model.__class__.__name__,
	          'config': model.get_config()
	      },
	      default=get_json_type).encode('utf8')

	  model_weights_group = f.create_group('model_weights')
	  model_layers = model.layers
	  topology.save_weights_to_hdf5_group(model_weights_group, model_layers)

	  if include_optimizer and hasattr(model, 'optimizer'):
	    if isinstance(model.optimizer, optimizers.TFOptimizer):
	      logging.warning(
	          'TensorFlow optimizers do not '
	          'make it possible to access '
	          'optimizer attributes or optimizer state '
	          'after instantiation. '
	          'As a result, we cannot save the optimizer '
	          'as part of the model save file.'
	          'You will have to compile your model again after loading it. '
	          'Prefer using a Keras optimizer instead '
	          '(see keras.io/optimizers).')
	    else:
	      f.attrs['training_config'] = json.dumps(
	          {
	              'optimizer_config': {
	                  'class_name': model.optimizer.__class__.__name__,
	                  'config': model.optimizer.get_config()
	              },
	              'loss': model.loss,
	              'metrics': model.metrics,
	              'sample_weight_mode': model.sample_weight_mode,
	              'loss_weights': model.loss_weights,
	          },
	          default=get_json_type).encode('utf8')

	      # Save optimizer weights.
	      symbolic_weights = getattr(model.optimizer, 'weights')
	      if symbolic_weights:
	        optimizer_weights_group = f.create_group('optimizer_weights')
	        weight_values = K.batch_get_value(symbolic_weights)
	        weight_names = []
	        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
	          # Default values of symbolic_weights is /variable for theano
	          if K.backend() == 'theano':
	            if hasattr(w, 'name') and w.name != '/variable':
	              name = str(w.name)
	            else:
	              name = 'param_' + str(i)
	          else:
	            if hasattr(w, 'name') and w.name:
	              name = str(w.name)
	            else:
	              name = 'param_' + str(i)
	          weight_names.append(name.encode('utf8'))
	        optimizer_weights_group.attrs['weight_names'] = weight_names
	        for name, val in zip(weight_names, weight_values):
	          param_dset = optimizer_weights_group.create_dataset(
	              name, val.shape, dtype=val.dtype)
	          if not val.shape:
	            # scalar
	            param_dset[()] = val
	          else:
	            param_dset[:] = val
	  f.flush()
	  f.close()


	def load_model(filepath, custom_objects=None, compile=True):  # pylint: disable=redefined-builtin
	  """Loads a model saved via `save_model`.

	  Arguments:
	      filepath: String, path to the saved model.
	      custom_objects: Optional dictionary mapping names
	          (strings) to custom classes or functions to be
	          considered during deserialization.
	      compile: Boolean, whether to compile the model
	          after loading.

	  Returns:
	      A Keras model instance. If an optimizer was found
	      as part of the saved model, the model is already
	      compiled. Otherwise, the model is uncompiled and
	      a warning will be displayed. When `compile` is set
	      to False, the compilation is omitted without any
	      warning.

	  Raises:
	      ImportError: if h5py is not available.
	      ValueError: In case of an invalid savefile.
	  """
	  if h5py is None:
	    raise ImportError('`load_model` requires h5py.')

	  if not custom_objects:
	    custom_objects = {}

	  def convert_custom_objects(obj):
	    """Handles custom object lookup.

	    Arguments:
	        obj: object, dict, or list.

	    Returns:
	        The same structure, where occurrences
	            of a custom object name have been replaced
	            with the custom object.
	    """
	    if isinstance(obj, list):
	      deserialized = []
	      for value in obj:
	        if value in custom_objects:
	          deserialized.append(custom_objects[value])
	        else:
	          deserialized.append(value)
	      return deserialized
	    if isinstance(obj, dict):
	      deserialized = {}
	      for key, value in obj.items():
	        deserialized[key] = []
	        if isinstance(value, list):
	          for element in value:
	            if element in custom_objects:
	              deserialized[key].append(custom_objects[element])
	            else:
	              deserialized[key].append(element)
	        elif value in custom_objects:
	          deserialized[key] = custom_objects[value]
	        else:
	          deserialized[key] = value
	      return deserialized
	    if obj in custom_objects:
	      return custom_objects[obj]
	    return obj

	  f = h5py.File(filepath, mode='r')

	  # instantiate model
	  model_config = f.attrs.get('model_config')
	  if model_config is None:
	    raise ValueError('No model found in config file.')
	  model_config = json.loads(model_config.decode('utf-8'))
	  model = model_from_config(model_config, custom_objects=custom_objects)

	  # set weights
	  topology.load_weights_from_hdf5_group(f['model_weights'], model.layers)

	  # Early return if compilation is not required.
	  if not compile:
	    f.close()
	    return model

	  # instantiate optimizer
	  training_config = f.attrs.get('training_config')
	  if training_config is None:
	    logging.warning('No training configuration found in save file: '
	                    'the model was *not* compiled. Compile it manually.')
	    f.close()
	    return model
	  training_config = json.loads(training_config.decode('utf-8'))
	  optimizer_config = training_config['optimizer_config']
	  optimizer = optimizers.deserialize(
	      optimizer_config, custom_objects=custom_objects)

	  # Recover loss functions and metrics.
	  loss = convert_custom_objects(training_config['loss'])
	  metrics = convert_custom_objects(training_config['metrics'])
	  sample_weight_mode = training_config['sample_weight_mode']
	  loss_weights = training_config['loss_weights']

	  # Compile model.
	  model.compile(
	      optimizer=optimizer,
	      loss=loss,
	      metrics=metrics,
	      loss_weights=loss_weights,
	      sample_weight_mode=sample_weight_mode)

	  # Set optimizer weights.
	  if 'optimizer_weights' in f:
	    # Build train function (to get weight updates).
	    if isinstance(model, Sequential):
	      model.model._make_train_function()
	    else:
	      model._make_train_function()
	    optimizer_weights_group = f['optimizer_weights']
	    optimizer_weight_names = [
	        n.decode('utf8') for n in optimizer_weights_group.attrs['weight_names']
	    ]
	    optimizer_weight_values = [
	        optimizer_weights_group[n] for n in optimizer_weight_names
	    ]
	    model.optimizer.set_weights(optimizer_weight_values)
	  f.close()
	  return model


	def model_from_config(config, custom_objects=None):
	  """Instantiates a Keras model from its config.

	  Arguments:
	      config: Configuration dictionary.
	      custom_objects: Optional dictionary mapping names
	          (strings) to custom classes or functions to be
	          considered during deserialization.

	  Returns:
	      A Keras model instance (uncompiled).

	  Raises:
	      TypeError if `config` is not a dictionary
	  """
	  if isinstance(config, list):
	    raise TypeError('`model_from_config` expects a dictionary, not a list. '
	                    'Maybe you meant to use '
	                    '`Sequential.from_config(config)`?')
	  return layer_module.deserialize(config, custom_objects=custom_objects)


	def model_from_yaml(yaml_string, custom_objects=None):
	  """Parses a yaml model configuration file and returns a model instance.

	  Arguments:
	      yaml_string: YAML string encoding a model configuration.
	      custom_objects: Optional dictionary mapping names
	          (strings) to custom classes or functions to be
	          considered during deserialization.

	  Returns:
	      A Keras model instance (uncompiled).

	  Raises:
	      ImportError: if yaml module is not found.
	  """
	  if yaml is None:
	    raise ImportError('Requires yaml module installed.')
	  config = yaml.load(yaml_string)
	  return layer_module.deserialize(config, custom_objects=custom_objects)


	def model_from_json(json_string, custom_objects=None):
	  """Parses a JSON model configuration file and returns a model instance.

	  Arguments:
	      json_string: JSON string encoding a model configuration.
	      custom_objects: Optional dictionary mapping names
	          (strings) to custom classes or functions to be
	          considered during deserialization.

	  Returns:
	      A Keras model instance (uncompiled).
	  """
	  config = json.loads(json_string)
	  return layer_module.deserialize(config, custom_objects=custom_objects)


	class Sequential(Model):
	  """Linear stack of layers.

	  Arguments:
	      layers: list of layers to add to the model.

	  # Note
	      The first layer passed to a Sequential model
	      should have a defined input shape. What that
	      means is that it should have received an `input_shape`
	      or `batch_input_shape` argument,
	      or for some type of layers (recurrent, Dense...)
	      an `input_dim` argument.

	  Example:

	      ```python
	          model = Sequential()
	          # first layer must have a defined input shape
	          model.add(Dense(32, input_dim=500))
	          # afterwards, Keras does automatic shape inference
	          model.add(Dense(32))

	          # also possible (equivalent to the above):
	          model = Sequential()
	          model.add(Dense(32, input_shape=(500,)))
	          model.add(Dense(32))

	          # also possible (equivalent to the above):
	          model = Sequential()
	          # here the batch dimension is None,
	          # which means any batch size will be accepted by the model.
	          model.add(Dense(32, batch_input_shape=(None, 500)))
	          model.add(Dense(32))
	      ```
	  """

	  def __init__(self, layers=None, name=None):
	    self.layers = []  # Stack of layers.
	    self.model = None  # Internal Model instance.
	    self.inputs = []  # List of input tensors
	    self.outputs = []  # List of length 1: the output tensor (unique).
	    self._trainable = True
	    self._initial_weights = None

	    # Model attributes.
	    self.inbound_nodes = []
	    self.outbound_nodes = []
	    self.built = False

	    # Set model name.
	    if not name:
	      prefix = 'sequential_'
	      name = prefix + str(K.get_uid(prefix))
	    self.name = name

	    # The following properties are not actually used by Keras;
	    # they exist for compatibility with TF's variable scoping mechanism.
	    self._updates = []
	    self._scope = None
	    self._reuse = None
	    self._base_name = name
	    self._graph = ops.get_default_graph()

	    # Add to the model any layers passed to the constructor.
	    if layers:
	      for layer in layers:
	        self.add(layer)

	  def add(self, layer):
	    """Adds a layer instance on top of the layer stack.

	    Arguments:
	        layer: layer instance.

	    Raises:
	        TypeError: If `layer` is not a layer instance.
	        ValueError: In case the `layer` argument does not
	            know its input shape.
	        ValueError: In case the `layer` argument has
	            multiple output tensors, or is already connected
	            somewhere else (forbidden in `Sequential` models).
	    """
	    if not isinstance(layer, Layer):
	      raise TypeError('The added layer must be '
	                      'an instance of class Layer. '
	                      'Found: ' + str(layer))
	    if not self.outputs:
	      # first layer in model: check that it is an input layer
	      if not layer.inbound_nodes:
	        # create an input layer
	        if not hasattr(layer, 'batch_input_shape'):
	          raise ValueError('The first layer in a '
	                           'Sequential model must '
	                           'get an `input_shape` or '
	                           '`batch_input_shape` argument.')
	        # Instantiate the input layer.
	        x = Input(
	            batch_shape=layer.batch_input_shape,
	            dtype=layer.dtype,
	            name=layer.name + '_input')
	        # This will build the current layer
	        # and create the node connecting the current layer
	        # to the input layer we just created.
	        layer(x)

	      if len(layer.inbound_nodes) != 1:
	        raise ValueError('A layer added to a Sequential model must '
	                         'not already be connected somewhere else. '
	                         'Model received layer ' + layer.name + ' which has ' +
	                         str(len(layer.inbound_nodes)) +
	                         ' pre-existing inbound connections.')

	      if len(layer.inbound_nodes[0].output_tensors) != 1:
	        raise ValueError('All layers in a Sequential model '
	                         'should have a single output tensor. '
	                         'For multi-output layers, '
	                         'use the functional API.')

	      self.outputs = [layer.inbound_nodes[0].output_tensors[0]]
	      self.inputs = topology.get_source_inputs(self.outputs[0])

	      # We create an input node, which we will keep updated
	      # as we add more layers
	      topology.Node(
	          outbound_layer=self,
	          inbound_layers=[],
	          node_indices=[],
	          tensor_indices=[],
	          input_tensors=self.inputs,
	          output_tensors=self.outputs,
	          # no model-level masking for now
	          input_masks=[None for _ in self.inputs],
	          output_masks=[None])
	    else:
	      output_tensor = layer(self.outputs[0])
	      if isinstance(output_tensor, list):
	        raise TypeError('All layers in a Sequential model '
	                        'should have a single output tensor. '
	                        'For multi-output layers, '
	                        'use the functional API.')
	      self.outputs = [output_tensor]
	      # update self.inbound_nodes
	      self.inbound_nodes[0].output_tensors = self.outputs
	      self.inbound_nodes[0].output_shapes = [K.int_shape(self.outputs[0])]

	    self.layers.append(layer)
	    self.built = False

	  def pop(self):
	    """Removes the last layer in the model.

	    Raises:
	        TypeError: if there are no layers in the model.
	    """
	    if not self.layers:
	      raise TypeError('There are no layers in the model.')

	    self.layers.pop()
	    if not self.layers:
	      self.outputs = []
	      self.inbound_nodes = []
	      self.outbound_nodes = []
	    else:
	      self.layers[-1].outbound_nodes = []
	      self.outputs = [self.layers[-1].output]
	      # update self.inbound_nodes
	      self.inbound_nodes[0].output_tensors = self.outputs
	      self.inbound_nodes[0].output_shapes = [K.int_shape(self.outputs[0])]
	    self.built = False

	  def get_layer(self, name=None, index=None):
	    """Retrieve a layer that is part of the model.

	    Returns a layer based on either its name (unique)
	    or its index in the graph. Indices are based on
	    order of horizontal graph traversal (bottom-up).

	    Arguments:
	        name: string, name of layer.
	        index: integer, index of layer.

	    Returns:
	        A layer instance.
	    """
	    if self.model is None:
	      self.build()
	    return self.model.get_layer(name, index)

	  def call(self, inputs, mask=None):
	    if self.model is None:
	      self.build()
	    return self.model.call(inputs, mask)

	  def build(self, input_shape=None):
	    if not self.inputs or not self.outputs:
	      raise TypeError('Sequential model cannot be built: model is empty.'
	                      ' Add some layers first.')
	    # actually create the model
	    self.model = Model(self.inputs, self.outputs[0], name=self.name + '_model')
	    self.model.trainable = self.trainable

	    # mirror model attributes
	    self.supports_masking = self.model.supports_masking
	    self._output_mask_cache = self.model._output_mask_cache
	    self._output_tensor_cache = self.model._output_tensor_cache
	    self._output_shape_cache = self.model._output_shape_cache
	    self.input_layers = self.model.input_layers
	    self.input_layers_node_indices = self.model.input_layers_node_indices
	    self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
	    self.output_layers = self.model.output_layers
	    self.output_layers_node_indices = self.model.output_layers_node_indices
	    self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
	    self.nodes_by_depth = self.model.nodes_by_depth
	    self.container_nodes = self.model.container_nodes
	    self.output_names = self.model.output_names
	    self.input_names = self.model.input_names
	    self._feed_input_names = self.model._feed_input_names
	    self._feed_inputs = self.model._feed_inputs

	    # Make sure child model callbacks
	    # will call the parent Sequential model.
	    self.model.callback_model = self

	    self.built = True

	  @property
	  def uses_learning_phase(self):
	    if self.model is None:
	      self.build()
	    return self.model.uses_learning_phase

	  def _gather_list_attr(self, attr):
	    all_attrs = []
	    for layer in self.layers:
	      all_attrs += getattr(layer, attr, [])
	    return all_attrs

	  @property
	  def trainable(self):
	    return self._trainable

	  @trainable.setter
	  def trainable(self, value):
	    if self.model:
	      self.model.trainable = value
	    self._trainable = value

	  @property
	  def trainable_weights(self):
	    if not self.trainable:
	      return []
	    return self._gather_list_attr('trainable_weights')

	  @property
	  def non_trainable_weights(self):
	    weights = self._gather_list_attr('non_trainable_weights')
	    if not self.trainable:
	      trainable_weights = self._gather_list_attr('trainable_weights')
	      return trainable_weights + weights
	    return weights

	  @property
	  def updates(self):
	    if self.model is None:
	      self.build()
	    return self.model.updates

	  @property
	  def state_updates(self):
	    if self.model is None:
	      self.build()
	    return self.model.state_updates

	  def get_updates_for(self, inputs):
	    if self.model is None:
	      self.build()
	    return self.model.get_updates_for(inputs)

	  @property
	  def losses(self):
	    if self.model is None:
	      self.build()
	    return self.model.losses

	  def get_losses_for(self, inputs):
	    if self.model is None:
	      self.build()
	    return self.model.get_losses_for(inputs)

	  @property
	  def regularizers(self):
	    if self.model is None:
	      self.build()
	    return self.model.regularizers

	  @property
	  def constraints(self):
	    if self.model is None:
	      self.build()
	    return self.model.constraints

	  def get_weights(self):
	    """Retrieves the weights of the model.

	    Returns:
	        A flat list of Numpy arrays
	        (one array per model weight).
	    """
	    if self.model is None:
	      self.build()
	    return self.model.get_weights()

	  def set_weights(self, weights):
	    """Sets the weights of the model.

	    Arguments:
	        weights: Should be a list
	            of Numpy arrays with shapes and types matching
	            the output of `model.get_weights()`.
	    """
	    if self.model is None:
	      self.build()
	    self.model.set_weights(weights)

	  def load_weights(self, filepath, by_name=False):
	    if h5py is None:
	      raise ImportError('`load_weights` requires h5py.')
	    f = h5py.File(filepath, mode='r')
	    if 'layer_names' not in f.attrs and 'model_weights' in f:
	      f = f['model_weights']
	    layers = self.layers
	    if by_name:
	      topology.load_weights_from_hdf5_group_by_name(f, layers)
	    else:
	      topology.load_weights_from_hdf5_group(f, layers)
	    if hasattr(f, 'close'):
	      f.close()

	  def save_weights(self, filepath, overwrite=True):
	    if h5py is None:
	      raise ImportError('`save_weights` requires h5py.')
	    # If file exists and should not be overwritten:
	    if not overwrite and os.path.isfile(filepath):
	      proceed = ask_to_proceed_with_overwrite(filepath)
	      if not proceed:
	        return
	    layers = self.layers
	    f = h5py.File(filepath, 'w')
	    topology.save_weights_to_hdf5_group(f, layers)
	    f.flush()
	    f.close()

	  def compile(self,
	              optimizer,
	              loss,
	              metrics=None,
	              sample_weight_mode=None,
	              **kwargs):
	    """Configures the learning process.

	    Arguments:
	        optimizer: str (name of optimizer) or optimizer object.
	            See [optimizers](/optimizers).
	        loss: str (name of objective function) or objective function.
	            See [losses](/losses).
	        metrics: list of metrics to be evaluated by the model
	            during training and testing.
	            Typically you will use `metrics=['accuracy']`.
	            See [metrics](/metrics).
	        sample_weight_mode: if you need to do timestep-wise
	            sample weighting (2D weights), set this to "temporal".
	            "None" defaults to sample-wise weights (1D).
	        **kwargs: for Theano backend, these are passed into K.function.
	            When using the Tensorflow backend, these are passed into
	            `tf.Session.run`.

	    Example:
	        ```python
	            model = Sequential()
	            model.add(Dense(32, input_shape=(500,)))
	            model.add(Dense(10, activation='softmax'))
	            model.compile(optimizer='rmsprop',
	                          loss='categorical_crossentropy',
	                          metrics=['accuracy'])
	        ```
	    """
	    # create the underlying model
	    self.build()
	    # call compile method of Model class
	    self.model.compile(
	        optimizer,
	        loss,
	        metrics=metrics,
	        sample_weight_mode=sample_weight_mode,
	        **kwargs)
	    self.optimizer = self.model.optimizer
	    self.loss = self.model.loss
	    self.total_loss = self.model.total_loss
	    self.loss_weights = self.model.loss_weights
	    self.metrics = self.model.metrics
	    self.metrics_tensors = self.model.metrics_tensors
	    self.metrics_names = self.model.metrics_names
	    self.sample_weight_mode = self.model.sample_weight_mode
	    self.sample_weights = self.model.sample_weights
	    self.targets = self.model.targets

	  def fit(self,
	          x,
	          y,
	          batch_size=32,
	          epochs=10,
	          verbose=1,
	          callbacks=None,
	          validation_split=0.,
	          validation_data=None,
	          shuffle=True,
	          class_weight=None,
	          sample_weight=None,
	          initial_epoch=0):
	    """Trains the model for a fixed number of epochs.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).
	        y: labels, as a Numpy array.
	        batch_size: integer. Number of samples per gradient update.
	        epochs: integer, the number of epochs to train the model.
	        verbose: 0 for no logging to stdout,
	            1 for progress bar logging, 2 for one log line per epoch.
	        callbacks: list of `keras.callbacks.Callback` instances.
	            List of callbacks to apply during training.
	            See [callbacks](/callbacks).
	        validation_split: float (0. < x < 1).
	            Fraction of the data to use as held-out validation data.
	        validation_data: tuple (x_val, y_val) or tuple
	            (x_val, y_val, val_sample_weights) to be used as held-out
	            validation data. Will override validation_split.
	        shuffle: boolean or str (for 'batch').
	            Whether to shuffle the samples at each epoch.
	            'batch' is a special option for dealing with the
	            limitations of HDF5 data; it shuffles in batch-sized chunks.
	        class_weight: dictionary mapping classes to a weight value,
	            used for scaling the loss function (during training only).
	        sample_weight: Numpy array of weights for
	            the training samples, used for scaling the loss function
	            (during training only). You can either pass a flat (1D)
	            Numpy array with the same length as the input samples
	            (1:1 mapping between weights and samples),
	            or in the case of temporal data,
	            you can pass a 2D array with shape (samples, sequence_length),
	            to apply a different weight to every timestep of every sample.
	            In this case you should make sure to specify
	            sample_weight_mode="temporal" in compile().
	        initial_epoch: epoch at which to start training
	            (useful for resuming a previous training run)

	    Returns:
	        A `History` object. Its `History.history` attribute is
	        a record of training loss values and metrics values
	        at successive epochs, as well as validation loss values
	        and validation metrics values (if applicable).

	    Raises:
	        RuntimeError: if the model was never compiled.
	    """
	    if self.model is None:
	      raise RuntimeError('The model needs to be compiled ' 'before being used.')
	    return self.model.fit(
	        x,
	        y,
	        batch_size=batch_size,
	        epochs=epochs,
	        verbose=verbose,
	        callbacks=callbacks,
	        validation_split=validation_split,
	        validation_data=validation_data,
	        shuffle=shuffle,
	        class_weight=class_weight,
	        sample_weight=sample_weight,
	        initial_epoch=initial_epoch)

	  def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
	    """Computes the loss on some input data, batch by batch.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).
	        y: labels, as a Numpy array.
	        batch_size: integer. Number of samples per gradient update.
	        verbose: verbosity mode, 0 or 1.
	        sample_weight: sample weights, as a Numpy array.

	    Returns:
	        Scalar test loss (if the model has no metrics)
	        or list of scalars (if the model computes other metrics).
	        The attribute `model.metrics_names` will give you
	        the display labels for the scalar outputs.

	    Raises:
	        RuntimeError: if the model was never compiled.
	    """
	    if self.model is None:
	      raise RuntimeError('The model needs to be compiled ' 'before being used.')
	    return self.model.evaluate(
	        x,
	        y,
	        batch_size=batch_size,
	        verbose=verbose,
	        sample_weight=sample_weight)

	  def predict(self, x, batch_size=32, verbose=0):
	    """Generates output predictions for the input samples.

	    The input samples are processed batch by batch.

	    Arguments:
	        x: the input data, as a Numpy array.
	        batch_size: integer.
	        verbose: verbosity mode, 0 or 1.

	    Returns:
	        A Numpy array of predictions.
	    """
	    if self.model is None:
	      self.build()
	    return self.model.predict(x, batch_size=batch_size, verbose=verbose)

	  def predict_on_batch(self, x):
	    """Returns predictions for a single batch of samples.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).

	    Returns:
	        A Numpy array of predictions.
	    """
	    if self.model is None:
	      self.build()
	    return self.model.predict_on_batch(x)

	  def train_on_batch(self, x, y, class_weight=None, sample_weight=None):
	    """Single gradient update over one batch of samples.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).
	        y: labels, as a Numpy array.
	        class_weight: dictionary mapping classes to a weight value,
	            used for scaling the loss function (during training only).
	        sample_weight: sample weights, as a Numpy array.

	    Returns:
	        Scalar training loss (if the model has no metrics)
	        or list of scalars (if the model computes other metrics).
	        The attribute `model.metrics_names` will give you
	        the display labels for the scalar outputs.

	    Raises:
	        RuntimeError: if the model was never compiled.
	    """
	    if self.model is None:
	      raise RuntimeError('The model needs to be compiled ' 'before being used.')
	    return self.model.train_on_batch(
	        x, y, sample_weight=sample_weight, class_weight=class_weight)

	  def test_on_batch(self, x, y, sample_weight=None):
	    """Evaluates the model over a single batch of samples.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).
	        y: labels, as a Numpy array.
	        sample_weight: sample weights, as a Numpy array.

	    Returns:
	        Scalar test loss (if the model has no metrics)
	        or list of scalars (if the model computes other metrics).
	        The attribute `model.metrics_names` will give you
	        the display labels for the scalar outputs.

	    Raises:
	        RuntimeError: if the model was never compiled.
	    """
	    if self.model is None:
	      raise RuntimeError('The model needs to be compiled ' 'before being used.')
	    return self.model.test_on_batch(x, y, sample_weight=sample_weight)

	  def predict_proba(self, x, batch_size=32, verbose=1):
	    """Generates class probability predictions for the input samples.

	    The input samples are processed batch by batch.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).
	        batch_size: integer.
	        verbose: verbosity mode, 0 or 1.

	    Returns:
	        A Numpy array of probability predictions.
	    """
	    preds = self.predict(x, batch_size, verbose)
	    if preds.min() < 0. or preds.max() > 1.:
	      logging.warning('Network returning invalid probability values. '
	                      'The last layer might not normalize predictions '
	                      'into probabilities '
	                      '(like softmax or sigmoid would).')
	    return preds

	  def predict_classes(self, x, batch_size=32, verbose=1):
	    """Generate class predictions for the input samples.

	    The input samples are processed batch by batch.

	    Arguments:
	        x: input data, as a Numpy array or list of Numpy arrays
	            (if the model has multiple inputs).
	        batch_size: integer.
	        verbose: verbosity mode, 0 or 1.

	    Returns:
	        A numpy array of class predictions.
	    """
	    proba = self.predict(x, batch_size=batch_size, verbose=verbose)
	    if proba.shape[-1] > 1:
	      return proba.argmax(axis=-1)
	    else:
	      return (proba > 0.5).astype('int32')

	  def fit_generator(self,
	                    generator,
	                    steps_per_epoch,
	                    epochs=1,
	                    verbose=1,
	                    callbacks=None,
	                    validation_data=None,
	                    validation_steps=None,
	                    class_weight=None,
	                    max_q_size=10,
	                    workers=1,
	                    pickle_safe=False,
	                    initial_epoch=0):
	    """Fits the model on data generated batch-by-batch by a Python generator.

	    The generator is run in parallel to the model, for efficiency.
	    For instance, this allows you to do real-time data augmentation
	    on images on CPU in parallel to training your model on GPU.

	    Arguments:
	        generator: A generator.
	            The output of the generator must be either
	            - a tuple (inputs, targets)
	            - a tuple (inputs, targets, sample_weights).
	            All arrays should contain the same number of samples.
	            The generator is expected to loop over its data
	            indefinitely. An epoch finishes when `steps_per_epoch`
	            batches have been seen by the model.
	        steps_per_epoch: Total number of steps (batches of samples)
	            to yield from `generator` before declaring one epoch
	            finished and starting the next epoch. It should typically
	            be equal to the number of unique samples of your dataset
	            divided by the batch size.
	        epochs: Integer, total number of iterations on the data.
	        verbose: Verbosity mode, 0, 1, or 2.
	        callbacks: List of callbacks to be called during training.
	        validation_data: This can be either
	            - A generator for the validation data
	            - A tuple (inputs, targets)
	            - A tuple (inputs, targets, sample_weights).
	        validation_steps: Only relevant if `validation_data`
	            is a generator.
	            Number of steps to yield from validation generator
	            at the end of every epoch. It should typically
	            be equal to the number of unique samples of your
	            validation dataset divided by the batch size.
	        class_weight: Dictionary mapping class indices to a weight
	            for the class.
	        max_q_size: Maximum size for the generator queue
	        workers: Maximum number of processes to spin up
	        pickle_safe: Ff True, use process based threading.
	            Note that because
	            this implementation relies on multiprocessing,
	            you should not pass
	            non picklable arguments to the generator
	            as they can't be passed
	            easily to children processes.
	        initial_epoch: Epoch at which to start training
	            (useful for resuming a previous training run)

	    Returns:
	        A `History` object.

	    Raises:
	        RuntimeError: if the model was never compiled.

	    Example:

	    ```python
	        def generate_arrays_from_file(path):
	            while 1:
	                f = open(path)
	                for line in f:
	                    # create Numpy arrays of input data
	                    # and labels, from each line in the file
	                    x, y = process_line(line)
	                    yield (x, y)
	                    f.close()

	        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
	                            steps_per_epoch=1000, epochs=10)
	    ```
	    """
	    if self.model is None:
	      raise RuntimeError('The model needs to be compiled ' 'before being used.')
	    return self.model.fit_generator(
	        generator,
	        steps_per_epoch,
	        epochs,
	        verbose=verbose,
	        callbacks=callbacks,
	        validation_data=validation_data,
	        validation_steps=validation_steps,
	        class_weight=class_weight,
	        max_q_size=max_q_size,
	        workers=workers,
	        pickle_safe=pickle_safe,
	        initial_epoch=initial_epoch)

	  def evaluate_generator(self,
	                         generator,
	                         steps,
	                         max_q_size=10,
	                         workers=1,
	                         pickle_safe=False):
	    """Evaluates the model on a data generator.

	    The generator should return the same kind of data
	    as accepted by `test_on_batch`.

	    Arguments:
	        generator: Generator yielding tuples (inputs, targets)
	            or (inputs, targets, sample_weights)
	        steps: Total number of steps (batches of samples)
	            to yield from `generator` before stopping.
	        max_q_size: maximum size for the generator queue
	        workers: maximum number of processes to spin up
	        pickle_safe: if True, use process based threading.
	            Note that because this implementation
	            relies on multiprocessing, you should not pass
	            non picklable arguments to the generator
	            as they can't be passed easily to children processes.

	    Returns:
	        Scalar test loss (if the model has no metrics)
	        or list of scalars (if the model computes other metrics).
	        The attribute `model.metrics_names` will give you
	        the display labels for the scalar outputs.

	    Raises:
	        RuntimeError: if the model was never compiled.
	    """
	    if self.model is None:
	      raise RuntimeError('The model needs to be compiled ' 'before being used.')
	    return self.model.evaluate_generator(
	        generator,
	        steps,
	        max_q_size=max_q_size,
	        workers=workers,
	        pickle_safe=pickle_safe)

	  def predict_generator(self,
	                        generator,
	                        steps,
	                        max_q_size=10,
	                        workers=1,
	                        pickle_safe=False,
	                        verbose=0):
	    """Generates predictions for the input samples from a data generator.

	    The generator should return the same kind of data as accepted by
	    `predict_on_batch`.

	    Arguments:
	        generator: generator yielding batches of input samples.
	        steps: Total number of steps (batches of samples)
	            to yield from `generator` before stopping.
	        max_q_size: maximum size for the generator queue
	        workers: maximum number of processes to spin up
	        pickle_safe: if True, use process based threading.
	            Note that because this implementation
	            relies on multiprocessing, you should not pass
	            non picklable arguments to the generator
	            as they can't be passed easily to children processes.
	        verbose: verbosity mode, 0 or 1.

	    Returns:
	        A Numpy array of predictions.
	    """
	    if self.model is None:
	      self.build()
	    return self.model.predict_generator(
	        generator,
	        steps,
	        max_q_size=max_q_size,
	        workers=workers,
	        pickle_safe=pickle_safe,
	        verbose=verbose)

	  def get_config(self):
	    config = []
	    for layer in self.layers:
	      config.append({
	          'class_name': layer.__class__.__name__,
	          'config': layer.get_config()
	      })
	    return copy.deepcopy(config)

	  @classmethod
	  def from_config(cls, config, custom_objects=None):
	    model = cls()
	    for conf in config:
	      layer = layer_module.deserialize(conf, custom_objects=custom_objects)
	      model.add(layer)
	    return model


## optimizer.py
class optimizers_py:

	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
	"""Keras optimizer classes (will eventually be replaced with core optimizers).
	"""

	def import_libs():

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import six
		from six.moves import zip  # pylint: disable=redefined-builtin

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object
		from tensorflow.python.training import optimizer as tf_optimizer_module


	def clip_norm(g, c, n):
	  if c > 0:
	    g = K.switch(n >= c, g * c / n, g)
	  return g


	class Optimizer(object):
	  """Abstract optimizer base class.

	  Note: this is the parent class of all optimizers, not an actual optimizer
	  that can be used for training models.

	  All Keras optimizers support the following keyword arguments:

	      clipnorm: float >= 0. Gradients will be clipped
	          when their L2 norm exceeds this value.
	      clipvalue: float >= 0. Gradients will be clipped
	          when their absolute value exceeds this value.
	  """

	  def __init__(self, **kwargs):
	    allowed_kwargs = {'clipnorm', 'clipvalue'}
	    for k in kwargs:
	      if k not in allowed_kwargs:
	        raise TypeError('Unexpected keyword argument '
	                        'passed to optimizer: ' + str(k))
	    self.__dict__.update(kwargs)
	    self.updates = []
	    self.weights = []

	  def get_updates(self, params, constraints, loss):
	    raise NotImplementedError

	  def get_gradients(self, loss, params):
	    grads = K.gradients(loss, params)
	    if hasattr(self, 'clipnorm') and self.clipnorm > 0:
	      norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
	      grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
	    if hasattr(self, 'clipvalue') and self.clipvalue > 0:
	      grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
	    return grads

	  def set_weights(self, weights):
	    """Sets the weights of the optimizer, from Numpy arrays.

	    Should only be called after computing the gradients
	    (otherwise the optimizer has no weights).

	    Arguments:
	        weights: a list of Numpy arrays. The number
	            of arrays and their shape must match
	            number of the dimensions of the weights
	            of the optimizer (i.e. it should match the
	            output of `get_weights`).

	    Raises:
	        ValueError: in case of incompatible weight shapes.
	    """
	    params = self.weights
	    weight_value_tuples = []
	    param_values = K.batch_get_value(params)
	    for pv, p, w in zip(param_values, params, weights):
	      if pv.shape != w.shape:
	        raise ValueError('Optimizer weight shape ' + str(pv.shape) +
	                         ' not compatible with '
	                         'provided weight shape ' + str(w.shape))
	      weight_value_tuples.append((p, w))
	    K.batch_set_value(weight_value_tuples)

	  def get_weights(self):
	    """Returns the current value of the weights of the optimizer.

	    Returns:
	        A list of numpy arrays.
	    """
	    return K.batch_get_value(self.weights)

	  def get_config(self):
	    config = {}
	    if hasattr(self, 'clipnorm'):
	      config['clipnorm'] = self.clipnorm
	    if hasattr(self, 'clipvalue'):
	      config['clipvalue'] = self.clipvalue
	    return config

	  @classmethod
	  def from_config(cls, config):
	    return cls(**config)


	class SGD(Optimizer):
	  """Stochastic gradient descent optimizer.

	  Includes support for momentum,
	  learning rate decay, and Nesterov momentum.

	  Arguments:
	      lr: float >= 0. Learning rate.
	      momentum: float >= 0. Parameter updates momentum.
	      decay: float >= 0. Learning rate decay over each update.
	      nesterov: boolean. Whether to apply Nesterov momentum.
	  """

	  def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False, **kwargs):
	    super(SGD, self).__init__(**kwargs)
	    self.iterations = K.variable(0., name='iterations')
	    self.lr = K.variable(lr, name='lr')
	    self.momentum = K.variable(momentum, name='momentum')
	    self.decay = K.variable(decay, name='decay')
	    self.initial_decay = decay
	    self.nesterov = nesterov

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    self.updates = []

	    lr = self.lr
	    if self.initial_decay > 0:
	      lr *= (1. / (1. + self.decay * self.iterations))
	      self.updates.append(K.update_add(self.iterations, 1))

	    # momentum
	    shapes = [K.int_shape(p) for p in params]
	    moments = [K.zeros(shape) for shape in shapes]
	    self.weights = [self.iterations] + moments
	    for p, g, m in zip(params, grads, moments):
	      v = self.momentum * m - lr * g  # velocity
	      self.updates.append(K.update(m, v))

	      if self.nesterov:
	        new_p = p + self.momentum * v - lr * g
	      else:
	        new_p = p + v

	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)

	      self.updates.append(K.update(p, new_p))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'momentum': float(K.get_value(self.momentum)),
	        'decay': float(K.get_value(self.decay)),
	        'nesterov': self.nesterov
	    }
	    base_config = super(SGD, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class RMSprop(Optimizer):
	  # pylint: disable=line-too-long
	  """RMSProp optimizer.

	  It is recommended to leave the parameters of this optimizer
	  at their default values
	  (except the learning rate, which can be freely tuned).

	  This optimizer is usually a good choice for recurrent
	  neural networks.

	  Arguments:
	      lr: float >= 0. Learning rate.
	      rho: float >= 0.
	      epsilon: float >= 0. Fuzz factor.
	      decay: float >= 0. Learning rate decay over each update.

	  References:
	      - [rmsprop: Divide the gradient by a running average of its recent
	        magnitude](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
	  """

	  # pylint: enable=line-too-long

	  def __init__(self, lr=0.001, rho=0.9, epsilon=1e-8, decay=0., **kwargs):
	    super(RMSprop, self).__init__(**kwargs)
	    self.lr = K.variable(lr, name='lr')
	    self.rho = K.variable(rho, name='rho')
	    self.epsilon = epsilon
	    self.decay = K.variable(decay, name='decay')
	    self.initial_decay = decay
	    self.iterations = K.variable(0., name='iterations')

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    shapes = [K.int_shape(p) for p in params]
	    accumulators = [K.zeros(shape) for shape in shapes]
	    self.weights = accumulators
	    self.updates = []

	    lr = self.lr
	    if self.initial_decay > 0:
	      lr *= (1. / (1. + self.decay * self.iterations))
	      self.updates.append(K.update_add(self.iterations, 1))

	    for p, g, a in zip(params, grads, accumulators):
	      # update accumulator
	      new_a = self.rho * a + (1. - self.rho) * K.square(g)
	      self.updates.append(K.update(a, new_a))
	      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)
	      self.updates.append(K.update(p, new_p))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'rho': float(K.get_value(self.rho)),
	        'decay': float(K.get_value(self.decay)),
	        'epsilon': self.epsilon
	    }
	    base_config = super(RMSprop, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class Adagrad(Optimizer):
	  # pylint: disable=line-too-long
	  """Adagrad optimizer.

	  It is recommended to leave the parameters of this optimizer
	  at their default values.

	  Arguments:
	      lr: float >= 0. Learning rate.
	      epsilon: float >= 0.
	      decay: float >= 0. Learning rate decay over each update.

	  References:
	      - [Adaptive Subgradient Methods for Online Learning and Stochastic
	        Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
	  """

	  # pylint: enable=line-too-long

	  def __init__(self, lr=0.01, epsilon=1e-8, decay=0., **kwargs):
	    super(Adagrad, self).__init__(**kwargs)
	    self.lr = K.variable(lr, name='lr')
	    self.epsilon = epsilon
	    self.decay = K.variable(decay, name='decay')
	    self.initial_decay = decay
	    self.iterations = K.variable(0., name='iterations')

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    shapes = [K.int_shape(p) for p in params]
	    accumulators = [K.zeros(shape) for shape in shapes]
	    self.weights = accumulators
	    self.updates = []

	    lr = self.lr
	    if self.initial_decay > 0:
	      lr *= (1. / (1. + self.decay * self.iterations))
	      self.updates.append(K.update_add(self.iterations, 1))

	    for p, g, a in zip(params, grads, accumulators):
	      new_a = a + K.square(g)  # update accumulator
	      self.updates.append(K.update(a, new_a))
	      new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)
	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)
	      self.updates.append(K.update(p, new_p))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'decay': float(K.get_value(self.decay)),
	        'epsilon': self.epsilon
	    }
	    base_config = super(Adagrad, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class Adadelta(Optimizer):
	  # pylint: disable=line-too-long
	  """Adadelta optimizer.

	  It is recommended to leave the parameters of this optimizer
	  at their default values.

	  Arguments:
	      lr: float >= 0. Learning rate.
	          It is recommended to leave it at the default value.
	      rho: float >= 0.
	      epsilon: float >= 0. Fuzz factor.
	      decay: float >= 0. Learning rate decay over each update.

	  References:
	      - [Adadelta - an adaptive learning rate
	        method](http://arxiv.org/abs/1212.5701)
	  """

	  # pylint: enable=line-too-long

	  def __init__(self, lr=1.0, rho=0.95, epsilon=1e-8, decay=0., **kwargs):
	    super(Adadelta, self).__init__(**kwargs)
	    self.lr = K.variable(lr, name='lr')
	    self.rho = rho
	    self.epsilon = epsilon
	    self.decay = K.variable(decay, name='decay')
	    self.initial_decay = decay
	    self.iterations = K.variable(0., name='iterations')

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    shapes = [K.int_shape(p) for p in params]
	    accumulators = [K.zeros(shape) for shape in shapes]
	    delta_accumulators = [K.zeros(shape) for shape in shapes]
	    self.weights = accumulators + delta_accumulators
	    self.updates = []

	    lr = self.lr
	    if self.initial_decay > 0:
	      lr *= (1. / (1. + self.decay * self.iterations))
	      self.updates.append(K.update_add(self.iterations, 1))

	    for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
	      # update accumulator
	      new_a = self.rho * a + (1. - self.rho) * K.square(g)
	      self.updates.append(K.update(a, new_a))

	      # use the new accumulator and the *old* delta_accumulator
	      update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)

	      new_p = p - lr * update
	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)
	      self.updates.append(K.update(p, new_p))

	      # update delta_accumulator
	      new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
	      self.updates.append(K.update(d_a, new_d_a))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'rho': self.rho,
	        'decay': float(K.get_value(self.decay)),
	        'epsilon': self.epsilon
	    }
	    base_config = super(Adadelta, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class Adam(Optimizer):
	  # pylint: disable=line-too-long
	  """Adam optimizer.

	  Default parameters follow those provided in the original paper.

	  Arguments:
	      lr: float >= 0. Learning rate.
	      beta_1: float, 0 < beta < 1. Generally close to 1.
	      beta_2: float, 0 < beta < 1. Generally close to 1.
	      epsilon: float >= 0. Fuzz factor.
	      decay: float >= 0. Learning rate decay over each update.

	  References:
	      - [Adam - A Method for Stochastic
	        Optimization](http://arxiv.org/abs/1412.6980v8)
	  """

	  # pylint: enable=line-too-long

	  def __init__(self,
	               lr=0.001,
	               beta_1=0.9,
	               beta_2=0.999,
	               epsilon=1e-8,
	               decay=0.,
	               **kwargs):
	    super(Adam, self).__init__(**kwargs)
	    self.iterations = K.variable(0, name='iterations')
	    self.lr = K.variable(lr, name='lr')
	    self.beta_1 = K.variable(beta_1, name='beta_1')
	    self.beta_2 = K.variable(beta_2, name='beta_2')
	    self.epsilon = epsilon
	    self.decay = K.variable(decay, name='decay')
	    self.initial_decay = decay

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    self.updates = [K.update_add(self.iterations, 1)]

	    lr = self.lr
	    if self.initial_decay > 0:
	      lr *= (1. / (1. + self.decay * self.iterations))

	    t = self.iterations + 1
	    lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
	                 (1. - K.pow(self.beta_1, t)))

	    shapes = [K.int_shape(p) for p in params]
	    ms = [K.zeros(shape) for shape in shapes]
	    vs = [K.zeros(shape) for shape in shapes]
	    self.weights = [self.iterations] + ms + vs

	    for p, g, m, v in zip(params, grads, ms, vs):
	      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
	      v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
	      p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

	      self.updates.append(K.update(m, m_t))
	      self.updates.append(K.update(v, v_t))

	      new_p = p_t
	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)
	      self.updates.append(K.update(p, new_p))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'beta_1': float(K.get_value(self.beta_1)),
	        'beta_2': float(K.get_value(self.beta_2)),
	        'decay': float(K.get_value(self.decay)),
	        'epsilon': self.epsilon
	    }
	    base_config = super(Adam, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class Adamax(Optimizer):
	  # pylint: disable=line-too-long
	  """Adamax optimizer from Adam paper's Section 7.

	  It is a variant of Adam based on the infinity norm.
	  Default parameters follow those provided in the paper.

	  Arguments:
	      lr: float >= 0. Learning rate.
	      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
	      epsilon: float >= 0. Fuzz factor.
	      decay: float >= 0. Learning rate decay over each update.

	  References:
	      - [Adam - A Method for Stochastic
	        Optimization](http://arxiv.org/abs/1412.6980v8)
	  """

	  # pylint: enable=line-too-long

	  def __init__(self,
	               lr=0.002,
	               beta_1=0.9,
	               beta_2=0.999,
	               epsilon=1e-8,
	               decay=0.,
	               **kwargs):
	    super(Adamax, self).__init__(**kwargs)
	    self.iterations = K.variable(0., name='iterations')
	    self.lr = K.variable(lr, name='lr')
	    self.beta_1 = K.variable(beta_1, name='beta_1')
	    self.beta_2 = K.variable(beta_2, name='beta_2')
	    self.epsilon = epsilon
	    self.decay = K.variable(decay, name='decay')
	    self.initial_decay = decay

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    self.updates = [K.update_add(self.iterations, 1)]

	    lr = self.lr
	    if self.initial_decay > 0:
	      lr *= (1. / (1. + self.decay * self.iterations))

	    t = self.iterations + 1
	    lr_t = lr / (1. - K.pow(self.beta_1, t))

	    shapes = [K.int_shape(p) for p in params]
	    # zero init of 1st moment
	    ms = [K.zeros(shape) for shape in shapes]
	    # zero init of exponentially weighted infinity norm
	    us = [K.zeros(shape) for shape in shapes]
	    self.weights = [self.iterations] + ms + us

	    for p, g, m, u in zip(params, grads, ms, us):

	      m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
	      u_t = K.maximum(self.beta_2 * u, K.abs(g))
	      p_t = p - lr_t * m_t / (u_t + self.epsilon)

	      self.updates.append(K.update(m, m_t))
	      self.updates.append(K.update(u, u_t))

	      new_p = p_t
	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)
	      self.updates.append(K.update(p, new_p))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'beta_1': float(K.get_value(self.beta_1)),
	        'beta_2': float(K.get_value(self.beta_2)),
	        'decay': float(K.get_value(self.decay)),
	        'epsilon': self.epsilon
	    }
	    base_config = super(Adamax, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class Nadam(Optimizer):
	  # pylint: disable=line-too-long
	  """Nesterov Adam optimizer.

	  Much like Adam is essentially RMSprop with momentum,
	  Nadam is Adam RMSprop with Nesterov momentum.

	  Default parameters follow those provided in the paper.
	  It is recommended to leave the parameters of this optimizer
	  at their default values.

	  Arguments:
	      lr: float >= 0. Learning rate.
	      beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
	      epsilon: float >= 0. Fuzz factor.

	  References:
	      - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
	      - [On the importance of initialization and momentum in deep
	        learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
	  """

	  # pylint: enable=line-too-long

	  def __init__(self,
	               lr=0.002,
	               beta_1=0.9,
	               beta_2=0.999,
	               epsilon=1e-8,
	               schedule_decay=0.004,
	               **kwargs):
	    super(Nadam, self).__init__(**kwargs)
	    self.iterations = K.variable(0., name='iterations')
	    self.m_schedule = K.variable(1., name='m_schedule')
	    self.lr = K.variable(lr, name='lr')
	    self.beta_1 = K.variable(beta_1, name='beta_1')
	    self.beta_2 = K.variable(beta_2, name='beta_2')
	    self.epsilon = epsilon
	    self.schedule_decay = schedule_decay

	  def get_updates(self, params, constraints, loss):
	    grads = self.get_gradients(loss, params)
	    self.updates = [K.update_add(self.iterations, 1)]

	    t = self.iterations + 1

	    # Due to the recommendations in [2], i.e. warming momentum schedule
	    momentum_cache_t = self.beta_1 * (1. - 0.5 *
	                                      (K.pow(0.96, t * self.schedule_decay)))
	    momentum_cache_t_1 = self.beta_1 * (1. - 0.5 *
	                                        (K.pow(0.96,
	                                               (t + 1) * self.schedule_decay)))
	    m_schedule_new = self.m_schedule * momentum_cache_t
	    m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
	    self.updates.append((self.m_schedule, m_schedule_new))

	    shapes = [K.int_shape(p) for p in params]
	    ms = [K.zeros(shape) for shape in shapes]
	    vs = [K.zeros(shape) for shape in shapes]

	    self.weights = [self.iterations] + ms + vs

	    for p, g, m, v in zip(params, grads, ms, vs):
	      # the following equations given in [1]
	      g_prime = g / (1. - m_schedule_new)
	      m_t = self.beta_1 * m + (1. - self.beta_1) * g
	      m_t_prime = m_t / (1. - m_schedule_next)
	      v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
	      v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
	      m_t_bar = (
	          1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_prime

	      self.updates.append(K.update(m, m_t))
	      self.updates.append(K.update(v, v_t))

	      p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
	      new_p = p_t

	      # apply constraints
	      if p in constraints:
	        c = constraints[p]
	        new_p = c(new_p)
	      self.updates.append(K.update(p, new_p))
	    return self.updates

	  def get_config(self):
	    config = {
	        'lr': float(K.get_value(self.lr)),
	        'beta_1': float(K.get_value(self.beta_1)),
	        'beta_2': float(K.get_value(self.beta_2)),
	        'epsilon': self.epsilon,
	        'schedule_decay': self.schedule_decay
	    }
	    base_config = super(Nadam, self).get_config()
	    return dict(list(base_config.items()) + list(config.items()))


	class TFOptimizer(Optimizer):
	  """Wrapper class for native TensorFlow optimizers.
	  """

	  def __init__(self, optimizer):  # pylint: disable=super-init-not-called
	    self.optimizer = optimizer
	    self.iterations = K.variable(0., name='iterations')
	    self.updates = []

	  def get_updates(self, params, constraints, loss):
	    if constraints:
	      raise ValueError('TF optimizers do not support '
	                       'weights constraints. Either remove '
	                       'all weights constraints in your model, '
	                       'or use a Keras optimizer.')
	    grads = self.optimizer.compute_gradients(loss, params)
	    opt_update = self.optimizer.apply_gradients(
	        grads, global_step=self.iterations)
	    self.updates.append(opt_update)
	    return self.updates

	  @property
	  def weights(self):
	    raise NotImplementedError

	  def get_config(self):
	    raise NotImplementedError

	  def from_config(self, config):
	    raise NotImplementedError


	# Aliases.

	# pylint: disable=invalid-name
	sgd = SGD
	rmsprop = RMSprop
	adagrad = Adagrad
	adadelta = Adadelta
	adam = Adam
	adamax = Adamax
	nadam = Nadam

	# pylint: enable=invalid-name


	def serialize(optimizer):
	  return serialize_keras_object(optimizer)


	def deserialize(config, custom_objects=None):
	  """Inverse of the `serialize` function.

	  Arguments:
	      config: Optimizer configuration dictionary.
	      custom_objects: Optional dictionary mapping
	          names (strings) to custom objects
	          (classes and functions)
	          to be considered during deserialization.

	  Returns:
	      A Keras Optimizer instance.
	  """
	  all_classes = {
	      'sgd': SGD,
	      'rmsprop': RMSprop,
	      'adagrad': Adagrad,
	      'adadelta': Adadelta,
	      'adam': Adam,
	      'adamax': Adamax,
	      'nadam': Nadam,
	      'tfoptimizer': TFOptimizer,
	  }
	  # Make deserialization case-insensitive for built-in optimizers.
	  if config['class_name'].lower() in all_classes:
	    config['class_name'] = config['class_name'].lower()
	  return deserialize_keras_object(
	      config,
	      module_objects=all_classes,
	      custom_objects=custom_objects,
	      printable_module_name='optimizer')


	def get(identifier):
	  """Retrieves a Keras Optimizer instance.

	  Arguments:
	      identifier: Optimizer identifier, one of
	          - String: name of an optimizer
	          - Dictionary: configuration dictionary.
	          - Keras Optimizer instance (it will be returned unchanged).
	          - TensorFlow Optimizer instance
	              (it will be wrapped as a Keras Optimizer).

	  Returns:
	      A Keras Optimizer instance.

	  Raises:
	      ValueError: If `identifier` cannot be interpreted.
	  """
	  # Wrap TF optimizer instances
	  if isinstance(identifier, tf_optimizer_module.Optimizer):
	    return TFOptimizer(identifier)
	  if isinstance(identifier, dict):
	    return deserialize(identifier)
	  elif isinstance(identifier, six.string_types):
	    config = {'class_name': str(identifier), 'config': {}}
	    return deserialize(config)
	  if isinstance(identifier, Optimizer):
	    return identifier
	  else:
	    raise ValueError('Could not interpret optimizer identifier:', identifier)


## regularizer.py
class regularizers_py:

	# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
	"""Keras built-in regularizers.
	"""
	def import_libs():

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import six

		from tensorflow.contrib.keras.python.keras import backend as K
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object


	class Regularizer(object):
	  """Regularizer base class.
	  """

	  def __call__(self, x):
	    return 0.

	  @classmethod
	  def from_config(cls, config):
	    return cls(**config)


	class L1L2(Regularizer):
	  """Regularizer for L1 and L2 regularization.

	  Arguments:
	      l1: Float; L1 regularization factor.
	      l2: Float; L2 regularization factor.
	  """

	  def __init__(self, l1=0., l2=0.):  # pylint: disable=redefined-outer-name
	    self.l1 = K.cast_to_floatx(l1)
	    self.l2 = K.cast_to_floatx(l2)

	  def __call__(self, x):
	    regularization = 0.
	    if self.l1:
	      regularization += K.sum(self.l1 * K.abs(x))
	    if self.l2:
	      regularization += K.sum(self.l2 * K.square(x))
	    return regularization

	  def get_config(self):
	    return {'l1': float(self.l1), 'l2': float(self.l2)}


	# Aliases.


	def l1(l=0.01):
	  return L1L2(l1=l)


	def l2(l=0.01):
	  return L1L2(l2=l)


	def l1_l2(l1=0.01, l2=0.01):  # pylint: disable=redefined-outer-name
	  return L1L2(l1=l1, l2=l2)


	def serialize(regularizer):
	  return serialize_keras_object(regularizer)


	def deserialize(config, custom_objects=None):
	  return deserialize_keras_object(
	      config,
	      module_objects=globals(),
	      custom_objects=custom_objects,
	      printable_module_name='regularizer')


	def get(identifier):
	  if identifier is None:
	    return None
	  if isinstance(identifier, dict):
	    return deserialize(identifier)
	  elif isinstance(identifier, six.string_types):
	    config = {'class_name': str(identifier), 'config': {}}
	    return deserialize(config)
	  elif callable(identifier):
	    return identifier
	  else:
	    raise ValueError('Could not interpret regularizer identifier:', identifier)


## testing_utils.py
class testing_utils_py:

	# Copyright 2016 The TensorFlow Authors. All Rights Reserved.

	"""Utilities for unit-testing Keras."""

	def import_libs():

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import numpy as np

		from tensorflow.contrib.keras.python import keras
		from tensorflow.python.util import tf_inspect


	def get_test_data(train_samples,
	                  test_samples,
	                  input_shape,
	                  num_classes):
	  """Generates test data to train a model on.

	  Arguments:
	    train_samples: Integer, how many training samples to generate.
	    test_samples: Integer, how many test samples to generate.
	    input_shape: Tuple of integers, shape of the inputs.
	    num_classes: Integer, number of classes for the data and targets.
	      Only relevant if `classification=True`.

	  Returns:
	    A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
	  """
	  num_sample = train_samples + test_samples
	  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
	  y = np.random.randint(0, num_classes, size=(num_sample,))
	  x = np.zeros((num_sample,) + input_shape)
	  for i in range(num_sample):
	    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)
	  return ((x[:train_samples], y[:train_samples]),
	          (x[train_samples:], y[train_samples:]))


	def layer_test(layer_cls, kwargs=None, input_shape=None, input_dtype=None,
	               input_data=None, expected_output=None,
	               expected_output_dtype=None):
	  """Test routine for a layer with a single input and single output.

	  Arguments:
	    layer_cls: Layer class object.
	    kwargs: Optional dictionary of keyword arguments for instantiating the
	      layer.
	    input_shape: Input shape tuple.
	    input_dtype: Data type of the input data.
	    input_data: Numpy array of input data.
	    expected_output: Shape tuple for the expected shape of the output.
	    expected_output_dtype: Data type expected for the output.

	  Returns:
	    The output data (Numpy array) returned by the layer, for additional
	    checks to be done by the calling code.
	  """
	  if input_data is None:
	    assert input_shape
	    if not input_dtype:
	      input_dtype = 'float32'
	    input_data_shape = list(input_shape)
	    for i, e in enumerate(input_data_shape):
	      if e is None:
	        input_data_shape[i] = np.random.randint(1, 4)
	    input_data = 10 * np.random.random(input_data_shape)
	    if input_dtype[:4] == 'float':
	      input_data -= 0.5
	    input_data = input_data.astype(input_dtype)
	  elif input_shape is None:
	    input_shape = input_data.shape
	  if input_dtype is None:
	    input_dtype = input_data.dtype
	  if expected_output_dtype is None:
	    expected_output_dtype = input_dtype

	  # instantiation
	  kwargs = kwargs or {}
	  layer = layer_cls(**kwargs)

	  # test get_weights , set_weights at layer level
	  weights = layer.get_weights()
	  layer.set_weights(weights)

	  # test and instantiation from weights
	  if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
	    kwargs['weights'] = weights
	    layer = layer_cls(**kwargs)

	  # test in functional API
	  x = keras.layers.Input(shape=input_shape[1:], dtype=input_dtype)
	  y = layer(x)
	  assert keras.backend.dtype(y) == expected_output_dtype

	  # check shape inference
	  model = keras.models.Model(x, y)
	  expected_output_shape = tuple(
	      layer._compute_output_shape(input_shape).as_list())  # pylint: disable=protected-access
	  actual_output = model.predict(input_data)
	  actual_output_shape = actual_output.shape
	  for expected_dim, actual_dim in zip(expected_output_shape,
	                                      actual_output_shape):
	    if expected_dim is not None:
	      assert expected_dim == actual_dim
	  if expected_output is not None:
	    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

	  # test serialization, weight setting at model level
	  model_config = model.get_config()
	  recovered_model = keras.models.Model.from_config(model_config)
	  if model.weights:
	    weights = model.get_weights()
	    recovered_model.set_weights(weights)
	    output = recovered_model.predict(input_data)
	    np.testing.assert_allclose(output, actual_output, rtol=1e-3)

	  # test training mode (e.g. useful for dropout tests)
	  model.compile('rmsprop', 'mse')
	  model.train_on_batch(input_data, actual_output)

	  # test as first layer in Sequential API
	  layer_config = layer.get_config()
	  layer_config['batch_input_shape'] = input_shape
	  layer = layer.__class__.from_config(layer_config)

	  model = keras.models.Sequential()
	  model.add(layer)
	  actual_output = model.predict(input_data)
	  actual_output_shape = actual_output.shape
	  for expected_dim, actual_dim in zip(expected_output_shape,
	                                      actual_output_shape):
	    if expected_dim is not None:
	      assert expected_dim == actual_dim
	  if expected_output is not None:
	    np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

	  # test serialization, weight setting at model level
	  model_config = model.get_config()
	  recovered_model = keras.models.Sequential.from_config(model_config)
	  if model.weights:
	    weights = model.get_weights()
	    recovered_model.set_weights(weights)
	    output = recovered_model.predict(input_data)
	    np.testing.assert_allclose(output, actual_output, rtol=1e-3)

	  # test training mode (e.g. useful for dropout tests)
	  model.compile('rmsprop', 'mse')
	  model.train_on_batch(input_data, actual_output)

	  # for further checks in the caller function
	  return actual_output


# wrappers
class wrappers:

	"""Keras API wrappers.
	"""
	class __init__:

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.wrappers import scikit_learn

	class scikit_learn:

		"""API wrapper allowing to use certain Keras models with the Scikit-Learn API.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import copy
			import types

			import numpy as np

			from tensorflow.contrib.keras.python.keras.models import Sequential
			from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical
			from tensorflow.python.util import tf_inspect


		class BaseWrapper(object):
		  """Base class for the Keras scikit-learn wrapper.

		  Warning: This class should not be used directly.
		  Use descendant classes instead.

		  Arguments:
		      build_fn: callable function or class instance
		      **sk_params: model parameters & fitting parameters

		  The build_fn should construct, compile and return a Keras model, which
		  will then be used to fit/predict. One of the following
		  three values could be passed to build_fn:
		  1. A function
		  2. An instance of a class that implements the __call__ method
		  3. None. This means you implement a class that inherits from either
		  `KerasClassifier` or `KerasRegressor`. The __call__ method of the
		  present class will then be treated as the default build_fn.

		  `sk_params` takes both model parameters and fitting parameters. Legal model
		  parameters are the arguments of `build_fn`. Note that like all other
		  estimators in scikit-learn, 'build_fn' should provide default values for
		  its arguments, so that you could create the estimator without passing any
		  values to `sk_params`.

		  `sk_params` could also accept parameters for calling `fit`, `predict`,
		  `predict_proba`, and `score` methods (e.g., `epochs`, `batch_size`).
		  fitting (predicting) parameters are selected in the following order:

		  1. Values passed to the dictionary arguments of
		  `fit`, `predict`, `predict_proba`, and `score` methods
		  2. Values passed to `sk_params`
		  3. The default values of the `keras.models.Sequential`
		  `fit`, `predict`, `predict_proba` and `score` methods

		  When using scikit-learn's `grid_search` API, legal tunable parameters are
		  those you could pass to `sk_params`, including fitting parameters.
		  In other words, you could use `grid_search` to search for the best
		  `batch_size` or `epochs` as well as the model parameters.
		  """

		  def __init__(self, build_fn=None, **sk_params):
		    self.build_fn = build_fn
		    self.sk_params = sk_params
		    self.check_params(sk_params)

		  def check_params(self, params):
		    """Checks for user typos in "params".

		    Arguments:
		        params: dictionary; the parameters to be checked

		    Raises:
		        ValueError: if any member of `params` is not a valid argument.
		    """
		    legal_params_fns = [
		        Sequential.fit, Sequential.predict, Sequential.predict_classes,
		        Sequential.evaluate
		    ]
		    if self.build_fn is None:
		      legal_params_fns.append(self.__call__)
		    elif (not isinstance(self.build_fn, types.FunctionType) and
		          not isinstance(self.build_fn, types.MethodType)):
		      legal_params_fns.append(self.build_fn.__call__)
		    else:
		      legal_params_fns.append(self.build_fn)

		    legal_params = []
		    for fn in legal_params_fns:
		      legal_params += tf_inspect.getargspec(fn)[0]
		    legal_params = set(legal_params)

		    for params_name in params:
		      if params_name not in legal_params:
		        if params_name != 'nb_epoch':
		          raise ValueError('{} is not a legal parameter'.format(params_name))

		  def get_params(self, **params):  # pylint: disable=unused-argument
		    """Gets parameters for this estimator.

		    Arguments:
		        **params: ignored (exists for API compatibility).

		    Returns:
		        Dictionary of parameter names mapped to their values.
		    """
		    res = copy.deepcopy(self.sk_params)
		    res.update({'build_fn': self.build_fn})
		    return res

		  def set_params(self, **params):
		    """Sets the parameters of this estimator.

		    Arguments:
		        **params: Dictionary of parameter names mapped to their values.

		    Returns:
		        self
		    """
		    self.check_params(params)
		    self.sk_params.update(params)
		    return self

		  def fit(self, x, y, **kwargs):
		    """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

		    Arguments:
		        x : array-like, shape `(n_samples, n_features)`
		            Training samples where n_samples in the number of samples
		            and n_features is the number of features.
		        y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
		            True labels for X.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments of `Sequential.fit`

		    Returns:
		        history : object
		            details about the training history at each epoch.
		    """
		    if self.build_fn is None:
		      self.model = self.__call__(**self.filter_sk_params(self.__call__))
		    elif (not isinstance(self.build_fn, types.FunctionType) and
		          not isinstance(self.build_fn, types.MethodType)):
		      self.model = self.build_fn(
		          **self.filter_sk_params(self.build_fn.__call__))
		    else:
		      self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

		    loss_name = self.model.loss
		    if hasattr(loss_name, '__name__'):
		      loss_name = loss_name.__name__
		    if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
		      y = to_categorical(y)

		    fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
		    fit_args.update(kwargs)

		    history = self.model.fit(x, y, **fit_args)

		    return history

		  def filter_sk_params(self, fn, override=None):
		    """Filters `sk_params` and return those in `fn`'s arguments.

		    Arguments:
		        fn : arbitrary function
		        override: dictionary, values to override sk_params

		    Returns:
		        res : dictionary dictionary containing variables
		            in both sk_params and fn's arguments.
		    """
		    override = override or {}
		    res = {}
		    fn_args = tf_inspect.getargspec(fn)[0]
		    for name, value in self.sk_params.items():
		      if name in fn_args:
		        res.update({name: value})
		    res.update(override)
		    return res


		class KerasClassifier(BaseWrapper):
		  """Implementation of the scikit-learn classifier API for Keras.
		  """

		  def fit(self, x, y, **kwargs):
		    """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

		    Arguments:
		        x : array-like, shape `(n_samples, n_features)`
		            Training samples where n_samples in the number of samples
		            and n_features is the number of features.
		        y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
		            True labels for X.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments of `Sequential.fit`

		    Returns:
		        history : object
		            details about the training history at each epoch.

		    Raises:
		        ValueError: In case of invalid shape for `y` argument.
		    """
		    y = np.array(y)
		    if len(y.shape) == 2 and y.shape[1] > 1:
		      self.classes_ = np.arange(y.shape[1])
		    elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
		      self.classes_ = np.unique(y)
		      y = np.searchsorted(self.classes_, y)
		    else:
		      raise ValueError('Invalid shape for y: ' + str(y.shape))
		    self.n_classes_ = len(self.classes_)
		    return super(KerasClassifier, self).fit(x, y, **kwargs)

		  def predict(self, x, **kwargs):
		    """Returns the class predictions for the given test data.

		    Arguments:
		        x: array-like, shape `(n_samples, n_features)`
		            Test samples where n_samples in the number of samples
		            and n_features is the number of features.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments
		            of `Sequential.predict_classes`.

		    Returns:
		        preds: array-like, shape `(n_samples,)`
		            Class predictions.
		    """
		    kwargs = self.filter_sk_params(Sequential.predict_classes, kwargs)
		    classes = self.model.predict_classes(x, **kwargs)
		    return self.classes_[classes]

		  def predict_proba(self, x, **kwargs):
		    """Returns class probability estimates for the given test data.

		    Arguments:
		        x: array-like, shape `(n_samples, n_features)`
		            Test samples where n_samples in the number of samples
		            and n_features is the number of features.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments
		            of `Sequential.predict_classes`.

		    Returns:
		        proba: array-like, shape `(n_samples, n_outputs)`
		            Class probability estimates.
		            In the case of binary classification,
		            tp match the scikit-learn API,
		            will return an array of shape '(n_samples, 2)'
		            (instead of `(n_sample, 1)` as in Keras).
		    """
		    kwargs = self.filter_sk_params(Sequential.predict_proba, kwargs)
		    probs = self.model.predict_proba(x, **kwargs)

		    # check if binary classification
		    if probs.shape[1] == 1:
		      # first column is probability of class 0 and second is of class 1
		      probs = np.hstack([1 - probs, probs])
		    return probs

		  def score(self, x, y, **kwargs):
		    """Returns the mean accuracy on the given test data and labels.

		    Arguments:
		        x: array-like, shape `(n_samples, n_features)`
		            Test samples where n_samples in the number of samples
		            and n_features is the number of features.
		        y: array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
		            True labels for x.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments of `Sequential.evaluate`.

		    Returns:
		        score: float
		            Mean accuracy of predictions on X wrt. y.

		    Raises:
		        ValueError: If the underlying model isn't configured to
		            compute accuracy. You should pass `metrics=["accuracy"]` to
		            the `.compile()` method of the model.
		    """
		    y = np.searchsorted(self.classes_, y)
		    kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)

		    loss_name = self.model.loss
		    if hasattr(loss_name, '__name__'):
		      loss_name = loss_name.__name__
		    if loss_name == 'categorical_crossentropy' and len(y.shape) != 2:
		      y = to_categorical(y)

		    outputs = self.model.evaluate(x, y, **kwargs)
		    if not isinstance(outputs, list):
		      outputs = [outputs]
		    for name, output in zip(self.model.metrics_names, outputs):
		      if name == 'acc':
		        return output
		    raise ValueError('The model is not configured to compute accuracy. '
		                     'You should pass `metrics=["accuracy"]` to '
		                     'the `model.compile()` method.')


		class KerasRegressor(BaseWrapper):
		  """Implementation of the scikit-learn regressor API for Keras.
		  """

		  def predict(self, x, **kwargs):
		    """Returns predictions for the given test data.

		    Arguments:
		        x: array-like, shape `(n_samples, n_features)`
		            Test samples where n_samples in the number of samples
		            and n_features is the number of features.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments of `Sequential.predict`.

		    Returns:
		        preds: array-like, shape `(n_samples,)`
		            Predictions.
		    """
		    kwargs = self.filter_sk_params(Sequential.predict, kwargs)
		    return np.squeeze(self.model.predict(x, **kwargs))

		  def score(self, x, y, **kwargs):
		    """Returns the mean loss on the given test data and labels.

		    Arguments:
		        x: array-like, shape `(n_samples, n_features)`
		            Test samples where n_samples in the number of samples
		            and n_features is the number of features.
		        y: array-like, shape `(n_samples,)`
		            True labels for X.
		        **kwargs: dictionary arguments
		            Legal arguments are the arguments of `Sequential.evaluate`.

		    Returns:
		        score: float
		            Mean accuracy of predictions on X wrt. y.
		    """
		    kwargs = self.filter_sk_params(Sequential.evaluate, kwargs)
		    loss = self.model.evaluate(x, y, **kwargs)
		    if isinstance(loss, list):
		      return loss[0]
		    return loss


# utils
class utils_folder:

	class __init__py:

		"""Keras utilities.
		"""

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.utils import conv_utils
		from tensorflow.contrib.keras.python.keras.utils import data_utils
		from tensorflow.contrib.keras.python.keras.utils import generic_utils
		from tensorflow.contrib.keras.python.keras.utils import io_utils
		from tensorflow.contrib.keras.python.keras.utils import np_utils
		from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import custom_object_scope
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import CustomObjectScope
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import get_custom_objects
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar
		from tensorflow.contrib.keras.python.keras.utils.generic_utils import serialize_keras_object
		from tensorflow.contrib.keras.python.keras.utils.io_utils import HDF5Matrix
		from tensorflow.contrib.keras.python.keras.utils.layer_utils import convert_all_kernels_in_model
		from tensorflow.contrib.keras.python.keras.utils.np_utils import normalize
		from tensorflow.contrib.keras.python.keras.utils.np_utils import to_categorical
		from tensorflow.contrib.keras.python.keras.utils.vis_utils import plot_model


		# Globally-importable utils.

	class conv_utils_py:


		"""Utilities used by convolution layers.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np
			from six.moves import range  # pylint: disable=redefined-builtin

			from tensorflow.contrib.keras.python.keras import backend as K


		def normalize_tuple(value, n, name):
		  """Transforms a single int or iterable of ints into an int tuple.

		  Arguments:
		      value: The value to validate and convert. Could an int, or any iterable
		        of ints.
		      n: The size of the tuple to be returned.
		      name: The name of the argument being validated, e.g. "strides" or
		        "kernel_size". This is only used to format error messages.

		  Returns:
		      A tuple of n integers.

		  Raises:
		      ValueError: If something else than an int/long or iterable thereof was
		      passed.
		  """
		  if isinstance(value, int):
		    return (value,) * n
		  else:
		    try:
		      value_tuple = tuple(value)
		    except TypeError:
		      raise ValueError('The `' + name + '` argument must be a tuple of ' +
		                       str(n) + ' integers. Received: ' + str(value))
		    if len(value_tuple) != n:
		      raise ValueError('The `' + name + '` argument must be a tuple of ' +
		                       str(n) + ' integers. Received: ' + str(value))
		    for single_value in value_tuple:
		      try:
		        int(single_value)
		      except ValueError:
		        raise ValueError('The `' + name + '` argument must be a tuple of ' +
		                         str(n) + ' integers. Received: ' + str(value) + ' '
		                         'including element ' + str(single_value) + ' of type' +
		                         ' ' + str(type(single_value)))
		  return value_tuple


		def normalize_data_format(value):
		  if value is None:
		    value = K.image_data_format()
		  data_format = value.lower()
		  if data_format not in {'channels_first', 'channels_last'}:
		    raise ValueError('The `data_format` argument must be one of '
		                     '"channels_first", "channels_last". Received: ' +
		                     str(value))
		  return data_format


		def normalize_padding(value):
		  padding = value.lower()
		  if padding not in {'valid', 'same', 'causal'}:
		    raise ValueError('The `padding` argument must be one of '
		                     '"valid", "same" (or "causal", only for `Conv1D). '
		                     'Received: ' + str(padding))
		  return padding


		def convert_kernel(kernel):
		  """Converts a Numpy kernel matrix from Theano format to TensorFlow format.

		  Also works reciprocally, since the transformation is its own inverse.

		  Arguments:
		      kernel: Numpy array (3D, 4D or 5D).

		  Returns:
		      The converted kernel.

		  Raises:
		      ValueError: in case of invalid kernel shape or invalid data_format.
		  """
		  kernel = np.asarray(kernel)
		  if not 3 <= kernel.ndim <= 5:
		    raise ValueError('Invalid kernel shape:', kernel.shape)
		  slices = [slice(None, None, -1) for _ in range(kernel.ndim)]
		  no_flip = (slice(None, None), slice(None, None))
		  slices[-2:] = no_flip
		  return np.copy(kernel[slices])


		def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
		  """Determines output length of a convolution given input length.

		  Arguments:
		      input_length: integer.
		      filter_size: integer.
		      padding: one of "same", "valid", "full".
		      stride: integer.
		      dilation: dilation rate, integer.

		  Returns:
		      The output length (integer).
		  """
		  if input_length is None:
		    return None
		  assert padding in {'same', 'valid', 'full', 'causal'}
		  dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
		  if padding == 'same':
		    output_length = input_length
		  elif padding == 'valid':
		    output_length = input_length - dilated_filter_size + 1
		  elif padding == 'full':
		    output_length = input_length + dilated_filter_size - 1
		  elif padding == 'causal':
		    output_length = input_length
		  return (output_length + stride - 1) // stride


		def conv_input_length(output_length, filter_size, padding, stride):
		  """Determines input length of a convolution given output length.

		  Arguments:
		      output_length: integer.
		      filter_size: integer.
		      padding: one of "same", "valid", "full".
		      stride: integer.

		  Returns:
		      The input length (integer).
		  """
		  if output_length is None:
		    return None
		  assert padding in {'same', 'valid', 'full'}
		  if padding == 'same':
		    pad = filter_size // 2
		  elif padding == 'valid':
		    pad = 0
		  elif padding == 'full':
		    pad = filter_size - 1
		  return (output_length - 1) * stride - 2 * pad + filter_size


		def deconv_length(dim_size, stride_size, kernel_size, padding):
		  if dim_size is None:
		    return None
		  dim_size *= stride_size
		  if padding == 'valid':
		    dim_size += max(kernel_size - stride_size, 0)
		  elif padding == 'full':
		    dim_size -= (stride_size + kernel_size - 2)
		  return dim_size

	class data_utils_py:

		"""Utilities for file download and caching."""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import hashlib
			import os
			import shutil
			import sys
			import tarfile
			import zipfile

			import six
			from six.moves.urllib.error import HTTPError
			from six.moves.urllib.error import URLError
			from six.moves.urllib.request import urlopen

			from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar


		if sys.version_info[0] == 2:

		  def urlretrieve(url, filename, reporthook=None, data=None):
		    """Replacement for `urlretrive` for Python 2.

		    Under Python 2, `urlretrieve` relies on `FancyURLopener` from legacy
		    `urllib` module, known to have issues with proxy management.

		    Arguments:
		        url: url to retrieve.
		        filename: where to store the retrieved data locally.
		        reporthook: a hook function that will be called once
		            on establishment of the network connection and once
		            after each block read thereafter.
		            The hook will be passed three arguments;
		            a count of blocks transferred so far,
		            a block size in bytes, and the total size of the file.
		        data: `data` argument passed to `urlopen`.
		    """

		    def chunk_read(response, chunk_size=8192, reporthook=None):
		      content_type = response.info().get('Content-Length')
		      total_size = -1
		      if content_type is not None:
		        total_size = int(content_type.strip())
		      count = 0
		      while 1:
		        chunk = response.read(chunk_size)
		        count += 1
		        if not chunk:
		          reporthook(count, total_size, total_size)
		          break
		        if reporthook:
		          reporthook(count, chunk_size, total_size)
		        yield chunk

		    response = urlopen(url, data)
		    with open(filename, 'wb') as fd:
		      for chunk in chunk_read(response, reporthook=reporthook):
		        fd.write(chunk)
		else:
		  from six.moves.urllib.request import urlretrieve  # pylint: disable=g-import-not-at-top


		def _extract_archive(file_path, path='.', archive_format='auto'):
		  """Extracts an archive if it matches tar, tar.gz, tar.bz, or zip formats.

		  Arguments:
		      file_path: path to the archive file
		      path: path to extract the archive file
		      archive_format: Archive format to try for extracting the file.
		          Options are 'auto', 'tar', 'zip', and None.
		          'tar' includes tar, tar.gz, and tar.bz files.
		          The default 'auto' is ['tar', 'zip'].
		          None or an empty list will return no matches found.

		  Returns:
		      True if a match was found and an archive extraction was completed,
		      False otherwise.
		  """
		  if archive_format is None:
		    return False
		  if archive_format is 'auto':
		    archive_format = ['tar', 'zip']
		  if isinstance(archive_format, six.string_types):
		    archive_format = [archive_format]

		  for archive_type in archive_format:
		    if archive_type is 'tar':
		      open_fn = tarfile.open
		      is_match_fn = tarfile.is_tarfile
		    if archive_type is 'zip':
		      open_fn = zipfile.ZipFile
		      is_match_fn = zipfile.is_zipfile

		    if is_match_fn(file_path):
		      with open_fn(file_path) as archive:
		        try:
		          archive.extractall(path)
		        except (tarfile.TarError, RuntimeError, KeyboardInterrupt):
		          if os.path.exists(path):
		            if os.path.isfile(path):
		              os.remove(path)
		            else:
		              shutil.rmtree(path)
		          raise
		      return True
		  return False


		def get_file(fname,
		             origin,
		             untar=False,
		             md5_hash=None,
		             file_hash=None,
		             cache_subdir='datasets',
		             hash_algorithm='auto',
		             extract=False,
		             archive_format='auto',
		             cache_dir=None):
		  """Downloads a file from a URL if it not already in the cache.

		  By default the file at the url `origin` is downloaded to the
		  cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
		  and given the filename `fname`. The final location of a file
		  `example.txt` would therefore be `~/.keras/datasets/example.txt`.

		  Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
		  Passing a hash will verify the file after download. The command line
		  programs `shasum` and `sha256sum` can compute the hash.

		  Arguments:
		      fname: Name of the file. If an absolute path `/path/to/file.txt` is
		          specified the file will be saved at that location.
		      origin: Original URL of the file.
		      untar: Deprecated in favor of 'extract'.
		          boolean, whether the file should be decompressed
		      md5_hash: Deprecated in favor of 'file_hash'.
		          md5 hash of the file for verification
		      file_hash: The expected hash string of the file after download.
		          The sha256 and md5 hash algorithms are both supported.
		      cache_subdir: Subdirectory under the Keras cache dir where the file is
		          saved. If an absolute path `/path/to/folder` is
		          specified the file will be saved at that location.
		      hash_algorithm: Select the hash algorithm to verify the file.
		          options are 'md5', 'sha256', and 'auto'.
		          The default 'auto' detects the hash algorithm in use.
		      extract: True tries extracting the file as an Archive, like tar or zip.
		      archive_format: Archive format to try for extracting the file.
		          Options are 'auto', 'tar', 'zip', and None.
		          'tar' includes tar, tar.gz, and tar.bz files.
		          The default 'auto' is ['tar', 'zip'].
		          None or an empty list will return no matches found.
		      cache_dir: Location to store cached files, when None it
		          defaults to the [Keras
		            Directory](/faq/#where-is-the-keras-configuration-filed-stored).

		  Returns:
		      Path to the downloaded file
		  """
		  if cache_dir is None:
		    cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
		  if md5_hash is not None and file_hash is None:
		    file_hash = md5_hash
		    hash_algorithm = 'md5'
		  datadir_base = os.path.expanduser(cache_dir)
		  if not os.access(datadir_base, os.W_OK):
		    datadir_base = os.path.join('/tmp', '.keras')
		  datadir = os.path.join(datadir_base, cache_subdir)
		  if not os.path.exists(datadir):
		    os.makedirs(datadir)

		  if untar:
		    untar_fpath = os.path.join(datadir, fname)
		    fpath = untar_fpath + '.tar.gz'
		  else:
		    fpath = os.path.join(datadir, fname)

		  download = False
		  if os.path.exists(fpath):
		    # File found; verify integrity if a hash was provided.
		    if file_hash is not None:
		      if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
		        print('A local file was found, but it seems to be '
		              'incomplete or outdated because the ' + hash_algorithm +
		              ' file hash does not match the original value of ' + file_hash +
		              ' so we will re-download the data.')
		        download = True
		  else:
		    download = True

		  if download:
		    print('Downloading data from', origin)

		    class ProgressTracker(object):
		      # Maintain progbar for the lifetime of download.
		      # This design was chosen for Python 2.7 compatibility.
		      progbar = None

		    def dl_progress(count, block_size, total_size):
		      if ProgressTracker.progbar is None:
		        if total_size is -1:
		          total_size = None
		        ProgressTracker.progbar = Progbar(total_size)
		      else:
		        ProgressTracker.progbar.update(count * block_size)

		    error_msg = 'URL fetch failure on {}: {} -- {}'
		    try:
		      try:
		        urlretrieve(origin, fpath, dl_progress)
		      except URLError as e:
		        raise Exception(error_msg.format(origin, e.errno, e.reason))
		      except HTTPError as e:
		        raise Exception(error_msg.format(origin, e.code, e.msg))
		    except (Exception, KeyboardInterrupt) as e:
		      if os.path.exists(fpath):
		        os.remove(fpath)
		      raise
		    ProgressTracker.progbar = None

		  if untar:
		    if not os.path.exists(untar_fpath):
		      _extract_archive(fpath, datadir, archive_format='tar')
		    return untar_fpath

		  if extract:
		    _extract_archive(fpath, datadir, archive_format)

		  return fpath


		def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
		  """Calculates a file sha256 or md5 hash.

		  Example:

		  ```python
		     >>> from keras.data_utils import _hash_file
		     >>> _hash_file('/path/to/file.zip')
		     'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
		  ```

		  Arguments:
		      fpath: path to the file being validated
		      algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
		          The default 'auto' detects the hash algorithm in use.
		      chunk_size: Bytes to read at a time, important for large files.

		  Returns:
		      The file hash
		  """
		  if (algorithm is 'sha256') or (algorithm is 'auto' and len(hash) is 64):
		    hasher = hashlib.sha256()
		  else:
		    hasher = hashlib.md5()

		  with open(fpath, 'rb') as fpath_file:
		    for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
		      hasher.update(chunk)

		  return hasher.hexdigest()


		def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
		  """Validates a file against a sha256 or md5 hash.

		  Arguments:
		      fpath: path to the file being validated
		      file_hash:  The expected hash string of the file.
		          The sha256 and md5 hash algorithms are both supported.
		      algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
		          The default 'auto' detects the hash algorithm in use.
		      chunk_size: Bytes to read at a time, important for large files.

		  Returns:
		      Whether the file is valid
		  """
		  if ((algorithm is 'sha256') or
		      (algorithm is 'auto' and len(file_hash) is 64)):
		    hasher = 'sha256'
		  else:
		    hasher = 'md5'

		  if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
		    return True
		  else:
		    return False

	class generic_utils_py:

		"""Python utilities required by Keras."""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import marshal
			import sys
			import time
			import types as python_types

			import numpy as np
			import six

			from tensorflow.python.util import tf_decorator
			from tensorflow.python.util import tf_inspect

		_GLOBAL_CUSTOM_OBJECTS = {}


		class CustomObjectScope(object):
		  """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

		  Code within a `with` statement will be able to access custom objects
		  by name. Changes to global custom objects persist
		  within the enclosing `with` statement. At end of the `with` statement,
		  global custom objects are reverted to state
		  at beginning of the `with` statement.

		  Example:

		  Consider a custom object `MyObject`

		  ```python
		      with CustomObjectScope({'MyObject':MyObject}):
		          layer = Dense(..., kernel_regularizer='MyObject')
		          # save, load, etc. will recognize custom object by name
		  ```
		  """

		  def __init__(self, *args):
		    self.custom_objects = args
		    self.backup = None

		  def __enter__(self):
		    self.backup = _GLOBAL_CUSTOM_OBJECTS.copy()
		    for objects in self.custom_objects:
		      _GLOBAL_CUSTOM_OBJECTS.update(objects)
		    return self

		  def __exit__(self, *args, **kwargs):
		    _GLOBAL_CUSTOM_OBJECTS.clear()
		    _GLOBAL_CUSTOM_OBJECTS.update(self.backup)


		def custom_object_scope(*args):
		  """Provides a scope that changes to `_GLOBAL_CUSTOM_OBJECTS` cannot escape.

		  Convenience wrapper for `CustomObjectScope`.
		  Code within a `with` statement will be able to access custom objects
		  by name. Changes to global custom objects persist
		  within the enclosing `with` statement. At end of the `with` statement,
		  global custom objects are reverted to state
		  at beginning of the `with` statement.

		  Example:

		  Consider a custom object `MyObject`

		  ```python
		      with custom_object_scope({'MyObject':MyObject}):
		          layer = Dense(..., kernel_regularizer='MyObject')
		          # save, load, etc. will recognize custom object by name
		  ```

		  Arguments:
		      *args: Variable length list of dictionaries of name,
		          class pairs to add to custom objects.

		  Returns:
		      Object of type `CustomObjectScope`.
		  """
		  return CustomObjectScope(*args)


		def get_custom_objects():
		  """Retrieves a live reference to the global dictionary of custom objects.

		  Updating and clearing custom objects using `custom_object_scope`
		  is preferred, but `get_custom_objects` can
		  be used to directly access `_GLOBAL_CUSTOM_OBJECTS`.

		  Example:

		  ```python
		      get_custom_objects().clear()
		      get_custom_objects()['MyObject'] = MyObject
		  ```

		  Returns:
		      Global dictionary of names to classes (`_GLOBAL_CUSTOM_OBJECTS`).
		  """
		  return _GLOBAL_CUSTOM_OBJECTS


		def serialize_keras_object(instance):
		  _, instance = tf_decorator.unwrap(instance)
		  if instance is None:
		    return None
		  if hasattr(instance, 'get_config'):
		    return {
		        'class_name': instance.__class__.__name__,
		        'config': instance.get_config()
		    }
		  if hasattr(instance, '__name__'):
		    return instance.__name__
		  else:
		    raise ValueError('Cannot serialize', instance)


		def deserialize_keras_object(identifier,
		                             module_objects=None,
		                             custom_objects=None,
		                             printable_module_name='object'):
		  if isinstance(identifier, dict):
		    # In this case we are dealing with a Keras config dictionary.
		    config = identifier
		    if 'class_name' not in config or 'config' not in config:
		      raise ValueError('Improper config format: ' + str(config))
		    class_name = config['class_name']
		    if custom_objects and class_name in custom_objects:
		      cls = custom_objects[class_name]
		    elif class_name in _GLOBAL_CUSTOM_OBJECTS:
		      cls = _GLOBAL_CUSTOM_OBJECTS[class_name]
		    else:
		      module_objects = module_objects or {}
		      cls = module_objects.get(class_name)
		      if cls is None:
		        raise ValueError('Unknown ' + printable_module_name + ': ' + class_name)
		    if hasattr(cls, 'from_config'):
		      arg_spec = tf_inspect.getargspec(cls.from_config)
		      custom_objects = custom_objects or {}

		      if 'custom_objects' in arg_spec.args:
		        return cls.from_config(
		            config['config'],
		            custom_objects=dict(
		                list(_GLOBAL_CUSTOM_OBJECTS.items()) +
		                list(custom_objects.items())))
		      with CustomObjectScope(custom_objects):
		        return cls.from_config(config['config'])
		    else:
		      # Then `cls` may be a function returning a class.
		      # in this case by convention `config` holds
		      # the kwargs of the function.
		      custom_objects = custom_objects or {}
		      with CustomObjectScope(custom_objects):
		        return cls(**config['config'])
		  elif isinstance(identifier, six.string_types):
		    function_name = identifier
		    if custom_objects and function_name in custom_objects:
		      fn = custom_objects.get(function_name)
		    elif function_name in _GLOBAL_CUSTOM_OBJECTS:
		      fn = _GLOBAL_CUSTOM_OBJECTS[function_name]
		    else:
		      fn = module_objects.get(function_name)
		      if fn is None:
		        raise ValueError('Unknown ' + printable_module_name + ':' +
		                         function_name)
		    return fn
		  else:
		    raise ValueError('Could not interpret serialized ' + printable_module_name +
		                     ': ' + identifier)


		def func_dump(func):
		  """Serializes a user defined function.

		  Arguments:
		      func: the function to serialize.

		  Returns:
		      A tuple `(code, defaults, closure)`.
		  """
		  code = marshal.dumps(func.__code__).decode('raw_unicode_escape')
		  defaults = func.__defaults__
		  if func.__closure__:
		    closure = tuple(c.cell_contents for c in func.__closure__)
		  else:
		    closure = None
		  return code, defaults, closure


		def func_load(code, defaults=None, closure=None, globs=None):
		  """Deserializes a user defined function.

		  Arguments:
		      code: bytecode of the function.
		      defaults: defaults of the function.
		      closure: closure of the function.
		      globs: dictionary of global objects.

		  Returns:
		      A function object.
		  """
		  if isinstance(code, (tuple, list)):  # unpack previous dump
		    code, defaults, closure = code
		    if isinstance(defaults, list):
		      defaults = tuple(defaults)
		  code = marshal.loads(code.encode('raw_unicode_escape'))
		  if globs is None:
		    globs = globals()
		  return python_types.FunctionType(
		      code, globs, name=code.co_name, argdefs=defaults, closure=closure)


		class Progbar(object):
		  """Displays a progress bar.

		  Arguments:
		      target: Total number of steps expected, None if unknown.
		      interval: Minimum visual progress update interval (in seconds).
		  """

		  def __init__(self, target, width=30, verbose=1, interval=0.05):
		    self.width = width
		    if target is None:
		      target = -1
		    self.target = target
		    self.sum_values = {}
		    self.unique_values = []
		    self.start = time.time()
		    self.last_update = 0
		    self.interval = interval
		    self.total_width = 0
		    self.seen_so_far = 0
		    self.verbose = verbose

		  def update(self, current, values=None, force=False):
		    """Updates the progress bar.

		    Arguments:
		        current: Index of current step.
		        values: List of tuples (name, value_for_last_step).
		            The progress bar will display averages for these values.
		        force: Whether to force visual progress update.
		    """
		    values = values or []
		    for k, v in values:
		      if k not in self.sum_values:
		        self.sum_values[k] = [
		            v * (current - self.seen_so_far), current - self.seen_so_far
		        ]
		        self.unique_values.append(k)
		      else:
		        self.sum_values[k][0] += v * (current - self.seen_so_far)
		        self.sum_values[k][1] += (current - self.seen_so_far)
		    self.seen_so_far = current

		    now = time.time()
		    if self.verbose == 1:
		      if not force and (now - self.last_update) < self.interval:
		        return

		      prev_total_width = self.total_width
		      sys.stdout.write('\b' * prev_total_width)
		      sys.stdout.write('\r')

		      if self.target is not -1:
		        numdigits = int(np.floor(np.log10(self.target))) + 1
		        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
		        bar = barstr % (current, self.target)
		        prog = float(current) / self.target
		        prog_width = int(self.width * prog)
		        if prog_width > 0:
		          bar += ('=' * (prog_width - 1))
		          if current < self.target:
		            bar += '>'
		          else:
		            bar += '='
		        bar += ('.' * (self.width - prog_width))
		        bar += ']'
		        sys.stdout.write(bar)
		        self.total_width = len(bar)

		      if current:
		        time_per_unit = (now - self.start) / current
		      else:
		        time_per_unit = 0
		      eta = time_per_unit * (self.target - current)
		      info = ''
		      if current < self.target and self.target is not -1:
		        info += ' - ETA: %ds' % eta
		      else:
		        info += ' - %ds' % (now - self.start)
		      for k in self.unique_values:
		        info += ' - %s:' % k
		        if isinstance(self.sum_values[k], list):
		          avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
		          if abs(avg) > 1e-3:
		            info += ' %.4f' % avg
		          else:
		            info += ' %.4e' % avg
		        else:
		          info += ' %s' % self.sum_values[k]

		      self.total_width += len(info)
		      if prev_total_width > self.total_width:
		        info += ((prev_total_width - self.total_width) * ' ')

		      sys.stdout.write(info)
		      sys.stdout.flush()

		      if current >= self.target:
		        sys.stdout.write('\n')

		    if self.verbose == 2:
		      if current >= self.target:
		        info = '%ds' % (now - self.start)
		        for k in self.unique_values:
		          info += ' - %s:' % k
		          avg = self.sum_values[k][0] / max(1, self.sum_values[k][1])
		          if avg > 1e-3:
		            info += ' %.4f' % avg
		          else:
		            info += ' %.4e' % avg
		        sys.stdout.write(info + '\n')

		    self.last_update = now

		  def add(self, n, values=None):
		    self.update(self.seen_so_far + n, values)

	class io_utils_py:

		"""Utilities related to disk I/O."""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from collections import defaultdict
			import sys

			import numpy as np


		try:
		  import h5py  # pylint:disable=g-import-not-at-top
		except ImportError:
		  h5py = None


		class HDF5Matrix(object):
		  """Representation of HDF5 dataset to be used instead of a Numpy array.

		  Example:

		  ```python
		      x_data = HDF5Matrix('input/file.hdf5', 'data')
		      model.predict(x_data)
		  ```

		  Providing `start` and `end` allows use of a slice of the dataset.

		  Optionally, a normalizer function (or lambda) can be given. This will
		  be called on every slice of data retrieved.

		  Arguments:
		      datapath: string, path to a HDF5 file
		      dataset: string, name of the HDF5 dataset in the file specified
		          in datapath
		      start: int, start of desired slice of the specified dataset
		      end: int, end of desired slice of the specified dataset
		      normalizer: function to be called on data when retrieved

		  Returns:
		      An array-like HDF5 dataset.
		  """
		  refs = defaultdict(int)

		  def __init__(self, datapath, dataset, start=0, end=None, normalizer=None):
		    if h5py is None:
		      raise ImportError('The use of HDF5Matrix requires '
		                        'HDF5 and h5py installed.')

		    if datapath not in list(self.refs.keys()):
		      f = h5py.File(datapath)
		      self.refs[datapath] = f
		    else:
		      f = self.refs[datapath]
		    self.data = f[dataset]
		    self.start = start
		    if end is None:
		      self.end = self.data.shape[0]
		    else:
		      self.end = end
		    self.normalizer = normalizer

		  def __len__(self):
		    return self.end - self.start

		  def __getitem__(self, key):
		    if isinstance(key, slice):
		      start, stop = key.start, key.stop
		      if start is None:
		        start = 0
		      if stop is None:
		        stop = self.data.shape[0]
		      if stop + self.start <= self.end:
		        idx = slice(start + self.start, stop + self.start)
		      else:
		        raise IndexError
		    elif isinstance(key, int):
		      if key + self.start < self.end:
		        idx = key + self.start
		      else:
		        raise IndexError
		    elif isinstance(key, np.ndarray):
		      if np.max(key) + self.start < self.end:
		        idx = (self.start + key).tolist()
		      else:
		        raise IndexError
		    elif isinstance(key, list):
		      if max(key) + self.start < self.end:
		        idx = [x + self.start for x in key]
		      else:
		        raise IndexError
		    else:
		      raise IndexError
		    if self.normalizer is not None:
		      return self.normalizer(self.data[idx])
		    else:
		      return self.data[idx]

		  @property
		  def shape(self):
		    return (self.end - self.start,) + self.data.shape[1:]


		def ask_to_proceed_with_overwrite(filepath):
		  """Produces a prompt asking about overwriting a file.

		  Arguments:
		      filepath: the path to the file to be overwritten.

		  Returns:
		      True if we can proceed with overwrite, False otherwise.
		  """
		  get_input = input
		  if sys.version_info[:2] <= (2, 7):
		    get_input = raw_input
		  overwrite = get_input('[WARNING] %s already exists - overwrite? '
		                        '[y/n]' % (filepath))
		  while overwrite not in ['y', 'n']:
		    overwrite = get_input('Enter "y" (overwrite) or "n" (cancel).')
		  if overwrite == 'n':
		    return False
		  print('[TIP] Next time specify overwrite=True!')
		  return True

	class layer_utils_py:

		"""Utilities related to Keras layers.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.utils.conv_utils import convert_kernel


		def print_summary(model, line_length=None, positions=None):
		  """Prints a summary of a model.

		  Arguments:
		      model: Keras model instance.
		      line_length: total length of printed lines
		      positions: relative or absolute positions of log elements in each line.
		          If not provided, defaults to `[.33, .55, .67, 1.]`.
		  """
		  if model.__class__.__name__ == 'Sequential':
		    sequential_like = True
		  else:
		    sequential_like = True
		    for v in model.nodes_by_depth.values():
		      if (len(v) > 1) or (len(v) == 1 and len(v[0].inbound_layers) > 1):
		        # If the model has multiple nodes or if the nodes have
		        # multiple inbound_layers, the model is no longer sequential.
		        sequential_like = False
		        break

		  if sequential_like:
		    line_length = line_length or 65
		    positions = positions or [.45, .85, 1.]
		    if positions[-1] <= 1:
		      positions = [int(line_length * p) for p in positions]
		    # header names for the different log elements
		    to_display = ['Layer (type)', 'Output Shape', 'Param #']
		  else:
		    line_length = line_length or 100
		    positions = positions or [.33, .55, .67, 1.]
		    if positions[-1] <= 1:
		      positions = [int(line_length * p) for p in positions]
		    # header names for the different log elements
		    to_display = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to']
		    relevant_nodes = []
		    for v in model.nodes_by_depth.values():
		      relevant_nodes += v

		  def print_row(fields, positions):
		    line = ''
		    for i in range(len(fields)):
		      if i > 0:
		        line = line[:-1] + ' '
		      line += str(fields[i])
		      line = line[:positions[i]]
		      line += ' ' * (positions[i] - len(line))
		    print(line)

		  print('_' * line_length)
		  print_row(to_display, positions)
		  print('=' * line_length)

		  def print_layer_summary(layer):
		    try:
		      output_shape = layer.output_shape
		    except AttributeError:
		      output_shape = 'multiple'
		    name = layer.name
		    cls_name = layer.__class__.__name__
		    fields = [name + ' (' + cls_name + ')', output_shape, layer.count_params()]
		    print_row(fields, positions)

		  def print_layer_summary_with_connections(layer):
		    """Prints a summary for a single layer.

		    Arguments:
		        layer: target layer.
		    """
		    try:
		      output_shape = layer.output_shape
		    except AttributeError:
		      output_shape = 'multiple'
		    connections = []
		    for node in layer.inbound_nodes:
		      if relevant_nodes and node not in relevant_nodes:
		        # node is not part of the current network
		        continue
		      for i in range(len(node.inbound_layers)):
		        inbound_layer = node.inbound_layers[i].name
		        inbound_node_index = node.node_indices[i]
		        inbound_tensor_index = node.tensor_indices[i]
		        connections.append(inbound_layer + '[' + str(inbound_node_index) + ']['
		                           + str(inbound_tensor_index) + ']')

		    name = layer.name
		    cls_name = layer.__class__.__name__
		    if not connections:
		      first_connection = ''
		    else:
		      first_connection = connections[0]
		    fields = [
		        name + ' (' + cls_name + ')', output_shape,
		        layer.count_params(), first_connection
		    ]
		    print_row(fields, positions)
		    if len(connections) > 1:
		      for i in range(1, len(connections)):
		        fields = ['', '', '', connections[i]]
		        print_row(fields, positions)

		  layers = model.layers
		  for i in range(len(layers)):
		    if sequential_like:
		      print_layer_summary(layers[i])
		    else:
		      print_layer_summary_with_connections(layers[i])
		    if i == len(layers) - 1:
		      print('=' * line_length)
		    else:
		      print('_' * line_length)

		  trainable_count = int(
		      np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
		  non_trainable_count = int(
		      np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

		  print('Total params: {:,}'.format(trainable_count + non_trainable_count))
		  print('Trainable params: {:,}'.format(trainable_count))
		  print('Non-trainable params: {:,}'.format(non_trainable_count))
		  print('_' * line_length)


		def convert_all_kernels_in_model(model):
		  """Converts all convolution kernels in a model from Theano to TensorFlow.

		  Also works from TensorFlow to Theano.

		  Arguments:
		      model: target model for the conversion.
		  """
		  # Note: SeparableConvolution not included
		  # since only supported by TF.
		  conv_classes = {
		      'Conv1D',
		      'Conv2D',
		      'Conv3D',
		      'Conv2DTranspose',
		  }
		  to_assign = []
		  for layer in model.layers:
		    if layer.__class__.__name__ in conv_classes:
		      original_kernel = K.get_value(layer.kernel)
		      converted_kernel = convert_kernel(original_kernel)
		      to_assign.append((layer.kernel, converted_kernel))
		  K.batch_set_value(to_assign)


		def convert_dense_weights_data_format(dense,
		                                      previous_feature_map_shape,
		                                      target_data_format='channels_first'):
		  """Utility useful when changing a convnet's `data_format`.

		  When porting the weights of a convnet from one data format to the other,
		  if the convnet includes a `Flatten` layer
		  (applied to the last convolutional feature map)
		  followed by a `Dense` layer, the weights of that `Dense` layer
		  should be updated to reflect the new dimension ordering.

		  Arguments:
		      dense: The target `Dense` layer.
		      previous_feature_map_shape: A shape tuple of 3 integers,
		          e.g. `(512, 7, 7)`. The shape of the convolutional
		          feature map right before the `Flatten` layer that
		          came before the target `Dense` layer.
		      target_data_format: One of "channels_last", "channels_first".
		          Set it "channels_last"
		          if converting a "channels_first" model to "channels_last",
		          or reciprocally.
		  """
		  assert target_data_format in {'channels_last', 'channels_first'}
		  kernel, bias = dense.get_weights()
		  for i in range(kernel.shape[1]):
		    if target_data_format == 'channels_first':
		      c, h, w = previous_feature_map_shape
		      original_fm_shape = (h, w, c)
		      ki = kernel[:, i].reshape(original_fm_shape)
		      ki = np.transpose(ki, (2, 0, 1))  # last -> first
		    else:
		      h, w, c = previous_feature_map_shape
		      original_fm_shape = (c, h, w)
		      ki = kernel[:, i].reshape(original_fm_shape)
		      ki = np.transpose(ki, (1, 2, 0))  # first -> last
		    kernel[:, i] = np.reshape(ki, (np.prod(previous_feature_map_shape),))
		  dense.set_weights([kernel, bias])

	class np_utils_py:

		"""Numpy-related utilities."""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np


		def to_categorical(y, num_classes=None):
		  """Converts a class vector (integers) to binary class matrix.

		  E.g. for use with categorical_crossentropy.

		  Arguments:
		      y: class vector to be converted into a matrix
		          (integers from 0 to num_classes).
		      num_classes: total number of classes.

		  Returns:
		      A binary matrix representation of the input.
		  """
		  y = np.array(y, dtype='int').ravel()
		  if not num_classes:
		    num_classes = np.max(y) + 1
		  n = y.shape[0]
		  categorical = np.zeros((n, num_classes))
		  categorical[np.arange(n), y] = 1
		  return categorical


		def normalize(x, axis=-1, order=2):
		  """Normalizes a Numpy array.

		  Arguments:
		      x: Numpy array to normalize.
		      axis: axis along which to normalize.
		      order: Normalization order (e.g. 2 for L2 norm).

		  Returns:
		      A normalized copy of the array.
		  """
		  l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
		  l2[l2 == 0] = 1
		  return x / np.expand_dims(l2, axis)

	class vis_utils_py:

		"""Utilities related to model visualization."""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import os
			import sys

		try:
		  # pydot-ng is a fork of pydot that is better maintained.
		  import pydot_ng as pydot  # pylint: disable=g-import-not-at-top
		except ImportError:
		  # Fall back on pydot if necessary.
		  # Silence a `print` statement that occurs in case of import error,
		  # by temporarily replacing sys.stdout.
		  _stdout = sys.stdout
		  sys.stdout = sys.stderr
		  try:
		    import pydot  # pylint: disable=g-import-not-at-top
		  except ImportError:
		    pydot = None
		  finally:
		    # Restore sys.stdout.
		    sys.stdout = _stdout


		def _check_pydot():
		  try:
		    # Attempt to create an image of a blank graph
		    # to check the pydot/graphviz installation.
		    pydot.Dot.create(pydot.Dot())
		  except Exception:
		    # pydot raises a generic Exception here,
		    # so no specific class can be caught.
		    raise ImportError('Failed to import pydot. You must install pydot'
		                      ' and graphviz for `pydotprint` to work.')


		def model_to_dot(model, show_shapes=False, show_layer_names=True, rankdir='TB'):
		  """Convert a Keras model to dot format.

		  Arguments:
		      model: A Keras model instance.
		      show_shapes: whether to display shape information.
		      show_layer_names: whether to display layer names.
		      rankdir: `rankdir` argument passed to PyDot,
		          a string specifying the format of the plot:
		          'TB' creates a vertical plot;
		          'LR' creates a horizontal plot.

		  Returns:
		      A `pydot.Dot` instance representing the Keras model.
		  """
		  from tensorflow.contrib.keras.python.keras.layers.wrappers import Wrapper  # pylint: disable=g-import-not-at-top
		  from tensorflow.contrib.keras.python.keras.models import Sequential  # pylint: disable=g-import-not-at-top

		  _check_pydot()
		  dot = pydot.Dot()
		  dot.set('rankdir', rankdir)
		  dot.set('concentrate', True)
		  dot.set_node_defaults(shape='record')

		  if isinstance(model, Sequential):
		    if not model.built:
		      model.build()
		    model = model.model
		  layers = model.layers

		  # Create graph nodes.
		  for layer in layers:
		    layer_id = str(id(layer))

		    # Append a wrapped layer's label to node's label, if it exists.
		    layer_name = layer.name
		    class_name = layer.__class__.__name__
		    if isinstance(layer, Wrapper):
		      layer_name = '{}({})'.format(layer_name, layer.layer.name)
		      child_class_name = layer.layer.__class__.__name__
		      class_name = '{}({})'.format(class_name, child_class_name)

		    # Create node's label.
		    if show_layer_names:
		      label = '{}: {}'.format(layer_name, class_name)
		    else:
		      label = class_name

		    # Rebuild the label as a table including input/output shapes.
		    if show_shapes:
		      try:
		        outputlabels = str(layer.output_shape)
		      except AttributeError:
		        outputlabels = 'multiple'
		      if hasattr(layer, 'input_shape'):
		        inputlabels = str(layer.input_shape)
		      elif hasattr(layer, 'input_shapes'):
		        inputlabels = ', '.join([str(ishape) for ishape in layer.input_shapes])
		      else:
		        inputlabels = 'multiple'
		      label = '%s\n|{input:|output:}|{{%s}|{%s}}' % (label, inputlabels,
		                                                     outputlabels)
		    node = pydot.Node(layer_id, label=label)
		    dot.add_node(node)

		  # Connect nodes with edges.
		  for layer in layers:
		    layer_id = str(id(layer))
		    for i, node in enumerate(layer.inbound_nodes):
		      node_key = layer.name + '_ib-' + str(i)
		      if node_key in model.container_nodes:
		        for inbound_layer in node.inbound_layers:
		          inbound_layer_id = str(id(inbound_layer))
		          layer_id = str(id(layer))
		          dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
		  return dot


		def plot_model(model,
		               to_file='model.png',
		               show_shapes=False,
		               show_layer_names=True,
		               rankdir='TB'):
		  """Converts a Keras model to dot format and save to a file.

		  Arguments:
		      model: A Keras model instance
		      to_file: File name of the plot image.
		      show_shapes: whether to display shape information.
		      show_layer_names: whether to display layer names.
		      rankdir: `rankdir` argument passed to PyDot,
		          a string specifying the format of the plot:
		          'TB' creates a vertical plot;
		          'LR' creates a horizontal plot.
		  """
		  dot = model_to_dot(model, show_shapes, show_layer_names, rankdir)
		  _, extension = os.path.splitext(to_file)
		  if not extension:
		    extension = 'png'
		  else:
		    extension = extension[1:]
		  dot.write(to_file, format=extension)


# preprocessing
class preprocessing_folder:

	class __init__py:
		"""Data preprocessing module.
		"""
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.preprocessing import image
		from tensorflow.contrib.keras.python.keras.preprocessing import sequence
		from tensorflow.contrib.keras.python.keras.preprocessing import text

	class image_py:

		"""Fairly basic set of tools for real-time data augmentation on image data.

		Can easily be extended to include new transformations,
		new preprocessing methods, etc...
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import os
			import re
			import threading

			import numpy as np
			from six.moves import range  # pylint: disable=redefined-builtin

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.python.platform import tf_logging as logging


		# pylint: disable=g-import-not-at-top
		try:
		  from PIL import Image as pil_image
		except ImportError:
		  pil_image = None
		try:
		  from scipy import linalg
		  import scipy.ndimage as ndi
		except ImportError:
		  linalg = None
		  ndi = None
		# pylint: enable=g-import-not-at-top


		def random_rotation(x,
		                    rg,
		                    row_axis=1,
		                    col_axis=2,
		                    channel_axis=0,
		                    fill_mode='nearest',
		                    cval=0.):
		  """Performs a random rotation of a Numpy image tensor.

		  Arguments:
		      x: Input tensor. Must be 3D.
		      rg: Rotation range, in degrees.
		      row_axis: Index of axis for rows in the input tensor.
		      col_axis: Index of axis for columns in the input tensor.
		      channel_axis: Index of axis for channels in the input tensor.
		      fill_mode: Points outside the boundaries of the input
		          are filled according to the given mode
		          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		      cval: Value used for points outside the boundaries
		          of the input if `mode='constant'`.

		  Returns:
		      Rotated Numpy image tensor.
		  """
		  theta = np.pi / 180 * np.random.uniform(-rg, rg)
		  rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
		                              [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

		  h, w = x.shape[row_axis], x.shape[col_axis]
		  transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
		  x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
		  return x


		def random_shift(x,
		                 wrg,
		                 hrg,
		                 row_axis=1,
		                 col_axis=2,
		                 channel_axis=0,
		                 fill_mode='nearest',
		                 cval=0.):
		  """Performs a random spatial shift of a Numpy image tensor.

		  Arguments:
		      x: Input tensor. Must be 3D.
		      wrg: Width shift range, as a float fraction of the width.
		      hrg: Height shift range, as a float fraction of the height.
		      row_axis: Index of axis for rows in the input tensor.
		      col_axis: Index of axis for columns in the input tensor.
		      channel_axis: Index of axis for channels in the input tensor.
		      fill_mode: Points outside the boundaries of the input
		          are filled according to the given mode
		          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		      cval: Value used for points outside the boundaries
		          of the input if `mode='constant'`.

		  Returns:
		      Shifted Numpy image tensor.
		  """
		  h, w = x.shape[row_axis], x.shape[col_axis]
		  tx = np.random.uniform(-hrg, hrg) * h
		  ty = np.random.uniform(-wrg, wrg) * w
		  translation_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

		  transform_matrix = translation_matrix  # no need to do offset
		  x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
		  return x


		def random_shear(x,
		                 intensity,
		                 row_axis=1,
		                 col_axis=2,
		                 channel_axis=0,
		                 fill_mode='nearest',
		                 cval=0.):
		  """Performs a random spatial shear of a Numpy image tensor.

		  Arguments:
		      x: Input tensor. Must be 3D.
		      intensity: Transformation intensity.
		      row_axis: Index of axis for rows in the input tensor.
		      col_axis: Index of axis for columns in the input tensor.
		      channel_axis: Index of axis for channels in the input tensor.
		      fill_mode: Points outside the boundaries of the input
		          are filled according to the given mode
		          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		      cval: Value used for points outside the boundaries
		          of the input if `mode='constant'`.

		  Returns:
		      Sheared Numpy image tensor.
		  """
		  shear = np.random.uniform(-intensity, intensity)
		  shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0],
		                           [0, 0, 1]])

		  h, w = x.shape[row_axis], x.shape[col_axis]
		  transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
		  x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
		  return x


		def random_zoom(x,
		                zoom_range,
		                row_axis=1,
		                col_axis=2,
		                channel_axis=0,
		                fill_mode='nearest',
		                cval=0.):
		  """Performs a random spatial zoom of a Numpy image tensor.

		  Arguments:
		      x: Input tensor. Must be 3D.
		      zoom_range: Tuple of floats; zoom range for width and height.
		      row_axis: Index of axis for rows in the input tensor.
		      col_axis: Index of axis for columns in the input tensor.
		      channel_axis: Index of axis for channels in the input tensor.
		      fill_mode: Points outside the boundaries of the input
		          are filled according to the given mode
		          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		      cval: Value used for points outside the boundaries
		          of the input if `mode='constant'`.

		  Returns:
		      Zoomed Numpy image tensor.

		  Raises:
		      ValueError: if `zoom_range` isn't a tuple.
		  """
		  if len(zoom_range) != 2:
		    raise ValueError('zoom_range should be a tuple or list of two floats. '
		                     'Received arg: ', zoom_range)

		  if zoom_range[0] == 1 and zoom_range[1] == 1:
		    zx, zy = 1, 1
		  else:
		    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
		  zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])

		  h, w = x.shape[row_axis], x.shape[col_axis]
		  transform_matrix = transform_matrix_offset_center(zoom_matrix, h, w)
		  x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
		  return x


		def random_channel_shift(x, intensity, channel_axis=0):
		  x = np.rollaxis(x, channel_axis, 0)
		  min_x, max_x = np.min(x), np.max(x)
		  channel_images = [
		      np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x,
		              max_x) for x_channel in x
		  ]
		  x = np.stack(channel_images, axis=0)
		  x = np.rollaxis(x, 0, channel_axis + 1)
		  return x


		def transform_matrix_offset_center(matrix, x, y):
		  o_x = float(x) / 2 + 0.5
		  o_y = float(y) / 2 + 0.5
		  offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
		  reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
		  transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
		  return transform_matrix


		def apply_transform(x,
		                    transform_matrix,
		                    channel_axis=0,
		                    fill_mode='nearest',
		                    cval=0.):
		  """Apply the image transformation specified by a matrix.

		  Arguments:
		      x: 2D numpy array, single image.
		      transform_matrix: Numpy array specifying the geometric transformation.
		      channel_axis: Index of axis for channels in the input tensor.
		      fill_mode: Points outside the boundaries of the input
		          are filled according to the given mode
		          (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
		      cval: Value used for points outside the boundaries
		          of the input if `mode='constant'`.

		  Returns:
		      The transformed version of the input.
		  """
		  x = np.rollaxis(x, channel_axis, 0)
		  final_affine_matrix = transform_matrix[:2, :2]
		  final_offset = transform_matrix[:2, 2]
		  channel_images = [
		      ndi.interpolation.affine_transform(
		          x_channel,
		          final_affine_matrix,
		          final_offset,
		          order=0,
		          mode=fill_mode,
		          cval=cval) for x_channel in x
		  ]
		  x = np.stack(channel_images, axis=0)
		  x = np.rollaxis(x, 0, channel_axis + 1)
		  return x


		def flip_axis(x, axis):
		  x = np.asarray(x).swapaxes(axis, 0)
		  x = x[::-1, ...]
		  x = x.swapaxes(0, axis)
		  return x


		def array_to_img(x, data_format=None, scale=True):
		  """Converts a 3D Numpy array to a PIL Image instance.

		  Arguments:
		      x: Input Numpy array.
		      data_format: Image data format.
		      scale: Whether to rescale image values
		          to be within [0, 255].

		  Returns:
		      A PIL Image instance.

		  Raises:
		      ImportError: if PIL is not available.
		      ValueError: if invalid `x` or `data_format` is passed.
		  """
		  if pil_image is None:
		    raise ImportError('Could not import PIL.Image. '
		                      'The use of `array_to_img` requires PIL.')
		  x = np.asarray(x, dtype=K.floatx())
		  if x.ndim != 3:
		    raise ValueError('Expected image array to have rank 3 (single image). '
		                     'Got array with shape:', x.shape)

		  if data_format is None:
		    data_format = K.image_data_format()
		  if data_format not in {'channels_first', 'channels_last'}:
		    raise ValueError('Invalid data_format:', data_format)

		  # Original Numpy array x has format (height, width, channel)
		  # or (channel, height, width)
		  # but target PIL image has format (width, height, channel)
		  if data_format == 'channels_first':
		    x = x.transpose(1, 2, 0)
		  if scale:
		    x = x + max(-np.min(x), 0)  # pylint: disable=g-no-augmented-assignment
		    x_max = np.max(x)
		    if x_max != 0:
		      x /= x_max
		    x *= 255
		  if x.shape[2] == 3:
		    # RGB
		    return pil_image.fromarray(x.astype('uint8'), 'RGB')
		  elif x.shape[2] == 1:
		    # grayscale
		    return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
		  else:
		    raise ValueError('Unsupported channel number: ', x.shape[2])


		def img_to_array(img, data_format=None):
		  """Converts a PIL Image instance to a Numpy array.

		  Arguments:
		      img: PIL Image instance.
		      data_format: Image data format.

		  Returns:
		      A 3D Numpy array.

		  Raises:
		      ValueError: if invalid `img` or `data_format` is passed.
		  """
		  if data_format is None:
		    data_format = K.image_data_format()
		  if data_format not in {'channels_first', 'channels_last'}:
		    raise ValueError('Unknown data_format: ', data_format)
		  # Numpy array x has format (height, width, channel)
		  # or (channel, height, width)
		  # but original PIL image has format (width, height, channel)
		  x = np.asarray(img, dtype=K.floatx())
		  if len(x.shape) == 3:
		    if data_format == 'channels_first':
		      x = x.transpose(2, 0, 1)
		  elif len(x.shape) == 2:
		    if data_format == 'channels_first':
		      x = x.reshape((1, x.shape[0], x.shape[1]))
		    else:
		      x = x.reshape((x.shape[0], x.shape[1], 1))
		  else:
		    raise ValueError('Unsupported image shape: ', x.shape)
		  return x

		# load image file into array with a given target_size rather than original size of image
		def load_img(path, grayscale=False, target_size=None):
		  """Loads an image into PIL format.

		  Arguments:
		      path: Path to image file
		      grayscale: Boolean, whether to load the image as grayscale.
		      target_size: Either `None` (default to original size)
		          or tuple of ints `(img_height, img_width)`.

		  Returns:
		      A PIL Image instance.

		  Raises:
		      ImportError: if PIL is not available.
		  """
		  if pil_image is None:
		    raise ImportError('Could not import PIL.Image. '
		                      'The use of `array_to_img` requires PIL.')
		  img = pil_image.open(path)
		  if grayscale:
		    if img.mode != 'L':
		      img = img.convert('L')
		  else:
		    if img.mode != 'RGB':
		      img = img.convert('RGB')
		  if target_size:
		    hw_tuple = (target_size[1], target_size[0])
		    if img.size != hw_tuple:
		      img = img.resize(hw_tuple)
		  return img


		def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
		  return [
		      os.path.join(root, f)
		      for root, _, files in os.walk(directory) for f in files
		      if re.match(r'([\w]+\.(?:' + ext + '))', f)
		  ]


		class ImageDataGenerator(object):
		  """Generate minibatches of image data with real-time data augmentation.

		  Arguments:
		      featurewise_center: set input mean to 0 over the dataset.
		      samplewise_center: set each sample mean to 0.
		      featurewise_std_normalization: divide inputs by std of the dataset.
		      samplewise_std_normalization: divide each input by its std.
		      zca_whitening: apply ZCA whitening.
		      zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
		      rotation_range: degrees (0 to 180).
		      width_shift_range: fraction of total width.
		      height_shift_range: fraction of total height.
		      shear_range: shear intensity (shear angle in radians).
		      zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
		          in the range [1-z, 1+z]. A sequence of two can be passed instead
		          to select this range.
		      channel_shift_range: shift range for each channels.
		      fill_mode: points outside the boundaries are filled according to the
		          given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
		          is 'nearest'.
		      cval: value used for points outside the boundaries when fill_mode is
		          'constant'. Default is 0.
		      horizontal_flip: whether to randomly flip images horizontally.
		      vertical_flip: whether to randomly flip images vertically.
		      rescale: rescaling factor. If None or 0, no rescaling is applied,
		          otherwise we multiply the data by the value provided
		          (before applying any other transformation).
		      preprocessing_function: function that will be implied on each input.
		          The function will run before any other modification on it.
		          The function should take one argument:
		          one image (Numpy tensor with rank 3),
		          and should output a Numpy tensor with the same shape.
		      data_format: 'channels_first' or 'channels_last'. In 'channels_first'
		        mode, the channels dimension
		          (the depth) is at index 1, in 'channels_last' mode it is at index 3.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		  """

		  def __init__(self,
		               featurewise_center=False,
		               samplewise_center=False,
		               featurewise_std_normalization=False,
		               samplewise_std_normalization=False,
		               zca_whitening=False,
		               zca_epsilon=1e-6,
		               rotation_range=0.,
		               width_shift_range=0.,
		               height_shift_range=0.,
		               shear_range=0.,
		               zoom_range=0.,
		               channel_shift_range=0.,
		               fill_mode='nearest',
		               cval=0.,
		               horizontal_flip=False,
		               vertical_flip=False,
		               rescale=None,
		               preprocessing_function=None,
		               data_format=None):
		    if data_format is None:
		      data_format = K.image_data_format()
		    self.featurewise_center = featurewise_center
		    self.samplewise_center = samplewise_center
		    self.featurewise_std_normalization = featurewise_std_normalization
		    self.samplewise_std_normalization = samplewise_std_normalization
		    self.zca_whitening = zca_whitening
		    self.zca_epsilon = zca_epsilon
		    self.rotation_range = rotation_range
		    self.width_shift_range = width_shift_range
		    self.height_shift_range = height_shift_range
		    self.shear_range = shear_range
		    self.zoom_range = zoom_range
		    self.channel_shift_range = channel_shift_range
		    self.fill_mode = fill_mode
		    self.cval = cval
		    self.horizontal_flip = horizontal_flip
		    self.vertical_flip = vertical_flip
		    self.rescale = rescale
		    self.preprocessing_function = preprocessing_function

		    if data_format not in {'channels_last', 'channels_first'}:
		      raise ValueError(
		          'data_format should be "channels_last" (channel after row and '
		          'column) or "channels_first" (channel before row and column). '
		          'Received arg: ', data_format)
		    self.data_format = data_format
		    if data_format == 'channels_first':
		      self.channel_axis = 1
		      self.row_axis = 2
		      self.col_axis = 3
		    if data_format == 'channels_last':
		      self.channel_axis = 3
		      self.row_axis = 1
		      self.col_axis = 2

		    self.mean = None
		    self.std = None
		    self.principal_components = None

		    if np.isscalar(zoom_range):
		      self.zoom_range = [1 - zoom_range, 1 + zoom_range]
		    elif len(zoom_range) == 2:
		      self.zoom_range = [zoom_range[0], zoom_range[1]]
		    else:
		      raise ValueError('zoom_range should be a float or '
		                       'a tuple or list of two floats. '
		                       'Received arg: ', zoom_range)

		  def flow(self,
		           x,
		           y=None,
		           batch_size=32,
		           shuffle=True,
		           seed=None,
		           save_to_dir=None,
		           save_prefix='',
		           save_format='png'):
		    return NumpyArrayIterator(
		        x,
		        y,
		        self,
		        batch_size=batch_size,
		        shuffle=shuffle,
		        seed=seed,
		        data_format=self.data_format,
		        save_to_dir=save_to_dir,
		        save_prefix=save_prefix,
		        save_format=save_format)

		  def flow_from_directory(self,
		                          directory,
		                          target_size=(256, 256),
		                          color_mode='rgb',
		                          classes=None,
		                          class_mode='categorical',
		                          batch_size=32,
		                          shuffle=True,
		                          seed=None,
		                          save_to_dir=None,
		                          save_prefix='',
		                          save_format='png',
		                          follow_links=False):
		    return DirectoryIterator(
		        directory,
		        self,
		        target_size=target_size,
		        color_mode=color_mode,
		        classes=classes,
		        class_mode=class_mode,
		        data_format=self.data_format,
		        batch_size=batch_size,
		        shuffle=shuffle,
		        seed=seed,
		        save_to_dir=save_to_dir,
		        save_prefix=save_prefix,
		        save_format=save_format,
		        follow_links=follow_links)

		  def standardize(self, x):
		    """Apply the normalization configuration to a batch of inputs.

		    Arguments:
		        x: batch of inputs to be normalized.

		    Returns:
		        The inputs, normalized.
		    """
		    if self.preprocessing_function:
		      x = self.preprocessing_function(x)
		    if self.rescale:
		      x *= self.rescale
		    # x is a single image, so it doesn't have image number at index 0
		    img_channel_axis = self.channel_axis - 1
		    if self.samplewise_center:
		      x -= np.mean(x, axis=img_channel_axis, keepdims=True)
		    if self.samplewise_std_normalization:
		      x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

		    if self.featurewise_center:
		      if self.mean is not None:
		        x -= self.mean
		      else:
		        logging.warning('This ImageDataGenerator specifies '
		                        '`featurewise_center`, but it hasn\'t'
		                        'been fit on any training data. Fit it '
		                        'first by calling `.fit(numpy_data)`.')
		    if self.featurewise_std_normalization:
		      if self.std is not None:
		        x /= (self.std + 1e-7)
		      else:
		        logging.warning('This ImageDataGenerator specifies '
		                        '`featurewise_std_normalization`, but it hasn\'t'
		                        'been fit on any training data. Fit it '
		                        'first by calling `.fit(numpy_data)`.')
		    if self.zca_whitening:
		      if self.principal_components is not None:
		        flatx = np.reshape(x, (x.size))
		        whitex = np.dot(flatx, self.principal_components)
		        x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
		      else:
		        logging.warning('This ImageDataGenerator specifies '
		                        '`zca_whitening`, but it hasn\'t'
		                        'been fit on any training data. Fit it '
		                        'first by calling `.fit(numpy_data)`.')
		    return x

		  def random_transform(self, x):
		    """Randomly augment a single image tensor.

		    Arguments:
		        x: 3D tensor, single image.

		    Returns:
		        A randomly transformed version of the input (same shape).

		    Raises:
		        ImportError: if Scipy is not available.
		    """
		    if ndi is None:
		      raise ImportError('Scipy is required for image transformations.')

		    # x is a single image, so it doesn't have image number at index 0
		    img_row_axis = self.row_axis - 1
		    img_col_axis = self.col_axis - 1
		    img_channel_axis = self.channel_axis - 1

		    # use composition of homographies
		    # to generate final transform that needs to be applied
		    if self.rotation_range:
		      theta = np.pi / 180 * np.random.uniform(-self.rotation_range,
		                                              self.rotation_range)
		    else:
		      theta = 0

		    if self.height_shift_range:
		      tx = np.random.uniform(-self.height_shift_range,
		                             self.height_shift_range) * x.shape[img_row_axis]
		    else:
		      tx = 0

		    if self.width_shift_range:
		      ty = np.random.uniform(-self.width_shift_range,
		                             self.width_shift_range) * x.shape[img_col_axis]
		    else:
		      ty = 0

		    if self.shear_range:
		      shear = np.random.uniform(-self.shear_range, self.shear_range)
		    else:
		      shear = 0

		    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
		      zx, zy = 1, 1
		    else:
		      zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)

		    transform_matrix = None
		    if theta != 0:
		      rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
		                                  [np.sin(theta),
		                                   np.cos(theta), 0], [0, 0, 1]])
		      transform_matrix = rotation_matrix

		    if tx != 0 or ty != 0:
		      shift_matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
		      transform_matrix = shift_matrix if transform_matrix is None else np.dot(
		          transform_matrix, shift_matrix)

		    if shear != 0:
		      shear_matrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0],
		                               [0, 0, 1]])
		      transform_matrix = shear_matrix if transform_matrix is None else np.dot(
		          transform_matrix, shear_matrix)

		    if zx != 1 or zy != 1:
		      zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
		      transform_matrix = zoom_matrix if transform_matrix is None else np.dot(
		          transform_matrix, zoom_matrix)

		    if transform_matrix is not None:
		      h, w = x.shape[img_row_axis], x.shape[img_col_axis]
		      transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
		      x = apply_transform(
		          x,
		          transform_matrix,
		          img_channel_axis,
		          fill_mode=self.fill_mode,
		          cval=self.cval)

		    if self.channel_shift_range != 0:
		      x = random_channel_shift(x, self.channel_shift_range, img_channel_axis)
		    if self.horizontal_flip:
		      if np.random.random() < 0.5:
		        x = flip_axis(x, img_col_axis)

		    if self.vertical_flip:
		      if np.random.random() < 0.5:
		        x = flip_axis(x, img_row_axis)

		    return x

		  def fit(self, x, augment=False, rounds=1, seed=None):
		    """Fits internal statistics to some sample data.

		    Required for featurewise_center, featurewise_std_normalization
		    and zca_whitening.

		    Arguments:
		        x: Numpy array, the data to fit on. Should have rank 4.
		            In case of grayscale data,
		            the channels axis should have value 1, and in case
		            of RGB data, it should have value 3.
		        augment: Whether to fit on randomly augmented samples
		        rounds: If `augment`,
		            how many augmentation passes to do over the data
		        seed: random seed.

		    Raises:
		        ValueError: in case of invalid input `x`.
		        ImportError: if Scipy is not available.
		    """
		    x = np.asarray(x, dtype=K.floatx())
		    if x.ndim != 4:
		      raise ValueError('Input to `.fit()` should have rank 4. '
		                       'Got array with shape: ' + str(x.shape))
		    if x.shape[self.channel_axis] not in {1, 3, 4}:
		      raise ValueError(
		          'Expected input to be images (as Numpy array) '
		          'following the data format convention "' + self.data_format + '" '
		          '(channels on axis ' + str(self.channel_axis) + '), i.e. expected '
		          'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
		          'However, it was passed an array with shape ' + str(x.shape) + ' (' +
		          str(x.shape[self.channel_axis]) + ' channels).')

		    if seed is not None:
		      np.random.seed(seed)

		    x = np.copy(x)
		    if augment:
		      ax = np.zeros(
		          tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
		      for r in range(rounds):
		        for i in range(x.shape[0]):
		          ax[i + r * x.shape[0]] = self.random_transform(x[i])
		      x = ax

		    if self.featurewise_center:
		      self.mean = np.mean(x, axis=(0, self.row_axis, self.col_axis))
		      broadcast_shape = [1, 1, 1]
		      broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
		      self.mean = np.reshape(self.mean, broadcast_shape)
		      x -= self.mean

		    if self.featurewise_std_normalization:
		      self.std = np.std(x, axis=(0, self.row_axis, self.col_axis))
		      broadcast_shape = [1, 1, 1]
		      broadcast_shape[self.channel_axis - 1] = x.shape[self.channel_axis]
		      self.std = np.reshape(self.std, broadcast_shape)
		      x /= (self.std + K.epsilon())

		    if self.zca_whitening:
		      if linalg is None:
		        raise ImportError('Scipy is required for zca_whitening.')

		      flat_x = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
		      sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
		      u, s, _ = linalg.svd(sigma)
		      self.principal_components = np.dot(
		          np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)


		class Iterator(object):
		  """Abstract base class for image data iterators.

		  Arguments:
		      n: Integer, total number of samples in the dataset to loop over.
		      batch_size: Integer, size of a batch.
		      shuffle: Boolean, whether to shuffle the data between epochs.
		      seed: Random seeding for data shuffling.
		  """
		  # create attributes including index_generator for each batch's sample indices
		  def __init__(self, n, batch_size, shuffle, seed):
		    self.n = n
		    self.batch_size = batch_size
		    self.shuffle = shuffle
		    self.batch_index = 0
		    self.total_batches_seen = 0
		    self.lock = threading.Lock()
		    self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

		  def reset(self):
		    self.batch_index = 0

			# 1. get index of all samples; 2. randomize index; 3. get current_index and batch_size to extract a batch of samples; 4. handle the last batch less than a batch_size
		  def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
		    # Ensure self.batch_index is 0.
		    self.reset()
		    while 1:
		      if seed is not None:
		        np.random.seed(seed + self.total_batches_seen)
		      if self.batch_index == 0:
		        index_array = np.arange(n)
		        if shuffle:
		          index_array = np.random.permutation(n)

		      current_index = (self.batch_index * batch_size) % n
		      if n > current_index + batch_size:
		        current_batch_size = batch_size
		        self.batch_index += 1
		      else:
		        current_batch_size = n - current_index
		        self.batch_index = 0
		      self.total_batches_seen += 1
		      yield (index_array[current_index:current_index + current_batch_size],
		             current_index, current_batch_size)

		  def __iter__(self):  # pylint: disable=non-iterator-returned
		    # Needed if we want to do something like:
		    # for x, y in data_gen.flow(...):
		    return self

		  def __next__(self, *args, **kwargs):
		    return self.next(*args, **kwargs)

		# __init__: convert arrays into batch_iterator
		# next: throw out a batch, in array
		class NumpyArrayIterator(Iterator):
		  """Iterator yielding data from a Numpy array.

		  Arguments:
		      x: Numpy array of input data.
		      y: Numpy array of targets data.
		      image_data_generator: Instance of `ImageDataGenerator`
		          to use for random transformations and normalization.
		      batch_size: Integer, size of a batch.
		      shuffle: Boolean, whether to shuffle the data between epochs.
		      seed: Random seed for data shuffling.
		      data_format: String, one of `channels_first`, `channels_last`.
		      save_to_dir: Optional directory where to save the pictures
		          being yielded, in a viewable format. This is useful
		          for visualizing the random transformations being
		          applied, for debugging purposes.
		      save_prefix: String prefix to use for saving sample
		          images (if `save_to_dir` is set).
		      save_format: Format to use for saving sample images
		          (if `save_to_dir` is set).
		  """

		  def __init__(self,
		               x,
		               y,
		               image_data_generator,
		               batch_size=32,
		               shuffle=False,
		               seed=None,
		               data_format=None,
		               save_to_dir=None,
		               save_prefix='',
		               save_format='png'):
		    if y is not None and len(x) != len(y):
		      raise ValueError('X (images tensor) and y (labels) '
		                       'should have the same length. '
		                       'Found: X.shape = %s, y.shape = %s' %
		                       (np.asarray(x).shape, np.asarray(y).shape))

		    if data_format is None:
		      data_format = K.image_data_format()
		    self.x = np.asarray(x, dtype=K.floatx())

		    if self.x.ndim != 4:
		      raise ValueError('Input data in `NumpyArrayIterator` '
		                       'should have rank 4. You passed an array '
		                       'with shape', self.x.shape)
		    channels_axis = 3 if data_format == 'channels_last' else 1
		    if self.x.shape[channels_axis] not in {1, 3, 4}:
		      raise ValueError(
		          'NumpyArrayIterator is set to use the '
		          'data format convention "' + data_format + '" '
		          '(channels on axis ' + str(channels_axis) + '), i.e. expected '
		          'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
		          'However, it was passed an array with shape ' + str(self.x.shape) +
		          ' (' + str(self.x.shape[channels_axis]) + ' channels).')
		    if y is not None:
		      self.y = np.asarray(y)
		    else:
		      self.y = None
		    self.image_data_generator = image_data_generator
		    self.data_format = data_format
		    self.save_to_dir = save_to_dir
		    self.save_prefix = save_prefix
		    self.save_format = save_format
		    super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle,
		                                             seed)

		  def next(self):
		    """For python 2.x.

		    Returns:
		        The next batch.
		    """
		    # Keeps under lock only the mechanism which advances
		    # the indexing of each batch.
		    with self.lock:
		      index_array, current_index, current_batch_size = next(
		          self.index_generator)
		    # The transformation of images is not under thread lock
		    # so it can be done in parallel
		    batch_x = np.zeros(
		        tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
		    for i, j in enumerate(index_array):
		      x = self.x[j]
		      x = self.image_data_generator.random_transform(x.astype(K.floatx()))
		      x = self.image_data_generator.standardize(x)
		      batch_x[i] = x
		    if self.save_to_dir:
		      for i in range(current_batch_size):
		        img = array_to_img(batch_x[i], self.data_format, scale=True)
		        fname = '{prefix}_{index}_{hash}.{format}'.format(
		            prefix=self.save_prefix,
		            index=current_index + i,
		            hash=np.random.randint(1e4),
		            format=self.save_format)
		        img.save(os.path.join(self.save_to_dir, fname))
		    if self.y is None:
		      return batch_x
		    batch_y = self.y[index_array]
		    return batch_x, batch_y

		# __init__: convert images of folders into batch_iterator
		# next: throw out a batch, in array
		class DirectoryIterator(Iterator):
		  """Iterator capable of reading images from a directory on disk.

		  Arguments: # important to read
		      directory: Path to the directory to read images from.
		          Each subdirectory in this directory will be
		          considered to contain images from one class,
		          or alternatively you could specify class subdirectories
		          via the `classes` argument.
		      image_data_generator: Instance of `ImageDataGenerator`
		          to use for random transformations and normalization.
		      target_size: tuple of integers, dimensions to resize input images to.
		      color_mode: One of `"rgb"`, `"grayscale"`. Color mode to read images.
		      classes: Optional list of strings, names of sudirectories
		          containing images from each class (e.g. `["dogs", "cats"]`).
		          It will be computed automatically if not set.
		      class_mode: Mode for yielding the targets:
		          `"binary"`: binary targets (if there are only two classes),
		          `"categorical"`: categorical targets,
		          `"sparse"`: integer targets,
		          `"input"`: targets are images identical to input images (mainly
		              used to work with autoencoders),
		          `None`: no targets get yielded (only input images are yielded).
		      batch_size: Integer, size of a batch.
		      shuffle: Boolean, whether to shuffle the data between epochs.
		      seed: Random seed for data shuffling.
		      data_format: String, one of `channels_first`, `channels_last`.
		      save_to_dir: Optional directory where to save the pictures
		          being yielded, in a viewable format. This is useful
		          for visualizing the random transformations being
		          applied, for debugging purposes.
		      save_prefix: String prefix to use for saving sample
		          images (if `save_to_dir` is set).
		      save_format: Format to use for saving sample images
		          (if `save_to_dir` is set).
		  """
		  # create a number of attributes and added Iterator super-class attributes
		  def __init__(self,
		               directory,
		               image_data_generator,
		               target_size=(256, 256),
		               color_mode='rgb',
		               classes=None,
		               class_mode='categorical',
		               batch_size=32,
		               shuffle=True,
		               seed=None,
		               data_format=None,
		               save_to_dir=None,
		               save_prefix='',
		               save_format='png',
		               follow_links=False):
		    if data_format is None:
		      data_format = K.image_data_format()
		    self.directory = directory
		    self.image_data_generator = image_data_generator
		    self.target_size = tuple(target_size)
		    if color_mode not in {'rgb', 'grayscale'}:
		      raise ValueError('Invalid color mode:', color_mode,
		                       '; expected "rgb" or "grayscale".')
		    self.color_mode = color_mode
		    self.data_format = data_format
		    if self.color_mode == 'rgb':
		      if self.data_format == 'channels_last':
		        self.image_shape = self.target_size + (3,)
		      else:
		        self.image_shape = (3,) + self.target_size
		    else:
		      if self.data_format == 'channels_last':
		        self.image_shape = self.target_size + (1,)
		      else:
		        self.image_shape = (1,) + self.target_size
		    self.classes = classes
		    if class_mode not in {'categorical', 'binary', 'sparse', 'input', None}:
		      raise ValueError('Invalid class_mode:', class_mode,
		                       '; expected one of "categorical", '
		                       '"binary", "sparse", "input"'
		                       ' or None.')
		    self.class_mode = class_mode
		    self.save_to_dir = save_to_dir
		    self.save_prefix = save_prefix
		    self.save_format = save_format

		    white_list_formats = {'png', 'jpg', 'jpeg', 'bmp'}

		    # first, count the number of samples and classes
		    self.samples = 0

		    if not classes:
		      classes = []
		      for subdir in sorted(os.listdir(directory)):
		        if os.path.isdir(os.path.join(directory, subdir)):
		          classes.append(subdir)
		    self.num_class = len(classes)
		    self.class_indices = dict(zip(classes, range(len(classes))))

		    def _recursive_list(subpath):
		      return sorted(
		          os.walk(subpath, followlinks=follow_links), key=lambda tpl: tpl[0])

		    for subdir in classes:
		      subpath = os.path.join(directory, subdir)
		      for root, _, files in _recursive_list(subpath):
		        for fname in files:
		          is_valid = False
		          for extension in white_list_formats:
		            if fname.lower().endswith('.' + extension):
		              is_valid = True
		              break
		          if is_valid:
		            self.samples += 1
		    print('Found %d images belonging to %d classes.' % (self.samples,
		                                                        self.num_class))

		    # second, build an index of the images in the different class subfolders
		    self.filenames = []
		    self.classes = np.zeros((self.samples,), dtype='int32')
		    i = 0
		    for subdir in classes:
		      subpath = os.path.join(directory, subdir)
		      for root, _, files in _recursive_list(subpath):
		        for fname in files:
		          is_valid = False
		          for extension in white_list_formats:
		            if fname.lower().endswith('.' + extension):
		              is_valid = True
		              break
		          if is_valid:
		            self.classes[i] = self.class_indices[subdir]
		            i += 1
		            # add filename relative to directory
		            absolute_path = os.path.join(root, fname)
		            self.filenames.append(os.path.relpath(absolute_path, directory))
		    super(DirectoryIterator, self).__init__(self.samples, batch_size, shuffle,
		                                            seed)

		  # 1. extract index for each batch samples, and current batch_size; 2. create empty batch (0s) with correct shape for batch_x; 3. get image data based on filename and index, and convert image data to array; 4. transform image array data with image_data_generator; 5. create batch_y based on 4 options: input, binary, sparse, categorical; 6. return batch_y, batch_x.
		  def next(self):
		    """For python 2.x.

		    Returns:
		        The next batch.
		    """
		    with self.lock:
		      index_array, current_index, current_batch_size = next(
		          self.index_generator)
		    # The transformation of images is not under thread lock
		    # so it can be done in parallel
		    batch_x = np.zeros(
		        (current_batch_size,) + self.image_shape, dtype=K.floatx())
		    grayscale = self.color_mode == 'grayscale'
		    # build batch of image data
		    for i, j in enumerate(index_array):
		      fname = self.filenames[j]
		      img = load_img(
		          os.path.join(self.directory, fname),
		          grayscale=grayscale,
		          target_size=self.target_size)
		      x = img_to_array(img, data_format=self.data_format)
		      x = self.image_data_generator.random_transform(x)
		      x = self.image_data_generator.standardize(x)
		      batch_x[i] = x
		    # optionally save augmented images to disk for debugging purposes
		    if self.save_to_dir:
		      for i in range(current_batch_size):
		        img = array_to_img(batch_x[i], self.data_format, scale=True)
		        fname = '{prefix}_{index}_{hash}.{format}'.format(
		            prefix=self.save_prefix,
		            index=current_index + i,
		            hash=np.random.randint(1e4),
		            format=self.save_format)
		        img.save(os.path.join(self.save_to_dir, fname))
		    # build batch of labels
		    if self.class_mode == 'input':
		      batch_y = batch_x.copy()
		    elif self.class_mode == 'sparse':
		      batch_y = self.classes[index_array]
		    elif self.class_mode == 'binary':
		      batch_y = self.classes[index_array].astype(K.floatx())
		    elif self.class_mode == 'categorical':
		      batch_y = np.zeros((len(batch_x), self.num_class), dtype=K.floatx())
		      for i, label in enumerate(self.classes[index_array]):
		        batch_y[i, label] = 1.
		    else:
		      return batch_x
		    return batch_x, batch_y

	class sequence_py:

		"""Preprocessing utilities for sequence data.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import random

			import numpy as np
			from six.moves import range  # pylint: disable=redefined-builtin


		def pad_sequences(sequences,
		                  maxlen=None,
		                  dtype='int32',
		                  padding='pre',
		                  truncating='pre',
		                  value=0.):
		  """Pads each sequence to the same length (length of the longest sequence).

		  If maxlen is provided, any sequence longer
		  than maxlen is truncated to maxlen.
		  Truncation happens off either the beginning (default) or
		  the end of the sequence.

		  Supports post-padding and pre-padding (default).

		  Arguments:
		      sequences: list of lists where each element is a sequence
		      maxlen: int, maximum length
		      dtype: type to cast the resulting sequence.
		      padding: 'pre' or 'post', pad either before or after each sequence.
		      truncating: 'pre' or 'post', remove values from sequences larger than
		          maxlen either in the beginning or in the end of the sequence
		      value: float, value to pad the sequences to the desired value.

		  Returns:
		      x: numpy array with dimensions (number_of_sequences, maxlen)

		  Raises:
		      ValueError: in case of invalid values for `truncating` or `padding`,
		          or in case of invalid shape for a `sequences` entry.
		  """
		  if not hasattr(sequences, '__len__'):
		    raise ValueError('`sequences` must be iterable.')
		  lengths = []
		  for x in sequences:
		    if not hasattr(x, '__len__'):
		      raise ValueError('`sequences` must be a list of iterables. '
		                       'Found non-iterable: ' + str(x))
		    lengths.append(len(x))

		  num_samples = len(sequences)
		  if maxlen is None:
		    maxlen = np.max(lengths)

		  # take the sample shape from the first non empty sequence
		  # checking for consistency in the main loop below.
		  sample_shape = tuple()
		  for s in sequences:
		    if len(s) > 0:  # pylint: disable=g-explicit-length-test
		      sample_shape = np.asarray(s).shape[1:]
		      break

		  x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
		  for idx, s in enumerate(sequences):
		    if not len(s):  # pylint: disable=g-explicit-length-test
		      continue  # empty list/array was found
		    if truncating == 'pre':
		      trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
		    elif truncating == 'post':
		      trunc = s[:maxlen]
		    else:
		      raise ValueError('Truncating type "%s" not understood' % truncating)

		    # check `trunc` has expected shape
		    trunc = np.asarray(trunc, dtype=dtype)
		    if trunc.shape[1:] != sample_shape:
		      raise ValueError(
		          'Shape of sample %s of sequence at position %s is different from '
		          'expected shape %s'
		          % (trunc.shape[1:], idx, sample_shape))

		    if padding == 'post':
		      x[idx, :len(trunc)] = trunc
		    elif padding == 'pre':
		      x[idx, -len(trunc):] = trunc
		    else:
		      raise ValueError('Padding type "%s" not understood' % padding)
		  return x


		def make_sampling_table(size, sampling_factor=1e-5):
		  """Generates a word rank-based probabilistic sampling table.

		  This generates an array where the ith element
		  is the probability that a word of rank i would be sampled,
		  according to the sampling distribution used in word2vec.

		  The word2vec formula is:
		      p(word) = min(1, sqrt(word.frequency/sampling_factor) /
		      (word.frequency/sampling_factor))

		  We assume that the word frequencies follow Zipf's law (s=1) to derive
		  a numerical approximation of frequency(rank):
		     frequency(rank) ~ 1/(rank * (log(rank) + gamma) + 1/2 - 1/(12*rank))
		      where gamma is the Euler-Mascheroni constant.

		  Arguments:
		      size: int, number of possible words to sample.
		      sampling_factor: the sampling factor in the word2vec formula.

		  Returns:
		      A 1D Numpy array of length `size` where the ith entry
		      is the probability that a word of rank i should be sampled.
		  """
		  gamma = 0.577
		  rank = np.array(list(range(size)))
		  rank[0] = 1
		  inv_fq = rank * (np.log(rank) + gamma) + 0.5 - 1. / (12. * rank)
		  f = sampling_factor * inv_fq

		  return np.minimum(1., f / np.sqrt(f))


		def skipgrams(sequence,
		              vocabulary_size,
		              window_size=4,
		              negative_samples=1.,
		              shuffle=True,
		              categorical=False,
		              sampling_table=None):
		  """Generates skipgram word pairs.

		  Takes a sequence (list of indexes of words),
		  returns couples of [word_index, other_word index] and labels (1s or 0s),
		  where label = 1 if 'other_word' belongs to the context of 'word',
		  and label=0 if 'other_word' is randomly sampled

		  Arguments:
		      sequence: a word sequence (sentence), encoded as a list
		          of word indices (integers). If using a `sampling_table`,
		          word indices are expected to match the rank
		          of the words in a reference dataset (e.g. 10 would encode
		          the 10-th most frequently occurring token).
		          Note that index 0 is expected to be a non-word and will be skipped.
		      vocabulary_size: int. maximum possible word index + 1
		      window_size: int. actually half-window.
		          The window of a word wi will be [i-window_size, i+window_size+1]
		      negative_samples: float >= 0. 0 for no negative (=random) samples.
		          1 for same number as positive samples. etc.
		      shuffle: whether to shuffle the word couples before returning them.
		      categorical: bool. if False, labels will be
		          integers (eg. [0, 1, 1 .. ]),
		          if True labels will be categorical eg. [[1,0],[0,1],[0,1] .. ]
		      sampling_table: 1D array of size `vocabulary_size` where the entry i
		          encodes the probabibily to sample a word of rank i.

		  Returns:
		      couples, labels: where `couples` are int pairs and
		          `labels` are either 0 or 1.

		  # Note
		      By convention, index 0 in the vocabulary is
		      a non-word and will be skipped.
		  """
		  couples = []
		  labels = []
		  for i, wi in enumerate(sequence):
		    if not wi:
		      continue
		    if sampling_table is not None:
		      if sampling_table[wi] < random.random():
		        continue

		    window_start = max(0, i - window_size)
		    window_end = min(len(sequence), i + window_size + 1)
		    for j in range(window_start, window_end):
		      if j != i:
		        wj = sequence[j]
		        if not wj:
		          continue
		        couples.append([wi, wj])
		        if categorical:
		          labels.append([0, 1])
		        else:
		          labels.append(1)

		  if negative_samples > 0:
		    num_negative_samples = int(len(labels) * negative_samples)
		    words = [c[0] for c in couples]
		    random.shuffle(words)

		    couples += [[words[i % len(words)],
		                 random.randint(1, vocabulary_size - 1)]
		                for i in range(num_negative_samples)]
		    if categorical:
		      labels += [[1, 0]] * num_negative_samples
		    else:
		      labels += [0] * num_negative_samples

		  if shuffle:
		    seed = random.randint(0, 10e6)
		    random.seed(seed)
		    random.shuffle(couples)
		    random.seed(seed)
		    random.shuffle(labels)

		  return couples, labels

	class text_py:

		"""Utilities for text input preprocessing.

		May benefit from a fast Cython rewrite.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from collections import OrderedDict
			import string
			import sys

			import numpy as np
			from six.moves import range  # pylint: disable=redefined-builtin
			from six.moves import zip  # pylint: disable=redefined-builtin

		if sys.version_info < (3,):
		  maketrans = string.maketrans
		else:
		  maketrans = str.maketrans


		def text_to_word_sequence(text,
		                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
		                          lower=True,
		                          split=' '):
		  """Converts a text to a sequence of words (or tokens).

		  Arguments:
		      text: Input text (string).
		      filters: Sequence of characters to filter out.
		      lower: Whether to convert the input to lowercase.
		      split: Sentence split marker (string).

		  Returns:
		      A list of words (or tokens).
		  """
		  if lower:
		    text = text.lower()
		  text = text.translate(maketrans(filters, split * len(filters)))
		  seq = text.split(split)
		  return [i for i in seq if i]


		def one_hot(text,
		            n,
		            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
		            lower=True,
		            split=' '):
		  seq = text_to_word_sequence(text, filters=filters, lower=lower, split=split)
		  return [(abs(hash(w)) % (n - 1) + 1) for w in seq]


		class Tokenizer(object):
		  """Text tokenization utility class.

		  This class allows to vectorize a text corpus, by turning each
		  text into either a sequence of integers (each integer being the index
		  of a token in a dictionary) or into a vector where the coefficient
		  for each token could be binary, based on word count, based on tf-idf...

		  Arguments:
		      num_words: the maximum number of words to keep, based
		          on word frequency. Only the most common `num_words` words will
		          be kept.
		      filters: a string where each element is a character that will be
		          filtered from the texts. The default is all punctuation, plus
		          tabs and line breaks, minus the `'` character.
		      lower: boolean. Whether to convert the texts to lowercase.
		      split: character or string to use for token splitting.
		      char_level: if True, every character will be treated as a token.

		  By default, all punctuation is removed, turning the texts into
		  space-separated sequences of words
		  (words maybe include the `'` character). These sequences are then
		  split into lists of tokens. They will then be indexed or vectorized.

		  `0` is a reserved index that won't be assigned to any word.
		  """

		  def __init__(self,
		               num_words=None,
		               filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
		               lower=True,
		               split=' ',
		               char_level=False):
		    self.word_counts = OrderedDict()
		    self.word_docs = {}
		    self.filters = filters
		    self.split = split
		    self.lower = lower
		    self.num_words = num_words
		    self.document_count = 0
		    self.char_level = char_level

		  def fit_on_texts(self, texts):
		    """Updates internal vocabulary based on a list of texts.

		    Required before using `texts_to_sequences` or `texts_to_matrix`.

		    Arguments:
		        texts: can be a list of strings,
		            or a generator of strings (for memory-efficiency)
		    """
		    self.document_count = 0
		    for text in texts:
		      self.document_count += 1
		      seq = text if self.char_level else text_to_word_sequence(
		          text, self.filters, self.lower, self.split)
		      for w in seq:
		        if w in self.word_counts:
		          self.word_counts[w] += 1
		        else:
		          self.word_counts[w] = 1
		      for w in set(seq):
		        if w in self.word_docs:
		          self.word_docs[w] += 1
		        else:
		          self.word_docs[w] = 1

		    wcounts = list(self.word_counts.items())
		    wcounts.sort(key=lambda x: x[1], reverse=True)
		    sorted_voc = [wc[0] for wc in wcounts]
		    # note that index 0 is reserved, never assigned to an existing word
		    self.word_index = dict(
		        list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

		    self.index_docs = {}
		    for w, c in list(self.word_docs.items()):
		      self.index_docs[self.word_index[w]] = c

		  def fit_on_sequences(self, sequences):
		    """Updates internal vocabulary based on a list of sequences.

		    Required before using `sequences_to_matrix`
		    (if `fit_on_texts` was never called).

		    Arguments:
		        sequences: A list of sequence.
		            A "sequence" is a list of integer word indices.
		    """
		    self.document_count = len(sequences)
		    self.index_docs = {}
		    for seq in sequences:
		      seq = set(seq)
		      for i in seq:
		        if i not in self.index_docs:
		          self.index_docs[i] = 1
		        else:
		          self.index_docs[i] += 1

		  def texts_to_sequences(self, texts):
		    """Transforms each text in texts in a sequence of integers.

		    Only top "num_words" most frequent words will be taken into account.
		    Only words known by the tokenizer will be taken into account.

		    Arguments:
		        texts: A list of texts (strings).

		    Returns:
		        A list of sequences.
		    """
		    res = []
		    for vect in self.texts_to_sequences_generator(texts):
		      res.append(vect)
		    return res

		  def texts_to_sequences_generator(self, texts):
		    """Transforms each text in texts in a sequence of integers.

		    Only top "num_words" most frequent words will be taken into account.
		    Only words known by the tokenizer will be taken into account.

		    Arguments:
		        texts: A list of texts (strings).

		    Yields:
		        Yields individual sequences.
		    """
		    num_words = self.num_words
		    for text in texts:
		      seq = text if self.char_level else text_to_word_sequence(
		          text, self.filters, self.lower, self.split)
		      vect = []
		      for w in seq:
		        i = self.word_index.get(w)
		        if i is not None:
		          if num_words and i >= num_words:
		            continue
		          else:
		            vect.append(i)
		      yield vect

		  def texts_to_matrix(self, texts, mode='binary'):
		    """Convert a list of texts to a Numpy matrix.

		    Arguments:
		        texts: list of strings.
		        mode: one of "binary", "count", "tfidf", "freq".

		    Returns:
		        A Numpy matrix.
		    """
		    sequences = self.texts_to_sequences(texts)
		    return self.sequences_to_matrix(sequences, mode=mode)

		  def sequences_to_matrix(self, sequences, mode='binary'):
		    """Converts a list of sequences into a Numpy matrix.

		    Arguments:
		        sequences: list of sequences
		            (a sequence is a list of integer word indices).
		        mode: one of "binary", "count", "tfidf", "freq"

		    Returns:
		        A Numpy matrix.

		    Raises:
		        ValueError: In case of invalid `mode` argument,
		            or if the Tokenizer requires to be fit to sample data.
		    """
		    if not self.num_words:
		      if self.word_index:
		        num_words = len(self.word_index) + 1
		      else:
		        raise ValueError('Specify a dimension (num_words argument), '
		                         'or fit on some text data first.')
		    else:
		      num_words = self.num_words

		    if mode == 'tfidf' and not self.document_count:
		      raise ValueError('Fit the Tokenizer on some data '
		                       'before using tfidf mode.')

		    x = np.zeros((len(sequences), num_words))
		    for i, seq in enumerate(sequences):
		      if not seq:
		        continue
		      counts = {}
		      for j in seq:
		        if j >= num_words:
		          continue
		        if j not in counts:
		          counts[j] = 1.
		        else:
		          counts[j] += 1
		      for j, c in list(counts.items()):
		        if mode == 'count':
		          x[i][j] = c
		        elif mode == 'freq':
		          x[i][j] = c / len(seq)
		        elif mode == 'binary':
		          x[i][j] = 1
		        elif mode == 'tfidf':
		          # Use weighting scheme 2 in
		          # https://en.wikipedia.org/wiki/Tf%E2%80%93idf
		          tf = 1 + np.log(c)
		          idf = np.log(1 + self.document_count /
		                       (1 + self.index_docs.get(j, 0)))
		          x[i][j] = tf * idf
		        else:
		          raise ValueError('Unknown vectorization mode:', mode)
		    return x


# layers
class layers_folder:

	class __init__py:

		"""Keras layers module.
		"""
		# pylint: disable=wildcard-import
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.engine import Input
		from tensorflow.contrib.keras.python.keras.engine import InputLayer
		from tensorflow.contrib.keras.python.keras.engine import InputSpec
		from tensorflow.contrib.keras.python.keras.engine import Layer
		from tensorflow.contrib.keras.python.keras.layers.advanced_activations import *
		from tensorflow.contrib.keras.python.keras.layers.convolutional import *
		from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import *
		from tensorflow.contrib.keras.python.keras.layers.core import *
		from tensorflow.contrib.keras.python.keras.layers.embeddings import *
		from tensorflow.contrib.keras.python.keras.layers.local import *
		from tensorflow.contrib.keras.python.keras.layers.merge import *
		from tensorflow.contrib.keras.python.keras.layers.noise import *
		from tensorflow.contrib.keras.python.keras.layers.normalization import *
		from tensorflow.contrib.keras.python.keras.layers.pooling import *
		from tensorflow.contrib.keras.python.keras.layers.recurrent import *
		from tensorflow.contrib.keras.python.keras.layers.serialization import deserialize
		from tensorflow.contrib.keras.python.keras.layers.serialization import serialize
		from tensorflow.contrib.keras.python.keras.layers.wrappers import *

	class advanced_activations_py:

		"""Layers that act as activation functions.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.python.framework import tensor_shape


		class LeakyReLU(Layer):
		  """Leaky version of a Rectified Linear Unit.

		  It allows a small gradient when the unit is not active:
		  `f(x) = alpha * x for x < 0`,
		  `f(x) = x for x >= 0`.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as the input.

		  Arguments:
		      alpha: float >= 0. Negative slope coefficient.

		  """

		  def __init__(self, alpha=0.3, **kwargs):
		    super(LeakyReLU, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.alpha = K.cast_to_floatx(alpha)

		  def call(self, inputs):
		    return K.relu(inputs, alpha=self.alpha)

		  def get_config(self):
		    config = {'alpha': self.alpha}
		    base_config = super(LeakyReLU, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class PReLU(Layer):
		  """Parametric Rectified Linear Unit.

		  It follows:
		  `f(x) = alpha * x for x < 0`,
		  `f(x) = x for x >= 0`,
		  where `alpha` is a learned array with the same shape as x.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as the input.

		  Arguments:
		      alpha_initializer: initializer function for the weights.
		      alpha_regularizer: regularizer for the weights.
		      alpha_constraint: constraint for the weights.
		      shared_axes: the axes along which to share learnable
		          parameters for the activation function.
		          For example, if the incoming feature maps
		          are from a 2D convolution
		          with output shape `(batch, height, width, channels)`,
		          and you wish to share parameters across space
		          so that each filter only has one set of parameters,
		          set `shared_axes=[1, 2]`.

		  """

		  def __init__(self,
		               alpha_initializer='zeros',
		               alpha_regularizer=None,
		               alpha_constraint=None,
		               shared_axes=None,
		               **kwargs):
		    super(PReLU, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.alpha_initializer = initializers.get(alpha_initializer)
		    self.alpha_regularizer = regularizers.get(alpha_regularizer)
		    self.alpha_constraint = constraints.get(alpha_constraint)
		    if shared_axes is None:
		      self.shared_axes = None
		    elif not isinstance(shared_axes, (list, tuple)):
		      self.shared_axes = [shared_axes]
		    else:
		      self.shared_axes = list(shared_axes)

		  def build(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    param_shape = input_shape[1:]
		    self.param_broadcast = [False] * len(param_shape)
		    if self.shared_axes is not None:
		      for i in self.shared_axes:
		        param_shape[i - 1] = 1
		        self.param_broadcast[i - 1] = True
		    self.alpha = self.add_weight(
		        shape=param_shape,
		        name='alpha',
		        initializer=self.alpha_initializer,
		        regularizer=self.alpha_regularizer,
		        constraint=self.alpha_constraint)
		    # Set input spec
		    axes = {}
		    if self.shared_axes:
		      for i in range(1, len(input_shape)):
		        if i not in self.shared_axes:
		          axes[i] = input_shape[i]
		    self.input_spec = InputSpec(ndim=len(input_shape), axes=axes)
		    self.built = True

		  def call(self, inputs, mask=None):
		    pos = K.relu(inputs)
		    if K.backend() == 'theano':
		      neg = (K.pattern_broadcast(self.alpha, self.param_broadcast) *
		             (inputs - K.abs(inputs)) * 0.5)
		    else:
		      neg = -self.alpha * K.relu(-inputs)
		    return pos + neg

		  def get_config(self):
		    config = {
		        'alpha_initializer': initializers.serialize(self.alpha_initializer),
		        'alpha_regularizer': regularizers.serialize(self.alpha_regularizer),
		        'alpha_constraint': constraints.serialize(self.alpha_constraint),
		        'shared_axes': self.shared_axes
		    }
		    base_config = super(PReLU, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ELU(Layer):
		  """Exponential Linear Unit.

		  It follows:
		  `f(x) =  alpha * (exp(x) - 1.) for x < 0`,
		  `f(x) = x for x >= 0`.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as the input.

		  Arguments:
		      alpha: scale for the negative factor.

		  """

		  def __init__(self, alpha=1.0, **kwargs):
		    super(ELU, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.alpha = K.cast_to_floatx(alpha)

		  def call(self, inputs):
		    return K.elu(inputs, self.alpha)

		  def get_config(self):
		    config = {'alpha': float(self.alpha)}
		    base_config = super(ELU, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ThresholdedReLU(Layer):
		  """Thresholded Rectified Linear Unit.

		  It follows:
		  `f(x) = x for x > theta`,
		  `f(x) = 0 otherwise`.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as the input.

		  Arguments:
		      theta: float >= 0. Threshold location of activation.

		  """

		  def __init__(self, theta=1.0, **kwargs):
		    super(ThresholdedReLU, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.theta = K.cast_to_floatx(theta)

		  def call(self, inputs, mask=None):
		    return inputs * K.cast(inputs > self.theta, K.floatx())

		  def get_config(self):
		    config = {'theta': float(self.theta)}
		    base_config = super(ThresholdedReLU, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class convolutional_recurrent_py:

		"""Convolutional-recurrent layers.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np

			from tensorflow.contrib.keras.python.keras import activations
			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.layers.recurrent import Recurrent
			from tensorflow.contrib.keras.python.keras.utils import conv_utils
			from tensorflow.python.framework import tensor_shape


		class ConvRecurrent2D(Recurrent):
		  """Abstract base class for convolutional recurrent layers.

		  Do not use in a model -- it's not a functional layer!

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of n integers, specifying the
		          dimensions of the convolution window.
		      strides: An integer or tuple/list of n integers,
		          specifying the strides of the convolution.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: One of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, time, ..., channels)`
		          while `channels_first` corresponds to
		          inputs with shape `(batch, time, channels, ...)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      dilation_rate: An integer or tuple/list of n integers, specifying
		          the dilation rate to use for dilated convolution.
		          Currently, specifying any `dilation_rate` value != 1 is
		          incompatible with specifying any `strides` value != 1.
		      return_sequences: Boolean. Whether to return the last output
		          in the output sequence, or the full sequence.
		      go_backwards: Boolean (default False).
		          If True, rocess the input sequence backwards.
		      stateful: Boolean (default False). If True, the last state
		          for each sample at index i in a batch will be used as initial
		          state for the sample of index i in the following batch.

		  Input shape:
		      5D tensor with shape `(num_samples, timesteps, channels, rows, cols)`.

		  Output shape:
		      - if `return_sequences`: 5D tensor with shape
		          `(num_samples, timesteps, channels, rows, cols)`.
		      - else, 4D tensor with shape `(num_samples, channels, rows, cols)`.

		  # Masking
		      This layer supports masking for input data with a variable number
		      of timesteps. To introduce masks to your data,
		      use an `Embedding` layer with the `mask_zero` parameter
		      set to `True`.
		      **Note:** for the time being, masking is only supported with Theano.

		  # Note on using statefulness in RNNs
		      You can set RNN layers to be 'stateful', which means that the states
		      computed for the samples in one batch will be reused as initial states
		      for the samples in the next batch.
		      This assumes a one-to-one mapping between
		      samples in different successive batches.

		      To enable statefulness:
		          - specify `stateful=True` in the layer constructor.
		          - specify a fixed batch size for your model, by passing
		              a `batch_input_size=(...)` to the first layer in your model.
		              This is the expected shape of your inputs *including the batch
		              size*.
		              It should be a tuple of integers, e.g. `(32, 10, 100)`.

		      To reset the states of your model, call `.reset_states()` on either
		      a specific layer, or on your entire model.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1),
		               padding='valid',
		               data_format=None,
		               dilation_rate=(1, 1),
		               return_sequences=False,
		               go_backwards=False,
		               stateful=False,
		               **kwargs):
		    super(ConvRecurrent2D, self).__init__(**kwargs)
		    self.filters = filters
		    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
		    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		    self.padding = conv_utils.normalize_padding(padding)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,
		                                                    'dilation_rate')
		    self.return_sequences = return_sequences
		    self.go_backwards = go_backwards
		    self.stateful = stateful
		    self.input_spec = [InputSpec(ndim=5)]
		    self.state_spec = None

		  def _compute_output_shape(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shape = input_shape[0]
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_first':
		      rows = input_shape[3]
		      cols = input_shape[4]
		    elif self.data_format == 'channels_last':
		      rows = input_shape[2]
		      cols = input_shape[3]
		    rows = conv_utils.conv_output_length(
		        rows,
		        self.kernel_size[0],
		        padding=self.padding,
		        stride=self.strides[0],
		        dilation=self.dilation_rate[0])
		    cols = conv_utils.conv_output_length(
		        cols,
		        self.kernel_size[1],
		        padding=self.padding,
		        stride=self.strides[1],
		        dilation=self.dilation_rate[1])
		    if self.return_sequences:
		      if self.data_format == 'channels_first':
		        return tensor_shape.TensorShape(
		            [input_shape[0], input_shape[1], self.filters, rows, cols])
		      elif self.data_format == 'channels_last':
		        return tensor_shape.TensorShape(
		            [input_shape[0], input_shape[1], rows, cols, self.filters])
		    else:
		      if self.data_format == 'channels_first':
		        return tensor_shape.TensorShape(
		            [input_shape[0], self.filters, rows, cols])
		      elif self.data_format == 'channels_last':
		        return tensor_shape.TensorShape(
		            [input_shape[0], rows, cols, self.filters])

		  def get_config(self):
		    config = {
		        'filters': self.filters,
		        'kernel_size': self.kernel_size,
		        'strides': self.strides,
		        'padding': self.padding,
		        'data_format': self.data_format,
		        'dilation_rate': self.dilation_rate,
		        'return_sequences': self.return_sequences,
		        'go_backwards': self.go_backwards,
		        'stateful': self.stateful
		    }
		    base_config = super(ConvRecurrent2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ConvLSTM2D(ConvRecurrent2D):
		  """Convolutional LSTM.

		  It is similar to an LSTM layer, but the input transformations
		  and recurrent transformations are both convolutional.

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of n integers, specifying the
		          dimensions of the convolution window.
		      strides: An integer or tuple/list of n integers,
		          specifying the strides of the convolution.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: One of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, time, ..., channels)`
		          while `channels_first` corresponds to
		          inputs with shape `(batch, time, channels, ...)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      dilation_rate: An integer or tuple/list of n integers, specifying
		          the dilation rate to use for dilated convolution.
		          Currently, specifying any `dilation_rate` value != 1 is
		          incompatible with specifying any `strides` value != 1.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      recurrent_activation: Activation function to use
		          for the recurrent step.
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix,
		          used for the linear transformation of the inputs..
		      recurrent_initializer: Initializer for the `recurrent_kernel`
		          weights matrix,
		          used for the linear transformation of the recurrent state..
		      bias_initializer: Initializer for the bias vector.
		      unit_forget_bias: Boolean.
		          If True, add 1 to the bias of the forget gate at initialization.
		          Use in combination with `bias_initializer="zeros"`.
		          This is recommended in [Jozefowicz et
		            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      recurrent_regularizer: Regularizer function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to
		          the `kernel` weights matrix.
		      recurrent_constraint: Constraint function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_constraint: Constraint function applied to the bias vector.
		      return_sequences: Boolean. Whether to return the last output
		          in the output sequence, or the full sequence.
		      go_backwards: Boolean (default False).
		          If True, rocess the input sequence backwards.
		      stateful: Boolean (default False). If True, the last state
		          for each sample at index i in a batch will be used as initial
		          state for the sample of index i in the following batch.
		      dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the inputs.
		      recurrent_dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the recurrent state.

		  Input shape:
		      - if data_format='channels_first'
		          5D tensor with shape:
		          `(samples,time, channels, rows, cols)`
		      - if data_format='channels_last'
		          5D tensor with shape:
		          `(samples,time, rows, cols, channels)`

		   Output shape:
		      - if `return_sequences`
		           - if data_format='channels_first'
		              5D tensor with shape:
		              `(samples, time, filters, output_row, output_col)`
		           - if data_format='channels_last'
		              5D tensor with shape:
		              `(samples, time, output_row, output_col, filters)`
		      - else
		          - if data_format ='channels_first'
		              4D tensor with shape:
		              `(samples, filters, output_row, output_col)`
		          - if data_format='channels_last'
		              4D tensor with shape:
		              `(samples, output_row, output_col, filters)`
		          where o_row and o_col depend on the shape of the filter and
		          the padding

		  Raises:
		      ValueError: in case of invalid constructor arguments.

		  References:
		      - [Convolutional LSTM Network: A Machine Learning Approach for
		      Precipitation Nowcasting](http://arxiv.org/abs/1506.04214v1)
		      The current implementation does not include the feedback loop on the
		      cells output
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1),
		               padding='valid',
		               data_format=None,
		               dilation_rate=(1, 1),
		               activation='tanh',
		               recurrent_activation='hard_sigmoid',
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               recurrent_initializer='orthogonal',
		               bias_initializer='zeros',
		               unit_forget_bias=True,
		               kernel_regularizer=None,
		               recurrent_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               recurrent_constraint=None,
		               bias_constraint=None,
		               return_sequences=False,
		               go_backwards=False,
		               stateful=False,
		               dropout=0.,
		               recurrent_dropout=0.,
		               **kwargs):
		    super(ConvLSTM2D, self).__init__(
		        filters,
		        kernel_size,
		        strides=strides,
		        padding=padding,
		        data_format=data_format,
		        dilation_rate=dilation_rate,
		        return_sequences=return_sequences,
		        go_backwards=go_backwards,
		        stateful=stateful,
		        **kwargs)
		    self.activation = activations.get(activation)
		    self.recurrent_activation = activations.get(recurrent_activation)
		    self.use_bias = use_bias

		    self.kernel_initializer = initializers.get(kernel_initializer)
		    self.recurrent_initializer = initializers.get(recurrent_initializer)
		    self.bias_initializer = initializers.get(bias_initializer)
		    self.unit_forget_bias = unit_forget_bias

		    self.kernel_regularizer = regularizers.get(kernel_regularizer)
		    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		    self.bias_regularizer = regularizers.get(bias_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)

		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.recurrent_constraint = constraints.get(recurrent_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		    self.dropout = min(1., max(0., dropout))
		    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
		    self.state_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

		  def build(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shape = input_shape[0]
		    input_shape = tuple(tensor_shape.TensorShape(input_shape).as_list())
		    batch_size = input_shape[0] if self.stateful else None
		    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:])

		    if self.stateful:
		      self.reset_states()
		    else:
		      # initial states: 2 all-zero tensor of shape (filters)
		      self.states = [None, None]

		    if self.data_format == 'channels_first':
		      channel_axis = 2
		    else:
		      channel_axis = -1
		    if input_shape[channel_axis] is None:
		      raise ValueError('The channel dimension of the inputs '
		                       'should be defined. Found `None`.')
		    input_dim = input_shape[channel_axis]
		    state_shape = [None] * 4
		    state_shape[channel_axis] = input_dim
		    state_shape = tuple(state_shape)
		    self.state_spec = [
		        InputSpec(shape=state_shape),
		        InputSpec(shape=state_shape)
		    ]
		    kernel_shape = self.kernel_size + (input_dim, self.filters * 4)
		    self.kernel_shape = kernel_shape
		    recurrent_kernel_shape = self.kernel_size + (self.filters, self.filters * 4)

		    self.kernel = self.add_weight(
		        shape=kernel_shape,
		        initializer=self.kernel_initializer,
		        name='kernel',
		        regularizer=self.kernel_regularizer,
		        constraint=self.kernel_constraint)
		    self.recurrent_kernel = self.add_weight(
		        shape=recurrent_kernel_shape,
		        initializer=self.recurrent_initializer,
		        name='recurrent_kernel',
		        regularizer=self.recurrent_regularizer,
		        constraint=self.recurrent_constraint)
		    if self.use_bias:
		      self.bias = self.add_weight(
		          shape=(self.filters * 4,),
		          initializer=self.bias_initializer,
		          name='bias',
		          regularizer=self.bias_regularizer,
		          constraint=self.bias_constraint)
		      if self.unit_forget_bias:
		        bias_value = np.zeros((self.filters * 4,))
		        bias_value[self.filters:self.filters * 2] = 1.
		        K.set_value(self.bias, bias_value)
		    else:
		      self.bias = None

		    self.kernel_i = self.kernel[:, :, :, :self.filters]
		    self.recurrent_kernel_i = self.recurrent_kernel[:, :, :, :self.filters]
		    self.kernel_f = self.kernel[:, :, :, self.filters:self.filters * 2]
		    self.recurrent_kernel_f = self.recurrent_kernel[:, :, :, self.filters:
		                                                    self.filters * 2]
		    self.kernel_c = self.kernel[:, :, :, self.filters * 2:self.filters * 3]
		    self.recurrent_kernel_c = self.recurrent_kernel[:, :, :, self.filters * 2:
		                                                    self.filters * 3]
		    self.kernel_o = self.kernel[:, :, :, self.filters * 3:]
		    self.recurrent_kernel_o = self.recurrent_kernel[:, :, :, self.filters * 3:]

		    if self.use_bias:
		      self.bias_i = self.bias[:self.filters]
		      self.bias_f = self.bias[self.filters:self.filters * 2]
		      self.bias_c = self.bias[self.filters * 2:self.filters * 3]
		      self.bias_o = self.bias[self.filters * 3:]
		    else:
		      self.bias_i = None
		      self.bias_f = None
		      self.bias_c = None
		      self.bias_o = None
		    self.built = True

		  def get_initial_state(self, inputs):
		    # (samples, timesteps, rows, cols, filters)
		    initial_state = K.zeros_like(inputs)
		    # (samples, rows, cols, filters)
		    initial_state = K.sum(initial_state, axis=1)
		    shape = list(self.kernel_shape)
		    shape[-1] = self.filters
		    initial_state = self.input_conv(
		        initial_state, K.zeros(tuple(shape)), padding=self.padding)

		    initial_states = [initial_state for _ in range(2)]
		    return initial_states

		  def reset_states(self):
		    if not self.stateful:
		      raise RuntimeError('Layer must be stateful.')
		    input_shape = self.input_spec[0].shape
		    output_shape = self._compute_output_shape(input_shape)

		    if not input_shape[0]:
		      raise ValueError('If a RNN is stateful, a complete '
		                       'input_shape must be provided '
		                       '(including batch size). '
		                       'Got input shape: ' + str(input_shape))

		    if self.return_sequences:
		      out_row, out_col, out_filter = output_shape[2:]
		    else:
		      out_row, out_col, out_filter = output_shape[1:]

		    if hasattr(self, 'states'):
		      K.set_value(self.states[0],
		                  np.zeros((input_shape[0], out_row, out_col, out_filter)))
		      K.set_value(self.states[1],
		                  np.zeros((input_shape[0], out_row, out_col, out_filter)))
		    else:
		      self.states = [
		          K.zeros((input_shape[0], out_row, out_col, out_filter)),
		          K.zeros((input_shape[0], out_row, out_col, out_filter))
		      ]

		  def get_constants(self, inputs, training=None):
		    constants = []
		    if self.implementation == 0 and 0 < self.dropout < 1:
		      ones = K.zeros_like(inputs)
		      ones = K.sum(ones, axis=1)
		      ones += 1

		      def dropped_inputs():
		        return K.dropout(ones, self.dropout)

		      dp_mask = [
		          K.in_train_phase(dropped_inputs, ones, training=training)
		          for _ in range(4)
		      ]
		      constants.append(dp_mask)
		    else:
		      constants.append([K.cast_to_floatx(1.) for _ in range(4)])

		    if 0 < self.recurrent_dropout < 1:
		      shape = list(self.kernel_shape)
		      shape[-1] = self.filters
		      ones = K.zeros_like(inputs)
		      ones = K.sum(ones, axis=1)
		      ones = self.input_conv(ones, K.zeros(shape), padding=self.padding)
		      ones += 1.

		      def dropped_inputs():  # pylint: disable=function-redefined
		        return K.dropout(ones, self.recurrent_dropout)

		      rec_dp_mask = [
		          K.in_train_phase(dropped_inputs, ones, training=training)
		          for _ in range(4)
		      ]
		      constants.append(rec_dp_mask)
		    else:
		      constants.append([K.cast_to_floatx(1.) for _ in range(4)])
		    return constants

		  def input_conv(self, x, w, b=None, padding='valid'):
		    conv_out = K.conv2d(
		        x,
		        w,
		        strides=self.strides,
		        padding=padding,
		        data_format=self.data_format,
		        dilation_rate=self.dilation_rate)
		    if b is not None:
		      conv_out = K.bias_add(conv_out, b, data_format=self.data_format)
		    return conv_out

		  def reccurent_conv(self, x, w):
		    conv_out = K.conv2d(
		        x, w, strides=(1, 1), padding='same', data_format=self.data_format)
		    return conv_out

		  def step(self, inputs, states):
		    assert len(states) == 4
		    h_tm1 = states[0]
		    c_tm1 = states[1]
		    dp_mask = states[2]
		    rec_dp_mask = states[3]

		    x_i = self.input_conv(
		        inputs * dp_mask[0], self.kernel_i, self.bias_i, padding=self.padding)
		    x_f = self.input_conv(
		        inputs * dp_mask[1], self.kernel_f, self.bias_f, padding=self.padding)
		    x_c = self.input_conv(
		        inputs * dp_mask[2], self.kernel_c, self.bias_c, padding=self.padding)
		    x_o = self.input_conv(
		        inputs * dp_mask[3], self.kernel_o, self.bias_o, padding=self.padding)
		    h_i = self.reccurent_conv(h_tm1 * rec_dp_mask[0], self.recurrent_kernel_i)
		    h_f = self.reccurent_conv(h_tm1 * rec_dp_mask[1], self.recurrent_kernel_f)
		    h_c = self.reccurent_conv(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c)
		    h_o = self.reccurent_conv(h_tm1 * rec_dp_mask[3], self.recurrent_kernel_o)

		    i = self.recurrent_activation(x_i + h_i)
		    f = self.recurrent_activation(x_f + h_f)
		    c = f * c_tm1 + i * self.activation(x_c + h_c)
		    o = self.recurrent_activation(x_o + h_o)
		    h = o * self.activation(c)
		    return h, [h, c]

		  def get_config(self):
		    config = {
		        'activation':
		            activations.serialize(self.activation),
		        'recurrent_activation':
		            activations.serialize(self.recurrent_activation),
		        'use_bias':
		            self.use_bias,
		        'kernel_initializer':
		            initializers.serialize(self.kernel_initializer),
		        'recurrent_initializer':
		            initializers.serialize(self.recurrent_initializer),
		        'bias_initializer':
		            initializers.serialize(self.bias_initializer),
		        'unit_forget_bias':
		            self.unit_forget_bias,
		        'kernel_regularizer':
		            regularizers.serialize(self.kernel_regularizer),
		        'recurrent_regularizer':
		            regularizers.serialize(self.recurrent_regularizer),
		        'bias_regularizer':
		            regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint':
		            constraints.serialize(self.kernel_constraint),
		        'recurrent_constraint':
		            constraints.serialize(self.recurrent_constraint),
		        'bias_constraint':
		            constraints.serialize(self.bias_constraint),
		        'dropout':
		            self.dropout,
		        'recurrent_dropout':
		            self.recurrent_dropout
		    }
		    base_config = super(ConvLSTM2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class convolutional_py:

		"""Keras convolution layers and image transformation layers.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import activations
			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			# imports for backwards namespace compatibility
			# pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling1D
			from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers.pooling import AveragePooling3D
			from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling1D
			from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers.pooling import MaxPooling3D
			# pylint: enable=unused-import
			from tensorflow.contrib.keras.python.keras.utils import conv_utils
			from tensorflow.python.framework import tensor_shape
			from tensorflow.python.layers import convolutional as tf_convolutional_layers


		class Conv1D(tf_convolutional_layers.Conv1D, Layer):
		  """1D convolution layer (e.g. temporal convolution).

		  This layer creates a convolution kernel that is convolved
		  with the layer input over a single spatial (or temporal) dimension
		  to produce a tensor of outputs.
		  If `use_bias` is True, a bias vector is created and added to the outputs.
		  Finally, if `activation` is not `None`,
		  it is applied to the outputs as well.

		  When using this layer as the first layer in a model,
		  provide an `input_shape` argument
		  (tuple of integers or `None`, e.g.
		  `(10, 128)` for sequences of 10 vectors of 128-dimensional vectors,
		  or `(None, 128)` for variable-length sequences of 128-dimensional vectors.

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of a single integer,
		          specifying the length of the 1D convolution window.
		      strides: An integer or tuple/list of a single integer,
		          specifying the stride length of the convolution.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: One of `"valid"`, `"causal"` or `"same"` (case-insensitive).
		          `"causal"` results in causal (dilated) convolutions, e.g. output[t]
		          does not depend on input[t+1:]. Useful when modeling temporal data
		          where the model should not violate the temporal order.
		          See [WaveNet: A Generative Model for Raw Audio, section
		            2.1](https://arxiv.org/abs/1609.03499).
		      dilation_rate: an integer or tuple/list of a single integer, specifying
		          the dilation rate to use for dilated convolution.
		          Currently, specifying any `dilation_rate` value != 1 is
		          incompatible with specifying any `strides` value != 1.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to the kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      3D tensor with shape: `(batch_size, steps, input_dim)`

		  Output shape:
		      3D tensor with shape: `(batch_size, new_steps, filters)`
		      `steps` value might have changed due to padding or strides.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=1,
		               padding='valid',
		               dilation_rate=1,
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    super(Conv1D, self).__init__(
		        filters=filters,
		        kernel_size=kernel_size,
		        strides=strides,
		        padding=padding,
		        data_format='channels_last',
		        dilation_rate=dilation_rate,
		        activation=activations.get(activation),
		        use_bias=use_bias,
		        kernel_initializer=initializers.get(kernel_initializer),
		        bias_initializer=initializers.get(bias_initializer),
		        kernel_regularizer=regularizers.get(kernel_regularizer),
		        bias_regularizer=regularizers.get(bias_regularizer),
		        activity_regularizer=regularizers.get(activity_regularizer),
		        **kwargs)
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		  def build(self, input_shape):
		    super(Conv1D, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.kernel_constraint:
		      self.constraints[self.kernel] = self.kernel_constraint
		    if self.use_bias and self.bias_constraint:
		      self.constraints[self.bias] = self.bias_constraint

		  def get_config(self):
		    config = {
		        'filters': self.filters,
		        'kernel_size': self.kernel_size,
		        'strides': self.strides,
		        'padding': self.padding,
		        'dilation_rate': self.dilation_rate,
		        'activation': activations.serialize(self.activation),
		        'use_bias': self.use_bias,
		        'kernel_initializer': initializers.serialize(self.kernel_initializer),
		        'bias_initializer': initializers.serialize(self.bias_initializer),
		        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint': constraints.serialize(self.kernel_constraint),
		        'bias_constraint': constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(Conv1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

		# 1. Layer of tf: access graph and create other attributes
		# 2. Layer of tf.keras: many other attributes
		# 3. Conv of tf: attributes
		# 4. Conv2D of tf: more attributes
		# 5. Conv2D of tf.keras: many more attributes
		class Conv2D(tf_convolutional_layers.Conv2D, Layer):
		  """2D convolution layer (e.g. spatial convolution over images).

		  This layer creates a convolution kernel that is convolved
		  with the layer input to produce a tensor of
		  outputs. If `use_bias` is True,
		  a bias vector is created and added to the outputs. Finally, if
		  `activation` is not `None`, it is applied to the outputs as well.

		  When using this layer as the first layer in a model,
		  provide the keyword argument `input_shape`
		  (tuple of integers, does not include the sample axis),
		  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
		  in `data_format="channels_last"`.

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of 2 integers, specifying the
		          width and height of the 2D convolution window.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		      strides: An integer or tuple/list of 2 integers,
		          specifying the strides of the convolution along the width and height.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: one of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      dilation_rate: an integer or tuple/list of 2 integers, specifying
		          the dilation rate to use for dilated convolution.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Currently, specifying any `dilation_rate` value != 1 is
		          incompatible with specifying any stride value != 1.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to the kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      4D tensor with shape:
		      `(samples, channels, rows, cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(samples, rows, cols, channels)` if data_format='channels_last'.

		  Output shape:
		      4D tensor with shape:
		      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		      `rows` and `cols` values might have changed due to padding.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1),
		               padding='valid',
		               data_format=None,
		               dilation_rate=(1, 1),
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    super(Conv2D, self).__init__(
		        filters=filters,
		        kernel_size=kernel_size,
		        strides=strides,
		        padding=padding,
		        data_format=data_format,
		        dilation_rate=dilation_rate,
		        activation=activations.get(activation),
		        use_bias=use_bias,
		        kernel_initializer=initializers.get(kernel_initializer),
		        bias_initializer=initializers.get(bias_initializer),
		        kernel_regularizer=regularizers.get(kernel_regularizer),
		        bias_regularizer=regularizers.get(bias_regularizer),
		        activity_regularizer=regularizers.get(activity_regularizer),
		        **kwargs)
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		  def build(self, input_shape):
		    super(Conv2D, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.kernel_constraint:
		      self.constraints[self.kernel] = self.kernel_constraint
		    if self.use_bias and self.bias_constraint:
		      self.constraints[self.bias] = self.bias_constraint

		  def get_config(self):
		    config = {
		        'filters': self.filters,
		        'kernel_size': self.kernel_size,
		        'strides': self.strides,
		        'padding': self.padding,
		        'data_format': self.data_format,
		        'dilation_rate': self.dilation_rate,
		        'activation': activations.serialize(self.activation),
		        'use_bias': self.use_bias,
		        'kernel_initializer': initializers.serialize(self.kernel_initializer),
		        'bias_initializer': initializers.serialize(self.bias_initializer),
		        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint': constraints.serialize(self.kernel_constraint),
		        'bias_constraint': constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(Conv2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Conv3D(tf_convolutional_layers.Conv3D, Layer):
		  """3D convolution layer (e.g. spatial convolution over volumes).

		  This layer creates a convolution kernel that is convolved
		  with the layer input to produce a tensor of
		  outputs. If `use_bias` is True,
		  a bias vector is created and added to the outputs. Finally, if
		  `activation` is not `None`, it is applied to the outputs as well.

		  When using this layer as the first layer in a model,
		  provide the keyword argument `input_shape`
		  (tuple of integers, does not include the sample axis),
		  e.g. `input_shape=(128, 128, 128, 3)` for 128x128x128 volumes
		  with a single channel,
		  in `data_format="channels_last"`.

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of 3 integers, specifying the
		          width and height of the 3D convolution window.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		      strides: An integer or tuple/list of 3 integers,
		          specifying the strides of the convolution along each spatial
		            dimension.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: one of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      dilation_rate: an integer or tuple/list of 3 integers, specifying
		          the dilation rate to use for dilated convolution.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Currently, specifying any `dilation_rate` value != 1 is
		          incompatible with specifying any stride value != 1.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to the kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      5D tensor with shape:
		      `(samples, channels, conv_dim1, conv_dim2, conv_dim3)` if
		        data_format='channels_first'
		      or 5D tensor with shape:
		      `(samples, conv_dim1, conv_dim2, conv_dim3, channels)` if
		        data_format='channels_last'.

		  Output shape:
		      5D tensor with shape:
		      `(samples, filters, new_conv_dim1, new_conv_dim2, new_conv_dim3)` if
		        data_format='channels_first'
		      or 5D tensor with shape:
		      `(samples, new_conv_dim1, new_conv_dim2, new_conv_dim3, filters)` if
		        data_format='channels_last'.
		      `new_conv_dim1`, `new_conv_dim2` and `new_conv_dim3` values might have
		        changed due to padding.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1, 1),
		               padding='valid',
		               data_format=None,
		               dilation_rate=(1, 1, 1),
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    super(Conv3D, self).__init__(
		        filters=filters,
		        kernel_size=kernel_size,
		        strides=strides,
		        padding=padding,
		        data_format=data_format,
		        dilation_rate=dilation_rate,
		        activation=activations.get(activation),
		        use_bias=use_bias,
		        kernel_initializer=initializers.get(kernel_initializer),
		        bias_initializer=initializers.get(bias_initializer),
		        kernel_regularizer=regularizers.get(kernel_regularizer),
		        bias_regularizer=regularizers.get(bias_regularizer),
		        activity_regularizer=regularizers.get(activity_regularizer),
		        **kwargs)
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		  def build(self, input_shape):
		    super(Conv3D, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.kernel_constraint:
		      self.constraints[self.kernel] = self.kernel_constraint
		    if self.use_bias and self.bias_constraint:
		      self.constraints[self.bias] = self.bias_constraint

		  def get_config(self):
		    config = {
		        'filters': self.filters,
		        'kernel_size': self.kernel_size,
		        'strides': self.strides,
		        'padding': self.padding,
		        'data_format': self.data_format,
		        'dilation_rate': self.dilation_rate,
		        'activation': activations.serialize(self.activation),
		        'use_bias': self.use_bias,
		        'kernel_initializer': initializers.serialize(self.kernel_initializer),
		        'bias_initializer': initializers.serialize(self.bias_initializer),
		        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint': constraints.serialize(self.kernel_constraint),
		        'bias_constraint': constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(Conv3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Conv2DTranspose(tf_convolutional_layers.Conv2DTranspose, Layer):
		  """Transposed convolution layer (sometimes called Deconvolution).

		  The need for transposed convolutions generally arises
		  from the desire to use a transformation going in the opposite direction
		  of a normal convolution, i.e., from something that has the shape of the
		  output of some convolution to something that has the shape of its input
		  while maintaining a connectivity pattern that is compatible with
		  said convolution.

		  When using this layer as the first layer in a model,
		  provide the keyword argument `input_shape`
		  (tuple of integers, does not include the sample axis),
		  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
		  in `data_format="channels_last"`.

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number of output filters in the convolution).
		      kernel_size: An integer or tuple/list of 2 integers, specifying the
		          width and height of the 2D convolution window.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		      strides: An integer or tuple/list of 2 integers,
		          specifying the strides of the convolution along the width and height.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: one of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      dilation_rate: an integer or tuple/list of 2 integers, specifying
		          the dilation rate to use for dilated convolution.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Currently, specifying any `dilation_rate` value != 1 is
		          incompatible with specifying any stride value != 1.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to the kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      4D tensor with shape:
		      `(batch, channels, rows, cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(batch, rows, cols, channels)` if data_format='channels_last'.

		  Output shape:
		      4D tensor with shape:
		      `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
		      `rows` and `cols` values might have changed due to padding.

		  References:
		      - [A guide to convolution arithmetic for deep
		        learning](https://arxiv.org/abs/1603.07285v1)
		      - [Deconvolutional
		        Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf)
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1),
		               padding='valid',
		               data_format=None,
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    super(Conv2DTranspose, self).__init__(
		        filters=filters,
		        kernel_size=kernel_size,
		        strides=strides,
		        padding=padding,
		        data_format=data_format,
		        activation=activations.get(activation),
		        use_bias=use_bias,
		        kernel_initializer=initializers.get(kernel_initializer),
		        bias_initializer=initializers.get(bias_initializer),
		        kernel_regularizer=regularizers.get(kernel_regularizer),
		        bias_regularizer=regularizers.get(bias_regularizer),
		        activity_regularizer=regularizers.get(activity_regularizer),
		        **kwargs)
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		  def build(self, input_shape):
		    super(Conv2DTranspose, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.kernel_constraint:
		      self.constraints[self.kernel] = self.kernel_constraint
		    if self.use_bias and self.bias_constraint:
		      self.constraints[self.bias] = self.bias_constraint

		  def get_config(self):
		    config = {
		        'filters': self.filters,
		        'kernel_size': self.kernel_size,
		        'strides': self.strides,
		        'padding': self.padding,
		        'data_format': self.data_format,
		        'activation': activations.serialize(self.activation),
		        'use_bias': self.use_bias,
		        'kernel_initializer': initializers.serialize(self.kernel_initializer),
		        'bias_initializer': initializers.serialize(self.bias_initializer),
		        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint': constraints.serialize(self.kernel_constraint),
		        'bias_constraint': constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(Conv2DTranspose, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class SeparableConv2D(tf_convolutional_layers.SeparableConv2D, Layer):
		  """Depthwise separable 2D convolution.

		  Separable convolutions consist in first performing
		  a depthwise spatial convolution
		  (which acts on each input channel separately)
		  followed by a pointwise convolution which mixes together the resulting
		  output channels. The `depth_multiplier` argument controls how many
		  output channels are generated per input channel in the depthwise step.

		  Intuitively, separable convolutions can be understood as
		  a way to factorize a convolution kernel into two smaller kernels,
		  or as an extreme version of an Inception block.

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of 2 integers, specifying the
		          width and height of the 2D convolution window.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		      strides: An integer or tuple/list of 2 integers,
		          specifying the strides of the convolution along the width and height.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: one of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      depth_multiplier: The number of depthwise convolution output channels
		          for each input channel.
		          The total number of depthwise convolution output
		          channels will be equal to `filterss_in * depth_multiplier`.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      depthwise_initializer: Initializer for the depthwise kernel matrix.
		      pointwise_initializer: Initializer for the pointwise kernel matrix.
		      bias_initializer: Initializer for the bias vector.
		      depthwise_regularizer: Regularizer function applied to
		          the depthwise kernel matrix.
		      pointwise_regularizer: Regularizer function applied to
		          the depthwise kernel matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      depthwise_constraint: Constraint function applied to
		          the depthwise kernel matrix.
		      pointwise_constraint: Constraint function applied to
		          the pointwise kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      4D tensor with shape:
		      `(batch, channels, rows, cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(batch, rows, cols, channels)` if data_format='channels_last'.

		  Output shape:
		      4D tensor with shape:
		      `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
		      `rows` and `cols` values might have changed due to padding.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1),
		               padding='valid',
		               data_format=None,
		               depth_multiplier=1,
		               activation=None,
		               use_bias=True,
		               depthwise_initializer='glorot_uniform',
		               pointwise_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               depthwise_regularizer=None,
		               pointwise_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               depthwise_constraint=None,
		               pointwise_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    super(SeparableConv2D, self).__init__(
		        filters=filters,
		        kernel_size=kernel_size,
		        strides=strides,
		        padding=padding,
		        data_format=data_format,
		        activation=activations.get(activation),
		        use_bias=use_bias,
		        depthwise_initializer=initializers.get(depthwise_initializer),
		        pointwise_initializer=initializers.get(pointwise_initializer),
		        bias_initializer=initializers.get(bias_initializer),
		        depthwise_regularizer=regularizers.get(depthwise_regularizer),
		        pointwise_regularizer=regularizers.get(pointwise_regularizer),
		        bias_regularizer=regularizers.get(bias_regularizer),
		        activity_regularizer=regularizers.get(activity_regularizer),
		        **kwargs)
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.depthwise_constraint = constraints.get(depthwise_constraint)
		    self.pointwise_constraint = constraints.get(pointwise_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		  def build(self, input_shape):
		    super(SeparableConv2D, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.depthwise_constraint:
		      self.constraints[self.depthwise_kernel] = self.depthwise_constraint
		    if self.pointwise_constraint:
		      self.constraints[self.pointwise_kernel] = self.pointwise_constraint
		    if self.use_bias and self.bias_constraint:
		      self.constraints[self.bias] = self.bias_constraint

		  def get_config(self):
		    config = {
		        'filters': self.filters,
		        'kernel_size': self.kernel_size,
		        'strides': self.strides,
		        'padding': self.padding,
		        'data_format': self.data_format,
		        'activation': activations.serialize(self.activation),
		        'use_bias': self.use_bias,
		        'depthwise_initializer': initializers.serialize(
		            self.depthwise_initializer),
		        'pointwise_initializer': initializers.serialize(
		            self.pointwise_initializer),
		        'bias_initializer': initializers.serialize(self.bias_initializer),
		        'depthwise_regularizer': regularizers.serialize(
		            self.depthwise_regularizer),
		        'pointwise_regularizer': regularizers.serialize(
		            self.pointwise_regularizer),
		        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'depthwise_constraint': constraints.serialize(
		            self.depthwise_constraint),
		        'pointwise_constraint': constraints.serialize(
		            self.pointwise_constraint),
		        'bias_constraint': constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(SeparableConv2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class UpSampling1D(Layer):
		  """Upsampling layer for 1D inputs.

		  Repeats each temporal step `size` times along the time axis.

		  Arguments:
		      size: integer. Upsampling factor.

		  Input shape:
		      3D tensor with shape: `(batch, steps, features)`.

		  Output shape:
		      3D tensor with shape: `(batch, upsampled_steps, features)`.
		  """

		  def __init__(self, size=2, **kwargs):
		    super(UpSampling1D, self).__init__(**kwargs)
		    self.size = int(size)
		    self.input_spec = InputSpec(ndim=3)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    size = self.size * input_shape[1] if input_shape[1] is not None else None
		    return tensor_shape.TensorShape([input_shape[0], size, input_shape[2]])

		  def call(self, inputs):
		    output = K.repeat_elements(inputs, self.size, axis=1)
		    return output

		  def get_config(self):
		    config = {'size': self.size}
		    base_config = super(UpSampling1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class UpSampling2D(Layer):
		  """Upsampling layer for 2D inputs.

		  Repeats the rows and columns of the data
		  by size[0] and size[1] respectively.

		  Arguments:
		      size: int, or tuple of 2 integers.
		          The upsampling factors for rows and columns.
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      4D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, rows, cols, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, rows, cols)`

		  Output shape:
		      4D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, upsampled_rows, upsampled_cols, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, upsampled_rows, upsampled_cols)`
		  """

		  def __init__(self, size=(2, 2), data_format=None, **kwargs):
		    super(UpSampling2D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.size = conv_utils.normalize_tuple(size, 2, 'size')
		    self.input_spec = InputSpec(ndim=4)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_first':
		      height = self.size[0] * input_shape[
		          2] if input_shape[2] is not None else None
		      width = self.size[1] * input_shape[
		          3] if input_shape[3] is not None else None
		      return tensor_shape.TensorShape(
		          [input_shape[0], input_shape[1], height, width])
		    else:
		      height = self.size[0] * input_shape[
		          1] if input_shape[1] is not None else None
		      width = self.size[1] * input_shape[
		          2] if input_shape[2] is not None else None
		      return tensor_shape.TensorShape(
		          [input_shape[0], height, width, input_shape[3]])

		  def call(self, inputs):
		    return K.resize_images(inputs, self.size[0], self.size[1], self.data_format)

		  def get_config(self):
		    config = {'size': self.size, 'data_format': self.data_format}
		    base_config = super(UpSampling2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class UpSampling3D(Layer):
		  """Upsampling layer for 3D inputs.

		  Repeats the 1st, 2nd and 3rd dimensions
		  of the data by size[0], size[1] and size[2] respectively.

		  Arguments:
		      size: int, or tuple of 3 integers.
		          The upsampling factors for dim1, dim2 and dim3.
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      5D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, dim1, dim2, dim3, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, dim1, dim2, dim3)`

		  Output shape:
		      5D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, upsampled_dim1, upsampled_dim2, upsampled_dim3, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, upsampled_dim1, upsampled_dim2, upsampled_dim3)`
		  """

		  def __init__(self, size=(2, 2, 2), data_format=None, **kwargs):
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.size = conv_utils.normalize_tuple(size, 3, 'size')
		    self.input_spec = InputSpec(ndim=5)
		    super(UpSampling3D, self).__init__(**kwargs)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_first':
		      dim1 = self.size[0] * input_shape[
		          2] if input_shape[2] is not None else None
		      dim2 = self.size[1] * input_shape[
		          3] if input_shape[3] is not None else None
		      dim3 = self.size[2] * input_shape[
		          4] if input_shape[4] is not None else None
		      return tensor_shape.TensorShape(
		          [input_shape[0], input_shape[1], dim1, dim2, dim3])
		    else:
		      dim1 = self.size[0] * input_shape[
		          1] if input_shape[1] is not None else None
		      dim2 = self.size[1] * input_shape[
		          2] if input_shape[2] is not None else None
		      dim3 = self.size[2] * input_shape[
		          3] if input_shape[3] is not None else None
		      return tensor_shape.TensorShape(
		          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

		  def call(self, inputs):
		    return K.resize_volumes(inputs, self.size[0], self.size[1], self.size[2],
		                            self.data_format)

		  def get_config(self):
		    config = {'size': self.size, 'data_format': self.data_format}
		    base_config = super(UpSampling3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ZeroPadding1D(Layer):
		  """Zero-padding layer for 1D input (e.g. temporal sequence).

		  Arguments:
		      padding: int, or tuple of int (length 2), or dictionary.
		          - If int:
		          How many zeros to add at the beginning and end of
		          the padding dimension (axis 1).
		          - If tuple of int (length 2):
		          How many zeros to add at the beginning and at the end of
		          the padding dimension (`(left_pad, right_pad)`).

		  Input shape:
		      3D tensor with shape `(batch, axis_to_pad, features)`

		  Output shape:
		      3D tensor with shape `(batch, padded_axis, features)`
		  """

		  def __init__(self, padding=1, **kwargs):
		    super(ZeroPadding1D, self).__init__(**kwargs)
		    self.padding = conv_utils.normalize_tuple(padding, 2, 'padding')
		    self.input_spec = InputSpec(ndim=3)

		  def _compute_output_shape(self, input_shape):
		    if input_shape[1] is not None:
		      length = input_shape[1] + self.padding[0] + self.padding[1]
		    else:
		      length = None
		    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

		  def call(self, inputs):
		    return K.temporal_padding(inputs, padding=self.padding)

		  def get_config(self):
		    config = {'padding': self.padding}
		    base_config = super(ZeroPadding1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ZeroPadding2D(Layer):
		  """Zero-padding layer for 2D input (e.g. picture).

		  This layer can add rows and columns or zeros
		  at the top, bottom, left and right side of an image tensor.

		  Arguments:
		      padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
		          - If int: the same symmetric padding
		              is applied to width and height.
		          - If tuple of 2 ints:
		              interpreted as two different
		              symmetric padding values for height and width:
		              `(symmetric_height_pad, symmetric_width_pad)`.
		          - If tuple of 2 tuples of 2 ints:
		              interpreted as
		              `((top_pad, bottom_pad), (left_pad, right_pad))`
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      4D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, rows, cols, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, rows, cols)`

		  Output shape:
		      4D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, padded_rows, padded_cols, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, padded_rows, padded_cols)`
		  """

		  def __init__(self, padding=(1, 1), data_format=None, **kwargs):
		    super(ZeroPadding2D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    if isinstance(padding, int):
		      self.padding = ((padding, padding), (padding, padding))
		    elif hasattr(padding, '__len__'):
		      if len(padding) != 2:
		        raise ValueError('`padding` should have two elements. '
		                         'Found: ' + str(padding))
		      height_padding = conv_utils.normalize_tuple(padding[0], 2,
		                                                  '1st entry of padding')
		      width_padding = conv_utils.normalize_tuple(padding[1], 2,
		                                                 '2nd entry of padding')
		      self.padding = (height_padding, width_padding)
		    else:
		      raise ValueError('`padding` should be either an int, '
		                       'a tuple of 2 ints '
		                       '(symmetric_height_pad, symmetric_width_pad), '
		                       'or a tuple of 2 tuples of 2 ints '
		                       '((top_pad, bottom_pad), (left_pad, right_pad)). '
		                       'Found: ' + str(padding))
		    self.input_spec = InputSpec(ndim=4)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_first':
		      if input_shape[2] is not None:
		        rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
		      else:
		        rows = None
		      if input_shape[3] is not None:
		        cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
		      else:
		        cols = None
		      return tensor_shape.TensorShape(
		          [input_shape[0], input_shape[1], rows, cols])
		    elif self.data_format == 'channels_last':
		      if input_shape[1] is not None:
		        rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
		      else:
		        rows = None
		      if input_shape[2] is not None:
		        cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
		      else:
		        cols = None
		      return tensor_shape.TensorShape(
		          [input_shape[0], rows, cols, input_shape[3]])

		  def call(self, inputs):
		    return K.spatial_2d_padding(
		        inputs, padding=self.padding, data_format=self.data_format)

		  def get_config(self):
		    config = {'padding': self.padding, 'data_format': self.data_format}
		    base_config = super(ZeroPadding2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ZeroPadding3D(Layer):
		  """Zero-padding layer for 3D data (spatial or spatio-temporal).

		  Arguments:
		      padding: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
		          - If int: the same symmetric padding
		              is applied to width and height.
		          - If tuple of 2 ints:
		              interpreted as two different
		              symmetric padding values for height and width:
		              `(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad)`.
		          - If tuple of 2 tuples of 2 ints:
		              interpreted as
		              `((left_dim1_pad, right_dim1_pad), (left_dim2_pad,
		                right_dim2_pad), (left_dim3_pad, right_dim3_pad))`
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      5D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, first_axis_to_pad, second_axis_to_pad, third_axis_to_pad,
		            depth)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, depth, first_axis_to_pad, second_axis_to_pad,
		            third_axis_to_pad)`

		  Output shape:
		      5D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, first_padded_axis, second_padded_axis, third_axis_to_pad,
		            depth)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, depth, first_padded_axis, second_padded_axis,
		            third_axis_to_pad)`
		  """

		  def __init__(self, padding=(1, 1, 1), data_format=None, **kwargs):
		    super(ZeroPadding3D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    if isinstance(padding, int):
		      self.padding = ((padding, padding), (padding, padding), (padding,
		                                                               padding))
		    elif hasattr(padding, '__len__'):
		      if len(padding) != 3:
		        raise ValueError('`padding` should have 3 elements. '
		                         'Found: ' + str(padding))
		      dim1_padding = conv_utils.normalize_tuple(padding[0], 2,
		                                                '1st entry of padding')
		      dim2_padding = conv_utils.normalize_tuple(padding[1], 2,
		                                                '2nd entry of padding')
		      dim3_padding = conv_utils.normalize_tuple(padding[2], 2,
		                                                '3rd entry of padding')
		      self.padding = (dim1_padding, dim2_padding, dim3_padding)
		    else:
		      raise ValueError(
		          '`padding` should be either an int, '
		          'a tuple of 3 ints '
		          '(symmetric_dim1_pad, symmetric_dim2_pad, symmetric_dim3_pad), '
		          'or a tuple of 3 tuples of 2 ints '
		          '((left_dim1_pad, right_dim1_pad),'
		          ' (left_dim2_pad, right_dim2_pad),'
		          ' (left_dim3_pad, right_dim2_pad)). '
		          'Found: ' + str(padding))
		    self.input_spec = InputSpec(ndim=5)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_first':
		      if input_shape[2] is not None:
		        dim1 = input_shape[2] + 2 * self.padding[0][0]
		      else:
		        dim1 = None
		      if input_shape[3] is not None:
		        dim2 = input_shape[3] + 2 * self.padding[1][0]
		      else:
		        dim2 = None
		      if input_shape[4] is not None:
		        dim3 = input_shape[4] + 2 * self.padding[2][0]
		      else:
		        dim3 = None
		      return tensor_shape.TensorShape(
		          [input_shape[0], input_shape[1], dim1, dim2, dim3])
		    elif self.data_format == 'channels_last':
		      if input_shape[1] is not None:
		        dim1 = input_shape[1] + 2 * self.padding[0][1]
		      else:
		        dim1 = None
		      if input_shape[2] is not None:
		        dim2 = input_shape[2] + 2 * self.padding[1][1]
		      else:
		        dim2 = None
		      if input_shape[3] is not None:
		        dim3 = input_shape[3] + 2 * self.padding[2][1]
		      else:
		        dim3 = None
		      return tensor_shape.TensorShape(
		          [input_shape[0], dim1, dim2, dim3, input_shape[4]])

		  def call(self, inputs):
		    return K.spatial_3d_padding(
		        inputs, padding=self.padding, data_format=self.data_format)

		  def get_config(self):
		    config = {'padding': self.padding, 'data_format': self.data_format}
		    base_config = super(ZeroPadding3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Cropping1D(Layer):
		  """Cropping layer for 1D input (e.g. temporal sequence).

		  It crops along the time dimension (axis 1).

		  Arguments:
		      cropping: int or tuple of int (length 2)
		          How many units should be trimmed off at the beginning and end of
		          the cropping dimension (axis 1).
		          If a single int is provided,
		          the same value will be used for both.

		  Input shape:
		      3D tensor with shape `(batch, axis_to_crop, features)`

		  Output shape:
		      3D tensor with shape `(batch, cropped_axis, features)`
		  """

		  def __init__(self, cropping=(1, 1), **kwargs):
		    super(Cropping1D, self).__init__(**kwargs)
		    self.cropping = conv_utils.normalize_tuple(cropping, 2, 'cropping')
		    self.input_spec = InputSpec(ndim=3)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if input_shape[1] is not None:
		      length = input_shape[1] - self.cropping[0] - self.cropping[1]
		    else:
		      length = None
		    return tensor_shape.TensorShape([input_shape[0], length, input_shape[2]])

		  def call(self, inputs):
		    if self.cropping[1] == 0:
		      return inputs[:, self.cropping[0]:, :]
		    else:
		      return inputs[:, self.cropping[0]:-self.cropping[1], :]

		  def get_config(self):
		    config = {'cropping': self.cropping}
		    base_config = super(Cropping1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Cropping2D(Layer):
		  """Cropping layer for 2D input (e.g. picture).

		  It crops along spatial dimensions, i.e. width and height.

		  Arguments:
		      cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
		          - If int: the same symmetric cropping
		              is applied to width and height.
		          - If tuple of 2 ints:
		              interpreted as two different
		              symmetric cropping values for height and width:
		              `(symmetric_height_crop, symmetric_width_crop)`.
		          - If tuple of 2 tuples of 2 ints:
		              interpreted as
		              `((top_crop, bottom_crop), (left_crop, right_crop))`
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      4D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, rows, cols, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, rows, cols)`

		  Output shape:
		      4D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, cropped_rows, cropped_cols, channels)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, channels, cropped_rows, cropped_cols)`

		  Examples:

		  ```python
		      # Crop the input 2D images or feature maps
		      model = Sequential()
		      model.add(Cropping2D(cropping=((2, 2), (4, 4)),
		                           input_shape=(28, 28, 3)))
		      # now model.output_shape == (None, 24, 20, 3)
		      model.add(Conv2D(64, (3, 3), padding='same))
		      model.add(Cropping2D(cropping=((2, 2), (2, 2))))
		      # now model.output_shape == (None, 20, 16. 64)
		  ```
		  """

		  def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
		    super(Cropping2D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    if isinstance(cropping, int):
		      self.cropping = ((cropping, cropping), (cropping, cropping))
		    elif hasattr(cropping, '__len__'):
		      if len(cropping) != 2:
		        raise ValueError('`cropping` should have two elements. '
		                         'Found: ' + str(cropping))
		      height_cropping = conv_utils.normalize_tuple(cropping[0], 2,
		                                                   '1st entry of cropping')
		      width_cropping = conv_utils.normalize_tuple(cropping[1], 2,
		                                                  '2nd entry of cropping')
		      self.cropping = (height_cropping, width_cropping)
		    else:
		      raise ValueError('`cropping` should be either an int, '
		                       'a tuple of 2 ints '
		                       '(symmetric_height_crop, symmetric_width_crop), '
		                       'or a tuple of 2 tuples of 2 ints '
		                       '((top_crop, bottom_crop), (left_crop, right_crop)). '
		                       'Found: ' + str(cropping))
		    self.input_spec = InputSpec(ndim=4)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    # pylint: disable=invalid-unary-operand-type
		    if self.data_format == 'channels_first':
		      return tensor_shape.TensorShape([
		          input_shape[0], input_shape[1],
		          input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
		          if input_shape[2] else None,
		          input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
		          if input_shape[3] else None
		      ])
		    else:
		      return tensor_shape.TensorShape([
		          input_shape[0],
		          input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
		          if input_shape[1] else None,
		          input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
		          if input_shape[2] else None, input_shape[3]
		      ])
		    # pylint: enable=invalid-unary-operand-type

		  def call(self, inputs):
		    # pylint: disable=invalid-unary-operand-type
		    if self.data_format == 'channels_first':
		      if self.cropping[0][1] == self.cropping[1][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:]
		      elif self.cropping[0][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
		                      -self.cropping[1][1]]
		      elif self.cropping[1][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
		                      self.cropping[1][0]:]
		      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
		                    self.cropping[1][0]:-self.cropping[1][1]]
		    else:
		      if self.cropping[0][1] == self.cropping[1][1] == 0:
		        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:, :]
		      elif self.cropping[0][1] == 0:
		        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
		                      -self.cropping[1][1], :]
		      elif self.cropping[1][1] == 0:
		        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
		                      self.cropping[1][0]:, :]
		      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
		          1][0]:-self.cropping[1][1], :]  # pylint: disable=invalid-unary-operand-type
		    # pylint: enable=invalid-unary-operand-type

		  def get_config(self):
		    config = {'cropping': self.cropping, 'data_format': self.data_format}
		    base_config = super(Cropping2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Cropping3D(Layer):
		  """Cropping layer for 3D data (e.g.

		  spatial or spatio-temporal).

		  Arguments:
		      cropping: int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
		          - If int: the same symmetric cropping
		              is applied to width and height.
		          - If tuple of 2 ints:
		              interpreted as two different
		              symmetric cropping values for height and width:
		              `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
		          - If tuple of 2 tuples of 2 ints:
		              interpreted as
		              `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
		                right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      5D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, first_axis_to_crop, second_axis_to_crop, third_axis_to_crop,
		            depth)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, depth, first_axis_to_crop, second_axis_to_crop,
		            third_axis_to_crop)`

		  Output shape:
		      5D tensor with shape:
		      - If `data_format` is `"channels_last"`:
		          `(batch, first_cropped_axis, second_cropped_axis, third_cropped_axis,
		            depth)`
		      - If `data_format` is `"channels_first"`:
		          `(batch, depth, first_cropped_axis, second_cropped_axis,
		            third_cropped_axis)`
		  """

		  def __init__(self,
		               cropping=((1, 1), (1, 1), (1, 1)),
		               data_format=None,
		               **kwargs):
		    super(Cropping3D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    if isinstance(cropping, int):
		      self.cropping = ((cropping, cropping), (cropping, cropping), (cropping,
		                                                                    cropping))
		    elif hasattr(cropping, '__len__'):
		      if len(cropping) != 3:
		        raise ValueError('`cropping` should have 3 elements. '
		                         'Found: ' + str(cropping))
		      dim1_cropping = conv_utils.normalize_tuple(cropping[0], 2,
		                                                 '1st entry of cropping')
		      dim2_cropping = conv_utils.normalize_tuple(cropping[1], 2,
		                                                 '2nd entry of cropping')
		      dim3_cropping = conv_utils.normalize_tuple(cropping[2], 2,
		                                                 '3rd entry of cropping')
		      self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
		    else:
		      raise ValueError(
		          '`cropping` should be either an int, '
		          'a tuple of 3 ints '
		          '(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop), '
		          'or a tuple of 3 tuples of 2 ints '
		          '((left_dim1_crop, right_dim1_crop),'
		          ' (left_dim2_crop, right_dim2_crop),'
		          ' (left_dim3_crop, right_dim2_crop)). '
		          'Found: ' + str(cropping))
		    self.input_spec = InputSpec(ndim=5)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    # pylint: disable=invalid-unary-operand-type
		    if self.data_format == 'channels_first':
		      if input_shape[2] is not None:
		        dim1 = input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
		      else:
		        dim1 = None
		      if input_shape[3] is not None:
		        dim2 = input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
		      else:
		        dim2 = None
		      if input_shape[4] is not None:
		        dim3 = input_shape[4] - self.cropping[2][0] - self.cropping[2][1]
		      else:
		        dim3 = None
		      return tensor_shape.TensorShape(
		          [input_shape[0], input_shape[1], dim1, dim2, dim3])
		    elif self.data_format == 'channels_last':
		      if input_shape[1] is not None:
		        dim1 = input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
		      else:
		        dim1 = None
		      if input_shape[2] is not None:
		        dim2 = input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
		      else:
		        dim2 = None
		      if input_shape[3] is not None:
		        dim3 = input_shape[3] - self.cropping[2][0] - self.cropping[2][1]
		      else:
		        dim3 = None
		      return tensor_shape.TensorShape(
		          [input_shape[0], dim1, dim2, dim3, input_shape[4]])
		    # pylint: enable=invalid-unary-operand-type

		  def call(self, inputs):
		    # pylint: disable=invalid-unary-operand-type
		    if self.data_format == 'channels_first':
		      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
		                      self.cropping[2][0]:]
		      elif self.cropping[0][1] == self.cropping[1][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:,
		                      self.cropping[2][0]:-self.cropping[2][1]]
		      elif self.cropping[1][1] == self.cropping[2][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
		                      self.cropping[1][0]:, self.cropping[2][0]:]
		      elif self.cropping[0][1] == self.cropping[2][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][0]:
		                      -self.cropping[1][1], self.cropping[2][0]:]
		      elif self.cropping[0][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:, self.cropping[1][
		            0]:-self.cropping[1][1], self.cropping[2][0]:-self.cropping[2][1]]
		      elif self.cropping[1][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
		                      cropping[1][0]:, self.cropping[2][0]:-self.cropping[2][1]]
		      elif self.cropping[2][1] == 0:
		        return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1], self.
		                      cropping[1][0]:-self.cropping[1][1], self.cropping[2][0]:]
		      return inputs[:, :, self.cropping[0][0]:-self.cropping[0][1],
		                    self.cropping[1][0]:-self.cropping[1][1], self.cropping[2][
		                        0]:-self.cropping[2][1]]
		    else:
		      if self.cropping[0][1] == self.cropping[1][1] == self.cropping[2][1] == 0:
		        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
		                      self.cropping[2][0]:, :]
		      elif self.cropping[0][1] == self.cropping[1][1] == 0:
		        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:,
		                      self.cropping[2][0]:-self.cropping[2][1], :]
		      elif self.cropping[1][1] == self.cropping[2][1] == 0:
		        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
		                      self.cropping[1][0]:, self.cropping[2][0]:, :]
		      elif self.cropping[0][1] == self.cropping[2][1] == 0:
		        return inputs[:, self.cropping[0][0]:, self.cropping[1][0]:
		                      -self.cropping[1][1], self.cropping[2][0]:, :]
		      elif self.cropping[0][1] == 0:
		        return inputs[:, self.cropping[0][0]:, self.cropping[1][
		            0]:-self.cropping[1][1], self.cropping[2][0]:
		                      -self.cropping[2][1], :]
		      elif self.cropping[1][1] == 0:
		        return inputs[:, self.cropping[0][
		            0]:-self.cropping[0][1], self.cropping[1][0]:, self.cropping[2][0]:
		                      -self.cropping[2][1], :]
		      elif self.cropping[2][1] == 0:
		        return inputs[:, self.cropping[0][0]:-self.cropping[0][1],
		                      self.cropping[1][0]:-self.cropping[1][1], self.cropping[
		                          2][0]:, :]
		      return inputs[:, self.cropping[0][0]:-self.cropping[0][1], self.cropping[
		          1][0]:-self.cropping[1][1], self.cropping[2][0]:  # pylint: disable=invalid-unary-operand-type
		                    -self.cropping[2][1], :]  # pylint: disable=invalid-unary-operand-type
		    # pylint: enable=invalid-unary-operand-type

		  def get_config(self):
		    config = {'cropping': self.cropping, 'data_format': self.data_format}
		    base_config = super(Cropping3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		# Aliases

		Convolution1D = Conv1D
		Convolution2D = Conv2D
		Convolution3D = Conv3D
		SeparableConvolution2D = SeparableConv2D
		Convolution2DTranspose = Conv2DTranspose
		Deconvolution2D = Deconv2D = Conv2DTranspose

	class core_py:

		"""Core Keras layers.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import copy
			import types as python_types

			import numpy as np

			from tensorflow.contrib.keras.python.keras import activations
			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object
			from tensorflow.contrib.keras.python.keras.utils.generic_utils import func_dump
			from tensorflow.contrib.keras.python.keras.utils.generic_utils import func_load
			from tensorflow.python.framework import tensor_shape
			from tensorflow.python.layers import core as tf_core_layers
			from tensorflow.python.util import tf_inspect


		class Masking(Layer):
		  """Masks a sequence by using a mask value to skip timesteps.

		  For each timestep in the input tensor (dimension #1 in the tensor),
		  if all values in the input tensor at that timestep
		  are equal to `mask_value`, then the timestep will be masked (skipped)
		  in all downstream layers (as long as they support masking).

		  If any downstream layer does not support masking yet receives such
		  an input mask, an exception will be raised.

		  Example:

		  Consider a Numpy data array `x` of shape `(samples, timesteps, features)`,
		  to be fed to a LSTM layer.
		  You want to mask timestep #3 and #5 because you lack data for
		  these timesteps. You can:

		      - set `x[:, 3, :] = 0.` and `x[:, 5, :] = 0.`
		      - insert a `Masking` layer with `mask_value=0.` before the LSTM layer:

		  ```python
		      model = Sequential()
		      model.add(Masking(mask_value=0., input_shape=(timesteps, features)))
		      model.add(LSTM(32))
		  ```
		  """

		  def __init__(self, mask_value=0., **kwargs):
		    super(Masking, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.mask_value = mask_value

		  def compute_mask(self, inputs, mask=None):
		    return K.any(K.not_equal(inputs, self.mask_value), axis=-1)

		  def call(self, inputs):
		    boolean_mask = K.any(
		        K.not_equal(inputs, self.mask_value), axis=-1, keepdims=True)
		    return inputs * K.cast(boolean_mask, K.floatx())

		  def get_config(self):
		    config = {'mask_value': self.mask_value}
		    base_config = super(Masking, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Dropout(tf_core_layers.Dropout, Layer):
		  """Applies Dropout to the input.

		  Dropout consists in randomly setting
		  a fraction `rate` of input units to 0 at each update during training time,
		  which helps prevent overfitting.

		  Arguments:
		      rate: float between 0 and 1. Fraction of the input units to drop.
		      noise_shape: 1D integer tensor representing the shape of the
		          binary dropout mask that will be multiplied with the input.
		          For instance, if your inputs have shape
		          `(batch_size, timesteps, features)` and
		          you want the dropout mask to be the same for all timesteps,
		          you can use `noise_shape=(batch_size, 1, features)`.
		      seed: A Python integer to use as random seed.
		  """

		  def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
		    self.supports_masking = True
		    # Inheritance call order:
		    # 1) tf.layers.Dropout, 2) keras.layers.Layer, 3) tf.layers.Layer
		    super(Dropout, self).__init__(**kwargs)

		  def call(self, inputs, training=None):
		    if training is None:
		      training = K.learning_phase()
		    output = super(Dropout, self).call(inputs, training=training)
		    if training is K.learning_phase():
		      output._uses_learning_phase = True  # pylint: disable=protected-access
		    return output

		  def get_config(self):
		    config = {'rate': self.rate}
		    base_config = super(Dropout, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class SpatialDropout1D(Dropout):
		  """Spatial 1D version of Dropout.

		  This version performs the same function as Dropout, however it drops
		  entire 1D feature maps instead of individual elements. If adjacent frames
		  within feature maps are strongly correlated (as is normally the case in
		  early convolution layers) then regular dropout will not regularize the
		  activations and will otherwise just result in an effective learning rate
		  decrease. In this case, SpatialDropout1D will help promote independence
		  between feature maps and should be used instead.

		  Arguments:
		      rate: float between 0 and 1. Fraction of the input units to drop.

		  Input shape:
		      3D tensor with shape:
		      `(samples, timesteps, channels)`

		  Output shape:
		      Same as input

		  References:
		      - [Efficient Object Localization Using Convolutional
		        Networks](https://arxiv.org/abs/1411.4280)
		  """

		  def __init__(self, rate, **kwargs):
		    super(SpatialDropout1D, self).__init__(rate, **kwargs)
		    self.input_spec = InputSpec(ndim=3)

		  def _get_noise_shape(self, inputs):
		    input_shape = K.shape(inputs)
		    noise_shape = (input_shape[0], 1, input_shape[2])
		    return noise_shape


		class SpatialDropout2D(Dropout):
		  """Spatial 2D version of Dropout.

		  This version performs the same function as Dropout, however it drops
		  entire 2D feature maps instead of individual elements. If adjacent pixels
		  within feature maps are strongly correlated (as is normally the case in
		  early convolution layers) then regular dropout will not regularize the
		  activations and will otherwise just result in an effective learning rate
		  decrease. In this case, SpatialDropout2D will help promote independence
		  between feature maps and should be used instead.

		  Arguments:
		      rate: float between 0 and 1. Fraction of the input units to drop.
		      data_format: 'channels_first' or 'channels_last'.
		          In 'channels_first' mode, the channels dimension
		          (the depth) is at index 1,
		          in 'channels_last' mode is it at index 3.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      4D tensor with shape:
		      `(samples, channels, rows, cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(samples, rows, cols, channels)` if data_format='channels_last'.

		  Output shape:
		      Same as input

		  References:
		      - [Efficient Object Localization Using Convolutional
		        Networks](https://arxiv.org/abs/1411.4280)
		  """

		  def __init__(self, rate, data_format=None, **kwargs):
		    super(SpatialDropout2D, self).__init__(rate, **kwargs)
		    if data_format is None:
		      data_format = K.image_data_format()
		    if data_format not in {'channels_last', 'channels_first'}:
		      raise ValueError('data_format must be in '
		                       '{"channels_last", "channels_first"}')
		    self.data_format = data_format
		    self.input_spec = InputSpec(ndim=4)

		  def _get_noise_shape(self, inputs):
		    input_shape = K.shape(inputs)
		    if self.data_format == 'channels_first':
		      noise_shape = (input_shape[0], input_shape[1], 1, 1)
		    elif self.data_format == 'channels_last':
		      noise_shape = (input_shape[0], 1, 1, input_shape[3])
		    else:
		      raise ValueError('Invalid data_format:', self.data_format)
		    return noise_shape


		class SpatialDropout3D(Dropout):
		  """Spatial 3D version of Dropout.

		  This version performs the same function as Dropout, however it drops
		  entire 3D feature maps instead of individual elements. If adjacent voxels
		  within feature maps are strongly correlated (as is normally the case in
		  early convolution layers) then regular dropout will not regularize the
		  activations and will otherwise just result in an effective learning rate
		  decrease. In this case, SpatialDropout3D will help promote independence
		  between feature maps and should be used instead.

		  Arguments:
		      rate: float between 0 and 1. Fraction of the input units to drop.
		      data_format: 'channels_first' or 'channels_last'.
		          In 'channels_first' mode, the channels dimension (the depth)
		          is at index 1, in 'channels_last' mode is it at index 4.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      5D tensor with shape:
		      `(samples, channels, dim1, dim2, dim3)` if data_format='channels_first'
		      or 5D tensor with shape:
		      `(samples, dim1, dim2, dim3, channels)` if data_format='channels_last'.

		  Output shape:
		      Same as input

		  References:
		      - [Efficient Object Localization Using Convolutional
		        Networks](https://arxiv.org/abs/1411.4280)
		  """

		  def __init__(self, rate, data_format=None, **kwargs):
		    super(SpatialDropout3D, self).__init__(rate, **kwargs)
		    if data_format is None:
		      data_format = K.image_data_format()
		    if data_format not in {'channels_last', 'channels_first'}:
		      raise ValueError('data_format must be in '
		                       '{"channels_last", "channels_first"}')
		    self.data_format = data_format
		    self.input_spec = InputSpec(ndim=5)

		  def _get_noise_shape(self, inputs):
		    input_shape = K.shape(inputs)
		    if self.data_format == 'channels_first':
		      noise_shape = (input_shape[0], input_shape[1], 1, 1, 1)
		    elif self.data_format == 'channels_last':
		      noise_shape = (input_shape[0], 1, 1, 1, input_shape[4])
		    else:
		      raise ValueError('Invalid data_format:', self.data_format)
		    return noise_shape


		class Activation(Layer):
		  """Applies an activation function to an output.

		  Arguments:
		      activation: name of activation function to use
		          or alternatively, a Theano or TensorFlow operation.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as input.
		  """

		  def __init__(self, activation, **kwargs):
		    super(Activation, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.activation = activations.get(activation)

		  def call(self, inputs):
		    return self.activation(inputs)

		  def get_config(self):
		    config = {'activation': activations.serialize(self.activation)}
		    base_config = super(Activation, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Reshape(Layer):
		  """Reshapes an output to a certain shape.

		  Arguments:
		      target_shape: target shape. Tuple of integers,
		          does not include the samples dimension (batch size).

		  Input shape:
		      Arbitrary, although all dimensions in the input shaped must be fixed.
		      Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      `(batch_size,) + target_shape`

		  Example:

		  ```python
		      # as first layer in a Sequential model
		      model = Sequential()
		      model.add(Reshape((3, 4), input_shape=(12,)))
		      # now: model.output_shape == (None, 3, 4)
		      # note: `None` is the batch dimension

		      # as intermediate layer in a Sequential model
		      model.add(Reshape((6, 2)))
		      # now: model.output_shape == (None, 6, 2)

		      # also supports shape inference using `-1` as dimension
		      model.add(Reshape((-1, 2, 2)))
		      # now: model.output_shape == (None, 3, 2, 2)
		  ```
		  """

		  def __init__(self, target_shape, **kwargs):
		    super(Reshape, self).__init__(**kwargs)
		    self.target_shape = tuple(target_shape)

		  def _fix_unknown_dimension(self, input_shape, output_shape):
		    """Find and replace a missing dimension in an output shape.

		    This is a near direct port of the internal Numpy function
		    `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

		    Arguments:
		        input_shape: shape of array being reshaped
		        output_shape: desired shape of the array with at most
		            a single -1 which indicates a dimension that should be
		            derived from the input shape.

		    Returns:
		        The new output shape with a -1 replaced with its computed value.

		        Raises a ValueError if the total array size of the output_shape is
		        different then the input_shape, or more then one unknown dimension
		        is specified.

		    Raises:
		        ValueError: in case of invalid values
		            for `input_shape` or `input_shape`.
		    """
		    output_shape = list(output_shape)
		    msg = 'total size of new array must be unchanged'

		    known, unknown = 1, None
		    for index, dim in enumerate(output_shape):
		      if dim < 0:
		        if unknown is None:
		          unknown = index
		        else:
		          raise ValueError('Can only specify one unknown dimension.')
		      else:
		        known *= dim

		    original = np.prod(input_shape, dtype=int)
		    if unknown is not None:
		      if known == 0 or original % known != 0:
		        raise ValueError(msg)
		      output_shape[unknown] = original // known
		    elif original != known:
		      raise ValueError(msg)
		    return output_shape

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    output_shape = [input_shape[0]]
		    output_shape += self._fix_unknown_dimension(input_shape[1:],
		                                                self.target_shape)
		    return tensor_shape.TensorShape(output_shape)

		  def call(self, inputs):
		    # In case the target shape is not fully defined,
		    # we need access to the shape of x.
		    target_shape = self.target_shape
		    if -1 in target_shape:
		      # target shape not fully defined
		      target_shape = self._compute_output_shape(inputs.get_shape())
		      target_shape = target_shape.as_list()[1:]
		    return K.reshape(inputs, (-1,) + tuple(target_shape))

		  def get_config(self):
		    config = {'target_shape': self.target_shape}
		    base_config = super(Reshape, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Permute(Layer):
		  """Permutes the dimensions of the input according to a given pattern.

		  Useful for e.g. connecting RNNs and convnets together.

		  Example:

		  ```python
		      model = Sequential()
		      model.add(Permute((2, 1), input_shape=(10, 64)))
		      # now: model.output_shape == (None, 64, 10)
		      # note: `None` is the batch dimension
		  ```

		  Arguments:
		      dims: Tuple of integers. Permutation pattern, does not include the
		          samples dimension. Indexing starts at 1.
		          For instance, `(2, 1)` permutes the first and second dimension
		          of the input.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same as the input shape, but with the dimensions re-ordered according
		      to the specified pattern.
		  """

		  def __init__(self, dims, **kwargs):
		    super(Permute, self).__init__(**kwargs)
		    self.dims = tuple(dims)
		    self.input_spec = InputSpec(ndim=len(self.dims) + 1)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    output_shape = copy.copy(input_shape)
		    for i, dim in enumerate(self.dims):
		      target_dim = input_shape[dim]
		      output_shape[i + 1] = target_dim
		    return tensor_shape.TensorShape(output_shape)

		  def call(self, inputs):
		    return K.permute_dimensions(inputs, (0,) + self.dims)

		  def get_config(self):
		    config = {'dims': self.dims}
		    base_config = super(Permute, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Flatten(Layer):
		  """Flattens the input. Does not affect the batch size.

		  Example:

		  ```python
		      model = Sequential()
		      model.add(Convolution2D(64, 3, 3,
		                              border_mode='same',
		                              input_shape=(3, 32, 32)))
		      # now: model.output_shape == (None, 64, 32, 32)

		      model.add(Flatten())
		      # now: model.output_shape == (None, 65536)
		  ```
		  """

		  def __init__(self, **kwargs):
		    super(Flatten, self).__init__(**kwargs)
		    self.input_spec = InputSpec(min_ndim=3)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if not all(input_shape[1:]):
		      raise ValueError('The shape of the input to "Flatten" '
		                       'is not fully defined '
		                       '(got ' + str(input_shape[1:]) + '. '
		                       'Make sure to pass a complete "input_shape" '
		                       'or "batch_input_shape" argument to the first '
		                       'layer in your model.')
		    return tensor_shape.TensorShape([input_shape[0], np.prod(input_shape[1:])])

		  def call(self, inputs):
		    outputs = K.batch_flatten(inputs)
		    outputs.set_shape(self._compute_output_shape(inputs.get_shape()))
		    return outputs


		class RepeatVector(Layer):
		  """Repeats the input n times.

		  Example:

		  ```python
		      model = Sequential()
		      model.add(Dense(32, input_dim=32))
		      # now: model.output_shape == (None, 32)
		      # note: `None` is the batch dimension

		      model.add(RepeatVector(3))
		      # now: model.output_shape == (None, 3, 32)
		  ```

		  Arguments:
		      n: integer, repetition factor.

		  Input shape:
		      2D tensor of shape `(num_samples, features)`.

		  Output shape:
		      3D tensor of shape `(num_samples, n, features)`.
		  """

		  def __init__(self, n, **kwargs):
		    super(RepeatVector, self).__init__(**kwargs)
		    self.n = n
		    self.input_spec = InputSpec(ndim=2)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    return tensor_shape.TensorShape([input_shape[0], self.n, input_shape[1]])

		  def call(self, inputs):
		    return K.repeat(inputs, self.n)

		  def get_config(self):
		    config = {'n': self.n}
		    base_config = super(RepeatVector, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Lambda(Layer):
		  """Wraps arbitrary expression as a `Layer` object.

		  Examples:

		  ```python
		      # add a x -> x^2 layer
		      model.add(Lambda(lambda x: x ** 2))
		  ```
		  ```python
		      # add a layer that returns the concatenation
		      # of the positive part of the input and
		      # the opposite of the negative part

		      def antirectifier(x):
		          x -= K.mean(x, axis=1, keepdims=True)
		          x = K.l2_normalize(x, axis=1)
		          pos = K.relu(x)
		          neg = K.relu(-x)
		          return K.concatenate([pos, neg], axis=1)

		      model.add(Lambda(antirectifier))
		  ```

		  Arguments:
		      function: The function to be evaluated.
		          Takes input tensor as first argument.
		      arguments: optional dictionary of keyword arguments to be passed
		          to the function.

		  Input shape:
		      Arbitrary. Use the keyword argument input_shape
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Specified by `output_shape` argument
		      (or auto-inferred when using TensorFlow).
		  """

		  def __init__(self, function, mask=None, arguments=None, **kwargs):
		    super(Lambda, self).__init__(**kwargs)
		    self.function = function
		    self.arguments = arguments if arguments else {}
		    if mask is not None:
		      self.supports_masking = True
		    self.mask = mask

		  def call(self, inputs, mask=None):
		    arguments = self.arguments
		    arg_spec = tf_inspect.getargspec(self.function)
		    if 'mask' in arg_spec.args:
		      arguments['mask'] = mask
		    return self.function(inputs, **arguments)

		  def compute_mask(self, inputs, mask=None):
		    if callable(self.mask):
		      return self.mask(inputs, mask)
		    return self.mask

		  def get_config(self):
		    if isinstance(self.function, python_types.LambdaType):
		      function = func_dump(self.function)
		      function_type = 'lambda'
		    else:
		      function = self.function.__name__
		      function_type = 'function'

		    config = {
		        'function': function,
		        'function_type': function_type,
		        'arguments': self.arguments
		    }
		    base_config = super(Lambda, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

		  @classmethod
		  def from_config(cls, config, custom_objects=None):
		    globs = globals()
		    if custom_objects:
		      globs = dict(list(globs.items()) + list(custom_objects.items()))
		    function_type = config.pop('function_type')
		    if function_type == 'function':
		      # Simple lookup in custom objects
		      function = deserialize_keras_object(
		          config['function'],
		          custom_objects=custom_objects,
		          printable_module_name='function in Lambda layer')
		    elif function_type == 'lambda':
		      # Unsafe deserialization from bytecode
		      function = func_load(config['function'], globs=globs)
		    else:
		      raise TypeError('Unknown function type:', function_type)

		    config['function'] = function
		    return cls(**config)


		class Dense(tf_core_layers.Dense, Layer):
		  """Just your regular densely-connected NN layer.

		  `Dense` implements the operation:
		  `output = activation(dot(input, kernel) + bias)`
		  where `activation` is the element-wise activation function
		  passed as the `activation` argument, `kernel` is a weights matrix
		  created by the layer, and `bias` is a bias vector created by the layer
		  (only applicable if `use_bias` is `True`).

		  Note: if the input to the layer has a rank greater than 2, then
		  it is flattened prior to the initial dot product with `kernel`.

		  Example:

		  ```python
		      # as first layer in a sequential model:
		      model = Sequential()
		      model.add(Dense(32, input_shape=(16,)))
		      # now the model will take as input arrays of shape (*, 16)
		      # and output arrays of shape (*, 32)

		      # after the first layer, you don't need to specify
		      # the size of the input anymore:
		      model.add(Dense(32))
		  ```

		  Arguments:
		      units: Positive integer, dimensionality of the output space.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to
		          the `kernel` weights matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      nD tensor with shape: `(batch_size, ..., input_dim)`.
		      The most common situation would be
		      a 2D input with shape `(batch_size, input_dim)`.

		  Output shape:
		      nD tensor with shape: `(batch_size, ..., units)`.
		      For instance, for a 2D input with shape `(batch_size, input_dim)`,
		      the output would have shape `(batch_size, units)`.
		  """

		  def __init__(self,
		               units,
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
		      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

		    # Inheritance call order:
		    # 1) tf.layers.Dense, 2) keras.layers.Layer, 3) tf.layers.Layer
		    super(Dense, self).__init__(
		        units,
		        activation=activations.get(activation),
		        use_bias=use_bias,
		        kernel_initializer=initializers.get(kernel_initializer),
		        bias_initializer=initializers.get(bias_initializer),
		        kernel_regularizer=regularizers.get(kernel_regularizer),
		        bias_regularizer=regularizers.get(bias_regularizer),
		        activity_regularizer=regularizers.get(activity_regularizer),
		        **kwargs)
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)
		    self.supports_masking = True

		  def build(self, input_shape):
		    super(Dense, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.kernel_constraint:
		      self.constraints[self.kernel] = self.kernel_constraint
		    if self.use_bias and self.bias_constraint:
		      self.constraints[self.bias] = self.bias_constraint

		  def get_config(self):
		    config = {
		        'units': self.units,
		        'activation': activations.serialize(self.activation),
		        'use_bias': self.use_bias,
		        'kernel_initializer': initializers.serialize(self.kernel_initializer),
		        'bias_initializer': initializers.serialize(self.bias_initializer),
		        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint': constraints.serialize(self.kernel_constraint),
		        'bias_constraint': constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(Dense, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class ActivityRegularization(Layer):
		  """Layer that applies an update to the cost function based input activity.

		  Arguments:
		      l1: L1 regularization factor (positive float).
		      l2: L2 regularization factor (positive float).

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as input.
		  """

		  def __init__(self, l1=0., l2=0., **kwargs):
		    super(ActivityRegularization, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.l1 = l1
		    self.l2 = l2
		    self.activity_regularizer = regularizers.L1L2(l1=l1, l2=l2)

		  def get_config(self):
		    config = {'l1': self.l1, 'l2': self.l2}
		    base_config = super(ActivityRegularization, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class embeddings_py:

		"""Embedding layer.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.python.framework import tensor_shape


		class Embedding(Layer):
		  """Turns positive integers (indexes) into dense vectors of fixed size.

		  eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

		  This layer can only be used as the first layer in a model.

		  Example:

		  ```python
		    model = Sequential()
		    model.add(Embedding(1000, 64, input_length=10))
		    # the model will take as input an integer matrix of size (batch,
		    input_length).
		    # the largest integer (i.e. word index) in the input should be no larger
		    than 999 (vocabulary size).
		    # now model.output_shape == (None, 10, 64), where None is the batch
		    dimension.

		    input_array = np.random.randint(1000, size=(32, 10))

		    model.compile('rmsprop', 'mse')
		    output_array = model.predict(input_array)
		    assert output_array.shape == (32, 10, 64)
		  ```

		  Arguments:
		    input_dim: int > 0. Size of the vocabulary,
		        i.e. maximum integer index + 1.
		    output_dim: int >= 0. Dimension of the dense embedding.
		    embeddings_initializer: Initializer for the `embeddings` matrix.
		    embeddings_regularizer: Regularizer function applied to
		          the `embeddings` matrix.
		    embeddings_constraint: Constraint function applied to
		          the `embeddings` matrix.
		    mask_zero: Whether or not the input value 0 is a special "padding"
		        value that should be masked out.
		        This is useful when using recurrent layers,
		        which may take variable length inputs.
		        If this is `True` then all subsequent layers
		        in the model need to support masking or an exception will be raised.
		        If mask_zero is set to True, as a consequence, index 0 cannot be
		        used in the vocabulary (input_dim should equal size of
		        vocabulary + 1).
		    input_length: Length of input sequences, when it is constant.
		        This argument is required if you are going to connect
		        `Flatten` then `Dense` layers upstream
		        (without it, the shape of the dense outputs cannot be computed).

		  Input shape:
		      2D tensor with shape: `(batch_size, sequence_length)`.

		  Output shape:
		      3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

		  References:
		      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
		        Networks](http://arxiv.org/abs/1512.05287)
		  """

		  def __init__(self,
		               input_dim,
		               output_dim,
		               embeddings_initializer='uniform',
		               embeddings_regularizer=None,
		               activity_regularizer=None,
		               embeddings_constraint=None,
		               mask_zero=False,
		               input_length=None,
		               **kwargs):
		    kwargs['dtype'] = 'int32'
		    if 'input_shape' not in kwargs:
		      if input_length:
		        kwargs['input_shape'] = (input_length,)
		      else:
		        kwargs['input_shape'] = (None,)
		    super(Embedding, self).__init__(**kwargs)

		    self.input_dim = input_dim
		    self.output_dim = output_dim
		    self.embeddings_initializer = initializers.get(embeddings_initializer)
		    self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)
		    self.embeddings_constraint = constraints.get(embeddings_constraint)
		    self.mask_zero = mask_zero
		    self.input_length = input_length

		  def build(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    self.embeddings = self.add_weight(
		        shape=(self.input_dim, self.output_dim),
		        initializer=self.embeddings_initializer,
		        name='embeddings',
		        regularizer=self.embeddings_regularizer,
		        constraint=self.embeddings_constraint)
		    self.built = True

		  def compute_mask(self, inputs, mask=None):
		    if not self.mask_zero:
		      return None
		    else:
		      return K.not_equal(inputs, 0)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if not self.input_length:
		      input_length = input_shape[1]
		    else:
		      input_length = self.input_length
		    return tensor_shape.TensorShape(
		        [input_shape[0], input_length, self.output_dim])

		  def call(self, inputs):
		    if K.dtype(inputs) != 'int32':
		      inputs = K.cast(inputs, 'int32')
		    out = K.gather(self.embeddings, inputs)
		    return out

		  def get_config(self):
		    config = {
		        'input_dim':
		            self.input_dim,
		        'output_dim':
		            self.output_dim,
		        'embeddings_initializer':
		            initializers.serialize(self.embeddings_initializer),
		        'embeddings_regularizer':
		            regularizers.serialize(self.embeddings_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'embeddings_constraint':
		            constraints.serialize(self.embeddings_constraint),
		        'mask_zero':
		            self.mask_zero,
		        'input_length':
		            self.input_length
		    }
		    base_config = super(Embedding, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class local_py:


		"""Locally-connected layers.
		"""
		def import_libs():

			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import activations
			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.contrib.keras.python.keras.utils import conv_utils
			from tensorflow.python.framework import tensor_shape


		class LocallyConnected1D(Layer):
		  """Locally-connected layer for 1D inputs.

		  The `LocallyConnected1D` layer works similarly to
		  the `Conv1D` layer, except that weights are unshared,
		  that is, a different set of filters is applied at each different patch
		  of the input.

		  Example:
		  ```python
		      # apply a unshared weight convolution 1d of length 3 to a sequence with
		      # 10 timesteps, with 64 output filters
		      model = Sequential()
		      model.add(LocallyConnected1D(64, 3, input_shape=(10, 32)))
		      # now model.output_shape == (None, 8, 64)
		      # add a new conv1d on top
		      model.add(LocallyConnected1D(32, 3))
		      # now model.output_shape == (None, 6, 32)
		  ```

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of a single integer,
		          specifying the length of the 1D convolution window.
		      strides: An integer or tuple/list of a single integer,
		          specifying the stride length of the convolution.
		          Specifying any stride value != 1 is incompatible with specifying
		          any `dilation_rate` value != 1.
		      padding: Currently only supports `"valid"` (case-insensitive).
		          `"same"` may be supported in the future.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to the kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      3D tensor with shape: `(batch_size, steps, input_dim)`

		  Output shape:
		      3D tensor with shape: `(batch_size, new_steps, filters)`
		      `steps` value might have changed due to padding or strides.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=1,
		               padding='valid',
		               data_format=None,
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    super(LocallyConnected1D, self).__init__(**kwargs)
		    self.filters = filters
		    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 1, 'kernel_size')
		    self.strides = conv_utils.normalize_tuple(strides, 1, 'strides')
		    self.padding = conv_utils.normalize_padding(padding)
		    if self.padding != 'valid':
		      raise ValueError('Invalid border mode for LocallyConnected1D '
		                       '(only "valid" is supported): ' + padding)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.activation = activations.get(activation)
		    self.use_bias = use_bias
		    self.kernel_initializer = initializers.get(kernel_initializer)
		    self.bias_initializer = initializers.get(bias_initializer)
		    self.kernel_regularizer = regularizers.get(kernel_regularizer)
		    self.bias_regularizer = regularizers.get(bias_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)
		    self.input_spec = InputSpec(ndim=3)

		  def build(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    input_dim = input_shape[2]
		    if input_dim is None:
		      raise ValueError('Axis 2 of input should be fully-defined. '
		                       'Found shape:', input_shape)
		    output_length = conv_utils.conv_output_length(
		        input_shape[1], self.kernel_size[0], self.padding, self.strides[0])
		    self.kernel_shape = (output_length, self.kernel_size[0] * input_dim,
		                         self.filters)
		    self.kernel = self.add_weight(
		        shape=self.kernel_shape,
		        initializer=self.kernel_initializer,
		        name='kernel',
		        regularizer=self.kernel_regularizer,
		        constraint=self.kernel_constraint)
		    if self.use_bias:
		      self.bias = self.add_weight(
		          shape=(output_length, self.filters),
		          initializer=self.bias_initializer,
		          name='bias',
		          regularizer=self.bias_regularizer,
		          constraint=self.bias_constraint)
		    else:
		      self.bias = None
		    self.input_spec = InputSpec(ndim=3, axes={2: input_dim})
		    self.built = True

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    length = conv_utils.conv_output_length(input_shape[1], self.kernel_size[0],
		                                           self.padding, self.strides[0])
		    return tensor_shape.TensorShape([input_shape[0], length, self.filters])

		  def call(self, inputs):
		    stride = self.strides[0]
		    output_length, feature_dim, filters = self.kernel_shape

		    xs = []
		    for i in range(output_length):
		      slice_length = slice(i * stride, i * stride + self.kernel_size[0])
		      xs.append(K.reshape(inputs[:, slice_length, :], (1, -1, feature_dim)))
		    x_aggregate = K.concatenate(xs, axis=0)
		    # Shape: `(output_length, batch_size, filters)`.
		    output = K.batch_dot(x_aggregate, self.kernel)
		    output = K.permute_dimensions(output, (1, 0, 2))

		    if self.use_bias:
		      output += K.reshape(self.bias, (1, output_length, filters))
		    if self.activation is not None:
		      output = self.activation(output)
		    return output

		  def get_config(self):
		    config = {
		        'filters':
		            self.filters,
		        'kernel_size':
		            self.kernel_size,
		        'strides':
		            self.strides,
		        'padding':
		            self.padding,
		        'activation':
		            activations.serialize(self.activation),
		        'use_bias':
		            self.use_bias,
		        'kernel_initializer':
		            initializers.serialize(self.kernel_initializer),
		        'bias_initializer':
		            initializers.serialize(self.bias_initializer),
		        'kernel_regularizer':
		            regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer':
		            regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint':
		            constraints.serialize(self.kernel_constraint),
		        'bias_constraint':
		            constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(LocallyConnected1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class LocallyConnected2D(Layer):
		  """Locally-connected layer for 2D inputs.

		  The `LocallyConnected2D` layer works similarly
		  to the `Conv2D` layer, except that weights are unshared,
		  that is, a different set of filters is applied at each
		  different patch of the input.

		  Examples:
		  ```python
		      # apply a 3x3 unshared weights convolution with 64 output filters on a
		      32x32 image
		      # with `data_format="channels_last"`:
		      model = Sequential()
		      model.add(LocallyConnected2D(64, (3, 3), input_shape=(32, 32, 3)))
		      # now model.output_shape == (None, 30, 30, 64)
		      # notice that this layer will consume (30*30)*(3*3*3*64) + (30*30)*64
		      parameters

		      # add a 3x3 unshared weights convolution on top, with 32 output filters:
		      model.add(LocallyConnected2D(32, (3, 3)))
		      # now model.output_shape == (None, 28, 28, 32)
		  ```

		  Arguments:
		      filters: Integer, the dimensionality of the output space
		          (i.e. the number output of filters in the convolution).
		      kernel_size: An integer or tuple/list of 2 integers, specifying the
		          width and height of the 2D convolution window.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		      strides: An integer or tuple/list of 2 integers,
		          specifying the strides of the convolution along the width and height.
		          Can be a single integer to specify the same value for
		          all spatial dimensions.
		      padding: Currently only support `"valid"` (case-insensitive).
		          `"same"` will be supported in future.
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix.
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to the kernel matrix.
		      bias_constraint: Constraint function applied to the bias vector.

		  Input shape:
		      4D tensor with shape:
		      `(samples, channels, rows, cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(samples, rows, cols, channels)` if data_format='channels_last'.

		  Output shape:
		      4D tensor with shape:
		      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
		      or 4D tensor with shape:
		      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
		      `rows` and `cols` values might have changed due to padding.
		  """

		  def __init__(self,
		               filters,
		               kernel_size,
		               strides=(1, 1),
		               padding='valid',
		               data_format=None,
		               activation=None,
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               bias_constraint=None,
		               **kwargs):
		    super(LocallyConnected2D, self).__init__(**kwargs)
		    self.filters = filters
		    self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
		    self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
		    self.padding = conv_utils.normalize_padding(padding)
		    if self.padding != 'valid':
		      raise ValueError('Invalid border mode for LocallyConnected2D '
		                       '(only "valid" is supported): ' + padding)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.activation = activations.get(activation)
		    self.use_bias = use_bias
		    self.kernel_initializer = initializers.get(kernel_initializer)
		    self.bias_initializer = initializers.get(bias_initializer)
		    self.kernel_regularizer = regularizers.get(kernel_regularizer)
		    self.bias_regularizer = regularizers.get(bias_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)
		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)
		    self.input_spec = InputSpec(ndim=4)

		  def build(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_last':
		      input_row, input_col = input_shape[1:-1]
		      input_filter = input_shape[3]
		    else:
		      input_row, input_col = input_shape[2:]
		      input_filter = input_shape[1]
		    if input_row is None or input_col is None:
		      raise ValueError('The spatial dimensions of the inputs to '
		                       ' a LocallyConnected2D layer '
		                       'should be fully-defined, but layer received '
		                       'the inputs shape ' + str(input_shape))

		    output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
		                                               self.padding, self.strides[0])
		    output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
		                                               self.padding, self.strides[1])
		    self.output_row = output_row
		    self.output_col = output_col
		    self.kernel_shape = (
		        output_row * output_col,
		        self.kernel_size[0] * self.kernel_size[1] * input_filter, self.filters)
		    self.kernel = self.add_weight(
		        shape=self.kernel_shape,
		        initializer=self.kernel_initializer,
		        name='kernel',
		        regularizer=self.kernel_regularizer,
		        constraint=self.kernel_constraint)
		    if self.use_bias:
		      self.bias = self.add_weight(
		          shape=(output_row, output_col, self.filters),
		          initializer=self.bias_initializer,
		          name='bias',
		          regularizer=self.bias_regularizer,
		          constraint=self.bias_constraint)
		    else:
		      self.bias = None
		    if self.data_format == 'channels_first':
		      self.input_spec = InputSpec(ndim=4, axes={1: input_filter})
		    else:
		      self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
		    self.built = True

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_first':
		      rows = input_shape[2]
		      cols = input_shape[3]
		    elif self.data_format == 'channels_last':
		      rows = input_shape[1]
		      cols = input_shape[2]
		    rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
		                                         self.padding, self.strides[0])
		    cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
		                                         self.padding, self.strides[1])

		    if self.data_format == 'channels_first':
		      return tensor_shape.TensorShape(
		          [input_shape[0], self.filters, rows, cols])
		    elif self.data_format == 'channels_last':
		      return tensor_shape.TensorShape(
		          [input_shape[0], rows, cols, self.filters])

		  def call(self, inputs):
		    stride_row, stride_col = self.strides
		    _, feature_dim, filters = self.kernel_shape

		    if self.data_format == 'channels_first':
		      if K.backend() == 'theano':
		        output = []
		        for i in range(self.output_row):
		          for j in range(self.output_col):
		            slice_row = slice(i * stride_row,
		                              i * stride_row + self.kernel_size[0])
		            slice_col = slice(j * stride_col,
		                              j * stride_col + self.kernel_size[1])
		            x_flatten = K.reshape(inputs[:, :, slice_row, slice_col],
		                                  (1, -1, feature_dim))
		            output.append(
		                K.dot(x_flatten, self.kernel[i * self.output_col + j, :, :]))
		        output = K.concatenate(output, axis=0)
		      else:
		        xs = []
		        for i in range(self.output_row):
		          for j in range(self.output_col):
		            slice_row = slice(i * stride_row,
		                              i * stride_row + self.kernel_size[0])
		            slice_col = slice(j * stride_col,
		                              j * stride_col + self.kernel_size[1])
		            xs.append(
		                K.reshape(inputs[:, :, slice_row, slice_col], (1, -1,
		                                                               feature_dim)))
		        x_aggregate = K.concatenate(xs, axis=0)
		        output = K.batch_dot(x_aggregate, self.kernel)
		      output = K.reshape(output, (self.output_row, self.output_col, -1,
		                                  filters))
		      output = K.permute_dimensions(output, (2, 3, 0, 1))

		    elif self.data_format == 'channels_last':
		      xs = []
		      for i in range(self.output_row):
		        for j in range(self.output_col):
		          slice_row = slice(i * stride_row,
		                            i * stride_row + self.kernel_size[0])
		          slice_col = slice(j * stride_col,
		                            j * stride_col + self.kernel_size[1])
		          xs.append(
		              K.reshape(inputs[:, slice_row, slice_col, :], (1, -1, feature_dim
		                                                            )))
		      x_aggregate = K.concatenate(xs, axis=0)
		      output = K.batch_dot(x_aggregate, self.kernel)
		      output = K.reshape(output, (self.output_row, self.output_col, -1,
		                                  filters))
		      output = K.permute_dimensions(output, (2, 0, 1, 3))

		    if self.use_bias:
		      if self.data_format == 'channels_first':
		        output += K.reshape(self.bias, (1, filters, self.output_row,
		                                        self.output_col))
		      elif self.data_format == 'channels_last':
		        output += K.reshape(self.bias, (1, self.output_row, self.output_col,
		                                        filters))
		    output = self.activation(output)
		    return output

		  def get_config(self):
		    config = {
		        'filters':
		            self.filters,
		        'kernel_size':
		            self.kernel_size,
		        'strides':
		            self.strides,
		        'padding':
		            self.padding,
		        'data_format':
		            self.data_format,
		        'activation':
		            activations.serialize(self.activation),
		        'use_bias':
		            self.use_bias,
		        'kernel_initializer':
		            initializers.serialize(self.kernel_initializer),
		        'bias_initializer':
		            initializers.serialize(self.bias_initializer),
		        'kernel_regularizer':
		            regularizers.serialize(self.kernel_regularizer),
		        'bias_regularizer':
		            regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint':
		            constraints.serialize(self.kernel_constraint),
		        'bias_constraint':
		            constraints.serialize(self.bias_constraint)
		    }
		    base_config = super(LocallyConnected2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class merge_py:

		# pylint: disable=not-callable
		# pylint: disable=redefined-builtin
		"""Layers can merge several input tensors into a single output tensor.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.engine.topology import Layer
			from tensorflow.python.framework import tensor_shape


		class _Merge(Layer):
		  """Generic merge layer for elementwise merge functions.

		  Used to implement `Sum`, `Average`, etc.

		  Arguments:
		      **kwargs: standard layer keyword arguments.
		  """

		  def __init__(self, **kwargs):
		    super(_Merge, self).__init__(**kwargs)
		    self.supports_masking = True

		  def _merge_function(self, inputs):
		    raise NotImplementedError

		  def _compute_elemwise_op_output_shape(self, shape1, shape2):
		    """Computes the shape of the resultant of an elementwise operation.

		    Arguments:
		        shape1: tuple or None. Shape of the first tensor
		        shape2: tuple or None. Shape of the second tensor

		    Returns:
		        expected output shape when an element-wise operation is
		        carried out on 2 tensors with shapes shape1 and shape2.
		        tuple or None.

		    Raises:
		        ValueError: if shape1 and shape2 are not compatible for
		            element-wise operations.
		    """
		    if None in [shape1, shape2]:
		      return None
		    elif len(shape1) < len(shape2):
		      return self._compute_elemwise_op_output_shape(shape2, shape1)
		    elif not shape2:
		      return shape1
		    output_shape = list(shape1[:-len(shape2)])
		    for i, j in zip(shape1[-len(shape2):], shape2):
		      if i is None or j is None:
		        output_shape.append(None)
		      elif i == 1:
		        output_shape.append(j)
		      elif j == 1:
		        output_shape.append(i)
		      else:
		        if i != j:
		          raise ValueError('Operands could not be broadcast '
		                           'together with shapes ' + str(shape1) + ' ' +
		                           str(shape2))
		        output_shape.append(i)
		    return tuple(output_shape)

		  def build(self, input_shape):
		    # Used purely for shape validation.
		    if not isinstance(input_shape, list):
		      raise ValueError('A merge layer should be called ' 'on a list of inputs.')
		    if len(input_shape) < 2:
		      raise ValueError('A merge layer should be called '
		                       'on a list of at least 2 inputs. '
		                       'Got ' + str(len(input_shape)) + ' inputs.')
		    input_shape = [tensor_shape.TensorShape(s).as_list() for s in input_shape]
		    batch_sizes = [s[0] for s in input_shape if s is not None]
		    batch_sizes = set(batch_sizes)
		    batch_sizes -= set([None])
		    if len(batch_sizes) > 1:
		      raise ValueError('Can not merge tensors with different '
		                       'batch sizes. Got tensors with shapes : ' +
		                       str(input_shape))
		    if input_shape[0] is None:
		      output_shape = None
		    else:
		      output_shape = input_shape[0][1:]
		    for i in range(1, len(input_shape)):
		      if input_shape[i] is None:
		        shape = None
		      else:
		        shape = input_shape[i][1:]
		      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
		    # If the inputs have different ranks, we have to reshape them
		    # to make them broadcastable.
		    if None not in input_shape and len(set(map(len, input_shape))) == 1:
		      self._reshape_required = False
		    else:
		      self._reshape_required = True
		    self.built = True

		  def call(self, inputs):
		    if self._reshape_required:
		      reshaped_inputs = []
		      input_ndims = list(map(K.ndim, inputs))
		      if None not in input_ndims:
		        # If ranks of all inputs are available,
		        # we simply expand each of them at axis=1
		        # until all of them have the same rank.
		        max_ndim = max(input_ndims)
		        for x in inputs:
		          x_ndim = K.ndim(x)
		          for _ in range(max_ndim - x_ndim):
		            x = K.expand_dims(x, 1)
		          reshaped_inputs.append(x)
		        return self._merge_function(reshaped_inputs)
		      else:
		        # Transpose all inputs so that batch size is the last dimension.
		        # (batch_size, dim1, dim2, ... ) -> (dim1, dim2, ... , batch_size)
		        transposed = False
		        for x in inputs:
		          x_ndim = K.ndim(x)
		          if x_ndim is None:
		            x_shape = K.shape(x)
		            batch_size = x_shape[0]
		            new_shape = K.concatenate([x_shape[1:], K.expand_dims(batch_size)])
		            x_transposed = K.reshape(x,
		                                     K.stack([batch_size,
		                                              K.prod(x_shape[1:])]))
		            x_transposed = K.permute_dimensions(x_transposed, (1, 0))
		            x_transposed = K.reshape(x_transposed, new_shape)
		            reshaped_inputs.append(x_transposed)
		            transposed = True
		          elif x_ndim > 1:
		            dims = list(range(1, x_ndim)) + [0]
		            reshaped_inputs.append(K.permute_dimensions(x, dims))
		            transposed = True
		          else:
		            # We don't transpose inputs if they are 1D vectors or scalars.
		            reshaped_inputs.append(x)
		        y = self._merge_function(reshaped_inputs)
		        y_ndim = K.ndim(y)
		        if transposed:
		          # If inputs have been transposed, we have to transpose the output too.
		          if y_ndim is None:
		            y_shape = K.shape(y)
		            y_ndim = K.shape(y_shape)[0]
		            batch_size = y_shape[y_ndim - 1]
		            new_shape = K.concatenate(
		                [K.expand_dims(batch_size), y_shape[:y_ndim - 1]])
		            y = K.reshape(y, (-1, batch_size))
		            y = K.permute_dimensions(y, (1, 0))
		            y = K.reshape(y, new_shape)
		          elif y_ndim > 1:
		            dims = [y_ndim - 1] + list(range(y_ndim - 1))
		            y = K.permute_dimensions(y, dims)
		        return y
		    else:
		      return self._merge_function(inputs)

		  def compute_output_shape(self, input_shape):
		    if input_shape[0] is None:
		      output_shape = None
		    else:
		      output_shape = input_shape[0][1:]
		    for i in range(1, len(input_shape)):
		      if input_shape[i] is None:
		        shape = None
		      else:
		        shape = input_shape[i][1:]
		      output_shape = self._compute_elemwise_op_output_shape(output_shape, shape)
		    batch_sizes = [s[0] for s in input_shape if s is not None]
		    batch_sizes = set(batch_sizes)
		    batch_sizes -= set([None])
		    if len(batch_sizes) == 1:
		      output_shape = (list(batch_sizes)[0],) + output_shape
		    else:
		      output_shape = (None,) + output_shape
		    return output_shape

		  def compute_mask(self, inputs, mask=None):
		    if mask is None:
		      return None
		    if not isinstance(mask, list):
		      raise ValueError('`mask` should be a list.')
		    if not isinstance(inputs, list):
		      raise ValueError('`inputs` should be a list.')
		    if len(mask) != len(inputs):
		      raise ValueError('The lists `inputs` and `mask` '
		                       'should have the same length.')
		    if all([m is None for m in mask]):
		      return None
		    masks = [K.expand_dims(m, 0) for m in mask if m is not None]
		    return K.all(K.concatenate(masks, axis=0), axis=0, keepdims=False)


		class Add(_Merge):
		  """Layer that adds a list of inputs.

		  It takes as input a list of tensors,
		  all of the same shape, and returns
		  a single tensor (also of the same shape).
		  """

		  def _merge_function(self, inputs):
		    output = inputs[0]
		    for i in range(1, len(inputs)):
		      output += inputs[i]
		    return output


		class Multiply(_Merge):
		  """Layer that multiplies (element-wise) a list of inputs.

		  It takes as input a list of tensors,
		  all of the same shape, and returns
		  a single tensor (also of the same shape).
		  """

		  def _merge_function(self, inputs):
		    output = inputs[0]
		    for i in range(1, len(inputs)):
		      output *= inputs[i]
		    return output


		class Average(_Merge):
		  """Layer that averages a list of inputs.

		  It takes as input a list of tensors,
		  all of the same shape, and returns
		  a single tensor (also of the same shape).
		  """

		  def _merge_function(self, inputs):
		    output = inputs[0]
		    for i in range(1, len(inputs)):
		      output += inputs[i]
		    return output / len(inputs)


		class Maximum(_Merge):
		  """Layer that computes the maximum (element-wise) a list of inputs.

		  It takes as input a list of tensors,
		  all of the same shape, and returns
		  a single tensor (also of the same shape).
		  """

		  def _merge_function(self, inputs):
		    output = inputs[0]
		    for i in range(1, len(inputs)):
		      output = K.maximum(output, inputs[i])
		    return output


		class Concatenate(_Merge):
		  """Layer that concatenates a list of inputs.

		  It takes as input a list of tensors,
		  all of the same shape expect for the concatenation axis,
		  and returns a single tensor, the concatenation of all inputs.

		  Arguments:
		      axis: Axis along which to concatenate.
		      **kwargs: standard layer keyword arguments.
		  """

		  def __init__(self, axis=-1, **kwargs):
		    super(Concatenate, self).__init__(**kwargs)
		    self.axis = axis
		    self.supports_masking = True

		  def build(self, input_shape):
		    # Used purely for shape validation.
		    if not isinstance(input_shape, list):
		      raise ValueError('`Concatenate` layer should be called '
		                       'on a list of inputs')
		    if all([shape is None for shape in input_shape]):
		      return
		    reduced_inputs_shapes = [
		        tensor_shape.TensorShape(shape).as_list() for shape in input_shape
		    ]
		    shape_set = set()
		    for i in range(len(reduced_inputs_shapes)):
		      del reduced_inputs_shapes[i][self.axis]
		      shape_set.add(tuple(reduced_inputs_shapes[i]))
		    if len(shape_set) > 1:
		      raise ValueError('`Concatenate` layer requires '
		                       'inputs with matching shapes '
		                       'except for the concat axis. '
		                       'Got inputs shapes: %s' % (input_shape))
		    self.built = True

		  def call(self, inputs):
		    if not isinstance(inputs, list):
		      raise ValueError('A `Concatenate` layer should be called '
		                       'on a list of inputs.')
		    return K.concatenate(inputs, axis=self.axis)

		  def _compute_output_shape(self, input_shape):
		    if not isinstance(input_shape, list):
		      raise ValueError('A `Concatenate` layer should be called '
		                       'on a list of inputs.')
		    input_shapes = input_shape
		    output_shape = tensor_shape.TensorShape(input_shapes[0]).as_list()
		    for shape in input_shapes[1:]:
		      shape = tensor_shape.TensorShape(shape).as_list()
		      if output_shape[self.axis] is None or shape[self.axis] is None:
		        output_shape[self.axis] = None
		        break
		      output_shape[self.axis] += shape[self.axis]
		    return tensor_shape.TensorShape(output_shape)

		  def compute_mask(self, inputs, mask=None):
		    if mask is None:
		      return None
		    if not isinstance(mask, list):
		      raise ValueError('`mask` should be a list.')
		    if not isinstance(inputs, list):
		      raise ValueError('`inputs` should be a list.')
		    if len(mask) != len(inputs):
		      raise ValueError('The lists `inputs` and `mask` '
		                       'should have the same length.')
		    if all([m is None for m in mask]):
		      return None
		    # Make a list of masks while making sure
		    # the dimensionality of each mask
		    # is the same as the corresponding input.
		    masks = []
		    for input_i, mask_i in zip(inputs, mask):
		      if mask_i is None:
		        # Input is unmasked. Append all 1s to masks,
		        # but cast it to bool first
		        masks.append(K.cast(K.ones_like(input_i), 'bool'))
		      elif K.ndim(mask_i) < K.ndim(input_i):
		        # Mask is smaller than the input, expand it
		        masks.append(K.expand_dims(mask_i))
		      else:
		        masks.append(mask_i)
		    concatenated = K.concatenate(masks, axis=self.axis)
		    return K.all(concatenated, axis=-1, keepdims=False)

		  def get_config(self):
		    config = {
		        'axis': self.axis,
		    }
		    base_config = super(Concatenate, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class Dot(_Merge):
		  """Layer that computes a dot product between samples in two tensors.

		  E.g. if applied to two tensors `a` and `b` of shape `(batch_size, n)`,
		  the output will be a tensor of shape `(batch_size, 1)`
		  where each entry `i` will be the dot product between
		  `a[i]` and `b[i]`.

		  Arguments:
		      axes: Integer or tuple of integers,
		          axis or axes along which to take the dot product.
		      normalize: Whether to L2-normalize samples along the
		          dot product axis before taking the dot product.
		          If set to True, then the output of the dot product
		          is the cosine proximity between the two samples.
		      **kwargs: Standard layer keyword arguments.
		  """

		  def __init__(self, axes, normalize=False, **kwargs):
		    super(Dot, self).__init__(**kwargs)
		    if not isinstance(axes, int):
		      if not isinstance(axes, (list, tuple)):
		        raise TypeError('Invalid type for `axes` - '
		                        'should be a list or an int.')
		      if len(axes) != 2:
		        raise ValueError('Invalid format for `axes` - '
		                         'should contain two elements.')
		      if not isinstance(axes[0], int) or not isinstance(axes[1], int):
		        raise ValueError('Invalid format for `axes` - '
		                         'list elements should be "int".')
		    self.axes = axes
		    self.normalize = normalize
		    self.supports_masking = True

		  def build(self, input_shape):
		    # Used purely for shape validation.
		    if not isinstance(input_shape, list) or len(input_shape) != 2:
		      raise ValueError('A `Dot` layer should be called '
		                       'on a list of 2 inputs.')
		    shape1 = tensor_shape.TensorShape(input_shape[0]).as_list()
		    shape2 = tensor_shape.TensorShape(input_shape[1]).as_list()
		    if shape1 is None or shape2 is None:
		      return
		    if isinstance(self.axes, int):
		      if self.axes < 0:
		        axes = [self.axes % len(shape1), self.axes % len(shape2)]
		      else:
		        axes = [self.axes] * 2
		    else:
		      axes = self.axes
		    if shape1[axes[0]] != shape2[axes[1]]:
		      raise ValueError('Dimension incompatibility '
		                       '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
		                       'Layer shapes: %s, %s' % (shape1, shape2))
		    self.built = True

		  def call(self, inputs):
		    x1 = inputs[0]
		    x2 = inputs[1]
		    if isinstance(self.axes, int):
		      if self.axes < 0:
		        axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
		      else:
		        axes = [self.axes] * 2
		    else:
		      axes = []
		      for i in range(len(self.axes)):
		        if self.axes[i] < 0:
		          axes.append(self.axes[i] % K.ndim(inputs[i]))
		        else:
		          axes.append(self.axes[i])
		    if self.normalize:
		      x1 = K.l2_normalize(x1, axis=axes[0])
		      x2 = K.l2_normalize(x2, axis=axes[1])
		    output = K.batch_dot(x1, x2, axes)
		    return output

		  def _compute_output_shape(self, input_shape):
		    if not isinstance(input_shape, list) or len(input_shape) != 2:
		      raise ValueError('A `Dot` layer should be called '
		                       'on a list of 2 inputs.')
		    shape1 = tensor_shape.TensorShape(input_shape[0]).as_list()
		    shape2 = tensor_shape.TensorShape(input_shape[1]).as_list()
		    if isinstance(self.axes, int):
		      if self.axes < 0:
		        axes = [self.axes % len(shape1), self.axes % len(shape2)]
		      else:
		        axes = [self.axes] * 2
		    else:
		      axes = self.axes
		    shape1.pop(axes[0])
		    shape2.pop(axes[1])
		    shape2.pop(0)
		    output_shape = shape1 + shape2
		    if len(output_shape) == 1:
		      output_shape += [1]
		    return tensor_shape.TensorShape(output_shape)

		  def compute_mask(self, inputs, mask=None):
		    return None

		  def get_config(self):
		    config = {
		        'axes': self.axes,
		        'normalize': self.normalize,
		    }
		    base_config = super(Dot, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		def add(inputs, **kwargs):
		  """Functional interface to the `Add` layer.

		  Arguments:
		      inputs: A list of input tensors (at least 2).
		      **kwargs: Standard layer keyword arguments.

		  Returns:
		      A tensor, the sum of the inputs.
		  """
		  return Add(**kwargs)(inputs)


		def multiply(inputs, **kwargs):
		  """Functional interface to the `Multiply` layer.

		  Arguments:
		      inputs: A list of input tensors (at least 2).
		      **kwargs: Standard layer keyword arguments.

		  Returns:
		      A tensor, the element-wise product of the inputs.
		  """
		  return Multiply(**kwargs)(inputs)


		def average(inputs, **kwargs):
		  """Functional interface to the `Average` layer.

		  Arguments:
		      inputs: A list of input tensors (at least 2).
		      **kwargs: Standard layer keyword arguments.

		  Returns:
		      A tensor, the average of the inputs.
		  """
		  return Average(**kwargs)(inputs)


		def maximum(inputs, **kwargs):
		  """Functional interface to the `Maximum` layer.

		  Arguments:
		      inputs: A list of input tensors (at least 2).
		      **kwargs: Standard layer keyword arguments.

		  Returns:
		      A tensor, the element-wise maximum of the inputs.
		  """
		  return Maximum(**kwargs)(inputs)


		def concatenate(inputs, axis=-1, **kwargs):
		  """Functional interface to the `Concatenate` layer.

		  Arguments:
		      inputs: A list of input tensors (at least 2).
		      axis: Concatenation axis.
		      **kwargs: Standard layer keyword arguments.

		  Returns:
		      A tensor, the concatenation of the inputs alongside axis `axis`.
		  """
		  return Concatenate(axis=axis, **kwargs)(inputs)


		def dot(inputs, axes, normalize=False, **kwargs):
		  """Functional interface to the `Dot` layer.

		  Arguments:
		      inputs: A list of input tensors (at least 2).
		      axes: Integer or tuple of integers,
		          axis or axes along which to take the dot product.
		      normalize: Whether to L2-normalize samples along the
		          dot product axis before taking the dot product.
		          If set to True, then the output of the dot product
		          is the cosine proximity between the two samples.
		      **kwargs: Standard layer keyword arguments.

		  Returns:
		      A tensor, the dot product of the samples from the inputs.
		  """
		  return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)


	class noise_py:

		"""Layers for regularization models via the addition of noise.
		"""
		def import_libs():

			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.engine import Layer


		class GaussianNoise(Layer):
		  """Apply additive zero-centered Gaussian noise.

		  This is useful to mitigate overfitting
		  (you could see it as a form of random data augmentation).
		  Gaussian Noise (GS) is a natural choice as corruption process
		  for real valued inputs.

		  As it is a regularization layer, it is only active at training time.

		  Arguments:
		      stddev: float, standard deviation of the noise distribution.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as input.
		  """

		  def __init__(self, stddev, **kwargs):
		    super(GaussianNoise, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.stddev = stddev

		  def call(self, inputs, training=None):

		    def noised():
		      return inputs + K.random_normal(
		          shape=K.shape(inputs), mean=0., stddev=self.stddev)

		    return K.in_train_phase(noised, inputs, training=training)

		  def get_config(self):
		    config = {'stddev': self.stddev}
		    base_config = super(GaussianNoise, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class GaussianDropout(Layer):
		  """Apply multiplicative 1-centered Gaussian noise.

		  As it is a regularization layer, it is only active at training time.

		  Arguments:
		      rate: float, drop probability (as with `Dropout`).
		          The multiplicative noise will have
		          standard deviation `sqrt(rate / (1 - rate))`.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as input.

		  References:
		      - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting
		        Srivastava, Hinton, et al.
		        2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
		  """

		  def __init__(self, rate, **kwargs):
		    super(GaussianDropout, self).__init__(**kwargs)
		    self.supports_masking = True
		    self.rate = rate

		  def call(self, inputs, training=None):
		    if 0 < self.rate < 1:

		      def noised():
		        stddev = np.sqrt(self.rate / (1.0 - self.rate))
		        return inputs * K.random_normal(
		            shape=K.shape(inputs), mean=1.0, stddev=stddev)

		      return K.in_train_phase(noised, inputs, training=training)
		    return inputs

		  def get_config(self):
		    config = {'rate': self.rate}
		    base_config = super(GaussianDropout, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class normalization_py:

		"""Normalization layers.
		"""
		def import_libs():

			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.python.layers import normalization as tf_normalization_layers


		class BatchNormalization(tf_normalization_layers.BatchNormalization, Layer):
		  """Batch normalization layer (Ioffe and Szegedy, 2014).

		  Normalize the activations of the previous layer at each batch,
		  i.e. applies a transformation that maintains the mean activation
		  close to 0 and the activation standard deviation close to 1.

		  Arguments:
		      axis: Integer, the axis that should be normalized
		          (typically the features axis).
		          For instance, after a `Conv2D` layer with
		          `data_format="channels_first"`,
		          set `axis=1` in `BatchNormalization`.
		      momentum: Momentum for the moving average.
		      epsilon: Small float added to variance to avoid dividing by zero.
		      center: If True, add offset of `beta` to normalized tensor.
		          If False, `beta` is ignored.
		      scale: If True, multiply by `gamma`.
		          If False, `gamma` is not used.
		          When the next layer is linear (also e.g. `nn.relu`),
		          this can be disabled since the scaling
		          will be done by the next layer.
		      beta_initializer: Initializer for the beta weight.
		      gamma_initializer: Initializer for the gamma weight.
		      moving_mean_initializer: Initializer for the moving mean.
		      moving_variance_initializer: Initializer for the moving variance.
		      beta_regularizer: Optional regularizer for the beta weight.
		      gamma_regularizer: Optional regularizer for the gamma weight.
		      beta_constraint: Optional constraint for the beta weight.
		      gamma_constraint: Optional constraint for the gamma weight.

		  Input shape:
		      Arbitrary. Use the keyword argument `input_shape`
		      (tuple of integers, does not include the samples axis)
		      when using this layer as the first layer in a model.

		  Output shape:
		      Same shape as input.

		  References:
		      - [Batch Normalization: Accelerating Deep Network Training by Reducing
		        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
		  """

		  def __init__(self,
		               axis=-1,
		               momentum=0.99,
		               epsilon=1e-3,
		               center=True,
		               scale=True,
		               beta_initializer='zeros',
		               gamma_initializer='ones',
		               moving_mean_initializer='zeros',
		               moving_variance_initializer='ones',
		               beta_regularizer=None,
		               gamma_regularizer=None,
		               beta_constraint=None,
		               gamma_constraint=None,
		               **kwargs):
		    self.supports_masking = True
		    super(BatchNormalization, self).__init__(
		        axis=axis,
		        momentum=momentum,
		        epsilon=epsilon,
		        center=center,
		        scale=scale,
		        beta_initializer=initializers.get(beta_initializer),
		        gamma_initializer=initializers.get(gamma_initializer),
		        moving_mean_initializer=initializers.get(moving_mean_initializer),
		        moving_variance_initializer=initializers.get(
		            moving_variance_initializer),
		        beta_regularizer=regularizers.get(beta_regularizer),
		        gamma_regularizer=regularizers.get(gamma_regularizer),
		        **kwargs
		    )
		    # TODO(fchollet): move weight constraint support to core layers.
		    self.beta_constraint = constraints.get(beta_constraint)
		    self.gamma_constraint = constraints.get(gamma_constraint)

		  def build(self, input_shape):
		    super(BatchNormalization, self).build(input_shape)
		    # TODO(fchollet): move weight constraint support to core layers.
		    if self.center and self.beta_constraint:
		      self.constraints[self.beta] = self.beta_constraint
		    if self.scale and self.gamma_constraint:
		      self.constraints[self.gamma] = self.gamma_constraint

		  def call(self, inputs, training=None):
		    if training is None:
		      training = K.learning_phase()
		    output = super(BatchNormalization, self).call(inputs, training=training)
		    if training is K.learning_phase():
		      output._uses_learning_phase = True  # pylint: disable=protected-access
		    return output

		  def get_config(self):
		    config = {
		        'axis': self.axis,
		        'momentum': self.momentum,
		        'epsilon': self.epsilon,
		        'center': self.center,
		        'scale': self.scale,
		        'beta_initializer': initializers.serialize(self.beta_initializer),
		        'gamma_initializer': initializers.serialize(self.gamma_initializer),
		        'moving_mean_initializer':
		            initializers.serialize(self.moving_mean_initializer),
		        'moving_variance_initializer':
		            initializers.serialize(self.moving_variance_initializer),
		        'beta_regularizer': regularizers.serialize(self.beta_regularizer),
		        'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
		        'beta_constraint': constraints.serialize(self.beta_constraint),
		        'gamma_constraint': constraints.serialize(self.gamma_constraint)
		    }
		    base_config = super(BatchNormalization, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

	class pooling_py:

		"""Pooling layers.
		"""

		def import_libs():

			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.contrib.keras.python.keras.utils import conv_utils
			from tensorflow.python.framework import tensor_shape
			from tensorflow.python.layers import pooling as tf_pooling_layers


		class MaxPooling1D(tf_pooling_layers.MaxPooling1D, Layer):
		  """Max pooling operation for temporal data.

		  Arguments:
		      pool_size: Integer, size of the max pooling windows.
		      strides: Integer, or None. Factor by which to downscale.
		          E.g. 2 will halve the input.
		          If None, it will default to `pool_size`.
		      padding: One of `"valid"` or `"same"` (case-insensitive).

		  Input shape:
		      3D tensor with shape: `(batch_size, steps, features)`.

		  Output shape:
		      3D tensor with shape: `(batch_size, downsampled_steps, features)`.
		  """

		  def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
		    if strides is None:
		      strides = pool_size
		    super(MaxPooling1D, self).__init__(pool_size, strides, padding, **kwargs)

		  def get_config(self):
		    config = {
		        'strides': self.strides,
		        'pool_size': self.pool_size,
		        'padding': self.padding
		    }
		    base_config = super(MaxPooling1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class AveragePooling1D(tf_pooling_layers.AveragePooling1D, Layer):
		  """Average pooling for temporal data.

		  Arguments:
		      pool_size: Integer, size of the max pooling windows.
		      strides: Integer, or None. Factor by which to downscale.
		          E.g. 2 will halve the input.
		          If None, it will default to `pool_size`.
		      padding: One of `"valid"` or `"same"` (case-insensitive).

		  Input shape:
		      3D tensor with shape: `(batch_size, steps, features)`.

		  Output shape:
		      3D tensor with shape: `(batch_size, downsampled_steps, features)`.
		  """

		  def __init__(self, pool_size=2, strides=None, padding='valid', **kwargs):
		    if strides is None:
		      strides = pool_size
		    super(AveragePooling1D, self).__init__(pool_size, strides, padding,
		                                           **kwargs)

		  def get_config(self):
		    config = {
		        'strides': self.strides,
		        'pool_size': self.pool_size,
		        'padding': self.padding
		    }
		    base_config = super(AveragePooling1D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class MaxPooling2D(tf_pooling_layers.MaxPooling2D, Layer):
		  """Max pooling operation for spatial data.

		  Arguments:
		      pool_size: integer or tuple of 2 integers,
		          factors by which to downscale (vertical, horizontal).
		          (2, 2) will halve the input in both spatial dimension.
		          If only one integer is specified, the same window length
		          will be used for both dimensions.
		      strides: Integer, tuple of 2 integers, or None.
		          Strides values.
		          If None, it will default to `pool_size`.
		      padding: One of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          4D tensor with shape:
		          `(batch_size, rows, cols, channels)`
		      - If `data_format='channels_first'`:
		          4D tensor with shape:
		          `(batch_size, channels, rows, cols)`

		  Output shape:
		      - If `data_format='channels_last'`:
		          4D tensor with shape:
		          `(batch_size, pooled_rows, pooled_cols, channels)`
		      - If `data_format='channels_first'`:
		          4D tensor with shape:
		          `(batch_size, channels, pooled_rows, pooled_cols)`
		  """

		  def __init__(self,
		               pool_size=(2, 2),
		               strides=None,
		               padding='valid',
		               data_format=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    if strides is None:
		      strides = pool_size
		    super(MaxPooling2D, self).__init__(pool_size, strides, padding, data_format,
		                                       **kwargs)

		  def get_config(self):
		    config = {
		        'pool_size': self.pool_size,
		        'padding': self.padding,
		        'strides': self.strides,
		        'data_format': self.data_format
		    }
		    base_config = super(MaxPooling2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class AveragePooling2D(tf_pooling_layers.AveragePooling2D, Layer):
		  """Average pooling operation for spatial data.

		  Arguments:
		      pool_size: integer or tuple of 2 integers,
		          factors by which to downscale (vertical, horizontal).
		          (2, 2) will halve the input in both spatial dimension.
		          If only one integer is specified, the same window length
		          will be used for both dimensions.
		      strides: Integer, tuple of 2 integers, or None.
		          Strides values.
		          If None, it will default to `pool_size`.
		      padding: One of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          4D tensor with shape:
		          `(batch_size, rows, cols, channels)`
		      - If `data_format='channels_first'`:
		          4D tensor with shape:
		          `(batch_size, channels, rows, cols)`

		  Output shape:
		      - If `data_format='channels_last'`:
		          4D tensor with shape:
		          `(batch_size, pooled_rows, pooled_cols, channels)`
		      - If `data_format='channels_first'`:
		          4D tensor with shape:
		          `(batch_size, channels, pooled_rows, pooled_cols)`
		  """

		  def __init__(self,
		               pool_size=(2, 2),
		               strides=None,
		               padding='valid',
		               data_format=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    if strides is None:
		      strides = pool_size
		    super(AveragePooling2D, self).__init__(pool_size, strides, padding,
		                                           data_format, **kwargs)

		  def get_config(self):
		    config = {
		        'pool_size': self.pool_size,
		        'padding': self.padding,
		        'strides': self.strides,
		        'data_format': self.data_format
		    }
		    base_config = super(AveragePooling2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class MaxPooling3D(tf_pooling_layers.MaxPooling3D, Layer):
		  """Max pooling operation for 3D data (spatial or spatio-temporal).

		  Arguments:
		      pool_size: tuple of 3 integers,
		          factors by which to downscale (dim1, dim2, dim3).
		          (2, 2, 2) will halve the size of the 3D input in each dimension.
		      strides: tuple of 3 integers, or None. Strides values.
		      padding: One of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          5D tensor with shape:
		          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		      - If `data_format='channels_first'`:
		          5D tensor with shape:
		          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

		  Output shape:
		      - If `data_format='channels_last'`:
		          5D tensor with shape:
		          `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
		      - If `data_format='channels_first'`:
		          5D tensor with shape:
		          `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
		  """

		  def __init__(self,
		               pool_size=(2, 2, 2),
		               strides=None,
		               padding='valid',
		               data_format=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    if strides is None:
		      strides = pool_size
		    super(MaxPooling3D, self).__init__(pool_size, strides, padding, data_format,
		                                       **kwargs)

		  def get_config(self):
		    config = {
		        'pool_size': self.pool_size,
		        'padding': self.padding,
		        'strides': self.strides,
		        'data_format': self.data_format
		    }
		    base_config = super(MaxPooling3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class AveragePooling3D(tf_pooling_layers.AveragePooling3D, Layer):
		  """Average pooling operation for 3D data (spatial or spatio-temporal).

		  Arguments:
		      pool_size: tuple of 3 integers,
		          factors by which to downscale (dim1, dim2, dim3).
		          (2, 2, 2) will halve the size of the 3D input in each dimension.
		      strides: tuple of 3 integers, or None. Strides values.
		      padding: One of `"valid"` or `"same"` (case-insensitive).
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          5D tensor with shape:
		          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		      - If `data_format='channels_first'`:
		          5D tensor with shape:
		          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

		  Output shape:
		      - If `data_format='channels_last'`:
		          5D tensor with shape:
		          `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
		      - If `data_format='channels_first'`:
		          5D tensor with shape:
		          `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`
		  """

		  def __init__(self,
		               pool_size=(2, 2, 2),
		               strides=None,
		               padding='valid',
		               data_format=None,
		               **kwargs):
		    if data_format is None:
		      data_format = K.image_data_format()
		    if strides is None:
		      strides = pool_size
		    super(AveragePooling3D, self).__init__(pool_size, strides, padding,
		                                           data_format, **kwargs)

		  def get_config(self):
		    config = {
		        'pool_size': self.pool_size,
		        'padding': self.padding,
		        'strides': self.strides,
		        'data_format': self.data_format
		    }
		    base_config = super(AveragePooling3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class _GlobalPooling1D(Layer):
		  """Abstract class for different global pooling 1D layers.
		  """

		  def __init__(self, **kwargs):
		    super(_GlobalPooling1D, self).__init__(**kwargs)
		    self.input_spec = InputSpec(ndim=3)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    return tensor_shape.TensorShape([input_shape[0], input_shape[2]])

		  def call(self, inputs):
		    raise NotImplementedError


		class GlobalAveragePooling1D(_GlobalPooling1D):
		  """Global average pooling operation for temporal data.

		  Input shape:
		      3D tensor with shape: `(batch_size, steps, features)`.

		  Output shape:
		      2D tensor with shape:
		      `(batch_size, channels)`
		  """

		  def call(self, inputs):
		    return K.mean(inputs, axis=1)


		class GlobalMaxPooling1D(_GlobalPooling1D):
		  """Global max pooling operation for temporal data.

		  Input shape:
		      3D tensor with shape: `(batch_size, steps, features)`.

		  Output shape:
		      2D tensor with shape:
		      `(batch_size, channels)`
		  """

		  def call(self, inputs):
		    return K.max(inputs, axis=1)


		class _GlobalPooling2D(Layer):
		  """Abstract class for different global pooling 2D layers.
		  """

		  def __init__(self, data_format=None, **kwargs):
		    super(_GlobalPooling2D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.input_spec = InputSpec(ndim=4)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_last':
		      return tensor_shape.TensorShape([input_shape[0], input_shape[3]])
		    else:
		      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

		  def call(self, inputs):
		    raise NotImplementedError

		  def get_config(self):
		    config = {'data_format': self.data_format}
		    base_config = super(_GlobalPooling2D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class GlobalAveragePooling2D(_GlobalPooling2D):
		  """Global average pooling operation for spatial data.

		  Arguments:
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          4D tensor with shape:
		          `(batch_size, rows, cols, channels)`
		      - If `data_format='channels_first'`:
		          4D tensor with shape:
		          `(batch_size, channels, rows, cols)`

		  Output shape:
		      2D tensor with shape:
		      `(batch_size, channels)`
		  """

		  def call(self, inputs):
		    if self.data_format == 'channels_last':
		      return K.mean(inputs, axis=[1, 2])
		    else:
		      return K.mean(inputs, axis=[2, 3])


		class GlobalMaxPooling2D(_GlobalPooling2D):
		  """Global max pooling operation for spatial data.

		  Arguments:
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, height, width, channels)` while `channels_first`
		          corresponds to inputs with shape
		          `(batch, channels, height, width)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          4D tensor with shape:
		          `(batch_size, rows, cols, channels)`
		      - If `data_format='channels_first'`:
		          4D tensor with shape:
		          `(batch_size, channels, rows, cols)`

		  Output shape:
		      2D tensor with shape:
		      `(batch_size, channels)`
		  """

		  def call(self, inputs):
		    if self.data_format == 'channels_last':
		      return K.max(inputs, axis=[1, 2])
		    else:
		      return K.max(inputs, axis=[2, 3])


		class _GlobalPooling3D(Layer):
		  """Abstract class for different global pooling 3D layers.
		  """

		  def __init__(self, data_format=None, **kwargs):
		    super(_GlobalPooling3D, self).__init__(**kwargs)
		    self.data_format = conv_utils.normalize_data_format(data_format)
		    self.input_spec = InputSpec(ndim=5)

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.data_format == 'channels_last':
		      return tensor_shape.TensorShape([input_shape[0], input_shape[4]])
		    else:
		      return tensor_shape.TensorShape([input_shape[0], input_shape[1]])

		  def call(self, inputs):
		    raise NotImplementedError

		  def get_config(self):
		    config = {'data_format': self.data_format}
		    base_config = super(_GlobalPooling3D, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class GlobalAveragePooling3D(_GlobalPooling3D):
		  """Global Average pooling operation for 3D data.

		  Arguments:
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          5D tensor with shape:
		          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		      - If `data_format='channels_first'`:
		          5D tensor with shape:
		          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

		  Output shape:
		      2D tensor with shape:
		      `(batch_size, channels)`
		  """

		  def call(self, inputs):
		    if self.data_format == 'channels_last':
		      return K.mean(inputs, axis=[1, 2, 3])
		    else:
		      return K.mean(inputs, axis=[2, 3, 4])


		class GlobalMaxPooling3D(_GlobalPooling3D):
		  """Global Max pooling operation for 3D data.

		  Arguments:
		      data_format: A string,
		          one of `channels_last` (default) or `channels_first`.
		          The ordering of the dimensions in the inputs.
		          `channels_last` corresponds to inputs with shape
		          `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		          while `channels_first` corresponds to inputs with shape
		          `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
		          It defaults to the `image_data_format` value found in your
		          Keras config file at `~/.keras/keras.json`.
		          If you never set it, then it will be "channels_last".

		  Input shape:
		      - If `data_format='channels_last'`:
		          5D tensor with shape:
		          `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
		      - If `data_format='channels_first'`:
		          5D tensor with shape:
		          `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

		  Output shape:
		      2D tensor with shape:
		      `(batch_size, channels)`
		  """

		  def call(self, inputs):
		    if self.data_format == 'channels_last':
		      return K.max(inputs, axis=[1, 2, 3])
		    else:
		      return K.max(inputs, axis=[2, 3, 4])


		# Aliases

		AvgPool1D = AveragePooling1D
		MaxPool1D = MaxPooling1D
		AvgPool2D = AveragePooling2D
		MaxPool2D = MaxPooling2D
		AvgPool3D = AveragePooling3D
		MaxPool3D = MaxPooling3D
		GlobalMaxPool1D = GlobalMaxPooling1D
		GlobalMaxPool2D = GlobalMaxPooling2D
		GlobalMaxPool3D = GlobalMaxPooling3D
		GlobalAvgPool1D = GlobalAveragePooling1D
		GlobalAvgPool2D = GlobalAveragePooling2D
		GlobalAvgPool3D = GlobalAveragePooling3D

	class recurrent_py:

		# pylint: disable=protected-access
		"""Recurrent layers.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np

			from tensorflow.contrib.keras.python.keras import activations
			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import constraints
			from tensorflow.contrib.keras.python.keras import initializers
			from tensorflow.contrib.keras.python.keras import regularizers
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.python.framework import tensor_shape


			# pylint: disable=access-member-before-definition


		def _time_distributed_dense(x,
		                            w,
		                            b=None,
		                            dropout=None,
		                            input_dim=None,
		                            output_dim=None,
		                            timesteps=None,
		                            training=None):
		  """Apply `y . w + b` for every temporal slice y of x.

		  Arguments:
		      x: input tensor.
		      w: weight matrix.
		      b: optional bias vector.
		      dropout: wether to apply dropout (same dropout mask
		          for every temporal slice of the input).
		      input_dim: integer; optional dimensionality of the input.
		      output_dim: integer; optional dimensionality of the output.
		      timesteps: integer; optional number of timesteps.
		      training: training phase tensor or boolean.

		  Returns:
		      Output tensor.
		  """
		  if not input_dim:
		    input_dim = K.shape(x)[2]
		  if not timesteps:
		    timesteps = K.shape(x)[1]
		  if not output_dim:
		    output_dim = K.shape(w)[1]

		  if dropout is not None and 0. < dropout < 1.:
		    # apply the same dropout pattern at every timestep
		    ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
		    dropout_matrix = K.dropout(ones, dropout)
		    expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
		    x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

		  # collapse time dimension and batch dimension together
		  x = K.reshape(x, (-1, input_dim))
		  x = K.dot(x, w)
		  if b is not None:
		    x = K.bias_add(x, b)
		  # reshape to 3D tensor
		  if K.backend() == 'tensorflow':
		    x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
		    x.set_shape([None, None, output_dim])
		  else:
		    x = K.reshape(x, (-1, timesteps, output_dim))
		  return x


		class Recurrent(Layer):
		  """Abstract base class for recurrent layers.

		  Do not use in a model -- it's not a valid layer!
		  Use its children classes `LSTM`, `GRU` and `SimpleRNN` instead.

		  All recurrent layers (`LSTM`, `GRU`, `SimpleRNN`) also
		  follow the specifications of this class and accept
		  the keyword arguments listed below.

		  Example:

		  ```python
		      # as the first layer in a Sequential model
		      model = Sequential()
		      model.add(LSTM(32, input_shape=(10, 64)))
		      # now model.output_shape == (None, 32)
		      # note: `None` is the batch dimension.

		      # for subsequent layers, no need to specify the input size:
		      model.add(LSTM(16))

		      # to stack recurrent layers, you must use return_sequences=True
		      # on any recurrent layer that feeds into another recurrent layer.
		      # note that you only need to specify the input size on the first layer.
		      model = Sequential()
		      model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))
		      model.add(LSTM(32, return_sequences=True))
		      model.add(LSTM(10))
		  ```

		  Arguments:
		      weights: list of Numpy arrays to set as initial weights.
		          The list should have 3 elements, of shapes:
		          `[(input_dim, output_dim), (output_dim, output_dim), (output_dim,)]`.
		      return_sequences: Boolean. Whether to return the last output
		          in the output sequence, or the full sequence.
		      go_backwards: Boolean (default False).
		          If True, process the input sequence backwards and return the
		          reversed sequence.
		      stateful: Boolean (default False). If True, the last state
		          for each sample at index i in a batch will be used as initial
		          state for the sample of index i in the following batch.
		      unroll: Boolean (default False).
		          If True, the network will be unrolled,
		          else a symbolic loop will be used.
		          Unrolling can speed-up a RNN,
		          although it tends to be more memory-intensive.
		          Unrolling is only suitable for short sequences.
		      implementation: one of {0, 1, or 2}.
		          If set to 0, the RNN will use
		          an implementation that uses fewer, larger matrix products,
		          thus running faster on CPU but consuming more memory.
		          If set to 1, the RNN will use more matrix products,
		          but smaller ones, thus running slower
		          (may actually be faster on GPU) while consuming less memory.
		          If set to 2 (LSTM/GRU only),
		          the RNN will combine the input gate,
		          the forget gate and the output gate into a single matrix,
		          enabling more time-efficient parallelization on the GPU.
		          Note: RNN dropout must be shared for all gates,
		          resulting in a slightly reduced regularization.
		      input_dim: dimensionality of the input (integer).
		          This argument (or alternatively, the keyword argument `input_shape`)
		          is required when using this layer as the first layer in a model.
		      input_length: Length of input sequences, to be specified
		          when it is constant.
		          This argument is required if you are going to connect
		          `Flatten` then `Dense` layers upstream
		          (without it, the shape of the dense outputs cannot be computed).
		          Note that if the recurrent layer is not the first layer
		          in your model, you would need to specify the input length
		          at the level of the first layer
		          (e.g. via the `input_shape` argument)

		  Input shape:s
		      3D tensor with shape `(batch_size, timesteps, input_dim)`,
		      (Optional) 2D tensors with shape `(batch_size, output_dim)`.

		  Output shape:
		      - if `return_sequences`: 3D tensor with shape
		          `(batch_size, timesteps, units)`.
		      - else, 2D tensor with shape `(batch_size, units)`.

		  # Masking
		      This layer supports masking for input data with a variable number
		      of timesteps. To introduce masks to your data,
		      use an `Embedding` layer with the `mask_zero` parameter
		      set to `True`.

		  # Note on using statefulness in RNNs
		      You can set RNN layers to be 'stateful', which means that the states
		      computed for the samples in one batch will be reused as initial states
		      for the samples in the next batch. This assumes a one-to-one mapping
		      between samples in different successive batches.

		      To enable statefulness:
		          - specify `stateful=True` in the layer constructor.
		          - specify a fixed batch size for your model, by passing
		              if sequential model:
		                `batch_input_shape=(...)` to the first layer in your model.
		              else for functional model with 1 or more Input layers:
		                `batch_shape=(...)` to all the first layers in your model.
		              This is the expected shape of your inputs
		              *including the batch size*.
		              It should be a tuple of integers, e.g. `(32, 10, 100)`.
		          - specify `shuffle=False` when calling fit().

		      To reset the states of your model, call `.reset_states()` on either
		      a specific layer, or on your entire model.

		  # Note on specifying the initial state of RNNs
		      You can specify the initial state of RNN layers symbolically by
		      calling them with the keyword argument `initial_state`. The value of
		      `initial_state` should be a tensor or list of tensors representing
		      the initial state of the RNN layer.

		      You can specify the initial state of RNN layers numerically by
		      calling `reset_states` with the keyword argument `states`. The value of
		      `states` should be a numpy array or list of numpy arrays representing
		      the initial state of the RNN layer.
		  """

		  def __init__(self,
		               return_sequences=False,
		               go_backwards=False,
		               stateful=False,
		               unroll=False,
		               implementation=0,
		               **kwargs):
		    super(Recurrent, self).__init__(**kwargs)
		    self.return_sequences = return_sequences
		    self.go_backwards = go_backwards
		    self.stateful = stateful
		    self.unroll = unroll
		    self.implementation = implementation
		    self.supports_masking = True
		    self.input_spec = [InputSpec(ndim=3)]
		    self.state_spec = None
		    self.dropout = 0
		    self.recurrent_dropout = 0

		  def _compute_output_shape(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shape = input_shape[0]
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.return_sequences:
		      return tensor_shape.TensorShape(
		          [input_shape[0], input_shape[1], self.units])
		    else:
		      return tensor_shape.TensorShape([input_shape[0], self.units])

		  def compute_mask(self, inputs, mask):
		    if self.return_sequences:
		      if isinstance(mask, list):
		        return mask[0]
		      return mask
		    else:
		      return None

		  def step(self, inputs, states):
		    raise NotImplementedError

		  def get_constants(self, inputs, training=None):
		    return []

		  def get_initial_state(self, inputs):
		    # build an all-zero tensor of shape (samples, output_dim)
		    initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
		    initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
		    initial_state = K.expand_dims(initial_state)  # (samples, 1)
		    initial_state = K.tile(initial_state, [1,
		                                           self.units])  # (samples, output_dim)
		    initial_state = [initial_state for _ in range(len(self.states))]
		    return initial_state

		  def preprocess_input(self, inputs, training=None):
		    return inputs

		  def __call__(self, inputs, initial_state=None, **kwargs):
		    # If `initial_state` is specified,
		    # and if it a Keras tensor,
		    # then add it to the inputs and temporarily
		    # modify the input spec to include the state.
		    if initial_state is None:
		      return super(Recurrent, self).__call__(inputs, **kwargs)

		    if not isinstance(initial_state, (list, tuple)):
		      initial_state = [initial_state]

		    is_keras_tensor = hasattr(initial_state[0], '_keras_history')
		    for tensor in initial_state:
		      if hasattr(tensor, '_keras_history') != is_keras_tensor:
		        raise ValueError('The initial state of an RNN layer cannot be'
		                         ' specified with a mix of Keras tensors and'
		                         ' non-Keras tensors')

		    if is_keras_tensor:
		      # Compute the full input spec, including state
		      input_spec = self.input_spec
		      state_spec = self.state_spec
		      if not isinstance(input_spec, list):
		        input_spec = [input_spec]
		      if not isinstance(state_spec, list):
		        state_spec = [state_spec]
		      self.input_spec = input_spec + state_spec

		      # Compute the full inputs, including state
		      inputs = [inputs] + list(initial_state)

		      # Perform the call
		      output = super(Recurrent, self).__call__(inputs, **kwargs)

		      # Restore original input spec
		      self.input_spec = input_spec
		      return output
		    else:
		      kwargs['initial_state'] = initial_state
		      return super(Recurrent, self).__call__(inputs, **kwargs)

		  def call(self, inputs, mask=None, training=None, initial_state=None):
		    # input shape: `(samples, time (padded with zeros), input_dim)`
		    # note that the .build() method of subclasses MUST define
		    # self.input_spec and self.state_spec with complete input shapes.
		    if isinstance(inputs, list):
		      initial_state = inputs[1:]
		      inputs = inputs[0]
		    elif initial_state is not None:
		      pass
		    elif self.stateful:
		      initial_state = self.states
		    else:
		      initial_state = self.get_initial_state(inputs)

		    if isinstance(mask, list):
		      mask = mask[0]

		    if len(initial_state) != len(self.states):
		      raise ValueError('Layer has ' + str(len(self.states)) +
		                       ' states but was passed ' + str(len(initial_state)) +
		                       ' initial states.')
		    input_shape = K.int_shape(inputs)
		    if self.unroll and input_shape[1] is None:
		      raise ValueError('Cannot unroll a RNN if the '
		                       'time dimension is undefined. \n'
		                       '- If using a Sequential model, '
		                       'specify the time dimension by passing '
		                       'an `input_shape` or `batch_input_shape` '
		                       'argument to your first layer. If your '
		                       'first layer is an Embedding, you can '
		                       'also use the `input_length` argument.\n'
		                       '- If using the functional API, specify '
		                       'the time dimension by passing a `shape` '
		                       'or `batch_shape` argument to your Input layer.')
		    constants = self.get_constants(inputs, training=None)
		    preprocessed_input = self.preprocess_input(inputs, training=None)
		    last_output, outputs, states = K.rnn(
		        self.step,
		        preprocessed_input,
		        initial_state,
		        go_backwards=self.go_backwards,
		        mask=mask,
		        constants=constants,
		        unroll=self.unroll)
		    if self.stateful:
		      updates = []
		      for i in range(len(states)):
		        updates.append((self.states[i], states[i]))
		      self.add_update(updates, inputs)

		    # Properly set learning phase
		    if 0 < self.dropout + self.recurrent_dropout:
		      last_output._uses_learning_phase = True
		      outputs._uses_learning_phase = True

		    if self.return_sequences:
		      return outputs
		    else:
		      return last_output

		  def reset_states(self, states=None):
		    if not self.stateful:
		      raise AttributeError('Layer must be stateful.')
		    batch_size = self.input_spec[0].shape[0]
		    if not batch_size:
		      raise ValueError('If a RNN is stateful, it needs to know '
		                       'its batch size. Specify the batch size '
		                       'of your input tensors: \n'
		                       '- If using a Sequential model, '
		                       'specify the batch size by passing '
		                       'a `batch_input_shape` '
		                       'argument to your first layer.\n'
		                       '- If using the functional API, specify '
		                       'the time dimension by passing a '
		                       '`batch_shape` argument to your Input layer.')
		    # initialize state if None
		    if self.states[0] is None:
		      self.states = [K.zeros((batch_size, self.units)) for _ in self.states]
		    elif states is None:
		      for state in self.states:
		        K.set_value(state, np.zeros((batch_size, self.units)))
		    else:
		      if not isinstance(states, (list, tuple)):
		        states = [states]
		      if len(states) != len(self.states):
		        raise ValueError('Layer ' + self.name + ' expects ' +
		                         str(len(self.states)) + ' states, '
		                         'but it received ' + str(len(states)) +
		                         ' state values. Input received: ' + str(states))
		      for index, (value, state) in enumerate(zip(states, self.states)):
		        if value.shape != (batch_size, self.units):
		          raise ValueError('State ' + str(index) +
		                           ' is incompatible with layer ' + self.name +
		                           ': expected shape=' + str((batch_size, self.units)) +
		                           ', found shape=' + str(value.shape))
		        K.set_value(state, value)

		  def get_config(self):
		    config = {
		        'return_sequences': self.return_sequences,
		        'go_backwards': self.go_backwards,
		        'stateful': self.stateful,
		        'unroll': self.unroll,
		        'implementation': self.implementation
		    }
		    base_config = super(Recurrent, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class SimpleRNN(Recurrent):
		  """Fully-connected RNN where the output is to be fed back to input.

		  Arguments:
		      units: Positive integer, dimensionality of the output space.
		      activation: Activation function to use.
		          If you don't specify anything, no activation is applied
		          If you pass None, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix,
		          used for the linear transformation of the inputs..
		      recurrent_initializer: Initializer for the `recurrent_kernel`
		          weights matrix,
		          used for the linear transformation of the recurrent state..
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      recurrent_regularizer: Regularizer function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to
		          the `kernel` weights matrix.
		      recurrent_constraint: Constraint function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_constraint: Constraint function applied to the bias vector.
		      dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the inputs.
		      recurrent_dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the recurrent state.

		  References:
		      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
		        Networks](http://arxiv.org/abs/1512.05287)
		  """

		  def __init__(self,
		               units,
		               activation='tanh',
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               recurrent_initializer='orthogonal',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               recurrent_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               recurrent_constraint=None,
		               bias_constraint=None,
		               dropout=0.,
		               recurrent_dropout=0.,
		               **kwargs):
		    super(SimpleRNN, self).__init__(**kwargs)
		    self.units = units
		    self.activation = activations.get(activation)
		    self.use_bias = use_bias

		    self.kernel_initializer = initializers.get(kernel_initializer)
		    self.recurrent_initializer = initializers.get(recurrent_initializer)
		    self.bias_initializer = initializers.get(bias_initializer)

		    self.kernel_regularizer = regularizers.get(kernel_regularizer)
		    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		    self.bias_regularizer = regularizers.get(bias_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)

		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.recurrent_constraint = constraints.get(recurrent_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		    self.dropout = min(1., max(0., dropout))
		    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
		    self.state_spec = InputSpec(shape=(None, self.units))

		  def build(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shape = input_shape[0]
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()

		    batch_size = input_shape[0] if self.stateful else None
		    self.input_dim = input_shape[2]
		    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

		    self.states = [None]
		    if self.stateful:
		      self.reset_states()

		    self.kernel = self.add_weight(
		        shape=(self.input_dim, self.units),
		        name='kernel',
		        initializer=self.kernel_initializer,
		        regularizer=self.kernel_regularizer,
		        constraint=self.kernel_constraint)
		    self.recurrent_kernel = self.add_weight(
		        shape=(self.units, self.units),
		        name='recurrent_kernel',
		        initializer=self.recurrent_initializer,
		        regularizer=self.recurrent_regularizer,
		        constraint=self.recurrent_constraint)
		    if self.use_bias:
		      self.bias = self.add_weight(
		          shape=(self.units,),
		          name='bias',
		          initializer=self.bias_initializer,
		          regularizer=self.bias_regularizer,
		          constraint=self.bias_constraint)
		    else:
		      self.bias = None
		    self.built = True

		  def preprocess_input(self, inputs, training=None):
		    if self.implementation > 0:
		      return inputs
		    else:
		      input_shape = inputs.get_shape().as_list()
		      input_dim = input_shape[2]
		      timesteps = input_shape[1]
		      return _time_distributed_dense(
		          inputs,
		          self.kernel,
		          self.bias,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)

		  def step(self, inputs, states):
		    if self.implementation == 0:
		      h = inputs
		    else:
		      if 0 < self.dropout < 1:
		        h = K.dot(inputs * states[1], self.kernel)
		      else:
		        h = K.dot(inputs, self.kernel)
		      if self.bias is not None:
		        h = K.bias_add(h, self.bias)

		    prev_output = states[0]
		    if 0 < self.recurrent_dropout < 1:
		      prev_output *= states[2]
		    output = h + K.dot(prev_output, self.recurrent_kernel)
		    if self.activation is not None:
		      output = self.activation(output)

		    # Properly set learning phase on output tensor.
		    if 0 < self.dropout + self.recurrent_dropout:
		      output._uses_learning_phase = True
		    return output, [output]

		  def get_constants(self, inputs, training=None):
		    constants = []
		    if self.implementation != 0 and 0 < self.dropout < 1:
		      input_shape = K.int_shape(inputs)
		      input_dim = input_shape[-1]
		      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
		      ones = K.tile(ones, (1, int(input_dim)))

		      def dropped_inputs():
		        return K.dropout(ones, self.dropout)

		      dp_mask = K.in_train_phase(dropped_inputs, ones, training=training)
		      constants.append(dp_mask)
		    else:
		      constants.append(K.cast_to_floatx(1.))

		    if 0 < self.recurrent_dropout < 1:
		      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
		      ones = K.tile(ones, (1, self.units))

		      def dropped_inputs():  # pylint: disable=function-redefined
		        return K.dropout(ones, self.recurrent_dropout)

		      rec_dp_mask = K.in_train_phase(dropped_inputs, ones, training=training)
		      constants.append(rec_dp_mask)
		    else:
		      constants.append(K.cast_to_floatx(1.))
		    return constants

		  def get_config(self):
		    config = {
		        'units':
		            self.units,
		        'activation':
		            activations.serialize(self.activation),
		        'use_bias':
		            self.use_bias,
		        'kernel_initializer':
		            initializers.serialize(self.kernel_initializer),
		        'recurrent_initializer':
		            initializers.serialize(self.recurrent_initializer),
		        'bias_initializer':
		            initializers.serialize(self.bias_initializer),
		        'kernel_regularizer':
		            regularizers.serialize(self.kernel_regularizer),
		        'recurrent_regularizer':
		            regularizers.serialize(self.recurrent_regularizer),
		        'bias_regularizer':
		            regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint':
		            constraints.serialize(self.kernel_constraint),
		        'recurrent_constraint':
		            constraints.serialize(self.recurrent_constraint),
		        'bias_constraint':
		            constraints.serialize(self.bias_constraint),
		        'dropout':
		            self.dropout,
		        'recurrent_dropout':
		            self.recurrent_dropout
		    }
		    base_config = super(SimpleRNN, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class GRU(Recurrent):
		  """Gated Recurrent Unit - Cho et al.

		  2014.

		  Arguments:
		      units: Positive integer, dimensionality of the output space.
		      activation: Activation function to use.
		          If you pass None, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      recurrent_activation: Activation function to use
		          for the recurrent step.
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix,
		          used for the linear transformation of the inputs..
		      recurrent_initializer: Initializer for the `recurrent_kernel`
		          weights matrix,
		          used for the linear transformation of the recurrent state..
		      bias_initializer: Initializer for the bias vector.
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      recurrent_regularizer: Regularizer function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to
		          the `kernel` weights matrix.
		      recurrent_constraint: Constraint function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_constraint: Constraint function applied to the bias vector.
		      dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the inputs.
		      recurrent_dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the recurrent state.

		  References:
		      - [On the Properties of Neural Machine Translation: Encoder-Decoder
		        Approaches](https://arxiv.org/abs/1409.1259)
		      - [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
		        Modeling](http://arxiv.org/abs/1412.3555v1)
		      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
		        Networks](http://arxiv.org/abs/1512.05287)
		  """

		  def __init__(self,
		               units,
		               activation='tanh',
		               recurrent_activation='hard_sigmoid',
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               recurrent_initializer='orthogonal',
		               bias_initializer='zeros',
		               kernel_regularizer=None,
		               recurrent_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               recurrent_constraint=None,
		               bias_constraint=None,
		               dropout=0.,
		               recurrent_dropout=0.,
		               **kwargs):
		    super(GRU, self).__init__(**kwargs)
		    self.units = units
		    self.activation = activations.get(activation)
		    self.recurrent_activation = activations.get(recurrent_activation)
		    self.use_bias = use_bias

		    self.kernel_initializer = initializers.get(kernel_initializer)
		    self.recurrent_initializer = initializers.get(recurrent_initializer)
		    self.bias_initializer = initializers.get(bias_initializer)

		    self.kernel_regularizer = regularizers.get(kernel_regularizer)
		    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		    self.bias_regularizer = regularizers.get(bias_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)

		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.recurrent_constraint = constraints.get(recurrent_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		    self.dropout = min(1., max(0., dropout))
		    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
		    self.state_spec = InputSpec(shape=(None, self.units))

		  def build(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shape = input_shape[0]
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    batch_size = input_shape[0] if self.stateful else None
		    self.input_dim = input_shape[2]
		    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

		    self.states = [None]
		    if self.stateful:
		      self.reset_states()

		    self.kernel = self.add_weight(
		        shape=(self.input_dim, self.units * 3),
		        name='kernel',
		        initializer=self.kernel_initializer,
		        regularizer=self.kernel_regularizer,
		        constraint=self.kernel_constraint)
		    self.recurrent_kernel = self.add_weight(
		        shape=(self.units, self.units * 3),
		        name='recurrent_kernel',
		        initializer=self.recurrent_initializer,
		        regularizer=self.recurrent_regularizer,
		        constraint=self.recurrent_constraint)

		    if self.use_bias:
		      self.bias = self.add_weight(
		          shape=(self.units * 3,),
		          name='bias',
		          initializer=self.bias_initializer,
		          regularizer=self.bias_regularizer,
		          constraint=self.bias_constraint)
		    else:
		      self.bias = None

		    self.kernel_z = self.kernel[:, :self.units]
		    self.recurrent_kernel_z = self.recurrent_kernel[:, :self.units]
		    self.kernel_r = self.kernel[:, self.units:self.units * 2]
		    self.recurrent_kernel_r = self.recurrent_kernel[:, self.units:
		                                                    self.units * 2]
		    self.kernel_h = self.kernel[:, self.units * 2:]
		    self.recurrent_kernel_h = self.recurrent_kernel[:, self.units * 2:]

		    if self.use_bias:
		      self.bias_z = self.bias[:self.units]
		      self.bias_r = self.bias[self.units:self.units * 2]
		      self.bias_h = self.bias[self.units * 2:]
		    else:
		      self.bias_z = None
		      self.bias_r = None
		      self.bias_h = None
		    self.built = True

		  def preprocess_input(self, inputs, training=None):
		    if self.implementation == 0:
		      input_shape = inputs.get_shape().as_list()
		      input_dim = input_shape[2]
		      timesteps = input_shape[1]

		      x_z = _time_distributed_dense(
		          inputs,
		          self.kernel_z,
		          self.bias_z,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      x_r = _time_distributed_dense(
		          inputs,
		          self.kernel_r,
		          self.bias_r,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      x_h = _time_distributed_dense(
		          inputs,
		          self.kernel_h,
		          self.bias_h,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      return K.concatenate([x_z, x_r, x_h], axis=2)
		    else:
		      return inputs

		  def get_constants(self, inputs, training=None):
		    constants = []
		    if self.implementation != 0 and 0 < self.dropout < 1:
		      input_shape = K.int_shape(inputs)
		      input_dim = input_shape[-1]
		      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
		      ones = K.tile(ones, (1, int(input_dim)))

		      def dropped_inputs():
		        return K.dropout(ones, self.dropout)

		      dp_mask = [
		          K.in_train_phase(dropped_inputs, ones, training=training)
		          for _ in range(3)
		      ]
		      constants.append(dp_mask)
		    else:
		      constants.append([K.cast_to_floatx(1.) for _ in range(3)])

		    if 0 < self.recurrent_dropout < 1:
		      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
		      ones = K.tile(ones, (1, self.units))

		      def dropped_inputs():  # pylint: disable=function-redefined
		        return K.dropout(ones, self.recurrent_dropout)

		      rec_dp_mask = [
		          K.in_train_phase(dropped_inputs, ones, training=training)
		          for _ in range(3)
		      ]
		      constants.append(rec_dp_mask)
		    else:
		      constants.append([K.cast_to_floatx(1.) for _ in range(3)])
		    return constants

		  def step(self, inputs, states):
		    h_tm1 = states[0]  # previous memory
		    dp_mask = states[1]  # dropout matrices for recurrent units
		    rec_dp_mask = states[2]

		    if self.implementation == 2:
		      matrix_x = K.dot(inputs * dp_mask[0], self.kernel)
		      if self.use_bias:
		        matrix_x = K.bias_add(matrix_x, self.bias)
		      matrix_inner = K.dot(h_tm1 * rec_dp_mask[0],
		                           self.recurrent_kernel[:, :2 * self.units])

		      x_z = matrix_x[:, :self.units]
		      x_r = matrix_x[:, self.units:2 * self.units]
		      recurrent_z = matrix_inner[:, :self.units]
		      recurrent_r = matrix_inner[:, self.units:2 * self.units]

		      z = self.recurrent_activation(x_z + recurrent_z)
		      r = self.recurrent_activation(x_r + recurrent_r)

		      x_h = matrix_x[:, 2 * self.units:]
		      recurrent_h = K.dot(r * h_tm1 * rec_dp_mask[0],
		                          self.recurrent_kernel[:, 2 * self.units:])
		      hh = self.activation(x_h + recurrent_h)
		    else:
		      if self.implementation == 0:
		        x_z = inputs[:, :self.units]
		        x_r = inputs[:, self.units:2 * self.units]
		        x_h = inputs[:, 2 * self.units:]
		      elif self.implementation == 1:
		        x_z = K.dot(inputs * dp_mask[0], self.kernel_z)
		        x_r = K.dot(inputs * dp_mask[1], self.kernel_r)
		        x_h = K.dot(inputs * dp_mask[2], self.kernel_h)
		        if self.use_bias:
		          x_z = K.bias_add(x_z, self.bias_z)
		          x_r = K.bias_add(x_r, self.bias_r)
		          x_h = K.bias_add(x_h, self.bias_h)
		      else:
		        raise ValueError('Unknown `implementation` mode.')
		      z = self.recurrent_activation(x_z + K.dot(h_tm1 * rec_dp_mask[0],
		                                                self.recurrent_kernel_z))
		      r = self.recurrent_activation(x_r + K.dot(h_tm1 * rec_dp_mask[1],
		                                                self.recurrent_kernel_r))

		      hh = self.activation(x_h + K.dot(r * h_tm1 * rec_dp_mask[2],
		                                       self.recurrent_kernel_h))
		    h = z * h_tm1 + (1 - z) * hh
		    if 0 < self.dropout + self.recurrent_dropout:
		      h._uses_learning_phase = True
		    return h, [h]

		  def get_config(self):
		    config = {
		        'units':
		            self.units,
		        'activation':
		            activations.serialize(self.activation),
		        'recurrent_activation':
		            activations.serialize(self.recurrent_activation),
		        'use_bias':
		            self.use_bias,
		        'kernel_initializer':
		            initializers.serialize(self.kernel_initializer),
		        'recurrent_initializer':
		            initializers.serialize(self.recurrent_initializer),
		        'bias_initializer':
		            initializers.serialize(self.bias_initializer),
		        'kernel_regularizer':
		            regularizers.serialize(self.kernel_regularizer),
		        'recurrent_regularizer':
		            regularizers.serialize(self.recurrent_regularizer),
		        'bias_regularizer':
		            regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint':
		            constraints.serialize(self.kernel_constraint),
		        'recurrent_constraint':
		            constraints.serialize(self.recurrent_constraint),
		        'bias_constraint':
		            constraints.serialize(self.bias_constraint),
		        'dropout':
		            self.dropout,
		        'recurrent_dropout':
		            self.recurrent_dropout
		    }
		    base_config = super(GRU, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


		class LSTM(Recurrent):
		  """Long-Short Term Memory unit - Hochreiter 1997.

		  For a step-by-step description of the algorithm, see
		  [this tutorial](http://deeplearning.net/tutorial/lstm.html).

		  Arguments:
		      units: Positive integer, dimensionality of the output space.
		      activation: Activation function to use.
		          If you pass None, no activation is applied
		          (ie. "linear" activation: `a(x) = x`).
		      recurrent_activation: Activation function to use
		          for the recurrent step.
		      use_bias: Boolean, whether the layer uses a bias vector.
		      kernel_initializer: Initializer for the `kernel` weights matrix,
		          used for the linear transformation of the inputs..
		      recurrent_initializer: Initializer for the `recurrent_kernel`
		          weights matrix,
		          used for the linear transformation of the recurrent state..
		      bias_initializer: Initializer for the bias vector.
		      unit_forget_bias: Boolean.
		          If True, add 1 to the bias of the forget gate at initialization.
		          Setting it to true will also force `bias_initializer="zeros"`.
		          This is recommended in [Jozefowicz et
		            al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
		      kernel_regularizer: Regularizer function applied to
		          the `kernel` weights matrix.
		      recurrent_regularizer: Regularizer function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_regularizer: Regularizer function applied to the bias vector.
		      activity_regularizer: Regularizer function applied to
		          the output of the layer (its "activation")..
		      kernel_constraint: Constraint function applied to
		          the `kernel` weights matrix.
		      recurrent_constraint: Constraint function applied to
		          the `recurrent_kernel` weights matrix.
		      bias_constraint: Constraint function applied to the bias vector.
		      dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the inputs.
		      recurrent_dropout: Float between 0 and 1.
		          Fraction of the units to drop for
		          the linear transformation of the recurrent state.

		  References:
		      - [Long short-term
		        memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
		        (original 1997 paper)
		      - [Supervised sequence labeling with recurrent neural
		        networks](http://www.cs.toronto.edu/~graves/preprint.pdf)
		      - [A Theoretically Grounded Application of Dropout in Recurrent Neural
		        Networks](http://arxiv.org/abs/1512.05287)
		  """

		  def __init__(self,
		               units,
		               activation='tanh',
		               recurrent_activation='hard_sigmoid',
		               use_bias=True,
		               kernel_initializer='glorot_uniform',
		               recurrent_initializer='orthogonal',
		               bias_initializer='zeros',
		               unit_forget_bias=True,
		               kernel_regularizer=None,
		               recurrent_regularizer=None,
		               bias_regularizer=None,
		               activity_regularizer=None,
		               kernel_constraint=None,
		               recurrent_constraint=None,
		               bias_constraint=None,
		               dropout=0.,
		               recurrent_dropout=0.,
		               **kwargs):
		    super(LSTM, self).__init__(**kwargs)
		    self.units = units
		    self.activation = activations.get(activation)
		    self.recurrent_activation = activations.get(recurrent_activation)
		    self.use_bias = use_bias

		    self.kernel_initializer = initializers.get(kernel_initializer)
		    self.recurrent_initializer = initializers.get(recurrent_initializer)
		    self.bias_initializer = initializers.get(bias_initializer)
		    self.unit_forget_bias = unit_forget_bias

		    self.kernel_regularizer = regularizers.get(kernel_regularizer)
		    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		    self.bias_regularizer = regularizers.get(bias_regularizer)
		    self.activity_regularizer = regularizers.get(activity_regularizer)

		    self.kernel_constraint = constraints.get(kernel_constraint)
		    self.recurrent_constraint = constraints.get(recurrent_constraint)
		    self.bias_constraint = constraints.get(bias_constraint)

		    self.dropout = min(1., max(0., dropout))
		    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
		    self.state_spec = [
		        InputSpec(shape=(None, self.units)),
		        InputSpec(shape=(None, self.units))
		    ]

		  def build(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shape = input_shape[0]
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    batch_size = input_shape[0] if self.stateful else None
		    self.input_dim = input_shape[2]
		    self.input_spec[0] = InputSpec(shape=(batch_size, None, self.input_dim))

		    self.states = [None, None]
		    if self.stateful:
		      self.reset_states()

		    self.kernel = self.add_weight(
		        shape=(self.input_dim, self.units * 4),
		        name='kernel',
		        initializer=self.kernel_initializer,
		        regularizer=self.kernel_regularizer,
		        constraint=self.kernel_constraint)
		    self.recurrent_kernel = self.add_weight(
		        shape=(self.units, self.units * 4),
		        name='recurrent_kernel',
		        initializer=self.recurrent_initializer,
		        regularizer=self.recurrent_regularizer,
		        constraint=self.recurrent_constraint)

		    if self.use_bias:
		      if self.unit_forget_bias:

		        def bias_initializer(_, *args, **kwargs):
		          return K.concatenate([
		              self.bias_initializer((self.units,), *args, **kwargs),
		              initializers.Ones()((self.units,), *args, **kwargs),
		              self.bias_initializer((self.units * 2,), *args, **kwargs),
		          ])
		      else:
		        bias_initializer = self.bias_initializer
		      self.bias = self.add_weight(
		          shape=(self.units * 4,),
		          name='bias',
		          initializer=bias_initializer,
		          regularizer=self.bias_regularizer,
		          constraint=self.bias_constraint)
		    else:
		      self.bias = None

		    self.kernel_i = self.kernel[:, :self.units]
		    self.kernel_f = self.kernel[:, self.units:self.units * 2]
		    self.kernel_c = self.kernel[:, self.units * 2:self.units * 3]
		    self.kernel_o = self.kernel[:, self.units * 3:]

		    self.recurrent_kernel_i = self.recurrent_kernel[:, :self.units]
		    self.recurrent_kernel_f = self.recurrent_kernel[:, self.units:
		                                                    self.units * 2]
		    self.recurrent_kernel_c = self.recurrent_kernel[:, self.units * 2:
		                                                    self.units * 3]
		    self.recurrent_kernel_o = self.recurrent_kernel[:, self.units * 3:]

		    if self.use_bias:
		      self.bias_i = self.bias[:self.units]
		      self.bias_f = self.bias[self.units:self.units * 2]
		      self.bias_c = self.bias[self.units * 2:self.units * 3]
		      self.bias_o = self.bias[self.units * 3:]
		    else:
		      self.bias_i = None
		      self.bias_f = None
		      self.bias_c = None
		      self.bias_o = None
		    self.built = True

		  def preprocess_input(self, inputs, training=None):
		    if self.implementation == 0:
		      input_shape = inputs.get_shape().as_list()
		      input_dim = input_shape[2]
		      timesteps = input_shape[1]

		      x_i = _time_distributed_dense(
		          inputs,
		          self.kernel_i,
		          self.bias_i,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      x_f = _time_distributed_dense(
		          inputs,
		          self.kernel_f,
		          self.bias_f,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      x_c = _time_distributed_dense(
		          inputs,
		          self.kernel_c,
		          self.bias_c,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      x_o = _time_distributed_dense(
		          inputs,
		          self.kernel_o,
		          self.bias_o,
		          self.dropout,
		          input_dim,
		          self.units,
		          timesteps,
		          training=training)
		      return K.concatenate([x_i, x_f, x_c, x_o], axis=2)
		    else:
		      return inputs

		  def get_constants(self, inputs, training=None):
		    constants = []
		    if self.implementation != 0 and 0 < self.dropout < 1:
		      input_shape = K.int_shape(inputs)
		      input_dim = input_shape[-1]
		      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
		      ones = K.tile(ones, (1, int(input_dim)))

		      def dropped_inputs():
		        return K.dropout(ones, self.dropout)

		      dp_mask = [
		          K.in_train_phase(dropped_inputs, ones, training=training)
		          for _ in range(4)
		      ]
		      constants.append(dp_mask)
		    else:
		      constants.append([K.cast_to_floatx(1.) for _ in range(4)])

		    if 0 < self.recurrent_dropout < 1:
		      ones = K.ones_like(K.reshape(inputs[:, 0, 0], (-1, 1)))
		      ones = K.tile(ones, (1, self.units))

		      def dropped_inputs():  # pylint: disable=function-redefined
		        return K.dropout(ones, self.recurrent_dropout)

		      rec_dp_mask = [
		          K.in_train_phase(dropped_inputs, ones, training=training)
		          for _ in range(4)
		      ]
		      constants.append(rec_dp_mask)
		    else:
		      constants.append([K.cast_to_floatx(1.) for _ in range(4)])
		    return constants

		  def step(self, inputs, states):
		    h_tm1 = states[0]
		    c_tm1 = states[1]
		    dp_mask = states[2]
		    rec_dp_mask = states[3]

		    if self.implementation == 2:
		      z = K.dot(inputs * dp_mask[0], self.kernel)
		      z += K.dot(h_tm1 * rec_dp_mask[0], self.recurrent_kernel)
		      if self.use_bias:
		        z = K.bias_add(z, self.bias)

		      z0 = z[:, :self.units]
		      z1 = z[:, self.units:2 * self.units]
		      z2 = z[:, 2 * self.units:3 * self.units]
		      z3 = z[:, 3 * self.units:]

		      i = self.recurrent_activation(z0)
		      f = self.recurrent_activation(z1)
		      c = f * c_tm1 + i * self.activation(z2)
		      o = self.recurrent_activation(z3)
		    else:
		      if self.implementation == 0:
		        x_i = inputs[:, :self.units]
		        x_f = inputs[:, self.units:2 * self.units]
		        x_c = inputs[:, 2 * self.units:3 * self.units]
		        x_o = inputs[:, 3 * self.units:]
		      elif self.implementation == 1:
		        x_i = K.dot(inputs * dp_mask[0], self.kernel_i) + self.bias_i
		        x_f = K.dot(inputs * dp_mask[1], self.kernel_f) + self.bias_f
		        x_c = K.dot(inputs * dp_mask[2], self.kernel_c) + self.bias_c
		        x_o = K.dot(inputs * dp_mask[3], self.kernel_o) + self.bias_o
		      else:
		        raise ValueError('Unknown `implementation` mode.')

		      i = self.recurrent_activation(x_i + K.dot(h_tm1 * rec_dp_mask[0],
		                                                self.recurrent_kernel_i))
		      f = self.recurrent_activation(x_f + K.dot(h_tm1 * rec_dp_mask[1],
		                                                self.recurrent_kernel_f))
		      c = f * c_tm1 + i * self.activation(
		          x_c + K.dot(h_tm1 * rec_dp_mask[2], self.recurrent_kernel_c))
		      o = self.recurrent_activation(x_o + K.dot(h_tm1 * rec_dp_mask[3],
		                                                self.recurrent_kernel_o))
		    h = o * self.activation(c)
		    if 0 < self.dropout + self.recurrent_dropout:
		      h._uses_learning_phase = True
		    return h, [h, c]

		  def get_config(self):
		    config = {
		        'units':
		            self.units,
		        'activation':
		            activations.serialize(self.activation),
		        'recurrent_activation':
		            activations.serialize(self.recurrent_activation),
		        'use_bias':
		            self.use_bias,
		        'kernel_initializer':
		            initializers.serialize(self.kernel_initializer),
		        'recurrent_initializer':
		            initializers.serialize(self.recurrent_initializer),
		        'bias_initializer':
		            initializers.serialize(self.bias_initializer),
		        'unit_forget_bias':
		            self.unit_forget_bias,
		        'kernel_regularizer':
		            regularizers.serialize(self.kernel_regularizer),
		        'recurrent_regularizer':
		            regularizers.serialize(self.recurrent_regularizer),
		        'bias_regularizer':
		            regularizers.serialize(self.bias_regularizer),
		        'activity_regularizer':
		            regularizers.serialize(self.activity_regularizer),
		        'kernel_constraint':
		            constraints.serialize(self.kernel_constraint),
		        'recurrent_constraint':
		            constraints.serialize(self.recurrent_constraint),
		        'bias_constraint':
		            constraints.serialize(self.bias_constraint),
		        'dropout':
		            self.dropout,
		        'recurrent_dropout':
		            self.recurrent_dropout
		    }
		    base_config = super(LSTM, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))


	class serialization_py:

		"""Layer serialization/deserialization functions.
		"""

		def import_libs():
			# pylint: disable=wildcard-import
			# pylint: disable=unused-import
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras.engine import Input
			from tensorflow.contrib.keras.python.keras.engine import InputLayer
			from tensorflow.contrib.keras.python.keras.layers.advanced_activations import *
			from tensorflow.contrib.keras.python.keras.layers.convolutional import *
			from tensorflow.contrib.keras.python.keras.layers.convolutional_recurrent import *
			from tensorflow.contrib.keras.python.keras.layers.core import *
			from tensorflow.contrib.keras.python.keras.layers.embeddings import *
			from tensorflow.contrib.keras.python.keras.layers.local import *
			from tensorflow.contrib.keras.python.keras.layers.merge import *
			from tensorflow.contrib.keras.python.keras.layers.noise import *
			from tensorflow.contrib.keras.python.keras.layers.normalization import *
			from tensorflow.contrib.keras.python.keras.layers.pooling import *
			from tensorflow.contrib.keras.python.keras.layers.recurrent import *
			from tensorflow.contrib.keras.python.keras.layers.wrappers import *
			from tensorflow.contrib.keras.python.keras.utils.generic_utils import deserialize_keras_object


		def serialize(layer):
		  return {'class_name': layer.__class__.__name__, 'config': layer.get_config()}


		def deserialize(config, custom_objects=None):
		  """Instantiates a layer from a config dictionary.

		  Arguments:
		      config: dict of the form {'class_name': str, 'config': dict}
		      custom_objects: dict mapping class names (or function names)
		          of custom (non-Keras) objects to class/functions

		  Returns:
		      Layer instance (may be Model, Sequential, Layer...)
		  """
		  from tensorflow.contrib.keras.python.keras import models  # pylint: disable=g-import-not-at-top
		  globs = globals()  # All layers.
		  globs['Model'] = models.Model
		  globs['Sequential'] = models.Sequential
		  return deserialize_keras_object(
		      config,
		      module_objects=globs,
		      custom_objects=custom_objects,
		      printable_module_name='layer')


	class wrappers_py:

		# pylint: disable=protected-access
		"""Wrapper layers: layers that augment the functionality of another layer.
		"""

		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import copy

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.engine import InputSpec
			from tensorflow.contrib.keras.python.keras.engine import Layer
			from tensorflow.python.framework import tensor_shape
			from tensorflow.python.util import tf_inspect


		class Wrapper(Layer):
		  """Abstract wrapper base class.

		  Wrappers take another layer and augment it in various ways.
		  Do not use this class as a layer, it is only an abstract base class.
		  Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

		  Arguments:
		      layer: The layer to be wrapped.
		  """

		  def __init__(self, layer, **kwargs):
		    self.layer = layer
		    super(Wrapper, self).__init__(**kwargs)

		  def build(self, input_shape=None):
		    self.built = True

		  @property
		  def activity_regularizer(self):
		    if hasattr(self.layer, 'activity_regularizer'):
		      return self.layer.activity_regularizer
		    else:
		      return None

		  @property
		  def trainable_weights(self):
		    return self.layer.trainable_weights

		  @property
		  def non_trainable_weights(self):
		    return self.layer.non_trainable_weights

		  @property
		  def updates(self):
		    if hasattr(self.layer, 'updates'):
		      return self.layer.updates
		    return []

		  def get_updates_for(self, inputs=None):
		    if inputs is None:
		      updates = self.layer.get_updates_for(None)
		      return updates + super(Wrapper, self).get_updates_for(None)
		    return super(Wrapper, self).get_updates_for(inputs)

		  @property
		  def losses(self):
		    if hasattr(self.layer, 'losses'):
		      return self.layer.losses
		    return []

		  def get_losses_for(self, inputs=None):
		    if inputs is None:
		      losses = self.layer.get_losses_for(None)
		      return losses + super(Wrapper, self).get_losses_for(None)
		    return super(Wrapper, self).get_losses_for(inputs)

		  @property
		  def constraints(self):
		    return self.layer.constraints

		  def get_weights(self):
		    return self.layer.get_weights()

		  def set_weights(self, weights):
		    self.layer.set_weights(weights)

		  def get_config(self):
		    config = {
		        'layer': {
		            'class_name': self.layer.__class__.__name__,
		            'config': self.layer.get_config()
		        }
		    }
		    base_config = super(Wrapper, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

		  @classmethod
		  def from_config(cls, config, custom_objects=None):
		    from tensorflow.contrib.keras.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
		    layer = deserialize_layer(
		        config.pop('layer'), custom_objects=custom_objects)
		    return cls(layer, **config)


		class TimeDistributed(Wrapper):
		  """This wrapper allows to apply a layer to every temporal slice of an input.

		  The input should be at least 3D, and the dimension of index one
		  will be considered to be the temporal dimension.

		  Consider a batch of 32 samples,
		  where each sample is a sequence of 10 vectors of 16 dimensions.
		  The batch input shape of the layer is then `(32, 10, 16)`,
		  and the `input_shape`, not including the samples dimension, is `(10, 16)`.

		  You can then use `TimeDistributed` to apply a `Dense` layer
		  to each of the 10 timesteps, independently:

		  ```python
		      # as the first layer in a model
		      model = Sequential()
		      model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
		      # now model.output_shape == (None, 10, 8)
		  ```

		  The output will then have shape `(32, 10, 8)`.

		  In subsequent layers, there is no need for the `input_shape`:

		  ```python
		      model.add(TimeDistributed(Dense(32)))
		      # now model.output_shape == (None, 10, 32)
		  ```

		  The output will then have shape `(32, 10, 32)`.

		  `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
		  for instance with a `Conv2D` layer:

		  ```python
		      model = Sequential()
		      model.add(TimeDistributed(Conv2D(64, (3, 3)),
		                                input_shape=(10, 299, 299, 3)))
		  ```

		  Arguments:
		      layer: a layer instance.
		  """

		  def __init__(self, layer, **kwargs):
		    super(TimeDistributed, self).__init__(layer, **kwargs)
		    self.supports_masking = True

		  def build(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    assert len(input_shape) >= 3
		    self.input_spec = InputSpec(shape=input_shape)
		    child_input_shape = [input_shape[0]] + input_shape[2:]
		    if not self.layer.built:
		      self.layer.build(child_input_shape)
		      self.layer.built = True
		    super(TimeDistributed, self).build()
		    self.built = True

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    child_input_shape = tensor_shape.TensorShape([input_shape[0]] +
		                                                 input_shape[2:])
		    child_output_shape = self.layer._compute_output_shape(  # pylint: disable=protected-access
		        child_input_shape).as_list()
		    timesteps = input_shape[1]
		    return tensor_shape.TensorShape([child_output_shape[0], timesteps] +
		                                    child_output_shape[1:])

		  def call(self, inputs, mask=None):
		    input_shape = K.int_shape(inputs)
		    if input_shape[0]:
		      # batch size matters, use rnn-based implementation
		      def step(x, _):
		        output = self.layer.call(x)
		        return output, []

		      _, outputs, _ = K.rnn(step, inputs, initial_states=[], unroll=False)
		      y = outputs
		    else:
		      # No batch size specified, therefore the layer will be able
		      # to process batches of any size.
		      # We can go with reshape-based implementation for performance.
		      input_length = input_shape[1]
		      if not input_length:
		        input_length = K.shape(inputs)[1]
		      # Shape: (num_samples * timesteps, ...)
		      inputs = K.reshape(inputs, (-1,) + input_shape[2:])
		      y = self.layer.call(inputs)  # (num_samples * timesteps, ...)
		      # Shape: (num_samples, timesteps, ...)
		      output_shape = self._compute_output_shape(input_shape).as_list()  # pylint: disable=protected-access
		      y = K.reshape(y, [-1, input_length] + output_shape[2:])

		    # Apply activity regularizer if any:
		    if (hasattr(self.layer, 'activity_regularizer') and
		        self.layer.activity_regularizer is not None):
		      regularization_loss = self.layer.activity_regularizer(y)
		      self.add_loss(regularization_loss, inputs)
		    return y


		class Bidirectional(Wrapper):
		  """Bidirectional wrapper for RNNs.

		  Arguments:
		      layer: `Recurrent` instance.
		      merge_mode: Mode by which outputs of the
		          forward and backward RNNs will be combined.
		          One of {'sum', 'mul', 'concat', 'ave', None}.
		          If None, the outputs will not be combined,
		          they will be returned as a list.

		  Raises:
		      ValueError: In case of invalid `merge_mode` argument.

		  Examples:

		  ```python
		      model = Sequential()
		      model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5,
		      10)))
		      model.add(Bidirectional(LSTM(10)))
		      model.add(Dense(5))
		      model.add(Activation('softmax'))
		      model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
		  ```
		  """

		  def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
		    super(Bidirectional, self).__init__(layer, **kwargs)
		    if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
		      raise ValueError('Invalid merge mode. '
		                       'Merge mode should be one of '
		                       '{"sum", "mul", "ave", "concat", None}')
		    self.forward_layer = copy.copy(layer)
		    config = layer.get_config()
		    config['go_backwards'] = not config['go_backwards']
		    self.backward_layer = layer.__class__.from_config(config)
		    self.forward_layer.name = 'forward_' + self.forward_layer.name
		    self.backward_layer.name = 'backward_' + self.backward_layer.name
		    self.merge_mode = merge_mode
		    if weights:
		      nw = len(weights)
		      self.forward_layer.initial_weights = weights[:nw // 2]
		      self.backward_layer.initial_weights = weights[nw // 2:]
		    self.stateful = layer.stateful
		    self.return_sequences = layer.return_sequences
		    self.supports_masking = True

		  def get_weights(self):
		    return self.forward_layer.get_weights() + self.backward_layer.get_weights()

		  def set_weights(self, weights):
		    nw = len(weights)
		    self.forward_layer.set_weights(weights[:nw // 2])
		    self.backward_layer.set_weights(weights[nw // 2:])

		  def _compute_output_shape(self, input_shape):
		    input_shape = tensor_shape.TensorShape(input_shape).as_list()
		    if self.merge_mode in ['sum', 'ave', 'mul']:
		      return self.forward_layer._compute_output_shape(input_shape)  # pylint: disable=protected-access
		    elif self.merge_mode == 'concat':
		      shape = self.forward_layer._compute_output_shape(input_shape).as_list()  # pylint: disable=protected-access
		      shape[-1] *= 2
		      return tensor_shape.TensorShape(shape)
		    elif self.merge_mode is None:
		      shape = self.forward_layer._compute_output_shape(input_shape)  # pylint: disable=protected-access
		      return [shape, copy.copy(shape)]

		  def call(self, inputs, training=None, mask=None):
		    kwargs = {}
		    func_args = tf_inspect.getargspec(self.layer.call).args
		    if 'training' in func_args:
		      kwargs['training'] = training
		    if 'mask' in func_args:
		      kwargs['mask'] = mask

		    y = self.forward_layer.call(inputs, **kwargs)
		    y_rev = self.backward_layer.call(inputs, **kwargs)
		    if self.return_sequences:
		      y_rev = K.reverse(y_rev, 1)
		    if self.merge_mode == 'concat':
		      output = K.concatenate([y, y_rev])
		    elif self.merge_mode == 'sum':
		      output = y + y_rev
		    elif self.merge_mode == 'ave':
		      output = (y + y_rev) / 2
		    elif self.merge_mode == 'mul':
		      output = y * y_rev
		    elif self.merge_mode is None:
		      output = [y, y_rev]

		    # Properly set learning phase
		    if 0 < self.layer.dropout + self.layer.recurrent_dropout:
		      if self.merge_mode is None:
		        for out in output:
		          out._uses_learning_phase = True
		      else:
		        output._uses_learning_phase = True
		    return output

		  def reset_states(self):
		    self.forward_layer.reset_states()
		    self.backward_layer.reset_states()

		  def build(self, input_shape):
		    with K.name_scope(self.forward_layer.name):
		      self.forward_layer.build(input_shape)
		    with K.name_scope(self.backward_layer.name):
		      self.backward_layer.build(input_shape)
		    self.built = True

		  def compute_mask(self, inputs, mask):
		    if self.return_sequences:
		      if not self.merge_mode:
		        return [mask, mask]
		      else:
		        return mask
		    else:
		      return None

		  @property
		  def trainable_weights(self):
		    if hasattr(self.forward_layer, 'trainable_weights'):
		      return (self.forward_layer.trainable_weights +
		              self.backward_layer.trainable_weights)
		    return []

		  @property
		  def non_trainable_weights(self):
		    if hasattr(self.forward_layer, 'non_trainable_weights'):
		      return (self.forward_layer.non_trainable_weights +
		              self.backward_layer.non_trainable_weights)
		    return []

		  @property
		  def updates(self):
		    if hasattr(self.forward_layer, 'updates'):
		      return self.forward_layer.updates + self.backward_layer.updates
		    return []

		  @property
		  def losses(self):
		    if hasattr(self.forward_layer, 'losses'):
		      return self.forward_layer.losses + self.backward_layer.losses
		    return []

		  @property
		  def constraints(self):
		    constraints = {}
		    if hasattr(self.forward_layer, 'constraints'):
		      constraints.update(self.forward_layer.constraints)
		      constraints.update(self.backward_layer.constraints)
		    return constraints

		  def get_config(self):
		    config = {'merge_mode': self.merge_mode}
		    base_config = super(Bidirectional, self).get_config()
		    return dict(list(base_config.items()) + list(config.items()))

# applications
class applications_folder:

	# in order to conveniently access VGG16 directly from tensorflow.contrib.keras.applications.VGG16
	class __init__py:

		"""Keras Applications: models with automatic loading of pre-trained weights.
		"""
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.applications.inception_v3 import InceptionV3
		from tensorflow.contrib.keras.python.keras.applications.resnet50 import ResNet50
		from tensorflow.contrib.keras.python.keras.applications.vgg16 import VGG16
		from tensorflow.contrib.keras.python.keras.applications.vgg19 import VGG19
		from tensorflow.contrib.keras.python.keras.applications.xception import Xception

	class imagenet_utils_py:

		"""Utilities used by models pre-trained on ImageNet.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import json

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		CLASS_INDEX = None
		CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


		def preprocess_input(x, data_format=None):
		  """Preprocesses a tensor encoding a batch of images.

		  Arguments:
		      x: input Numpy tensor, 4D.
		      data_format: data format of the image tensor.

		  Returns:
		      Preprocessed tensor.
		  """
		  if data_format is None:
		    data_format = K.image_data_format()
		  assert data_format in {'channels_last', 'channels_first'}

		  if data_format == 'channels_first':
		    # 'RGB'->'BGR'
		    x = x[:, ::-1, :, :]
		    # Zero-center by mean pixel
		    x[:, 0, :, :] -= 103.939
		    x[:, 1, :, :] -= 116.779
		    x[:, 2, :, :] -= 123.68
		  else:
		    # 'RGB'->'BGR'
		    x = x[:, :, :, ::-1]
		    # Zero-center by mean pixel
		    x[:, :, :, 0] -= 103.939
		    x[:, :, :, 1] -= 116.779
		    x[:, :, :, 2] -= 123.68
		  return x


		def decode_predictions(preds, top=5):
		  """Decodes the prediction of an ImageNet model.

		  Arguments:
		      preds: Numpy tensor encoding a batch of predictions.
		      top: integer, how many top-guesses to return.

		  Returns:
		      A list of lists of top class prediction tuples
		      `(class_name, class_description, score)`.
		      One list of tuples per sample in batch input.

		  Raises:
		      ValueError: in case of invalid shape of the `pred` array
		          (must be 2D).
		  """
		  global CLASS_INDEX
		  if len(preds.shape) != 2 or preds.shape[1] != 1000:
		    raise ValueError('`decode_predictions` expects '
		                     'a batch of predictions '
		                     '(i.e. a 2D array of shape (samples, 1000)). '
		                     'Found array with shape: ' + str(preds.shape))
		  if CLASS_INDEX is None:
		    fpath = get_file(
		        'imagenet_class_index.json', CLASS_INDEX_PATH, cache_subdir='models')
		    CLASS_INDEX = json.load(open(fpath))
		  results = []
		  for pred in preds:
		    top_indices = pred.argsort()[-top:][::-1]
		    result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
		    result.sort(key=lambda x: x[2], reverse=True)
		    results.append(result)
		  return results


		def _obtain_input_shape(input_shape, default_size, min_size, data_format,
		                        include_top):
		  """Internal utility to compute/validate an ImageNet model's input shape.

		  Arguments:
		      input_shape: either None (will return the default network input shape),
		          or a user-provided shape to be validated.
		      default_size: default input width/height for the model.
		      min_size: minimum input width/height accepted by the model.
		      data_format: image data format to use.
		      include_top: whether the model is expected to
		          be linked to a classifier via a Flatten layer.

		  Returns:
		      An integer shape tuple (may include None entries).

		  Raises:
		      ValueError: in case of invalid argument values.
		  """
		  if data_format == 'channels_first':
		    default_shape = (3, default_size, default_size)
		  else:
		    default_shape = (default_size, default_size, 3)
		  if include_top:
		    if input_shape is not None:
		      if input_shape != default_shape:
		        raise ValueError('When setting`include_top=True`, '
		                         '`input_shape` should be ' + str(default_shape) + '.')
		    input_shape = default_shape
		  else:
		    if data_format == 'channels_first':
		      if input_shape is not None:
		        if len(input_shape) != 3:
		          raise ValueError('`input_shape` must be a tuple of three integers.')
		        if input_shape[0] != 3:
		          raise ValueError('The input must have 3 channels; got '
		                           '`input_shape=' + str(input_shape) + '`')
		        if ((input_shape[1] is not None and input_shape[1] < min_size) or
		            (input_shape[2] is not None and input_shape[2] < min_size)):
		          raise ValueError('Input size must be at least ' + str(min_size) + 'x'
		                           + str(min_size) + ', got '
		                           '`input_shape=' + str(input_shape) + '`')
		      else:
		        input_shape = (3, None, None)
		    else:
		      if input_shape is not None:
		        if len(input_shape) != 3:
		          raise ValueError('`input_shape` must be a tuple of three integers.')
		        if input_shape[-1] != 3:
		          raise ValueError('The input must have 3 channels; got '
		                           '`input_shape=' + str(input_shape) + '`')
		        if ((input_shape[0] is not None and input_shape[0] < min_size) or
		            (input_shape[1] is not None and input_shape[1] < min_size)):
		          raise ValueError('Input size must be at least ' + str(min_size) + 'x'
		                           + str(min_size) + ', got '
		                           '`input_shape=' + str(input_shape) + '`')
		      else:
		        input_shape = (None, None, 3)
		  return input_shape

	class inception_v3_py:

		# pylint: disable=invalid-name
		"""Inception V3 model for Keras.

		Note that the input image format for this model is different than for
		the VGG16 and ResNet models (299x299 instead of 224x224),
		and that the input preprocessing function is also different (same as Xception).

		# Reference

		- [Rethinking the Inception Architecture for Computer
		Vision](http://arxiv.org/abs/1512.00567)

		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import layers
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
			from tensorflow.contrib.keras.python.keras.layers import Activation
			from tensorflow.contrib.keras.python.keras.layers import AveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
			from tensorflow.contrib.keras.python.keras.layers import Conv2D
			from tensorflow.contrib.keras.python.keras.layers import Dense
			from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import Input
			from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
			from tensorflow.contrib.keras.python.keras.models import Model
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
		WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.5/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'


		def conv2d_bn(x,
		              filters,
		              num_row,
		              num_col,
		              padding='same',
		              strides=(1, 1),
		              name=None):
		  """Utility function to apply conv + BN.

		  Arguments:
		      x: input tensor.
		      filters: filters in `Conv2D`.
		      num_row: height of the convolution kernel.
		      num_col: width of the convolution kernel.
		      padding: padding mode in `Conv2D`.
		      strides: strides in `Conv2D`.
		      name: name of the ops; will become `name + '_conv'`
		          for the convolution and `name + '_bn'` for the
		          batch norm layer.

		  Returns:
		      Output tensor after applying `Conv2D` and `BatchNormalization`.
		  """
		  if name is not None:
		    bn_name = name + '_bn'
		    conv_name = name + '_conv'
		  else:
		    bn_name = None
		    conv_name = None
		  if K.image_data_format() == 'channels_first':
		    bn_axis = 1
		  else:
		    bn_axis = 3
		  x = Conv2D(
		      filters, (num_row, num_col),
		      strides=strides,
		      padding=padding,
		      use_bias=False,
		      name=conv_name)(x)
		  x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
		  x = Activation('relu', name=name)(x)
		  return x


		def InceptionV3(include_top=True,
		                weights='imagenet',
		                input_tensor=None,
		                input_shape=None,
		                pooling=None,
		                classes=1000):
		  """Instantiates the Inception v3 architecture.

		  Optionally loads weights pre-trained
		  on ImageNet. Note that when using TensorFlow,
		  for best performance you should set
		  `image_data_format="channels_last"` in your Keras config
		  at ~/.keras/keras.json.
		  The model and the weights are compatible with both
		  TensorFlow and Theano. The data format
		  convention used by the model is the one
		  specified in your Keras config file.
		  Note that the default input image size for this model is 299x299.

		  Arguments:
		      include_top: whether to include the fully-connected
		          layer at the top of the network.
		      weights: one of `None` (random initialization)
		          or "imagenet" (pre-training on ImageNet).
		      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
		          to use as image input for the model.
		      input_shape: optional shape tuple, only to be specified
		          if `include_top` is False (otherwise the input shape
		          has to be `(299, 299, 3)` (with `channels_last` data format)
		          or `(3, 299, 299)` (with `channels_first` data format).
		          It should have exactly 3 inputs channels,
		          and width and height should be no smaller than 139.
		          E.g. `(150, 150, 3)` would be one valid value.
		      pooling: Optional pooling mode for feature extraction
		          when `include_top` is `False`.
		          - `None` means that the output of the model will be
		              the 4D tensor output of the
		              last convolutional layer.
		          - `avg` means that global average pooling
		              will be applied to the output of the
		              last convolutional layer, and thus
		              the output of the model will be a 2D tensor.
		          - `max` means that global max pooling will
		              be applied.
		      classes: optional number of classes to classify images
		          into, only to be specified if `include_top` is True, and
		          if no `weights` argument is specified.

		  Returns:
		      A Keras model instance.

		  Raises:
		      ValueError: in case of invalid argument for `weights`,
		          or invalid input shape.
		  """
		  if weights not in {'imagenet', None}:
		    raise ValueError('The `weights` argument should be either '
		                     '`None` (random initialization) or `imagenet` '
		                     '(pre-training on ImageNet).')

		  if weights == 'imagenet' and include_top and classes != 1000:
		    raise ValueError('If using `weights` as imagenet with `include_top`'
		                     ' as true, `classes` should be 1000')

		  # Determine proper input shape
		  input_shape = _obtain_input_shape(
		      input_shape,
		      default_size=299,
		      min_size=139,
		      data_format=K.image_data_format(),
		      include_top=include_top)

		  if input_tensor is None:
		    img_input = Input(shape=input_shape)
		  else:
		    img_input = Input(tensor=input_tensor, shape=input_shape)

		  if K.image_data_format() == 'channels_first':
		    channel_axis = 1
		  else:
		    channel_axis = 3

		  x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
		  x = conv2d_bn(x, 32, 3, 3, padding='valid')
		  x = conv2d_bn(x, 64, 3, 3)
		  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		  x = conv2d_bn(x, 80, 1, 1, padding='valid')
		  x = conv2d_bn(x, 192, 3, 3, padding='valid')
		  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		  # mixed 0, 1, 2: 35 x 35 x 256
		  branch1x1 = conv2d_bn(x, 64, 1, 1)

		  branch5x5 = conv2d_bn(x, 48, 1, 1)
		  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

		  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

		  branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		  branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
		  x = layers.concatenate(
		      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
		      axis=channel_axis,
		      name='mixed0')

		  # mixed 1: 35 x 35 x 256
		  branch1x1 = conv2d_bn(x, 64, 1, 1)

		  branch5x5 = conv2d_bn(x, 48, 1, 1)
		  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

		  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

		  branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
		  x = layers.concatenate(
		      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
		      axis=channel_axis,
		      name='mixed1')

		  # mixed 2: 35 x 35 x 256
		  branch1x1 = conv2d_bn(x, 64, 1, 1)

		  branch5x5 = conv2d_bn(x, 48, 1, 1)
		  branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

		  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

		  branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		  branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
		  x = layers.concatenate(
		      [branch1x1, branch5x5, branch3x3dbl, branch_pool],
		      axis=channel_axis,
		      name='mixed2')

		  # mixed 3: 17 x 17 x 768
		  branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

		  branch3x3dbl = conv2d_bn(x, 64, 1, 1)
		  branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
		  branch3x3dbl = conv2d_bn(
		      branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

		  branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		  x = layers.concatenate(
		      [branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

		  # mixed 4: 17 x 17 x 768
		  branch1x1 = conv2d_bn(x, 192, 1, 1)

		  branch7x7 = conv2d_bn(x, 128, 1, 1)
		  branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
		  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

		  branch7x7dbl = conv2d_bn(x, 128, 1, 1)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

		  branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		  x = layers.concatenate(
		      [branch1x1, branch7x7, branch7x7dbl, branch_pool],
		      axis=channel_axis,
		      name='mixed4')

		  # mixed 5, 6: 17 x 17 x 768
		  for i in range(2):
		    branch1x1 = conv2d_bn(x, 192, 1, 1)

		    branch7x7 = conv2d_bn(x, 160, 1, 1)
		    branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
		    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

		    branch7x7dbl = conv2d_bn(x, 160, 1, 1)
		    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
		    branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

		    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		    x = layers.concatenate(
		        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
		        axis=channel_axis,
		        name='mixed' + str(5 + i))

		  # mixed 7: 17 x 17 x 768
		  branch1x1 = conv2d_bn(x, 192, 1, 1)

		  branch7x7 = conv2d_bn(x, 192, 1, 1)
		  branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
		  branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

		  branch7x7dbl = conv2d_bn(x, 192, 1, 1)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
		  branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

		  branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		  branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		  x = layers.concatenate(
		      [branch1x1, branch7x7, branch7x7dbl, branch_pool],
		      axis=channel_axis,
		      name='mixed7')

		  # mixed 8: 8 x 8 x 1280
		  branch3x3 = conv2d_bn(x, 192, 1, 1)
		  branch3x3 = conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

		  branch7x7x3 = conv2d_bn(x, 192, 1, 1)
		  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
		  branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
		  branch7x7x3 = conv2d_bn(
		      branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

		  branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
		  x = layers.concatenate(
		      [branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

		  # mixed 9: 8 x 8 x 2048
		  for i in range(2):
		    branch1x1 = conv2d_bn(x, 320, 1, 1)

		    branch3x3 = conv2d_bn(x, 384, 1, 1)
		    branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
		    branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
		    branch3x3 = layers.concatenate(
		        [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

		    branch3x3dbl = conv2d_bn(x, 448, 1, 1)
		    branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
		    branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
		    branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
		    branch3x3dbl = layers.concatenate(
		        [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

		    branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
		    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		    x = layers.concatenate(
		        [branch1x1, branch3x3, branch3x3dbl, branch_pool],
		        axis=channel_axis,
		        name='mixed' + str(9 + i))
		  if include_top:
		    # Classification block
		    x = GlobalAveragePooling2D(name='avg_pool')(x)
		    x = Dense(classes, activation='softmax', name='predictions')(x)
		  else:
		    if pooling == 'avg':
		      x = GlobalAveragePooling2D()(x)
		    elif pooling == 'max':
		      x = GlobalMaxPooling2D()(x)

		  # Ensure that the model takes into account
		  # any potential predecessors of `input_tensor`.
		  if input_tensor is not None:
		    inputs = get_source_inputs(input_tensor)
		  else:
		    inputs = img_input
		  # Create model.
		  model = Model(inputs, x, name='inception_v3')

		  # load weights
		  if weights == 'imagenet':
		    if include_top:
		      weights_path = get_file(
		          'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
		          WEIGHTS_PATH,
		          cache_subdir='models',
		          md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
		    else:
		      weights_path = get_file(
		          'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
		          WEIGHTS_PATH_NO_TOP,
		          cache_subdir='models',
		          md5_hash='bcbd6486424b2319ff4ef7d526e38f63')
		    model.load_weights(weights_path)
		  return model


		def preprocess_input(x):
		  x /= 255.
		  x -= 0.5
		  x *= 2.
		  return x

	class resnet50_py:

		# pylint: disable=invalid-name
		"""ResNet50 model for Keras.

		# Reference:

		- [Deep Residual Learning for Image
		Recognition](https://arxiv.org/abs/1512.03385)

		Adapted from code contributed by BigMoyan.
		"""
		def import_libs():

			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import layers
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
			from tensorflow.contrib.keras.python.keras.layers import Activation
			from tensorflow.contrib.keras.python.keras.layers import AveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
			from tensorflow.contrib.keras.python.keras.layers import Conv2D
			from tensorflow.contrib.keras.python.keras.layers import Dense
			from tensorflow.contrib.keras.python.keras.layers import Flatten
			from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import Input
			from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import ZeroPadding2D
			from tensorflow.contrib.keras.python.keras.models import Model
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
		WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


		def identity_block(input_tensor, kernel_size, filters, stage, block):
		  """The identity block is the block that has no conv layer at shortcut.

		  Arguments:
		      input_tensor: input tensor
		      kernel_size: default 3, the kernel size of middle conv layer at main path
		      filters: list of integers, the filterss of 3 conv layer at main path
		      stage: integer, current stage label, used for generating layer names
		      block: 'a','b'..., current block label, used for generating layer names

		  Returns:
		      Output tensor for the block.
		  """
		  filters1, filters2, filters3 = filters
		  if K.image_data_format() == 'channels_last':
		    bn_axis = 3
		  else:
		    bn_axis = 1
		  conv_name_base = 'res' + str(stage) + block + '_branch'
		  bn_name_base = 'bn' + str(stage) + block + '_branch'

		  x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
		  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
		  x = Activation('relu')(x)

		  x = Conv2D(
		      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
		  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
		  x = Activation('relu')(x)

		  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
		  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

		  x = layers.add([x, input_tensor])
		  x = Activation('relu')(x)
		  return x


		def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2,
		                                                                          2)):
		  """conv_block is the block that has a conv layer at shortcut.

		  Arguments:
		      input_tensor: input tensor
		      kernel_size: default 3, the kernel size of middle conv layer at main path
		      filters: list of integers, the filterss of 3 conv layer at main path
		      stage: integer, current stage label, used for generating layer names
		      block: 'a','b'..., current block label, used for generating layer names
		      strides: Tuple of integers.

		  Returns:
		      Output tensor for the block.

		  Note that from stage 3, the first conv layer at main path is with
		  strides=(2,2)
		  And the shortcut should have strides=(2,2) as well
		  """
		  filters1, filters2, filters3 = filters
		  if K.image_data_format() == 'channels_last':
		    bn_axis = 3
		  else:
		    bn_axis = 1
		  conv_name_base = 'res' + str(stage) + block + '_branch'
		  bn_name_base = 'bn' + str(stage) + block + '_branch'

		  x = Conv2D(
		      filters1, (1, 1), strides=strides,
		      name=conv_name_base + '2a')(input_tensor)
		  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
		  x = Activation('relu')(x)

		  x = Conv2D(
		      filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
		  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
		  x = Activation('relu')(x)

		  x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
		  x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

		  shortcut = Conv2D(
		      filters3, (1, 1), strides=strides,
		      name=conv_name_base + '1')(input_tensor)
		  shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

		  x = layers.add([x, shortcut])
		  x = Activation('relu')(x)
		  return x


		def ResNet50(include_top=True,
		             weights='imagenet',
		             input_tensor=None,
		             input_shape=None,
		             pooling=None,
		             classes=1000):
		  """Instantiates the ResNet50 architecture.

		  Optionally loads weights pre-trained
		  on ImageNet. Note that when using TensorFlow,
		  for best performance you should set
		  `image_data_format="channels_last"` in your Keras config
		  at ~/.keras/keras.json.

		  The model and the weights are compatible with both
		  TensorFlow and Theano. The data format
		  convention used by the model is the one
		  specified in your Keras config file.

		  Arguments:
		      include_top: whether to include the fully-connected
		          layer at the top of the network.
		      weights: one of `None` (random initialization)
		          or "imagenet" (pre-training on ImageNet).
		      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
		          to use as image input for the model.
		      input_shape: optional shape tuple, only to be specified
		          if `include_top` is False (otherwise the input shape
		          has to be `(224, 224, 3)` (with `channels_last` data format)
		          or `(3, 224, 224)` (with `channels_first` data format).
		          It should have exactly 3 inputs channels,
		          and width and height should be no smaller than 197.
		          E.g. `(200, 200, 3)` would be one valid value.
		      pooling: Optional pooling mode for feature extraction
		          when `include_top` is `False`.
		          - `None` means that the output of the model will be
		              the 4D tensor output of the
		              last convolutional layer.
		          - `avg` means that global average pooling
		              will be applied to the output of the
		              last convolutional layer, and thus
		              the output of the model will be a 2D tensor.
		          - `max` means that global max pooling will
		              be applied.
		      classes: optional number of classes to classify images
		          into, only to be specified if `include_top` is True, and
		          if no `weights` argument is specified.

		  Returns:
		      A Keras model instance.

		  Raises:
		      ValueError: in case of invalid argument for `weights`,
		          or invalid input shape.
		  """
		  if weights not in {'imagenet', None}:
		    raise ValueError('The `weights` argument should be either '
		                     '`None` (random initialization) or `imagenet` '
		                     '(pre-training on ImageNet).')

		  if weights == 'imagenet' and include_top and classes != 1000:
		    raise ValueError('If using `weights` as imagenet with `include_top`'
		                     ' as true, `classes` should be 1000')

		  # Determine proper input shape
		  input_shape = _obtain_input_shape(
		      input_shape,
		      default_size=224,
		      min_size=197,
		      data_format=K.image_data_format(),
		      include_top=include_top)

		  if input_tensor is None:
		    img_input = Input(shape=input_shape)
		  else:
		    img_input = Input(tensor=input_tensor, shape=input_shape)

		  if K.image_data_format() == 'channels_last':
		    bn_axis = 3
		  else:
		    bn_axis = 1

		  x = ZeroPadding2D((3, 3))(img_input)
		  x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
		  x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
		  x = Activation('relu')(x)
		  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

		  x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
		  x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
		  x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

		  x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
		  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
		  x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
		  x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

		  x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
		  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
		  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
		  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
		  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
		  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

		  x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
		  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
		  x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

		  x = AveragePooling2D((7, 7), name='avg_pool')(x)

		  if include_top:
		    x = Flatten()(x)
		    x = Dense(classes, activation='softmax', name='fc1000')(x)
		  else:
		    if pooling == 'avg':
		      x = GlobalAveragePooling2D()(x)
		    elif pooling == 'max':
		      x = GlobalMaxPooling2D()(x)

		  # Ensure that the model takes into account
		  # any potential predecessors of `input_tensor`.
		  if input_tensor is not None:
		    inputs = get_source_inputs(input_tensor)
		  else:
		    inputs = img_input
		  # Create model.
		  model = Model(inputs, x, name='resnet50')

		  # load weights
		  if weights == 'imagenet':
		    if include_top:
		      weights_path = get_file(
		          'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
		          WEIGHTS_PATH,
		          cache_subdir='models',
		          md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
		    else:
		      weights_path = get_file(
		          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
		          WEIGHTS_PATH_NO_TOP,
		          cache_subdir='models',
		          md5_hash='a268eb855778b3df3c7506639542a6af')
		    model.load_weights(weights_path)
		  return model

	# 1. import libs
	# 2. method VGG16() instantiate a vgg16 model, not compiled though
	class vgg16_py:

		# pylint: disable=invalid-name
		"""VGG16 model for Keras.

		# Reference

		- [Very Deep Convolutional Networks for Large-Scale Image
		Recognition](https://arxiv.org/abs/1409.1556)

		"""

		def import_libs():

			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
			from tensorflow.contrib.keras.python.keras.layers import Conv2D
			from tensorflow.contrib.keras.python.keras.layers import Dense
			from tensorflow.contrib.keras.python.keras.layers import Flatten
			from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import Input
			from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
			from tensorflow.contrib.keras.python.keras.models import Model
			from tensorflow.contrib.keras.python.keras.utils import layer_utils
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
		WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

		# 1. create layers and tensors onto graph, from first to the last
		# 2. construct a model using first (input tensor) and last tensor
		# 3. load selected weights to the model
		# 4. return this model
		def VGG16(include_top=True,
		          weights='imagenet',
		          input_tensor=None,
		          input_shape=None,
		          pooling=None,
		          classes=1000):
		  """
		  Instantiates the VGG16 architecture:
		  	Optionally loads weights pre-trained
		  	on ImageNet. Note that when using TensorFlow,
		  	for best performance you should set
		  	`image_data_format="channels_last"` in your Keras config
		  	at ~/.keras/keras.json.

		  	The model and the weights are compatible with both
		  	TensorFlow and Theano. The data format
		  	convention used by the model is the one
		  	specified in your Keras config file.

		  Arguments:
		      include_top: whether to include the 3 fully-connected
		          layers at the top of the network.
		      weights: one of `None` (random initialization)
		          or "imagenet" (pre-training on ImageNet).
		      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
		          to use as image input for the model.
		      input_shape: optional shape tuple, only to be specified
		          if `include_top` is False (otherwise the input shape
		          has to be `(224, 224, 3)` (with `channels_last` data format)
		          or `(3, 224, 224)` (with `channels_first` data format).
		          It should have exactly 3 inputs channels,
		          and width and height should be no smaller than 48.
		          E.g. `(200, 200, 3)` would be one valid value.
		      pooling: Optional pooling mode for feature extraction
		          when `include_top` is `False`.
		          - `None` means that the output of the model will be
		              the 4D tensor output of the
		              last convolutional layer.
		          - `avg` means that global average pooling
		              will be applied to the output of the
		              last convolutional layer, and thus
		              the output of the model will be a 2D tensor.
		          - `max` means that global max pooling will
		              be applied.
		      classes: optional number of classes to classify images
		          into, only to be specified if `include_top` is True, and
		          if no `weights` argument is specified.

		  Returns:
		      A Keras model instance.

		  Raises:
		      ValueError: in case of invalid argument for `weights`,
		          or invalid input shape.
		  """
		  if weights not in {'imagenet', None}:
		    raise ValueError('The `weights` argument should be either '
		                     '`None` (random initialization) or `imagenet` '
		                     '(pre-training on ImageNet).')

		  if weights == 'imagenet' and include_top and classes != 1000:
		    raise ValueError('If using `weights` as imagenet with `include_top`'
		                     ' as true, `classes` should be 1000')

		  # Determine proper input shape for imagenet dataset
		  input_shape = _obtain_input_shape(
		      input_shape,
		      default_size=224,
		      min_size=48,
		      data_format=K.image_data_format(),
		      include_top=include_top)

		  # build Input layer and return input tensor with input_shape
		  if input_tensor is None:
		    img_input = Input(shape=input_shape)
		  else:
		    img_input = Input(tensor=input_tensor, shape=input_shape)

		  # Block 1
		  x = Conv2D(
		      64, (3, 3), activation='relu', padding='same',
		      name='block1_conv1')(img_input)
		  x = Conv2D(
		      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		  # Block 2
		  x = Conv2D(
		      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
		  x = Conv2D(
		      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		  # Block 3
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		  # Block 4
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		  # Block 5
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

		  if include_top:
		    # Classification block
		    x = Flatten(name='flatten')(x)
		    x = Dense(4096, activation='relu', name='fc1')(x)
		    x = Dense(4096, activation='relu', name='fc2')(x)
		    x = Dense(classes, activation='softmax', name='predictions')(x)
		  else:
		    if pooling == 'avg':
		      x = GlobalAveragePooling2D()(x)
		    elif pooling == 'max':
		      x = GlobalMaxPooling2D()(x)

		  # Ensure that the model takes into account ?
		  # any potential predecessors of `input_tensor`.?

		  # build model object with inputs tensor and final tensor
		  if input_tensor is not None:
		    inputs = get_source_inputs(input_tensor)
		  else:
		    inputs = img_input
		  # Create model.
		  model = Model(inputs, x, name='vgg16')

		  # there are 2 versions of weights (with or without last 3 dense layers)
		  # load the selected weights to model
		  if weights == 'imagenet':
		    if include_top:
		      weights_path = get_file(
		          'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
		          WEIGHTS_PATH,
		          cache_subdir='models')
		    else:
		      weights_path = get_file(
		          'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
		          WEIGHTS_PATH_NO_TOP,
		          cache_subdir='models')
		    model.load_weights(weights_path)
		    if K.backend() == 'theano':
		      layer_utils.convert_all_kernels_in_model(model)

			# prepare the model if image shape (channel, width, height)
		    if K.image_data_format() == 'channels_first':
		      if include_top:
		        maxpool = model.get_layer(name='block5_pool')
		        shape = maxpool.output_shape[1:]
		        dense = model.get_layer(name='fc1')
		        layer_utils.convert_dense_weights_data_format(dense, shape,
		                                                      'channels_first')
		  return model

	class vgg19_py:

		# pylint: disable=invalid-name
		"""VGG19 model for Keras.

		# Reference

		- [Very Deep Convolutional Networks for Large-Scale Image
		Recognition](https://arxiv.org/abs/1409.1556)

		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import preprocess_input  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
			from tensorflow.contrib.keras.python.keras.layers import Conv2D
			from tensorflow.contrib.keras.python.keras.layers import Dense
			from tensorflow.contrib.keras.python.keras.layers import Flatten
			from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import Input
			from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
			from tensorflow.contrib.keras.python.keras.models import Model
			from tensorflow.contrib.keras.python.keras.utils import layer_utils
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels.h5'
		WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


		def VGG19(include_top=True,
		          weights='imagenet',
		          input_tensor=None,
		          input_shape=None,
		          pooling=None,
		          classes=1000):
		  """Instantiates the VGG19 architecture.

		  Optionally loads weights pre-trained
		  on ImageNet. Note that when using TensorFlow,
		  for best performance you should set
		  `image_data_format="channels_last"` in your Keras config
		  at ~/.keras/keras.json.

		  The model and the weights are compatible with both
		  TensorFlow and Theano. The data format
		  convention used by the model is the one
		  specified in your Keras config file.

		  Arguments:
		      include_top: whether to include the 3 fully-connected
		          layers at the top of the network.
		      weights: one of `None` (random initialization)
		          or "imagenet" (pre-training on ImageNet).
		      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
		          to use as image input for the model.
		      input_shape: optional shape tuple, only to be specified
		          if `include_top` is False (otherwise the input shape
		          has to be `(224, 224, 3)` (with `channels_last` data format)
		          or `(3, 224, 224)` (with `channels_first` data format).
		          It should have exactly 3 inputs channels,
		          and width and height should be no smaller than 48.
		          E.g. `(200, 200, 3)` would be one valid value.
		      pooling: Optional pooling mode for feature extraction
		          when `include_top` is `False`.
		          - `None` means that the output of the model will be
		              the 4D tensor output of the
		              last convolutional layer.
		          - `avg` means that global average pooling
		              will be applied to the output of the
		              last convolutional layer, and thus
		              the output of the model will be a 2D tensor.
		          - `max` means that global max pooling will
		              be applied.
		      classes: optional number of classes to classify images
		          into, only to be specified if `include_top` is True, and
		          if no `weights` argument is specified.

		  Returns:
		      A Keras model instance.

		  Raises:
		      ValueError: in case of invalid argument for `weights`,
		          or invalid input shape.
		  """

		  if weights not in {'imagenet', None}:
		    raise ValueError('The `weights` argument should be either '
		                     '`None` (random initialization) or `imagenet` '
		                     '(pre-training on ImageNet).')

		  if weights == 'imagenet' and include_top and classes != 1000:
		    raise ValueError('If using `weights` as imagenet with `include_top`'
		                     ' as true, `classes` should be 1000')
		  # Determine proper input shape
		  input_shape = _obtain_input_shape(
		      input_shape,
		      default_size=224,
		      min_size=48,
		      data_format=K.image_data_format(),
		      include_top=include_top)

		  if input_tensor is None:
		    img_input = Input(shape=input_shape)
		  else:
		    img_input = Input(tensor=input_tensor, shape=input_shape)

		  # Block 1
		  x = Conv2D(
		      64, (3, 3), activation='relu', padding='same',
		      name='block1_conv1')(img_input)
		  x = Conv2D(
		      64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

		  # Block 2
		  x = Conv2D(
		      128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
		  x = Conv2D(
		      128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

		  # Block 3
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
		  x = Conv2D(
		      256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

		  # Block 4
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

		  # Block 5
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
		  x = Conv2D(
		      512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
		  x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

		  if include_top:
		    # Classification block
		    x = Flatten(name='flatten')(x)
		    x = Dense(4096, activation='relu', name='fc1')(x)
		    x = Dense(4096, activation='relu', name='fc2')(x)
		    x = Dense(classes, activation='softmax', name='predictions')(x)
		  else:
		    if pooling == 'avg':
		      x = GlobalAveragePooling2D()(x)
		    elif pooling == 'max':
		      x = GlobalMaxPooling2D()(x)

		  # Ensure that the model takes into account
		  # any potential predecessors of `input_tensor`.
		  if input_tensor is not None:
		    inputs = get_source_inputs(input_tensor)
		  else:
		    inputs = img_input
		  # Create model.
		  model = Model(inputs, x, name='vgg19')

		  # load weights
		  if weights == 'imagenet':
		    if include_top:
		      weights_path = get_file(
		          'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
		          WEIGHTS_PATH,
		          cache_subdir='models')
		    else:
		      weights_path = get_file(
		          'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
		          WEIGHTS_PATH_NO_TOP,
		          cache_subdir='models')
		    model.load_weights(weights_path)
		    if K.backend() == 'theano':
		      layer_utils.convert_all_kernels_in_model(model)

		    if K.image_data_format() == 'channels_first':
		      if include_top:
		        maxpool = model.get_layer(name='block5_pool')
		        shape = maxpool.output_shape[1:]
		        dense = model.get_layer(name='fc1')
		        layer_utils.convert_dense_weights_data_format(dense, shape,
		                                                      'channels_first')
		  return model

	class xception_py:

		# pylint: disable=invalid-name
		"""Xception V1 model for Keras.

		On ImageNet, this model gets to a top-1 validation accuracy of 0.790
		and a top-5 validation accuracy of 0.945.

		Do note that the input image format for this model is different than for
		the VGG16 and ResNet models (299x299 instead of 224x224),
		and that the input preprocessing function
		is also different (same as Inception V3).

		Also do note that this model is only available for the TensorFlow backend,
		due to its reliance on `SeparableConvolution` layers.

		# Reference

		- [Xception: Deep Learning with Depthwise Separable
		Convolutions](https://arxiv.org/abs/1610.02357)

		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import layers
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import _obtain_input_shape
			from tensorflow.contrib.keras.python.keras.applications.imagenet_utils import decode_predictions  # pylint: disable=unused-import
			from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
			from tensorflow.contrib.keras.python.keras.layers import Activation
			from tensorflow.contrib.keras.python.keras.layers import BatchNormalization
			from tensorflow.contrib.keras.python.keras.layers import Conv2D
			from tensorflow.contrib.keras.python.keras.layers import Dense
			from tensorflow.contrib.keras.python.keras.layers import GlobalAveragePooling2D
			from tensorflow.contrib.keras.python.keras.layers import GlobalMaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import Input
			from tensorflow.contrib.keras.python.keras.layers import MaxPooling2D
			from tensorflow.contrib.keras.python.keras.layers import SeparableConv2D
			from tensorflow.contrib.keras.python.keras.models import Model
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file
			from tensorflow.python.platform import tf_logging as logging


		TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
		TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


		def Xception(include_top=True,
		             weights='imagenet',
		             input_tensor=None,
		             input_shape=None,
		             pooling=None,
		             classes=1000):
		  """Instantiates the Xception architecture.

		  Optionally loads weights pre-trained
		  on ImageNet. This model is available for TensorFlow only,
		  and can only be used with inputs following the TensorFlow
		  data format `(width, height, channels)`.
		  You should set `image_data_format="channels_last"` in your Keras config
		  located at ~/.keras/keras.json.

		  Note that the default input image size for this model is 299x299.

		  Arguments:
		      include_top: whether to include the fully-connected
		          layer at the top of the network.
		      weights: one of `None` (random initialization)
		          or "imagenet" (pre-training on ImageNet).
		      input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
		          to use as image input for the model.
		      input_shape: optional shape tuple, only to be specified
		          if `include_top` is False (otherwise the input shape
		          has to be `(299, 299, 3)`.
		          It should have exactly 3 inputs channels,
		          and width and height should be no smaller than 71.
		          E.g. `(150, 150, 3)` would be one valid value.
		      pooling: Optional pooling mode for feature extraction
		          when `include_top` is `False`.
		          - `None` means that the output of the model will be
		              the 4D tensor output of the
		              last convolutional layer.
		          - `avg` means that global average pooling
		              will be applied to the output of the
		              last convolutional layer, and thus
		              the output of the model will be a 2D tensor.
		          - `max` means that global max pooling will
		              be applied.
		      classes: optional number of classes to classify images
		          into, only to be specified if `include_top` is True, and
		          if no `weights` argument is specified.

		  Returns:
		      A Keras model instance.

		  Raises:
		      ValueError: in case of invalid argument for `weights`,
		          or invalid input shape.
		      RuntimeError: If attempting to run this model with a
		          backend that does not support separable convolutions.
		  """
		  if weights not in {'imagenet', None}:
		    raise ValueError('The `weights` argument should be either '
		                     '`None` (random initialization) or `imagenet` '
		                     '(pre-training on ImageNet).')

		  if weights == 'imagenet' and include_top and classes != 1000:
		    raise ValueError('If using `weights` as imagenet with `include_top`'
		                     ' as true, `classes` should be 1000')

		  if K.backend() != 'tensorflow':
		    raise RuntimeError('The Xception model is only available with '
		                       'the TensorFlow backend.')
		  if K.image_data_format() != 'channels_last':
		    logging.warning(
		        'The Xception model is only available for the '
		        'input data format "channels_last" '
		        '(width, height, channels). '
		        'However your settings specify the default '
		        'data format "channels_first" (channels, width, height). '
		        'You should set `image_data_format="channels_last"` in your Keras '
		        'config located at ~/.keras/keras.json. '
		        'The model being returned right now will expect inputs '
		        'to follow the "channels_last" data format.')
		    K.set_image_data_format('channels_last')
		    old_data_format = 'channels_first'
		  else:
		    old_data_format = None

		  # Determine proper input shape
		  input_shape = _obtain_input_shape(
		      input_shape,
		      default_size=299,
		      min_size=71,
		      data_format=K.image_data_format(),
		      include_top=include_top)

		  if input_tensor is None:
		    img_input = Input(shape=input_shape)
		  else:
		    img_input = Input(tensor=input_tensor, shape=input_shape)

		  x = Conv2D(
		      32, (3, 3), strides=(2, 2), use_bias=False,
		      name='block1_conv1')(img_input)
		  x = BatchNormalization(name='block1_conv1_bn')(x)
		  x = Activation('relu', name='block1_conv1_act')(x)
		  x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
		  x = BatchNormalization(name='block1_conv2_bn')(x)
		  x = Activation('relu', name='block1_conv2_act')(x)

		  residual = Conv2D(
		      128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
		  residual = BatchNormalization()(residual)

		  x = SeparableConv2D(
		      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
		  x = BatchNormalization(name='block2_sepconv1_bn')(x)
		  x = Activation('relu', name='block2_sepconv2_act')(x)
		  x = SeparableConv2D(
		      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
		  x = BatchNormalization(name='block2_sepconv2_bn')(x)

		  x = MaxPooling2D(
		      (3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
		  x = layers.add([x, residual])

		  residual = Conv2D(
		      256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
		  residual = BatchNormalization()(residual)

		  x = Activation('relu', name='block3_sepconv1_act')(x)
		  x = SeparableConv2D(
		      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
		  x = BatchNormalization(name='block3_sepconv1_bn')(x)
		  x = Activation('relu', name='block3_sepconv2_act')(x)
		  x = SeparableConv2D(
		      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
		  x = BatchNormalization(name='block3_sepconv2_bn')(x)

		  x = MaxPooling2D(
		      (3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
		  x = layers.add([x, residual])

		  residual = Conv2D(
		      728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
		  residual = BatchNormalization()(residual)

		  x = Activation('relu', name='block4_sepconv1_act')(x)
		  x = SeparableConv2D(
		      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
		  x = BatchNormalization(name='block4_sepconv1_bn')(x)
		  x = Activation('relu', name='block4_sepconv2_act')(x)
		  x = SeparableConv2D(
		      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
		  x = BatchNormalization(name='block4_sepconv2_bn')(x)

		  x = MaxPooling2D(
		      (3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
		  x = layers.add([x, residual])

		  for i in range(8):
		    residual = x
		    prefix = 'block' + str(i + 5)

		    x = Activation('relu', name=prefix + '_sepconv1_act')(x)
		    x = SeparableConv2D(
		        728, (3, 3), padding='same', use_bias=False,
		        name=prefix + '_sepconv1')(x)
		    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
		    x = Activation('relu', name=prefix + '_sepconv2_act')(x)
		    x = SeparableConv2D(
		        728, (3, 3), padding='same', use_bias=False,
		        name=prefix + '_sepconv2')(x)
		    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
		    x = Activation('relu', name=prefix + '_sepconv3_act')(x)
		    x = SeparableConv2D(
		        728, (3, 3), padding='same', use_bias=False,
		        name=prefix + '_sepconv3')(x)
		    x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

		    x = layers.add([x, residual])

		  residual = Conv2D(
		      1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
		  residual = BatchNormalization()(residual)

		  x = Activation('relu', name='block13_sepconv1_act')(x)
		  x = SeparableConv2D(
		      728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
		  x = BatchNormalization(name='block13_sepconv1_bn')(x)
		  x = Activation('relu', name='block13_sepconv2_act')(x)
		  x = SeparableConv2D(
		      1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
		  x = BatchNormalization(name='block13_sepconv2_bn')(x)

		  x = MaxPooling2D(
		      (3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
		  x = layers.add([x, residual])

		  x = SeparableConv2D(
		      1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
		  x = BatchNormalization(name='block14_sepconv1_bn')(x)
		  x = Activation('relu', name='block14_sepconv1_act')(x)

		  x = SeparableConv2D(
		      2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
		  x = BatchNormalization(name='block14_sepconv2_bn')(x)
		  x = Activation('relu', name='block14_sepconv2_act')(x)

		  if include_top:
		    x = GlobalAveragePooling2D(name='avg_pool')(x)
		    x = Dense(classes, activation='softmax', name='predictions')(x)
		  else:
		    if pooling == 'avg':
		      x = GlobalAveragePooling2D()(x)
		    elif pooling == 'max':
		      x = GlobalMaxPooling2D()(x)

		  # Ensure that the model takes into account
		  # any potential predecessors of `input_tensor`.
		  if input_tensor is not None:
		    inputs = get_source_inputs(input_tensor)
		  else:
		    inputs = img_input
		  # Create model.
		  model = Model(inputs, x, name='xception')

		  # load weights
		  if weights == 'imagenet':
		    if include_top:
		      weights_path = get_file(
		          'xception_weights_tf_dim_ordering_tf_kernels.h5',
		          TF_WEIGHTS_PATH,
		          cache_subdir='models')
		    else:
		      weights_path = get_file(
		          'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
		          TF_WEIGHTS_PATH_NO_TOP,
		          cache_subdir='models')
		    model.load_weights(weights_path)

		  if old_data_format:
		    K.set_image_data_format(old_data_format)
		  return model


		def preprocess_input(x):
		  x /= 255.
		  x -= 0.5
		  x *= 2.
		  return x


# engine
class engine_folder:

	class __init__py:
		"""The Keras Engine: graph topology and training loop functionality.
		"""
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.engine.topology import get_source_inputs
		from tensorflow.contrib.keras.python.keras.engine.topology import Input
		from tensorflow.contrib.keras.python.keras.engine.topology import InputLayer
		from tensorflow.contrib.keras.python.keras.engine.topology import InputSpec
		from tensorflow.contrib.keras.python.keras.engine.topology import Layer
		from tensorflow.contrib.keras.python.keras.engine.training import Model


		# Note: topology.Node is an internal class,
		# it isn't meant to be used by Keras users.

	class topology_py:

		# pylint: disable=protected-access
		"""Base layer code and base model (Container) code.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import copy
			import json
			import os
			import re

			import numpy as np
			from six.moves import zip  # pylint: disable=redefined-builtin

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.utils import conv_utils
			from tensorflow.contrib.keras.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
			from tensorflow.contrib.keras.python.keras.utils.layer_utils import print_summary as print_layer_summary
			from tensorflow.python.framework import ops
			from tensorflow.python.framework import tensor_shape
			from tensorflow.python.layers import base as tf_base_layers
			from tensorflow.python.platform import tf_logging as logging
			from tensorflow.python.util import tf_inspect


		# pylint: disable=g-import-not-at-top
		try:
		  import h5py
		except ImportError:
		  h5py = None

		try:
		  import yaml
		except ImportError:
		  yaml = None
		# pylint: enable=g-import-not-at-top

		InputSpec = tf_base_layers.InputSpec  # pylint: disable=invalid-name


		class Node(object):
		  """A `Node` describes the connectivity between two layers.

		  Each time a layer is connected to some new input,
		  a node is added to `layer.inbound_nodes`.
		  Each time the output of a layer is used by another layer,
		  a node is added to `layer.outbound_nodes`.

		  Arguments:
		      outbound_layer: the layer that takes
		          `input_tensors` and turns them into `output_tensors`
		          (the node gets created when the `call`
		          method of the layer was called).
		      inbound_layers: a list of layers, the same length as `input_tensors`,
		          the layers from where `input_tensors` originate.
		      node_indices: a list of integers, the same length as `inbound_layers`.
		          `node_indices[i]` is the origin node of `input_tensors[i]`
		          (necessary since each inbound layer might have several nodes,
		          e.g. if the layer is being shared with a different data stream).
		      tensor_indices: a list of integers,
		          the same length as `inbound_layers`.
		          `tensor_indices[i]` is the index of `input_tensors[i]` within the
		          output of the inbound layer
		          (necessary since each inbound layer might
		          have multiple tensor outputs, with each one being
		          independently manipulable).
		      input_tensors: list of input tensors.
		      output_tensors: list of output tensors.
		      input_masks: list of input masks (a mask can be a tensor, or None).
		      output_masks: list of output masks (a mask can be a tensor, or None).
		      arguments: dictionary of keyword arguments that were passed to the
		          `call` method of the layer at the call that created the node.

		  `node_indices` and `tensor_indices` are basically fine-grained coordinates
		  describing the origin of the `input_tensors`.

		  A node from layer A to layer B is added to:
		      A.outbound_nodes
		      B.inbound_nodes
		  """

		  def __init__(self,
		               outbound_layer,
		               inbound_layers,
		               node_indices,
		               tensor_indices,
		               input_tensors,
		               output_tensors,
		               input_masks,
		               output_masks,
		               arguments=None):
		    # Layer instance (NOT a list).
		    # this is the layer that takes a list of input tensors
		    # and turns them into a list of output tensors.
		    # the current node will be added to
		    # the inbound_nodes of outbound_layer.
		    self.outbound_layer = outbound_layer

		    # The following 3 properties describe where
		    # the input tensors come from: which layers,
		    # and for each layer, which node and which
		    # tensor output of each node.

		    # List of layer instances.
		    self.inbound_layers = inbound_layers
		    # List of integers, 1:1 mapping with inbound_layers.
		    self.node_indices = node_indices
		    # List of integers, 1:1 mapping with inbound_layers.
		    self.tensor_indices = tensor_indices

		    # Following 2 properties:
		    # tensor inputs and outputs of outbound_layer.

		    # List of tensors. 1:1 mapping with inbound_layers.
		    self.input_tensors = input_tensors
		    # List of tensors, created by outbound_layer.call().
		    self.output_tensors = output_tensors

		    # Following 2 properties: input and output masks.
		    # List of tensors, 1:1 mapping with input_tensor.
		    self.input_masks = input_masks
		    # List of tensors, created by outbound_layer.compute_mask().
		    self.output_masks = output_masks

		    # Following 2 properties: input and output shapes.

		    # List of shape tuples, shapes of input_tensors.
		    self.input_shapes = [K.int_shape(x) for x in input_tensors]
		    # List of shape tuples, shapes of output_tensors.
		    self.output_shapes = [K.int_shape(x) for x in output_tensors]

		    # Optional keyword arguments to layer's `call`.
		    self.arguments = arguments

		    # Add nodes to all layers involved.
		    for layer in inbound_layers:
		      if layer is not None:
		        layer.outbound_nodes.append(self)
		    outbound_layer.inbound_nodes.append(self)

		  def get_config(self):
		    inbound_names = []
		    for layer in self.inbound_layers:
		      if layer:
		        inbound_names.append(layer.name)
		      else:
		        inbound_names.append(None)
		    return {
		        'outbound_layer':
		            self.outbound_layer.name if self.outbound_layer else None,
		        'inbound_layers':
		            inbound_names,
		        'node_indices':
		            self.node_indices,
		        'tensor_indices':
		            self.tensor_indices
		    }


		class Layer(tf_base_layers.Layer):
		  """Abstract base layer class.

		  # Properties
		      name: String, must be unique within a model.
		      input_spec: List of InputSpec class instances
		          each entry describes one required input:
		              - ndim
		              - dtype
		          A layer with `n` input tensors must have
		          an `input_spec` of length `n`.
		      trainable: Boolean, whether the layer weights
		          will be updated during training.
		      uses_learning_phase: Whether any operation
		          of the layer uses `K.in_training_phase()`
		          or `K.in_test_phase()`.
		      input_shape: Shape tuple. Provided for convenience,
		          but note that there may be cases in which this
		          attribute is ill-defined (e.g. a shared layer
		          with multiple input shapes), in which case
		          requesting `input_shape` will raise an Exception.
		          Prefer using `layer.get_input_shape_for(input_shape)`,
		          or `layer.get_input_shape_at(node_index)`.
		      output_shape: Shape tuple. See above.
		      inbound_nodes: List of nodes.
		      outbound_nodes: List of nodes.
		      input, output: Input/output tensor(s). Note that if the layer is used
		          more than once (shared layer), this is ill-defined
		          and will raise an exception. In such cases, use
		          `layer.get_input_at(node_index)`.
		      input_mask, output_mask: Same as above, for masks.
		      trainable_weights: List of variables.
		      non_trainable_weights: List of variables.
		      weights: The concatenation of the lists trainable_weights and
		          non_trainable_weights (in this order).
		      constraints: Dict mapping weights to constraints.

		  # Methods
		      call(x, mask=None): Where the layer's logic lives.
		      __call__(x, mask=None): Wrapper around the layer logic (`call`).
		          If x is a Keras tensor:
		              - Connect current layer with last layer from tensor:
		                  `self._add_inbound_node(last_layer)`
		              - Add layer to tensor history
		          If layer is not built:
		              - Build from inputs shape
		      get_weights()
		      set_weights(weights)
		      get_config()
		      count_params()
		      _compute_output_shape(input_shape)
		      compute_mask(x, mask)
		      get_input_at(node_index)
		      get_output_at(node_index)
		      get_input_shape_at(node_index)
		      get_output_shape_at(node_index)
		      get_input_mask_at(node_index)
		      get_output_mask_at(node_index)

		  # Class Methods
		      from_config(config)

		  # Internal methods:
		      build(input_shape)
		      _add_inbound_node(layer, index=0)
		  """

		  def __init__(self, **kwargs):
		    # These properties should be set by the user via keyword arguments.
		    # note that 'dtype', 'input_shape' and 'batch_input_shape'
		    # are only applicable to input layers: do not pass these keywords
		    # to non-input layers.
		    allowed_kwargs = {
		        'input_shape',
		        'batch_input_shape',
		        'batch_size',
		        'dtype',
		        'name',
		        'trainable',
		        'weights',
		    }
		    # Validate optional keyword arguments.
		    for kwarg in kwargs:
		      if kwarg not in allowed_kwargs:
		        raise TypeError('Keyword argument not understood:', kwarg)

		    # Get layer name.
		    name = kwargs.get('name')

		    # Get `trainable` status.
		    trainable = kwargs.get('trainable', True)

		    # Get `dtype`.
		    dtype = kwargs.get('dtype')
		    if dtype is None:
		      dtype = K.floatx()

		    # Call super, which will set all properties common to Keras layers
		    # and core TF layers.
		    super(Layer, self).__init__(name=name, dtype=dtype, trainable=trainable)

		    # Add properties that are Keras-only for now.
		    self.input_spec = None
		    self.supports_masking = False
		    self._constraints = {}  # dict {tensor: constraint instance}

		    # These lists will be filled via successive calls
		    # to self._add_inbound_node().
		    self.inbound_nodes = []
		    self.outbound_nodes = []

		    # Manage input shape information if passed.
		    if 'input_shape' in kwargs or 'batch_input_shape' in kwargs:
		      # In this case we will later create an input layer
		      # to insert before the current layer
		      if 'batch_input_shape' in kwargs:
		        batch_input_shape = tuple(kwargs['batch_input_shape'])
		      elif 'input_shape' in kwargs:
		        if 'batch_size' in kwargs:
		          batch_size = kwargs['batch_size']
		        else:
		          batch_size = None
		        batch_input_shape = (batch_size,) + tuple(kwargs['input_shape'])
		      self.batch_input_shape = batch_input_shape

		    # Manage initial weight values if passed.
		    if 'weights' in kwargs:
		      self._initial_weights = kwargs['weights']
		    else:
		      self._initial_weights = None

		  @property
		  def constraints(self):
		    return self._constraints

		  @constraints.setter
		  def constraints(self, constraints):
		    self._constraints = constraints

		  def add_weight(self,
		                 name,
		                 shape,
		                 dtype=None,
		                 initializer=None,
		                 regularizer=None,
		                 trainable=True,
		                 constraint=None):
		    """Adds a weight variable to the layer.

		    Arguments:
		        name: String, the name for the weight variable.
		        shape: The shape tuple of the weight.
		        dtype: The dtype of the weight.
		        initializer: An Initializer instance (callable).
		        regularizer: An optional Regularizer instance.
		        trainable: A boolean, whether the weight should
		            be trained via backprop or not (assuming
		            that the layer itself is also trainable).
		        constraint: An optional Constraint instance.

		    Returns:
		        The created weight variable.
		    """
		    if dtype is None:
		      dtype = K.floatx()
		    weight = self.add_variable(
		        name, shape, dtype=dtype,
		        initializer=initializer, regularizer=regularizer, trainable=trainable)
		    if constraint is not None:
		      self.constraints[weight] = constraint
		    return weight

		  def call(self, inputs, **kwargs):  # pylint: disable=unused-argument
		    """This is where the layer's logic lives.

		    Arguments:
		        inputs: Input tensor, or list/tuple of input tensors.
		        **kwargs: Additional keyword arguments.

		    Returns:
		        A tensor or list/tuple of tensors.
		    """
		    return inputs

		  # 0. call tf.keras.Layer.__call__() on input tensor
		  # 1. tf Layer __call__ is called
		  # 2. tf.keras.Conv2D.build() is called
		  # 3. link to tf.layers.Conv2D.build is called, inside, create kernel tensor and bias tensor
		  # 4. back to tf.keras.Conv2D.build, constraints attributes are added
		  # 5. tf.layers.convolutional.Conv2D.call is called:
		  		# create outputs from nn.convolution()
				# add bias tensor onto outputs
				# run activation on outputs
		  # back to tf.layers.__call__():
		  # 6. run _add_inbound_node:
		  		# fill in inbound_layers, node_indices, input_tensors, outputs_tensors based on info from input_tensors
				# create a node and store this node inside inbound_layer
				# add _keras_history to output_tensors
		  # 7. other attributes added

		  def __call__(self, inputs, **kwargs):
		    """Wrapper around self.call(), for handling internal references.

		    If a Keras tensor is passed:
		        - We call self._add_inbound_node().
		        - If necessary, we `build` the layer to match
		            the shape of the input(s).
		        - We update the _keras_history of the output tensor(s)
		            with the current layer.
		            This is done as part of _add_inbound_node().

		    Arguments:
		        inputs: Can be a tensor or list/tuple of tensors.
		        **kwargs: Additional keyword arguments to be passed to `call()`.

		    Returns:
		        Output of the layer's `call` method.

		    Raises:
		        ValueError: in case the layer is missing shape information
		            for its `build` call.
		    """
		    if isinstance(inputs, list):
		      inputs = inputs[:]

		    # Handle mask propagation.
		    previous_mask = _collect_previous_mask(inputs)
		    user_kwargs = copy.copy(kwargs)
		    if not _is_all_none(previous_mask):
		      # The previous layer generated a mask.
		      if 'mask' in tf_inspect.getargspec(self.call).args:
		        if 'mask' not in kwargs:
		          # If mask is explicitly passed to __call__,
		          # we should override the default mask.
		          kwargs['mask'] = previous_mask

		    # Actually call the layer (optionally building it).
		    output = super(Layer, self).__call__(inputs, **kwargs)

		    # Handle mask computation.
		    with K.name_scope(self.name):
		      output_mask = self.compute_mask(inputs, previous_mask)

		    # If the layer returns tensors from its inputs, unmodified,
		    # we copy them to avoid loss of tensor metadata.
		    output_ls = _to_list(output)
		    inputs_ls = _to_list(inputs)
		    output_ls_copy = []
		    for x in output_ls:
		      if x in inputs_ls:
		        x = K.identity(x)
		      output_ls_copy.append(x)
		    if len(output_ls_copy) == 1:
		      output = output_ls_copy[0]
		    else:
		      output = output_ls_copy

		    # Add an inbound node to the layer, so that it keeps track
		    # of the call and of all new variables created during the call.
		    # This also updates the layer history of the output tensor(s).
		    # If the input tensor(s) had not previous Keras history,
		    # this does nothing.
		    self._add_inbound_node(
		        input_tensors=inputs,
		        output_tensors=output,
		        input_masks=previous_mask,
		        output_masks=output_mask,
		        arguments=user_kwargs)

		    # Optionally load weight values that were specified at layer instantiation.
		    if hasattr(self, '_initial_weights') and self._initial_weights is not None:
		      self.set_weights(self._initial_weights)
		      del self._initial_weights
		    return output

		  def _add_inbound_node(self,
		                        input_tensors,
		                        output_tensors,
		                        input_masks,
		                        output_masks,
		                        arguments=None):
		    """Internal method to create an inbound node for the layer.

		    Arguments:
		        input_tensors: list of input tensors.
		        output_tensors: list of output tensors.
		        input_masks: list of input masks (a mask can be a tensor, or None).
		        output_masks: list of output masks (a mask can be a tensor, or None).
		        arguments: dictionary of keyword arguments that were passed to the
		            `call` method of the layer at the call that created the node.
		    """
		    input_tensors = _to_list(input_tensors)
		    output_tensors = _to_list(output_tensors)
		    input_masks = _to_list(input_masks)
		    output_masks = _to_list(output_masks)

		    # Collect input tensor(s) coordinates.
		    inbound_layers = []
		    node_indices = []
		    tensor_indices = []
		    for x in input_tensors:
		      if hasattr(x, '_keras_history'):
		        inbound_layer, node_index, tensor_index = x._keras_history
		        inbound_layers.append(inbound_layer)
		        node_indices.append(node_index)
		        tensor_indices.append(tensor_index)
		      else:
		        inbound_layers.append(None)
		        node_indices.append(None)
		        tensor_indices.append(None)

		    # Create node, add it to inbound nodes.
		    Node(
		        self,
		        inbound_layers=inbound_layers,
		        node_indices=node_indices,
		        tensor_indices=tensor_indices,
		        input_tensors=input_tensors,
		        output_tensors=output_tensors,
		        input_masks=input_masks,
		        output_masks=output_masks,
		        arguments=arguments)

		    # Update tensor history and `_uses_learning_phase`.
		    for i in range(len(output_tensors)):
		      uses_lp = any(
		          [getattr(x, '_uses_learning_phase', False) for x in input_tensors])
		      uses_lp = getattr(self, 'uses_learning_phase', False) or uses_lp
		      output_tensors[i]._uses_learning_phase = getattr(
		          output_tensors[i], '_uses_learning_phase', False) or uses_lp
		      output_tensors[i]._keras_history = (self, len(self.inbound_nodes) - 1, i)

		  def _compute_output_shape(self, input_shape):
		    """Computes the output shape of the layer.

		    Assumes that the layer will be built
		    to match that input shape provided.

		    Arguments:
		        input_shape: Shape tuple (tuple of integers)
		            or list of shape tuples (one per output tensor of the layer).
		            Shape tuples can include None for free dimensions,
		            instead of an integer.

		    Returns:
		        An input shape tuple.
		    """
		    if isinstance(input_shape, list):
		      return [tensor_shape.TensorShape(shape) for shape in input_shape]
		    else:
		      return tensor_shape.TensorShape(input_shape)

		  def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
		    """Computes an output mask tensor.

		    Arguments:
		        inputs: Tensor or list of tensors.
		        mask: Tensor or list of tensors.

		    Returns:
		        None or a tensor (or list of tensors,
		            one per output tensor of the layer).
		    """
		    if not self.supports_masking:
		      if mask is not None:
		        if isinstance(mask, list):
		          if any(m is not None for m in mask):
		            raise TypeError('Layer ' + self.name + ' does not support masking, '
		                            'but was passed an input_mask: ' + str(mask))
		        else:
		          raise TypeError('Layer ' + self.name + ' does not support masking, '
		                          'but was passed an input_mask: ' + str(mask))
		      # masking not explicitly supported: return None as mask
		      return None
		    # if masking is explicitly supported, by default
		    # carry over the input mask
		    return mask

		  def build(self, input_shape):  # pylint: disable=unused-argument
		    """Creates the layer weights.

		    Must be implemented on all layers that have weights.

		    Arguments:
		        input_shape: Keras tensor (future input to layer)
		            or list/tuple of Keras tensors to reference
		            for weight shape computations.
		    """
		    self.built = True

		  def _get_node_attribute_at_index(self, node_index, attr, attr_name):
		    """Retrieves an attribute (e.g. input_tensors) from a node.

		    This is used to implement the methods:
		        - get_input_shape_at
		        - get_output_shape_at
		        - get_input_at
		        etc...

		    Arguments:
		        node_index: Integer index of the node from which
		            to retrieve the attribute.
		        attr: Exact node attribute name.
		        attr_name: Human-readable attribute name, for error messages.

		    Returns:
		        The layer's attribute `attr` at the node of index `node_index`.

		    Raises:
		        RuntimeError: If the layer has no inbound nodes.
		        ValueError: If the index is does not match any node.
		    """
		    if not self.inbound_nodes:
		      raise RuntimeError('The layer has never been called '
		                         'and thus has no defined ' + attr_name + '.')
		    if not len(self.inbound_nodes) > node_index:
		      raise ValueError('Asked to get ' + attr_name + ' at node ' +
		                       str(node_index) + ', but the layer has only ' +
		                       str(len(self.inbound_nodes)) + ' inbound nodes.')
		    values = getattr(self.inbound_nodes[node_index], attr)
		    if len(values) == 1:
		      return values[0]
		    else:
		      return values

		  def get_input_shape_at(self, node_index):
		    """Retrieves the input shape(s) of a layer at a given node.

		    Arguments:
		        node_index: Integer, index of the node
		            from which to retrieve the attribute.
		            E.g. `node_index=0` will correspond to the
		            first time the layer was called.

		    Returns:
		        A shape tuple
		        (or list of shape tuples if the layer has multiple inputs).
		    """
		    return self._get_node_attribute_at_index(node_index, 'input_shapes',
		                                             'input shape')

		  def get_output_shape_at(self, node_index):
		    """Retrieves the output shape(s) of a layer at a given node.

		    Arguments:
		        node_index: Integer, index of the node
		            from which to retrieve the attribute.
		            E.g. `node_index=0` will correspond to the
		            first time the layer was called.

		    Returns:
		        A shape tuple
		        (or list of shape tuples if the layer has multiple outputs).
		    """
		    return self._get_node_attribute_at_index(node_index, 'output_shapes',
		                                             'output shape')

		  def get_input_at(self, node_index):
		    """Retrieves the input tensor(s) of a layer at a given node.

		    Arguments:
		        node_index: Integer, index of the node
		            from which to retrieve the attribute.
		            E.g. `node_index=0` will correspond to the
		            first time the layer was called.

		    Returns:
		        A tensor (or list of tensors if the layer has multiple inputs).
		    """
		    return self._get_node_attribute_at_index(node_index, 'input_tensors',
		                                             'input')

		  def get_output_at(self, node_index):
		    """Retrieves the output tensor(s) of a layer at a given node.

		    Arguments:
		        node_index: Integer, index of the node
		            from which to retrieve the attribute.
		            E.g. `node_index=0` will correspond to the
		            first time the layer was called.

		    Returns:
		        A tensor (or list of tensors if the layer has multiple outputs).
		    """
		    return self._get_node_attribute_at_index(node_index, 'output_tensors',
		                                             'output')

		  def get_input_mask_at(self, node_index):
		    """Retrieves the input mask tensor(s) of a layer at a given node.

		    Arguments:
		        node_index: Integer, index of the node
		            from which to retrieve the attribute.
		            E.g. `node_index=0` will correspond to the
		            first time the layer was called.

		    Returns:
		        A mask tensor
		        (or list of tensors if the layer has multiple inputs).
		    """
		    return self._get_node_attribute_at_index(node_index, 'input_masks',
		                                             'input mask')

		  def get_output_mask_at(self, node_index):
		    """Retrieves the output mask tensor(s) of a layer at a given node.

		    Arguments:
		        node_index: Integer, index of the node
		            from which to retrieve the attribute.
		            E.g. `node_index=0` will correspond to the
		            first time the layer was called.

		    Returns:
		        A mask tensor
		        (or list of tensors if the layer has multiple outputs).
		    """
		    return self._get_node_attribute_at_index(node_index, 'output_masks',
		                                             'output mask')

		  @property
		  def input(self):
		    """Retrieves the input tensor(s) of a layer.

		    Only applicable if the layer has exactly one inbound node,
		    i.e. if it is connected to one incoming layer.

		    Returns:
		        Input tensor or list of input tensors.

		    Raises:
		        AttributeError: if the layer is connected to
		        more than one incoming layers.
		    """
		    if len(self.inbound_nodes) > 1:
		      raise AttributeError('Layer ' + self.name +
		                           ' has multiple inbound nodes, '
		                           'hence the notion of "layer input" '
		                           'is ill-defined. '
		                           'Use `get_input_at(node_index)` instead.')
		    elif not self.inbound_nodes:
		      raise AttributeError('Layer ' + self.name +
		                           ' is not connected, no input to return.')
		    return self._get_node_attribute_at_index(0, 'input_tensors', 'input')

		  @property
		  def output(self):
		    """Retrieves the output tensor(s) of a layer.

		    Only applicable if the layer has exactly one inbound node,
		    i.e. if it is connected to one incoming layer.

		    Returns:
		        Output tensor or list of output tensors.

		    Raises:
		        AttributeError: if the layer is connected to
		        more than one incoming layers.
		    """
		    if not self.inbound_nodes:
		      raise AttributeError('Layer ' + self.name + ' has no inbound nodes.')
		    if len(self.inbound_nodes) > 1:
		      raise AttributeError('Layer ' + self.name +
		                           ' has multiple inbound nodes, '
		                           'hence the notion of "layer output" '
		                           'is ill-defined. '
		                           'Use `get_output_at(node_index)` instead.')
		    return self._get_node_attribute_at_index(0, 'output_tensors', 'output')

		  @property
		  def input_mask(self):
		    """Retrieves the input mask tensor(s) of a layer.

		    Only applicable if the layer has exactly one inbound node,
		    i.e. if it is connected to one incoming layer.

		    Returns:
		        Input mask tensor (potentially None) or list of input
		        mask tensors.

		    Raises:
		        AttributeError: if the layer is connected to
		        more than one incoming layers.
		    """
		    if len(self.inbound_nodes) != 1:
		      raise AttributeError('Layer ' + self.name +
		                           ' has multiple inbound nodes, ' +
		                           'hence the notion of "layer input mask" '
		                           'is ill-defined. '
		                           'Use `get_input_mask_at(node_index)` '
		                           'instead.')
		    return self._get_node_attribute_at_index(0, 'input_masks', 'input mask')

		  @property
		  def output_mask(self):
		    """Retrieves the output mask tensor(s) of a layer.

		    Only applicable if the layer has exactly one inbound node,
		    i.e. if it is connected to one incoming layer.

		    Returns:
		        Output mask tensor (potentially None) or list of output
		        mask tensors.

		    Raises:
		        AttributeError: if the layer is connected to
		        more than one incoming layers.
		    """
		    if len(self.inbound_nodes) != 1:
		      raise AttributeError('Layer ' + self.name +
		                           ' has multiple inbound nodes, '
		                           'hence the notion of "layer output mask" '
		                           'is ill-defined. '
		                           'Use `get_output_mask_at(node_index)` '
		                           'instead.')
		    return self._get_node_attribute_at_index(0, 'output_masks', 'output mask')

		  @property
		  def input_shape(self):
		    """Retrieves the input shape(s) of a layer.

		    Only applicable if the layer has exactly one inbound node,
		    i.e. if it is connected to one incoming layer.

		    Returns:
		        Input shape, as `TensorShape`
		        (or list of `TensorShape`, one tuple per input tensor).

		    Raises:
		        AttributeError: if the layer is connected to
		        more than one incoming layers.
		    """
		    if not self.inbound_nodes:
		      raise AttributeError('The layer has never been called '
		                           'and thus has no defined input shape.')
		    all_input_shapes = set(
		        [str(node.input_shapes) for node in self.inbound_nodes])
		    if len(all_input_shapes) == 1:
		      input_shapes = self.inbound_nodes[0].input_shapes
		      if len(input_shapes) == 1:
		        return tuple(tensor_shape.TensorShape(input_shapes[0]).as_list())
		      else:
		        return [
		            tuple(tensor_shape.TensorShape(shape).as_list())
		            for shape in input_shapes
		        ]
		    else:
		      raise AttributeError('The layer "' + str(self.name) +
		                           ' has multiple inbound nodes, '
		                           'with different input shapes. Hence '
		                           'the notion of "input shape" is '
		                           'ill-defined for the layer. '
		                           'Use `get_input_shape_at(node_index)` '
		                           'instead.')

		  @property
		  def output_shape(self):
		    """Retrieves the output shape(s) of a layer.

		    Only applicable if the layer has one inbound node,
		    or if all inbound nodes have the same output shape.

		    Returns:
		        Output shape, as `TensorShape`
		        (or list of `TensorShape`, one tuple per output tensor).

		    Raises:
		        AttributeError: if the layer is connected to
		        more than one incoming layers.
		    """
		    if not self.inbound_nodes:
		      raise AttributeError('The layer has never been called '
		                           'and thus has no defined output shape.')
		    all_output_shapes = set(
		        [str(node.output_shapes) for node in self.inbound_nodes])
		    if len(all_output_shapes) == 1:
		      output_shapes = self.inbound_nodes[0].output_shapes
		      if len(output_shapes) == 1:
		        return tuple(tensor_shape.TensorShape(output_shapes[0]).as_list())
		      else:
		        return [
		            tuple(tensor_shape.TensorShape(shape).as_list())
		            for shape in output_shapes
		        ]
		    else:
		      raise AttributeError('The layer "' + str(self.name) +
		                           ' has multiple inbound nodes, '
		                           'with different output shapes. Hence '
		                           'the notion of "output shape" is '
		                           'ill-defined for the layer. '
		                           'Use `get_output_shape_at(node_index)` '
		                           'instead.')

		  def set_weights(self, weights):
		    """Sets the weights of the layer, from Numpy arrays.

		    Arguments:
		        weights: a list of Numpy arrays. The number
		            of arrays and their shape must match
		            number of the dimensions of the weights
		            of the layer (i.e. it should match the
		            output of `get_weights`).

		    Raises:
		        ValueError: If the provided weights list does not match the
		            layer's specifications.
		    """
		    params = self.weights
		    if len(params) != len(weights):
		      raise ValueError('You called `set_weights(weights)` on layer "' +
		                       self.name + '" with a  weight list of length ' +
		                       str(len(weights)) + ', but the layer was expecting ' +
		                       str(len(params)) + ' weights. Provided weights: ' +
		                       str(weights)[:50] + '...')
		    if not params:
		      return
		    weight_value_tuples = []
		    param_values = K.batch_get_value(params)
		    for pv, p, w in zip(param_values, params, weights):
		      if pv.shape != w.shape:
		        raise ValueError('Layer weight shape ' + str(pv.shape) +
		                         ' not compatible with '
		                         'provided weight shape ' + str(w.shape))
		      weight_value_tuples.append((p, w))
		    K.batch_set_value(weight_value_tuples)

		  def get_weights(self):
		    """Returns the current weights of the layer.

		    Returns:
		        Weights values as a list of numpy arrays.
		    """
		    params = self.weights
		    return K.batch_get_value(params)

		  def get_config(self):
		    """Returns the config of the layer.

		    A layer config is a Python dictionary (serializable)
		    containing the configuration of a layer.
		    The same layer can be reinstantiated later
		    (without its trained weights) from this configuration.

		    The config of a layer does not include connectivity
		    information, nor the layer class name. These are handled
		    by `Container` (one layer of abstraction above).

		    Returns:
		        Python dictionary.
		    """
		    config = {'name': self.name, 'trainable': self.trainable}
		    if hasattr(self, 'batch_input_shape'):
		      config['batch_input_shape'] = self.batch_input_shape
		    if hasattr(self, 'dtype'):
		      config['dtype'] = self.dtype
		    return config

		  @classmethod
		  def from_config(cls, config):
		    """Creates a layer from its config.

		    This method is the reverse of `get_config`,
		    capable of instantiating the same layer from the config
		    dictionary. It does not handle layer connectivity
		    (handled by Container), nor weights (handled by `set_weights`).

		    Arguments:
		        config: A Python dictionary, typically the
		            output of get_config.

		    Returns:
		        A layer instance.
		    """
		    return cls(**config)

		  def count_params(self):
		    """Count the total number of scalars composing the weights.

		    Returns:
		        An integer count.

		    Raises:
		        RuntimeError: if the layer isn't yet built
		            (in which case its weights aren't yet defined).
		    """
		    if not self.built:
		      if self.__class__.__name__ == 'Sequential':
		        self.build()  # pylint: disable=no-value-for-parameter
		      else:
		        raise RuntimeError('You tried to call `count_params` on ' + self.name +
		                           ', but the layer isn\'t built. '
		                           'You can build it manually via: `' + self.name +
		                           '.build(batch_input_shape)`.')
		    return sum([K.count_params(p) for p in self.weights])


		class InputLayer(Layer):
		  """Layer to be used as an entry point into a graph.

		  It can either wrap an existing tensor (pass an `input_tensor` argument)
		  or create its a placeholder tensor (pass arguments `input_shape`
		  or `batch_input_shape` as well as `dtype`).

		  Arguments:
		      input_shape: Shape tuple, not including the batch axis.
		      batch_size: Optional input batch size (integer or None).
		      batch_input_shape: Shape tuple, including the batch axis.
		      dtype: Datatype of the input.
		      input_tensor: Optional tensor to use as layer input
		          instead of creating a placeholder.
		      sparse: Boolean, whether the placeholder created
		          is meant to be sparse.
		      name: Name of the layer (string).
		  """

		  def __init__(self,
		               input_shape=None,
		               batch_size=None,
		               batch_input_shape=None,
		               dtype=None,
		               input_tensor=None,
		               sparse=False,
		               name=None):
		    if not name:
		      prefix = 'input'
		      name = prefix + '_' + str(K.get_uid(prefix))
		    if not dtype:
		      if input_tensor is None:
		        dtype = K.floatx()
		      else:
		        dtype = K.dtype(input_tensor)
		    super(InputLayer, self).__init__(dtype=dtype, name=name)
		    self.built = True
		    self.sparse = sparse

		    if input_shape and batch_input_shape:
		      raise ValueError('Only provide the input_shape OR '
		                       'batch_input_shape argument to '
		                       'InputLayer, not both at the same time.')
		    if input_tensor is not None:
		      # Attempt automatic input shape inference.
		      try:
		        batch_input_shape = K.int_shape(input_tensor)
		      except TypeError:
		        if not input_shape and not batch_input_shape:
		          raise ValueError('InputLayer was provided '
		                           'an input_tensor argument, '
		                           'but its input shape cannot be '
		                           'automatically inferred. '
		                           'You should pass an input_shape or '
		                           'batch_input_shape argument.')
		    if not batch_input_shape:
		      if not input_shape:
		        raise ValueError('An Input layer should be passed either '
		                         'a `batch_input_shape` or an `input_shape`.')
		      else:
		        batch_input_shape = (batch_size,) + tuple(input_shape)
		    else:
		      batch_input_shape = tuple(batch_input_shape)
		    self.batch_input_shape = batch_input_shape

			# create a placeholder as input tensor
		    if input_tensor is None:
		      self.is_placeholder = True
		      input_tensor = K.placeholder(
		          shape=batch_input_shape,
		          dtype=dtype,
		          sparse=self.sparse,
		          name=self.name)
		    else:
		      self.is_placeholder = False

		    # save InputLayer as history inside input_tensor
		    input_tensor._uses_learning_phase = False
		    input_tensor._keras_history = (self, 0, 0)

			# Create an input node and add to Inputlayer.outbound_node
		    Node(
		        self,
		        inbound_layers=[],
		        node_indices=[],
		        tensor_indices=[],
		        input_tensors=[input_tensor],
		        output_tensors=[input_tensor],
		        input_masks=[None],
		        output_masks=[None])

		  def get_config(self):
		    config = {
		        'batch_input_shape': self.batch_input_shape,
		        'dtype': self.dtype,
		        'sparse': self.sparse,
		        'name': self.name
		    }
		    return config

		# 1. Layer of tf, build a graph and other attributes
		# 2. Layer of tf.keras, build more attributes
		# 3. InputLayer of tf.keras, build input tensor as placeholder, and
		# 4. save InputLayer as content of history stored in input_tensor._keras_history
		# 5. create an input node and store input_tensor as input-output-tensors, store InputLayer as inbound_layers, outbound_layers
		# 6. then save this Node inside outbound_layers.inbound_nodes or inbound_layers.outbound_nodes
		# 7. input_tensor can be accessed through InputLayer->Node->tensors
		# 8. return input_tensor

		def Input(  # pylint: disable=invalid-name
		    shape=None,
		    batch_shape=None,
		    name=None,
		    dtype=K.floatx(),
		    sparse=False,
		    tensor=None):
		  """
		  `Input()` is used to instantiate a Keras tensor.

			  A Keras tensor is a tensor object from the underlying backend
			  (Theano or TensorFlow), which we augment with certain
			  attributes that allow us to build a Keras model
			  just by knowing the inputs and outputs of the model.

			  For instance, if a, b and c are Keras tensors,
			  it becomes possible to do:
			  `model = Model(input=[a, b], output=c)`

			  The added Keras attribute is:
			      `_keras_history`: Last layer applied to the tensor.
			          the entire layer graph is retrievable from that layer,
			          recursively.

		  Arguments:
		      shape: A shape tuple (integer), not including the batch size.
		          For instance, `shape=(32,)` indicates that the expected input
		          will be batches of 32-dimensional vectors.
		      batch_shape: A shape tuple (integer), including the batch size.
		          For instance, `batch_shape=(10, 32)` indicates that
		          the expected input will be batches of 10 32-dimensional vectors.
		          `batch_shape=(None, 32)` indicates batches of an arbitrary number
		          of 32-dimensional vectors.
		      name: An optional name string for the layer.
		          Should be unique in a model (do not reuse the same name twice).
		          It will be autogenerated if it isn't provided.
		      dtype: The data type expected by the input, as a string
		          (`float32`, `float64`, `int32`...)
		      sparse: A boolean specifying whether the placeholder
		          to be created is sparse.
		      tensor: Optional existing tensor to wrap into the `Input` layer.
		          If set, the layer will not create a placeholder tensor.

		  Returns:
		      A tensor.

		  Example:

		      ```python
		      # this is a logistic regression in Keras
		      x = Input(shape=(32,))
		      y = Dense(16, activation='softmax')(x)
		      model = Model(x, y)
		      ```
		  """
		  if not batch_shape and tensor is None:
		    assert shape, ('Please provide to Input either a `shape`'
		                   ' or a `batch_shape` argument. Note that '
		                   '`shape` does not include the batch '
		                   'dimension.')
		  if shape and not batch_shape:
		    batch_shape = (None,) + tuple(shape)
		  input_layer = InputLayer(
		      batch_input_shape=batch_shape,
		      name=name,
		      dtype=dtype,
		      sparse=sparse,
		      input_tensor=tensor)
		  # Return tensor including `_keras_history`.
		  # Note that in this case train_output and test_output are the same pointer.
		  outputs = input_layer.inbound_nodes[0].output_tensors
		  if len(outputs) == 1:
		    return outputs[0]
		  else:
		    return outputs


		class Container(Layer):
		  """A Container is a directed acyclic graph of layers.

		  It is the topological form of a "model". A Model
		  is simply a Container with added training routines.

		  # Properties
		      name
		      inputs
		      outputs
		      input_layers
		      output_layers
		      input_spec (list of class instances)
		          each entry describes one required input:
		              - ndim
		              - dtype
		      trainable (boolean)
		      input_shape
		      output_shape
		      inbound_nodes: list of nodes
		      outbound_nodes: list of nodes
		      trainable_weights (list of variables)
		      non_trainable_weights (list of variables)
		      constraints (list of tuples (weight, constraint))

		  # Methods
		      summary
		      get_layer
		      get_weights
		      set_weights
		      get_config
		      compute_output_shape

		  # Class Methods
		      from_config
		  """

		  def __init__(self, inputs, outputs, name=None):  # pylint: disable=super-init-not-called
		    # Handle `name` argument.
		    if not name:
		      prefix = self.__class__.__name__.lower()
		      name = prefix + '_' + str(K.get_uid(prefix))
		    self.name = name
		    self.supports_masking = False
		    self.trainable = True
		    self._per_input_losses = {}
		    self._per_input_updates = {}

		    # The following properties are not actually used by Keras;
		    # they exist for compatibility with TF.
		    self._updates = []
		    self._scope = None
		    self._reuse = None
		    self._base_name = name
		    self._graph = ops.get_default_graph()

		    # Container-specific properties.
		    if isinstance(inputs, (list, tuple)):
		      self.inputs = list(inputs)  # Tensor or list of tensors.
		    else:
		      self.inputs = [inputs]
		    if isinstance(outputs, (list, tuple)):
		      self.outputs = list(outputs)
		    else:
		      self.outputs = [outputs]

		    # Check for redundancy in inputs.
		    inputs_set = set(self.inputs)
		    if len(inputs_set) != len(self.inputs):
		      raise ValueError('The list of inputs passed to the model '
		                       'is redundant. '
		                       'All inputs should only appear once.'
		                       ' Found: ' + str(self.inputs))

		    # List of initial layers (1 to 1 mapping with self.inputs,
		    # hence the same layer might appear twice)
		    self.input_layers = []
		    self.input_layers_node_indices = []
		    self.input_layers_tensor_indices = []
		    # list of layers (1 to 1 mapping with self.inputs,
		    # hence the same layer might appear twice)
		    self.output_layers = []
		    self.output_layers_node_indices = []
		    self.output_layers_tensor_indices = []
		    # all layers in order of horizontal graph traversal.
		    # Entries are unique. Includes input and output layers.
		    self.layers = []

		    # This is for performance optimization
		    # when calling the Container on new inputs.
		    # every time the Container is called on a set on input tensors,
		    # we compute the output tensors,
		    # output masks and output shapes in one pass,
		    # then cache them here. When of of these output is queried later,
		    # we retrieve it from there instead of recomputing it.
		    self._output_mask_cache = {}
		    self._output_tensor_cache = {}
		    self._output_shape_cache = {}

		    # User-provided arguments validation.
		    for x in self.inputs:
		      # Check that x is a Keras tensor.
		      if not hasattr(x, '_keras_history'):
		        cls_name = self.__class__.__name__
		        raise TypeError('Input tensors to a ' + cls_name + ' ' +
		                        'must be Keras tensors. Found: ' + str(x) +
		                        ' (missing Keras metadata).')
		      # Check that x is an input tensor.
		      layer, node_index, tensor_index = x._keras_history
		      if len(layer.inbound_nodes) > 1 or (
		          layer.inbound_nodes and layer.inbound_nodes[0].inbound_layers):
		        cls_name = self.__class__.__name__
		        logging.warning(cls_name + ' inputs must come from '
		                        'a Keras Input layer, '
		                        'they cannot be the output of '
		                        'a previous non-Input layer. '
		                        'Here, a tensor specified as '
		                        'input to "' + self.name + '" was not an Input tensor, '
		                        'it was generated by layer ' + layer.name + '.\n'
		                        'Note that input tensors are '
		                        'instantiated via `tensor = Input(shape)`.\n'
		                        'The tensor that caused the issue was: ' + str(x.name))
		    for x in self.outputs:
		      if not hasattr(x, '_keras_history'):
		        cls_name = self.__class__.__name__
		        raise TypeError('Output tensors to a ' + cls_name + ' must be '
		                        'Keras tensors. Found: ' + str(x))
		    # Build self.output_layers:
		    for x in self.outputs:
		      layer, node_index, tensor_index = x._keras_history
		      self.output_layers.append(layer)
		      self.output_layers_node_indices.append(node_index)
		      self.output_layers_tensor_indices.append(tensor_index)

		    # Fill in the output mask cache.
		    masks = []
		    for x in self.inputs:
		      layer, node_index, tensor_index = x._keras_history
		      node = layer.inbound_nodes[node_index]
		      mask = node.output_masks[tensor_index]
		      masks.append(mask)
		    mask_cache_key = ','.join([str(id(x)) for x in self.inputs])
		    mask_cache_key += '_' + ','.join([str(id(x)) for x in masks])
		    masks = []
		    for x in self.outputs:
		      layer, node_index, tensor_index = x._keras_history
		      node = layer.inbound_nodes[node_index]
		      mask = node.output_masks[tensor_index]
		      masks.append(mask)
		    if len(masks) == 1:
		      mask = masks[0]
		    else:
		      mask = masks
		    self._output_mask_cache[mask_cache_key] = mask

		    # Build self.input_layers:
		    for x in self.inputs:
		      layer, node_index, tensor_index = x._keras_history
		      # It's supposed to be an input layer, so only one node
		      # and one tensor output.
		      assert node_index == 0
		      assert tensor_index == 0
		      self.input_layers.append(layer)
		      self.input_layers_node_indices.append(node_index)
		      self.input_layers_tensor_indices.append(tensor_index)

		    # Build self.input_names and self.output_names.
		    self.input_names = []
		    self.output_names = []
		    self._feed_input_names = []
		    self._feed_inputs = []
		    self._feed_input_shapes = []
		    for i, layer in enumerate(self.input_layers):
		      self.input_names.append(layer.name)
		      if layer.is_placeholder:
		        self._feed_input_names.append(layer.name)
		        self._feed_inputs.append(layer.input)
		        self._feed_input_shapes.append(K.int_shape(self.inputs[i]))
		    for layer in self.output_layers:
		      self.output_names.append(layer.name)

		    self.internal_input_shapes = [K.int_shape(x) for x in self.inputs]
		    self.internal_output_shapes = [K.int_shape(x) for x in self.outputs]

		    # Container_nodes: set of nodes included in the graph
		    # (not all nodes included in the layers
		    # are relevant to the current graph).
		    container_nodes = set()  # ids of all nodes relevant to the Container
		    nodes_depths = {}  # dict {node: depth value}
		    layers_depths = {}  # dict {layer: depth value}
		    layer_indices = {}  # dict {layer: index in traversal}
		    nodes_in_decreasing_depth = []

		    def build_map_of_graph(tensor,
		                           finished_nodes,
		                           nodes_in_progress,
		                           layer=None,
		                           node_index=None,
		                           tensor_index=None):
		      """Builds a map of the graph of layers.

		      This recursively updates the map `layer_indices`,
		      the list `nodes_in_decreasing_depth` and the set `container_nodes`.

		      Arguments:
		          tensor: Some tensor in a graph.
		          finished_nodes: Set of nodes whose subgraphs have been traversed
		              completely. Useful to prevent duplicated work.
		          nodes_in_progress: Set of nodes that are currently active on the
		              recursion stack. Useful to detect cycles.
		          layer: Layer from which `tensor` comes from. If not provided,
		              will be obtained from `tensor._keras_history`.
		          node_index: Node index from which `tensor` comes from.
		          tensor_index: Tensor_index from which `tensor` comes from.

		      Raises:
		          RuntimeError: if a cycle is detected.
		      """
		      if not layer or node_index is None or tensor_index is None:
		        layer, node_index, tensor_index = tensor._keras_history
		      node = layer.inbound_nodes[node_index]

		      # Prevent cycles.
		      if node in nodes_in_progress:
		        raise RuntimeError('The tensor ' + str(tensor) + ' at layer "' +
		                           layer.name + '" is part of a cycle.')

		      # Don't repeat work for shared subgraphs
		      if node in finished_nodes:
		        return

		      node_key = layer.name + '_ib-' + str(node_index)
		      # Update container_nodes.
		      container_nodes.add(node_key)

		      # Store the traversal order for layer sorting.
		      if layer not in layer_indices:
		        layer_indices[layer] = len(layer_indices)

		      nodes_in_progress.add(node)

		      # Propagate to all previous tensors connected to this node.
		      for i in range(len(node.inbound_layers)):
		        x = node.input_tensors[i]
		        layer = node.inbound_layers[i]
		        node_index = node.node_indices[i]
		        tensor_index = node.tensor_indices[i]
		        build_map_of_graph(x, finished_nodes, nodes_in_progress, layer,
		                           node_index, tensor_index)

		      finished_nodes.add(node)
		      nodes_in_progress.remove(node)

		      nodes_in_decreasing_depth.append(node)

		    finished_nodes = set()
		    nodes_in_progress = set()
		    for x in self.outputs:
		      build_map_of_graph(x, finished_nodes, nodes_in_progress)

		    for node in reversed(nodes_in_decreasing_depth):
		      # If the depth is not set, the node has no outbound nodes (depth 0).
		      depth = nodes_depths.setdefault(node, 0)

		      # Update the depth of the corresponding layer
		      previous_depth = layers_depths.get(node.outbound_layer, 0)
		      # If we've seen this layer before at a higher depth,
		      # we should use that depth instead of the node depth.
		      # This is necessary for shared layers that have inputs at different
		      # depth levels in the graph.
		      depth = max(depth, previous_depth)
		      layers_depths[node.outbound_layer] = depth
		      nodes_depths[node] = depth

		      # Update the depth of inbound nodes.
		      for i in range(len(node.inbound_layers)):
		        inbound_layer = node.inbound_layers[i]
		        node_index = node.node_indices[i]
		        inbound_node = inbound_layer.inbound_nodes[node_index]
		        previous_depth = nodes_depths.get(inbound_node, 0)
		        nodes_depths[inbound_node] = max(depth + 1, previous_depth)

		    # Build a dict {depth: list of nodes with this depth}
		    nodes_by_depth = {}
		    for node, depth in nodes_depths.items():
		      if depth not in nodes_by_depth:
		        nodes_by_depth[depth] = []
		      nodes_by_depth[depth].append(node)

		    # Build a dict {depth: list of layers with this depth}
		    layers_by_depth = {}
		    for layer, depth in layers_depths.items():
		      if depth not in layers_by_depth:
		        layers_by_depth[depth] = []
		      layers_by_depth[depth].append(layer)

		    # Get sorted list of layer depths.
		    depth_keys = list(layers_by_depth.keys())
		    depth_keys.sort(reverse=True)

		    # Set self.layers and self.layers_by_depth.
		    layers = []
		    for depth in depth_keys:
		      layers_for_depth = layers_by_depth[depth]
		      # Container.layers needs to have a deterministic order:
		      # here we order them by traversal order.
		      layers_for_depth.sort(key=lambda x: layer_indices[x])
		      for layer in layers_for_depth:
		        layers.append(layer)
		    self.layers = layers
		    self.layers_by_depth = layers_by_depth

		    # Get sorted list of node depths.
		    depth_keys = list(nodes_by_depth.keys())
		    depth_keys.sort(reverse=True)

		    # Check that all tensors required are computable.
		    # computable_tensors: all tensors in the graph
		    # that can be computed from the inputs provided.
		    computable_tensors = []
		    for x in self.inputs:
		      computable_tensors.append(x)

		    layers_with_complete_input = []  # To provide a better error msg.
		    for depth in depth_keys:
		      for node in nodes_by_depth[depth]:
		        layer = node.outbound_layer
		        if layer:
		          for x in node.input_tensors:
		            if x not in computable_tensors:
		              raise RuntimeError('Graph disconnected: '
		                                 'cannot obtain value for tensor ' + str(x) +
		                                 ' at layer "' + layer.name + '". '
		                                 'The following previous layers '
		                                 'were accessed without issue: ' +
		                                 str(layers_with_complete_input))
		          for x in node.output_tensors:
		            computable_tensors.append(x)
		          layers_with_complete_input.append(layer.name)

		    # Set self.nodes and self.nodes_by_depth.
		    self.container_nodes = container_nodes
		    self.nodes_by_depth = nodes_by_depth

		    # Ensure name unicity, which will be crucial for serialization
		    # (since serialized nodes refer to layers by their name).
		    all_names = [layer.name for layer in self.layers]
		    for name in all_names:
		      if all_names.count(name) != 1:
		        raise RuntimeError('The name "' + name + '" is used ' +
		                           str(all_names.count(name)) + ' times in the model. '
		                           'All layer names should be unique.')

		    # Layer parameters.
		    # The new container starts with a single inbound node
		    # for its inputs, and no outbound nodes.
		    self.outbound_nodes = []  # Will be appended to by future calls to __call__
		    self.inbound_nodes = [
		    ]  # Will be appended to below, and by future calls to __call__
		    # Create the node linking internal inputs to internal outputs.
		    Node(
		        outbound_layer=self,
		        inbound_layers=[],
		        node_indices=[],
		        tensor_indices=[],
		        input_tensors=self.inputs,
		        output_tensors=self.outputs,
		        # No container-level masking for now.
		        input_masks=[None for _ in self.inputs],
		        output_masks=[None for _ in self.outputs])
		    self.built = True

		    # The following are implemented as property functions:
		    # self.constraints
		    # self.trainable_weights
		    # self.non_trainable_weights
		    # self.input_spec

		  def get_layer(self, name=None, index=None):
		    """Retrieves a layer based on either its name (unique) or index.

		    Indices are based on order of horizontal graph traversal (bottom-up).

		    Arguments:
		        name: String, name of layer.
		        index: Integer, index of layer.

		    Returns:
		        A layer instance.

		    Raises:
		        ValueError: In case of invalid layer name or index.
		    """
		    # It would be unreliable to build a dictionary
		    # based on layer names, because names can potentially
		    # be changed at any point by the user
		    # without the container being notified of it.
		    if index is not None:
		      if len(self.layers) <= index:
		        raise ValueError('Was asked to retrieve layer at index ' + str(index) +
		                         ' but model only has ' + str(len(self.layers)) +
		                         ' layers.')
		      else:
		        return self.layers[index]
		    else:
		      if not name:
		        raise ValueError('Provide either a layer name or layer index.')
		    layer = None
		    for layer in self.layers:
		      if layer.name == name:
		        return layer
		    if not layer:
		      raise ValueError('No such layer: ' + name)

		  @property
		  def updates(self):
		    """Retrieve the model's updates.

		    Will only include updates that are either
		    unconditional, or conditional on inputs to this model
		    (e.g. will not include updates that depend on tensors
		    that aren't inputs to this model).

		    Returns:
		        A list of update ops.
		    """
		    updates = []
		    for layer in self.layers:
		      if hasattr(layer, 'updates'):
		        # Collect updates that are dependent on inputs
		        # that are part of the model.
		        for node_index, node in enumerate(layer.inbound_nodes):
		          node_key = layer.name + '_ib-' + str(node_index)
		          if node_key in self.container_nodes:
		            # The model owns this layer node.
		            inputs = node.input_tensors
		            updates += layer.get_updates_for(inputs)
		        # Collect unconditional updates.
		        updates += layer.get_updates_for(None)
		    return updates

		  @property
		  def losses(self):
		    """Retrieve the model's losses.

		    Will only include losses that are either
		    unconditional, or conditional on inputs to this model
		    (e.g. will not include losses that depend on tensors
		    that aren't inputs to this model).

		    Returns:
		        A list of loss tensors.
		    """
		    losses = []
		    # Retrieve losses for all internal layers.
		    for layer in self.layers:
		      if hasattr(layer, 'losses'):
		        # Collect losses that are dependent on inputs
		        # that are part of the model.
		        for node_index, node in enumerate(layer.inbound_nodes):
		          node_key = layer.name + '_ib-' + str(node_index)
		          if node_key in self.container_nodes:
		            # The model owns this layer node.
		            inputs = node.input_tensors
		            losses += layer.get_losses_for(inputs)
		        # Collect unconditional losses.
		        losses += layer.get_losses_for(None)
		    # Add any potential unconditional model-level loss.
		    losses += self.get_losses_for(None)
		    return losses

		  @property
		  def uses_learning_phase(self):
		    return any([x._uses_learning_phase for x in self.outputs])

		  @property
		  def stateful(self):
		    return any([(hasattr(layer, 'stateful') and layer.stateful)
		                for layer in self.layers])

		  def reset_states(self):
		    for layer in self.layers:
		      if hasattr(layer, 'reset_states') and getattr(layer, 'stateful', False):
		        layer.reset_states()

		  @property
		  def state_updates(self):
		    """Returns the `updates` from all layers that are stateful.

		    This is useful for separating training updates and
		    state updates, e.g. when we need to update a layer's internal state
		    during prediction.

		    Returns:
		        A list of update ops.
		    """
		    state_updates = []
		    for layer in self.layers:
		      if getattr(layer, 'stateful', False):
		        if hasattr(layer, 'updates'):
		          state_updates += layer.updates
		    return state_updates

		  @property
		  def constraints(self):
		    cons = {}
		    for layer in self.layers:
		      for key, value in layer.constraints.items():
		        if key in cons and cons[key] != value:
		          raise ValueError('Received multiple constraints '
		                           'for one weight tensor: ' + str(key))
		        cons[key] = value
		    return cons

		  @property
		  def trainable_weights(self):
		    if not self.trainable:
		      return []
		    weights = []
		    for layer in self.layers:
		      weights += layer.trainable_weights
		    return weights

		  @property
		  def non_trainable_weights(self):
		    weights = []
		    for layer in self.layers:
		      weights += layer.non_trainable_weights
		    if not self.trainable:
		      trainable_weights = []
		      for layer in self.layers:
		        trainable_weights += layer.trainable_weights
		      return trainable_weights + weights
		    return weights

		  def get_weights(self):
		    """Retrieves the weights of the model.

		    Returns:
		        A flat list of Numpy arrays.
		    """
		    weights = []
		    for layer in self.layers:
		      weights += layer.weights
		    return K.batch_get_value(weights)

		  def set_weights(self, weights):
		    """Sets the weights of the model.

		    Arguments:
		        weights: A list of Numpy arrays with shapes and types matching
		            the output of `model.get_weights()`.
		    """
		    tuples = []
		    for layer in self.layers:
		      num_param = len(layer.weights)
		      layer_weights = weights[:num_param]
		      for sw, w in zip(layer.weights, layer_weights):
		        tuples.append((sw, w))
		      weights = weights[num_param:]
		    K.batch_set_value(tuples)

		  @property
		  def input_spec(self):
		    """Gets the model's input specs.

		    Returns:
		        A list of `InputSpec` instances (one per input to the model)
		            or a single instance if the model has only one input.
		    """
		    specs = []
		    for layer in getattr(self, 'input_layers', []):
		      if layer.input_spec is None:
		        specs.append(None)
		      else:
		        if not isinstance(layer.input_spec, list):
		          raise TypeError('Layer ' + layer.name +
		                          ' has an input_spec attribute that '
		                          'is not a list. We expect a list. '
		                          'Found input_spec = ' + str(layer.input_spec))
		        specs += layer.input_spec
		    if len(specs) == 1:
		      return specs[0]
		    return specs

		  def call(self, inputs, mask=None):
		    """Call the model on new inputs.

		    In this case `call` just reapplies
		    all ops in the graph to the new inputs
		    (e.g. build a new computational graph from the provided inputs).

		    A model is callable on non-Keras tensors.

		    Arguments:
		        inputs: A tensor or list of tensors.
		        mask: A mask or list of masks. A mask can be
		            either a tensor or None (no mask).

		    Returns:
		        A tensor if there is a single output, or
		        a list of tensors if there are more than one outputs.
		    """
		    inputs = _to_list(inputs)
		    if mask is None:
		      masks = [None for _ in range(len(inputs))]
		    else:
		      masks = _to_list(mask)
		    cache_key = ','.join([str(id(x)) for x in inputs])
		    cache_key += '_' + ','.join([str(id(x)) for x in masks])
		    if cache_key in self._output_tensor_cache:
		      return self._output_tensor_cache[cache_key]
		    else:
		      output_tensors, _, _ = self.run_internal_graph(inputs, masks)
		      return output_tensors

		  def compute_mask(self, inputs, mask):
		    inputs = _to_list(inputs)
		    if mask is None:
		      masks = [None for _ in range(len(inputs))]
		    else:
		      masks = _to_list(mask)
		    cache_key = ','.join([str(id(x)) for x in inputs])
		    cache_key += '_' + ','.join([str(id(x)) for x in masks])
		    if cache_key in self._output_mask_cache:
		      return self._output_mask_cache[cache_key]
		    else:
		      _, output_masks, _ = self.run_internal_graph(inputs, masks)
		      return output_masks

		  def _compute_output_shape(self, input_shape):
		    if isinstance(input_shape, list):
		      input_shapes = []
		      for shape in input_shape:
		        if shape is not None:
		          input_shapes.append(tuple(tensor_shape.TensorShape(shape).as_list()))
		        else:
		          input_shapes.append(None)
		    else:
		      if input_shape is not None:
		        input_shapes = [tuple(tensor_shape.TensorShape(input_shape).as_list())]
		      else:
		        input_shapes = [None]

		    if len(input_shapes) != len(self.input_layers):
		      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
		                       ': model has ' + str(len(self.input_layers)) +
		                       ' tensor inputs.')

		    cache_key = ','.join([str(x) for x in input_shapes])
		    if cache_key in self._output_shape_cache:
		      output_shapes = self._output_shape_cache[cache_key]
		      if isinstance(output_shapes, list):
		        if len(output_shapes) == 1:
		          return tensor_shape.TensorShape(output_shapes[0])
		        else:
		          return [tensor_shape.TensorShape(shape) for shape in output_shapes]
		      else:
		        return tensor_shape.TensorShape(output_shapes)
		    else:
		      # Bad luck, we have to run the graph manually.
		      layers_to_output_shapes = {}
		      for i in range(len(input_shapes)):
		        layer = self.input_layers[i]
		        input_shape = input_shapes[i]
		        # It's an input layer: compute_output_shape is identity,
		        # and there is only one node and one tensor output.
		        shape_key = layer.name + '_0_0'
		        layers_to_output_shapes[shape_key] = input_shape

		      depth_keys = list(self.nodes_by_depth.keys())
		      depth_keys.sort(reverse=True)
		      # Iterate over nodes, by depth level.
		      if len(depth_keys) > 1:
		        for depth in depth_keys:
		          nodes = self.nodes_by_depth[depth]
		          for node in nodes:
		            # This is always a single layer, never a list.
		            layer = node.outbound_layer
		            if layer in self.input_layers:
		              # We've already covered the input layers
		              # a few lines above.
		              continue
		            # Potentially redundant list,
		            # same size of node.input_tensors.
		            input_shapes = []
		            for j in range(len(node.inbound_layers)):
		              inbound_layer = node.inbound_layers[j]
		              node_index = node.node_indices[j]
		              tensor_index = node.tensor_indices[j]
		              shape_key = inbound_layer.name + '_%s_%s' % (node_index,
		                                                           tensor_index)
		              input_shape = layers_to_output_shapes[shape_key]
		              input_shapes.append(input_shape)

		            if len(input_shapes) == 1:
		              output_shape = layer._compute_output_shape(input_shapes[0])
		            else:
		              output_shape = layer._compute_output_shape(input_shapes)
		            if isinstance(output_shape, list):
		              output_shapes = [
		                  tuple(tensor_shape.TensorShape(shape).as_list())
		                  for shape in output_shape
		              ]
		            else:
		              output_shapes = [
		                  tuple(tensor_shape.TensorShape(output_shape).as_list())
		              ]

		            node_index = layer.inbound_nodes.index(node)
		            for j in range(len(output_shapes)):
		              shape_key = layer.name + '_%s_%s' % (node_index, j)
		              layers_to_output_shapes[shape_key] = output_shapes[j]

		      # Read final output shapes from layers_to_output_shapes.
		      output_shapes = []
		      output_shape_keys = []
		      for i in range(len(self.output_layers)):
		        layer = self.output_layers[i]
		        node_index = self.output_layers_node_indices[i]
		        tensor_index = self.output_layers_tensor_indices[i]
		        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
		        output_shape_keys.append(shape_key)

		      for i, key in enumerate(output_shape_keys):
		        assert key in layers_to_output_shapes
		        output_shapes.append(layers_to_output_shapes[key])
		      # Store in cache.
		      self._output_shape_cache[cache_key] = output_shapes
		      if isinstance(output_shapes, list):
		        if len(output_shapes) == 1:
		          return tensor_shape.TensorShape(output_shapes[0])
		        else:
		          return [tensor_shape.TensorShape(shape) for shape in output_shapes]
		      else:
		        return tensor_shape.TensorShape(output_shapes)

		  def run_internal_graph(self, inputs, masks=None):
		    """Computes output tensors for new inputs.

		    # Note:
		        - Expects `inputs` to be a list (potentially with 1 element).
		        - Can be run on non-Keras tensors.

		    Arguments:
		        inputs: List of tensors
		        masks: List of masks (tensors or None).

		    Returns:
		        Three lists: output_tensors, output_masks, output_shapes
		    """
		    if masks is None:
		      masks = [None for _ in range(len(inputs))]

		    # Dictionary mapping reference tensors to tuples
		    # (computed tensor, compute mask)
		    # we assume a 1:1 mapping from tensor to mask
		    # TODO(fchollet): raise exception when a `.compute_mask()` call
		    # does not return a list the same size as `call`
		    tensor_map = {}
		    for x, y, mask in zip(self.inputs, inputs, masks):
		      tensor_map[str(id(x))] = (y, mask)

		    depth_keys = list(self.nodes_by_depth.keys())
		    depth_keys.sort(reverse=True)
		    for depth in depth_keys:
		      nodes = self.nodes_by_depth[depth]
		      for node in nodes:
		        # This is always a single layer, never a list.
		        layer = node.outbound_layer

		        reference_input_tensors = node.input_tensors
		        reference_output_tensors = node.output_tensors

		        # If all previous input tensors are available in tensor_map,
		        # then call node.inbound_layer on them.
		        computed_data = []  # List of tuples (input, mask).
		        for x in reference_input_tensors:
		          if str(id(x)) in tensor_map:
		            computed_data.append(tensor_map[str(id(x))])

		        if len(computed_data) == len(reference_input_tensors):
		          # call layer
		          with K.name_scope(layer.name):
		            if node.arguments:
		              kwargs = node.arguments
		            else:
		              kwargs = {}
		            if len(computed_data) == 1:
		              computed_tensor, computed_mask = computed_data[0]
		              if 'mask' in tf_inspect.getargspec(layer.call).args:
		                if 'mask' not in kwargs:
		                  kwargs['mask'] = computed_mask
		              output_tensors = _to_list(layer.call(computed_tensor, **kwargs))
		              output_masks = _to_list(
		                  layer.compute_mask(computed_tensor, computed_mask))
		              computed_tensors = [computed_tensor]
		              computed_masks = [computed_mask]
		            else:
		              computed_tensors = [x[0] for x in computed_data]
		              computed_masks = [x[1] for x in computed_data]
		              if 'mask' in tf_inspect.getargspec(layer.call).args:
		                if 'mask' not in kwargs:
		                  kwargs['mask'] = computed_masks
		              output_tensors = _to_list(layer.call(computed_tensors, **kwargs))
		              output_masks = _to_list(
		                  layer.compute_mask(computed_tensors, computed_masks))

		            # Apply activity regularizer if any:
		            if hasattr(layer, 'activity_regularizer'
		                      ) and layer.activity_regularizer is not None:
		              regularization_losses = [
		                  layer.activity_regularizer(x) for x in computed_tensors
		              ]
		              layer.add_loss(regularization_losses, computed_tensors)

		          # Update model updates and losses:
		          # Keep track of updates that depend on the inputs
		          # (e.g. BN updates).
		          self.add_update(layer.get_updates_for(computed_tensors), inputs)
		          # Keep track of unconditional updates (e.g. a counter).
		          self.add_update(layer.get_updates_for(None), None)
		          # Keep track of losses that depend on the inputs
		          # (e.g. activity regularizers).
		          self.add_loss(layer.get_losses_for(computed_tensors), inputs)
		          # Keep track of unconditional losses
		          # (e.g. weight regularizers).
		          self.add_loss(layer.get_losses_for(None), None)

		          # Update `_uses_learning_phase`.
		          if len(computed_tensors) == 1:
		            uses_learning_phase = getattr(computed_tensors[0],
		                                          '_uses_learning_phase', False)
		          else:
		            uses_learning_phase = any([
		                getattr(x, '_uses_learning_phase', False)
		                for x in computed_tensors
		            ])
		          for x in output_tensors:
		            x._uses_learning_phase = getattr(x, '_uses_learning_phase',
		                                             False) or uses_learning_phase

		          # Update tensor_map.
		          for x, y, mask in zip(reference_output_tensors, output_tensors,
		                                output_masks):
		            tensor_map[str(id(x))] = (y, mask)

		    output_tensors = []
		    output_masks = []
		    output_shapes = []
		    for x in self.outputs:
		      assert str(id(x)) in tensor_map, 'Could not compute output ' + str(x)
		      tensor, mask = tensor_map[str(id(x))]
		      output_shapes.append(K.int_shape(x))
		      output_tensors.append(tensor)
		      output_masks.append(mask)

		    # Update cache;
		    # keys are based on ids on input tensors and inputs masks.
		    cache_key = ','.join([str(id(x)) for x in inputs])
		    cache_key += '_' + ','.join([str(id(x)) for x in masks])

		    if len(output_tensors) == 1:
		      output_tensors = output_tensors[0]
		      self._output_tensor_cache[cache_key] = output_tensors
		    else:
		      self._output_tensor_cache[cache_key] = output_tensors

		    if len(output_masks) == 1:
		      output_masks = output_masks[0]
		      self._output_mask_cache[cache_key] = output_masks
		    else:
		      self._output_mask_cache[cache_key] = output_masks

		    if output_shapes is not None:
		      input_shapes = [K.int_shape(x) for x in inputs]
		      cache_key = ','.join([str(x) for x in input_shapes])
		      if len(output_shapes) == 1:
		        output_shapes = output_shapes[0]
		        self._output_shape_cache[cache_key] = output_shapes
		      else:
		        self._output_shape_cache[cache_key] = output_shapes
		    return output_tensors, output_masks, output_shapes

		  def get_config(self):
		    config = {
		        'name': self.name,
		    }
		    node_conversion_map = {}
		    for layer in self.layers:
		      if issubclass(layer.__class__, Container):
		        # Containers start with a pre-existing node
		        # linking their input to output.
		        kept_nodes = 1
		      else:
		        kept_nodes = 0
		      for original_node_index, node in enumerate(layer.inbound_nodes):
		        node_key = layer.name + '_ib-' + str(original_node_index)
		        if node_key in self.container_nodes:
		          node_conversion_map[node_key] = kept_nodes
		          kept_nodes += 1
		    layer_configs = []
		    for layer in self.layers:  # From the earliest layers on.
		      layer_class_name = layer.__class__.__name__
		      layer_config = layer.get_config()
		      filtered_inbound_nodes = []
		      for original_node_index, node in enumerate(layer.inbound_nodes):
		        node_key = layer.name + '_ib-' + str(original_node_index)
		        if node_key in self.container_nodes:
		          # The node is relevant to the model:
		          # add to filtered_inbound_nodes.
		          if node.arguments:
		            try:
		              json.dumps(node.arguments)
		              kwargs = node.arguments
		            except TypeError:
		              logging.warning(
		                  'Layer ' + layer.name +
		                  ' was passed non-serializable keyword arguments: ' +
		                  str(node.arguments) + '. They will not be included '
		                  'in the serialized model (and thus will be missing '
		                  'at deserialization time).')
		              kwargs = {}
		          else:
		            kwargs = {}
		          if node.inbound_layers:
		            node_data = []
		            for i in range(len(node.inbound_layers)):
		              inbound_layer = node.inbound_layers[i]
		              node_index = node.node_indices[i]
		              tensor_index = node.tensor_indices[i]
		              node_key = inbound_layer.name + '_ib-' + str(node_index)
		              new_node_index = node_conversion_map.get(node_key, 0)
		              node_data.append(
		                  [inbound_layer.name, new_node_index, tensor_index, kwargs])
		            filtered_inbound_nodes.append(node_data)
		      layer_configs.append({
		          'name': layer.name,
		          'class_name': layer_class_name,
		          'config': layer_config,
		          'inbound_nodes': filtered_inbound_nodes,
		      })
		    config['layers'] = layer_configs

		    # Gather info about inputs and outputs.
		    model_inputs = []
		    for i in range(len(self.input_layers)):
		      layer = self.input_layers[i]
		      node_index = self.input_layers_node_indices[i]
		      node_key = layer.name + '_ib-' + str(node_index)
		      new_node_index = node_conversion_map[node_key]
		      tensor_index = self.input_layers_tensor_indices[i]
		      model_inputs.append([layer.name, new_node_index, tensor_index])
		    config['input_layers'] = model_inputs
		    model_outputs = []
		    for i in range(len(self.output_layers)):
		      layer = self.output_layers[i]
		      node_index = self.output_layers_node_indices[i]
		      node_key = layer.name + '_ib-' + str(node_index)
		      new_node_index = node_conversion_map[node_key]
		      tensor_index = self.output_layers_tensor_indices[i]
		      model_outputs.append([layer.name, new_node_index, tensor_index])
		    config['output_layers'] = model_outputs
		    return copy.deepcopy(config)

		  @classmethod
		  def from_config(cls, config, custom_objects=None):
		    """Instantiates a Model from its config (output of `get_config()`).

		    Arguments:
		        config: Model config dictionary.
		        custom_objects: Optional dictionary mapping names
		            (strings) to custom classes or functions to be
		            considered during deserialization.

		    Returns:
		        A model instance.

		    Raises:
		        ValueError: In case of improperly formatted config dict.
		    """
		    # layer instances created during
		    # the graph reconstruction process
		    created_layers = {}

		    def process_layer(layer_data):
		      """Deserialize a layer, then call it on appropriate inputs.

		      Arguments:
		          layer_data: layer config dict.

		      Raises:
		          ValueError: In case of improperly formatted `layer_data` dict.
		      """
		      layer_name = layer_data['name']

		      # Instantiate layer.
		      from tensorflow.contrib.keras.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
		      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
		      created_layers[layer_name] = layer

		      # Gather layer inputs.
		      inbound_nodes_data = layer_data['inbound_nodes']
		      for node_data in inbound_nodes_data:
		        input_tensors = []
		        for input_data in node_data:
		          inbound_layer_name = input_data[0]
		          inbound_node_index = input_data[1]
		          inbound_tensor_index = input_data[2]
		          if len(input_data) == 3:
		            kwargs = {}
		          elif len(input_data) == 4:
		            kwargs = input_data[3]
		          else:
		            raise ValueError('Improperly formatted model config.')
		          if inbound_layer_name not in created_layers:
		            raise ValueError('Missing layer: ' + inbound_layer_name)
		          inbound_layer = created_layers[inbound_layer_name]
		          inbound_node = inbound_layer.inbound_nodes[inbound_node_index]
		          input_tensors.append(
		              inbound_node.output_tensors[inbound_tensor_index])
		        # Call layer on its inputs, thus creating the node
		        # and building the layer if needed.
		        if input_tensors:
		          if len(input_tensors) == 1:
		            layer(input_tensors[0], **kwargs)
		          else:
		            layer(input_tensors, **kwargs)

		    for layer_data in config['layers']:
		      process_layer(layer_data)

		    name = config.get('name')
		    input_tensors = []
		    output_tensors = []
		    for layer_data in config['input_layers']:
		      layer_name, node_index, tensor_index = layer_data
		      assert layer_name in created_layers
		      layer = created_layers[layer_name]
		      layer_output_tensors = layer.inbound_nodes[node_index].output_tensors
		      input_tensors.append(layer_output_tensors[tensor_index])
		    for layer_data in config['output_layers']:
		      layer_name, node_index, tensor_index = layer_data
		      assert layer_name in created_layers
		      layer = created_layers[layer_name]
		      layer_output_tensors = layer.inbound_nodes[node_index].output_tensors
		      output_tensors.append(layer_output_tensors[tensor_index])
		    return cls(inputs=input_tensors, outputs=output_tensors, name=name)

		  def save(self, filepath, overwrite=True, include_optimizer=True):
		    """Save the model to a single HDF5 file.

		    The savefile includes:
		        - The model architecture, allowing to re-instantiate the model.
		        - The model weights.
		        - The state of the optimizer, allowing to resume training
		            exactly where you left off.

		    This allows you to save the entirety of the state of a model
		    in a single file.

		    Saved models can be reinstantiated via `keras.models.load_model`.
		    The model returned by `load_model`
		    is a compiled model ready to be used (unless the saved model
		    was never compiled in the first place).

		    Arguments:
		        filepath: String, path to the file to save the weights to.
		        overwrite: Whether to silently overwrite any existing file at the
		            target location, or provide the user with a manual prompt.
		        include_optimizer: If True, save optimizer's state together.

		    Example:

		    ```python
		    from keras.models import load_model

		    model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
		    del model  # deletes the existing model

		    # returns a compiled model
		    # identical to the previous one
		    model = load_model('my_model.h5')
		    ```
		    """
		    from tensorflow.contrib.keras.python.keras.models import save_model  # pylint: disable=g-import-not-at-top
		    save_model(self, filepath, overwrite, include_optimizer)

		  def save_weights(self, filepath, overwrite=True):
		    """Dumps all layer weights to a HDF5 file.

		    The weight file has:
		        - `layer_names` (attribute), a list of strings
		            (ordered names of model layers).
		        - For every layer, a `group` named `layer.name`
		            - For every such layer group, a group attribute `weight_names`,
		                a list of strings
		                (ordered names of weights tensor of the layer).
		            - For every weight in the layer, a dataset
		                storing the weight value, named after the weight tensor.

		    Arguments:
		        filepath: String, path to the file to save the weights to.
		        overwrite: Whether to silently overwrite any existing file at the
		            target location, or provide the user with a manual prompt.

		    Raises:
		        ImportError: If h5py is not available.
		    """
		    if h5py is None:
		      raise ImportError('`save_weights` requires h5py.')
		    # If file exists and should not be overwritten:
		    if not overwrite and os.path.isfile(filepath):
		      proceed = ask_to_proceed_with_overwrite(filepath)
		      if not proceed:
		        return
		    f = h5py.File(filepath, 'w')
		    save_weights_to_hdf5_group(f, self.layers)
		    f.flush()
		    f.close()

		  # by_name = True, most used in transfer learning
		  def load_weights(self, filepath, by_name=False):
		    """Loads all layer weights from a HDF5 save file.

		    If `by_name` is False (default) weights are loaded
		    based on the network's topology, meaning the architecture
		    should be the same as when the weights were saved.
		    Note that layers that don't have weights are not taken
		    into account in the topological ordering, so adding or
		    removing layers is fine as long as they don't have weights.

		    If `by_name` is True, weights are loaded into layers
		    only if they share the same name. This is useful
		    for fine-tuning or transfer-learning models where
		    some of the layers have changed.

		    Arguments:
		        filepath: String, path to the weights file to load.
		        by_name: Boolean, whether to load weights by name
		            or by topological order.

		    Raises:
		        ImportError: If h5py is not available.
		    """
		    if h5py is None:
		      raise ImportError('`load_weights` requires h5py.')
		    f = h5py.File(filepath, mode='r')
		    if 'layer_names' not in f.attrs and 'model_weights' in f:
		      f = f['model_weights']
		    if by_name:
		      load_weights_from_hdf5_group_by_name(f, self.layers)
		    else:
		      load_weights_from_hdf5_group(f, self.layers)

		    if hasattr(f, 'close'):
		      f.close()

		  def _updated_config(self):
		    """Util hared between different serialization methods.

		    Returns:
		        Model config with Keras version information added.
		    """
		    from tensorflow.contrib.keras.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

		    config = self.get_config()
		    model_config = {
		        'class_name': self.__class__.__name__,
		        'config': config,
		        'keras_version': keras_version,
		        'backend': K.backend()
		    }
		    return model_config

		  def to_json(self, **kwargs):
		    """Returns a JSON string containing the network configuration.

		    To load a network from a JSON save file, use
		    `keras.models.model_from_json(json_string, custom_objects={})`.

		    Arguments:
		        **kwargs: Additional keyword arguments
		            to be passed to `json.dumps()`.

		    Returns:
		        A JSON string.
		    """

		    def get_json_type(obj):
		      # If obj is any numpy type
		      if type(obj).__module__ == np.__name__:
		        return obj.item()

		      # If obj is a python 'type'
		      if type(obj).__name__ == type.__name__:
		        return obj.__name__

		      raise TypeError('Not JSON Serializable:', obj)

		    model_config = self._updated_config()
		    return json.dumps(model_config, default=get_json_type, **kwargs)

		  def to_yaml(self, **kwargs):
		    """Returns a yaml string containing the network configuration.

		    To load a network from a yaml save file, use
		    `keras.models.model_from_yaml(yaml_string, custom_objects={})`.

		    `custom_objects` should be a dictionary mapping
		    the names of custom losses / layers / etc to the corresponding
		    functions / classes.

		    Arguments:
		        **kwargs: Additional keyword arguments
		            to be passed to `yaml.dump()`.

		    Returns:
		        A YAML string.

		    Raises:
		        ImportError: if yaml module is not found.
		    """
		    if yaml is None:
		      raise ImportError('Requires yaml module installed.')
		    return yaml.dump(self._updated_config(), **kwargs)

		  def summary(self, line_length=None, positions=None):
		    print_layer_summary(self, line_length=line_length, positions=positions)


		def get_source_inputs(tensor, layer=None, node_index=None):
		  """Returns the list of input tensors necessary to compute `tensor`.

		  Output will always be a list of tensors
		  (potentially with 1 element).

		  Arguments:
		      tensor: The tensor to start from.
		      layer: Origin layer of the tensor. Will be
		          determined via tensor._keras_history if not provided.
		      node_index: Origin node index of the tensor.

		  Returns:
		      List of input tensors.
		  """
		  if not hasattr(tensor, '_keras_history'):
		    return tensor

		  if layer is None or node_index:
		    layer, node_index, _ = tensor._keras_history
		  if not layer.inbound_nodes:
		    return [tensor]
		  else:
		    node = layer.inbound_nodes[node_index]
		    if not node.inbound_layers:
		      # Reached an Input layer, stop recursion.
		      return node.input_tensors
		    else:
		      source_tensors = []
		      for i in range(len(node.inbound_layers)):
		        x = node.input_tensors[i]
		        layer = node.inbound_layers[i]
		        node_index = node.node_indices[i]
		        previous_sources = get_source_inputs(x, layer, node_index)
		        # Avoid input redundancy.
		        for x in previous_sources:
		          if x not in source_tensors:
		            source_tensors.append(x)
		      return source_tensors


		def _to_list(x):
		  """Normalizes a list/tensor into a list.

		  If a tensor is passed, we return
		  a list of size 1 containing the tensor.

		  Arguments:
		      x: target object to be normalized.

		  Returns:
		      A list.
		  """
		  if isinstance(x, list):
		    return x
		  return [x]


		def _object_list_uid(object_list):
		  object_list = _to_list(object_list)
		  return ', '.join([str(abs(id(x))) for x in object_list])


		def _is_all_none(iterable_or_element):
		  if not isinstance(iterable_or_element, (list, tuple)):
		    iterable = [iterable_or_element]
		  else:
		    iterable = iterable_or_element
		  for element in iterable:
		    if element is not None:
		      return False
		  return True


		def _collect_previous_mask(input_tensors):
		  """Retrieves the output mask(s) of the previous node.

		  Arguments:
		      input_tensors: A tensor or list of tensors.

		  Returns:
		      A mask tensor or list of mask tensors.
		  """
		  input_tensors = _to_list(input_tensors)
		  masks = []
		  for x in input_tensors:
		    if hasattr(x, '_keras_history'):
		      inbound_layer, node_index, tensor_index = x._keras_history
		      node = inbound_layer.inbound_nodes[node_index]
		      mask = node.output_masks[tensor_index]
		      masks.append(mask)
		    else:
		      masks.append(None)
		  if len(masks) == 1:
		    return masks[0]
		  return masks


		def _to_snake_case(name):
		  intermediate = re.sub('(.)([A-Z][a-z0-9]+)', r'\1_\2', name)
		  insecure = re.sub('([a-z])([A-Z])', r'\1_\2', intermediate).lower()
		  # If the class is private the name starts with "_" which is not secure
		  # for creating scopes. We prefix the name with "private" in this case.
		  if insecure[0] != '_':
		    return insecure
		  return 'private' + insecure


		def _collect_input_shape(input_tensors):
		  """Collects the output shape(s) of a list of Keras tensors.

		  Arguments:
		      input_tensors: list of input tensors (or single input tensor).

		  Returns:
		      List of shape tuples (or single tuple), one tuple per input.
		  """
		  input_tensors = _to_list(input_tensors)
		  shapes = []
		  for x in input_tensors:
		    shapes.append(K.int_shape(x))
		  if len(shapes) == 1:
		    return shapes[0]
		  return shapes


		def save_weights_to_hdf5_group(f, layers):
		  from tensorflow.contrib.keras.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top

		  f.attrs['layer_names'] = [layer.name.encode('utf8') for layer in layers]
		  f.attrs['backend'] = K.backend().encode('utf8')
		  f.attrs['keras_version'] = str(keras_version).encode('utf8')

		  for layer in layers:
		    g = f.create_group(layer.name)
		    symbolic_weights = layer.weights
		    weight_values = K.batch_get_value(symbolic_weights)
		    weight_names = []
		    for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
		      if hasattr(w, 'name') and w.name:
		        name = str(w.name)
		      else:
		        name = 'param_' + str(i)
		      weight_names.append(name.encode('utf8'))
		    g.attrs['weight_names'] = weight_names
		    for name, val in zip(weight_names, weight_values):
		      param_dset = g.create_dataset(name, val.shape, dtype=val.dtype)
		      if not val.shape:
		        # scalar
		        param_dset[()] = val
		      else:
		        param_dset[:] = val


		def preprocess_weights_for_loading(layer,
		                                   weights,
		                                   original_keras_version=None,
		                                   original_backend=None):
		  """Converts layers weights from Keras 1 format to Keras 2.

		  Arguments:
		      layer: Layer instance.
		      weights: List of weights values (Numpy arrays).
		      original_keras_version: Keras version for the weights, as a string.
		      original_backend: Keras backend the weights were trained with,
		          as a string.

		  Returns:
		      A list of weights values (Numpy arrays).
		  """
		  if original_keras_version == '1':
		    if layer.__class__.__name__ == 'Bidirectional':
		      num_weights_per_layer = len(weights) // 2

		      forward_weights = preprocess_weights_for_loading(
		          layer.forward_layer, weights[:num_weights_per_layer],
		          original_keras_version, original_backend)
		      backward_weights = preprocess_weights_for_loading(
		          layer.backward_layer, weights[num_weights_per_layer:],
		          original_keras_version, original_backend)
		      weights = forward_weights + backward_weights

		    if layer.__class__.__name__ == 'TimeDistributed':
		      weights = preprocess_weights_for_loading(
		          layer.layer, weights, original_keras_version, original_backend)

		    if layer.__class__.__name__ == 'Conv1D':
		      shape = weights[0].shape
		      # Handle Keras 1.1 format
		      if shape[:2] != (layer.kernel_size[0], 1) or shape[3] != layer.filters:
		        # Legacy shape:
		        # (filters, input_dim, filter_length, 1)
		        assert shape[0] == layer.filters and shape[2:] == (layer.kernel_size[0],
		                                                           1)
		        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
		      weights[0] = weights[0][:, 0, :, :]

		    if layer.__class__.__name__ == 'Conv2D':
		      if layer.data_format == 'channels_first':
		        # old: (filters, stack_size, kernel_rows, kernel_cols)
		        # new: (kernel_rows, kernel_cols, stack_size, filters)
		        weights[0] = np.transpose(weights[0], (2, 3, 1, 0))

		    if layer.__class__.__name__ == 'Conv2DTranspose':
		      if layer.data_format == 'channels_last':
		        # old: (kernel_rows, kernel_cols, stack_size, filters)
		        # new: (kernel_rows, kernel_cols, filters, stack_size)
		        weights[0] = np.transpose(weights[0], (0, 1, 3, 2))
		      if layer.data_format == 'channels_first':
		        # old: (filters, stack_size, kernel_rows, kernel_cols)
		        # new: (kernel_rows, kernel_cols, filters, stack_size)
		        weights[0] = np.transpose(weights[0], (2, 3, 0, 1))

		    if layer.__class__.__name__ == 'Conv3D':
		      if layer.data_format == 'channels_first':
		        # old: (filters, stack_size, ...)
		        # new: (..., stack_size, filters)
		        weights[0] = np.transpose(weights[0], (2, 3, 4, 1, 0))

		    if layer.__class__.__name__ == 'GRU':
		      if len(weights) == 9:
		        kernel = np.concatenate([weights[0], weights[3], weights[6]], axis=-1)
		        recurrent_kernel = np.concatenate(
		            [weights[1], weights[4], weights[7]], axis=-1)
		        bias = np.concatenate([weights[2], weights[5], weights[8]], axis=-1)
		        weights = [kernel, recurrent_kernel, bias]

		    if layer.__class__.__name__ == 'LSTM':
		      if len(weights) == 12:
		        # old: i, c, f, o
		        # new: i, f, c, o
		        kernel = np.concatenate(
		            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
		        recurrent_kernel = np.concatenate(
		            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
		        bias = np.concatenate(
		            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
		        weights = [kernel, recurrent_kernel, bias]

		    if layer.__class__.__name__ == 'ConvLSTM2D':
		      if len(weights) == 12:
		        kernel = np.concatenate(
		            [weights[0], weights[6], weights[3], weights[9]], axis=-1)
		        recurrent_kernel = np.concatenate(
		            [weights[1], weights[7], weights[4], weights[10]], axis=-1)
		        bias = np.concatenate(
		            [weights[2], weights[8], weights[5], weights[11]], axis=-1)
		        if layer.data_format == 'channels_first':
		          # old: (filters, stack_size, kernel_rows, kernel_cols)
		          # new: (kernel_rows, kernel_cols, stack_size, filters)
		          kernel = np.transpose(kernel, (2, 3, 1, 0))
		          recurrent_kernel = np.transpose(recurrent_kernel, (2, 3, 1, 0))
		        weights = [kernel, recurrent_kernel, bias]

		  conv_layers = ['Conv1D', 'Conv2D', 'Conv3D', 'Conv2DTranspose', 'ConvLSTM2D']
		  if layer.__class__.__name__ in conv_layers:
		    if original_backend and K.backend() != original_backend:
		      weights[0] = conv_utils.convert_kernel(weights[0])
		      if layer.__class__.__name__ == 'ConvLSTM2D':
		        weights[1] = conv_utils.convert_kernel(weights[1])
		    if K.int_shape(layer.weights[0]) != weights[0].shape:
		      weights[0] = np.transpose(weights[0], (3, 2, 0, 1))
		      if layer.__class__.__name__ == 'ConvLSTM2D':
		        weights[1] = np.transpose(weights[1], (3, 2, 0, 1))
		  return weights


		def load_weights_from_hdf5_group(f, layers):
		  """Implements topological (order-based) weight loading.

		  Arguments:
		      f: A pointer to a HDF5 group.
		      layers: a list of target layers.

		  Raises:
		      ValueError: in case of mismatch between provided layers
		          and weights file.
		  """
		  if 'keras_version' in f.attrs:
		    original_keras_version = f.attrs['keras_version'].decode('utf8')
		  else:
		    original_keras_version = '1'
		  if 'backend' in f.attrs:
		    original_backend = f.attrs['backend'].decode('utf8')
		  else:
		    original_backend = None

		  filtered_layers = []
		  for layer in layers:
		    weights = layer.weights
		    if weights:
		      filtered_layers.append(layer)

		  layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
		  filtered_layer_names = []
		  for name in layer_names:
		    g = f[name]
		    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
		    if weight_names:
		      filtered_layer_names.append(name)
		  layer_names = filtered_layer_names
		  if len(layer_names) != len(filtered_layers):
		    raise ValueError('You are trying to load a weight file '
		                     'containing ' + str(len(layer_names)) +
		                     ' layers into a model with ' + str(len(filtered_layers)) +
		                     ' layers.')

		  # We batch weight value assignments in a single backend call
		  # which provides a speedup in TensorFlow.
		  weight_value_tuples = []
		  for k, name in enumerate(layer_names):
		    g = f[name]
		    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
		    weight_values = [g[weight_name] for weight_name in weight_names]
		    layer = filtered_layers[k]
		    symbolic_weights = layer.weights
		    weight_values = preprocess_weights_for_loading(
		        layer, weight_values, original_keras_version, original_backend)
		    if len(weight_values) != len(symbolic_weights):
		      raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
		                       '" in the current model) was found to '
		                       'correspond to layer ' + name + ' in the save file. '
		                       'However the new layer ' + layer.name + ' expects ' +
		                       str(len(symbolic_weights)) +
		                       ' weights, but the saved weights have ' +
		                       str(len(weight_values)) + ' elements.')
		    weight_value_tuples += zip(symbolic_weights, weight_values)
		  K.batch_set_value(weight_value_tuples)


		def load_weights_from_hdf5_group_by_name(f, layers):
		  """Implements name-based weight loading.

		  (instead of topological weight loading).

		  Layers that have no matching name are skipped.

		  Arguments:
		      f: A pointer to a HDF5 group.
		      layers: a list of target layers.

		  Raises:
		      ValueError: in case of mismatch between provided layers
		          and weights file.
		  """
		  if 'keras_version' in f.attrs:
		    original_keras_version = f.attrs['keras_version'].decode('utf8')
		  else:
		    original_keras_version = '1'
		  if 'backend' in f.attrs:
		    original_backend = f.attrs['backend'].decode('utf8')
		  else:
		    original_backend = None

		  # New file format.
		  layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

		  # Reverse index of layer name to list of layers with name.
		  index = {}
		  for layer in layers:
		    if layer.name:
		      index.setdefault(layer.name, []).append(layer)

		  # We batch weight value assignments in a single backend call
		  # which provides a speedup in TensorFlow.
		  weight_value_tuples = []
		  for k, name in enumerate(layer_names):
		    g = f[name]
		    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
		    weight_values = [g[weight_name] for weight_name in weight_names]

		    for layer in index.get(name, []):
		      symbolic_weights = layer.weights
		      weight_values = preprocess_weights_for_loading(
		          layer, weight_values, original_keras_version, original_backend)
		      if len(weight_values) != len(symbolic_weights):
		        raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
		                         '") expects ' + str(len(symbolic_weights)) +
		                         ' weight(s), but the saved weights' + ' have ' +
		                         str(len(weight_values)) + ' element(s).')
		      # Set values.
		      for i in range(len(weight_values)):
		        weight_value_tuples.append((symbolic_weights[i], weight_values[i]))
		  K.batch_set_value(weight_value_tuples)

	class training_py:
		"""Keras training and evaluation routines.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import copy
			import multiprocessing
			import threading
			import time

			import numpy as np
			import six

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras import callbacks as cbks
			from tensorflow.contrib.keras.python.keras import losses
			from tensorflow.contrib.keras.python.keras import metrics as metrics_module
			from tensorflow.contrib.keras.python.keras import optimizers
			from tensorflow.contrib.keras.python.keras.engine.topology import Container
			from tensorflow.contrib.keras.python.keras.utils.generic_utils import Progbar
			from tensorflow.python.platform import tf_logging as logging


		# pylint: disable=g-import-not-at-top
		try:
		  import queue
		except ImportError:
		  import Queue as queue
		# pylint: enable=g-import-not-at-top


		def _standardize_input_data(data,
		                            names,
		                            shapes=None,
		                            check_batch_axis=True,
		                            exception_prefix=''):
		  """Normalizes inputs and targets provided by users.

		  Users may pass data as a list of arrays, dictionary of arrays,
		  or as a single array. We normalize this to an ordered list of
		  arrays (same order as `names`), while checking that the provided
		  arrays have shapes that match the network's expectations.

		  Arguments:
		      data: User-provided input data (polymorphic).
		      names: List of expected array names.
		      shapes: Optional list of expected array shapes.
		      check_batch_axis: Boolean; whether to check that
		          the batch axis of the arrays matches the expected
		          value found in `shapes`.
		      exception_prefix: String prefix used for exception formatting.

		  Returns:
		      List of standardized input arrays (one array per model input).

		  Raises:
		      ValueError: in case of improperly formatted user-provided data.
		  """
		  if not names:
		    return []
		  if data is None:
		    return [None for _ in range(len(names))]
		  if isinstance(data, dict):
		    arrays = []
		    for name in names:
		      if name not in data:
		        raise ValueError('No data provided for "' + name +
		                         '". Need data for each key in: ' + str(names))
		      arrays.append(data[name])
		  elif isinstance(data, list):
		    if len(data) != len(names):
		      if data and hasattr(data[0], 'shape'):
		        raise ValueError(
		            'Error when checking model ' + exception_prefix +
		            ': the list of Numpy arrays '
		            'that you are passing to your model '
		            'is not the size the model expected. '
		            'Expected to see ' + str(len(names)) + ' arrays but instead got '
		            'the following list of ' + str(len(data)) + ' arrays: ' +
		            str(data)[:200] + '...')
		      else:
		        if len(names) == 1:
		          data = [np.asarray(data)]
		        else:
		          raise ValueError('Error when checking model ' + exception_prefix +
		                           ': you are passing a list as '
		                           'input to your model, '
		                           'but the model expects '
		                           'a list of ' + str(len(names)) +
		                           ' Numpy arrays instead. '
		                           'The list you passed was: ' + str(data)[:200])
		    arrays = data
		  else:
		    if not hasattr(data, 'shape'):
		      raise TypeError('Error when checking model ' + exception_prefix +
		                      ': data should be a Numpy array, '
		                      'or list/dict of Numpy arrays. '
		                      'Found: ' + str(data)[:200] + '...')
		    if len(names) > 1:
		      # Case: model expects multiple inputs but only received
		      # a single Numpy array.
		      raise ValueError('The model expects ' + str(len(names)) + exception_prefix
		                       + ' arrays, but only received one array. '
		                       'Found: array with shape ' + str(data.shape))
		    arrays = [data]

		  # Make arrays at least 2D.
		  for i in range(len(names)):
		    array = arrays[i]
		    if len(array.shape) == 1:
		      array = np.expand_dims(array, 1)
		      arrays[i] = array

		  # Check shapes compatibility.
		  if shapes:
		    for i in range(len(names)):
		      if shapes[i] is None:
		        continue
		      array = arrays[i]
		      if len(array.shape) != len(shapes[i]):
		        raise ValueError(
		            'Error when checking ' + exception_prefix + ': expected ' + names[i]
		            + ' to have ' + str(len(shapes[i])) +
		            ' dimensions, but got array with shape ' + str(array.shape))
		      for j, (dim, ref_dim) in enumerate(zip(array.shape, shapes[i])):
		        if not j and not check_batch_axis:
		          # skip the first axis
		          continue
		        if ref_dim:
		          if ref_dim != dim:
		            raise ValueError('Error when checking ' + exception_prefix +
		                             ': expected ' + names[i] + ' to have shape ' +
		                             str(shapes[i]) + ' but got array with shape ' +
		                             str(array.shape))
		  return arrays


		def _standardize_sample_or_class_weights(x_weight, output_names, weight_type):
		  """Maps `sample_weight` or `class_weight` to model outputs.

		  Arguments:
		      x_weight: User-provided `sample_weight` or `class_weight` argument.
		      output_names: List of output names (strings) in the model.
		      weight_type: A string used purely for exception printing.

		  Returns:
		      A list of `sample_weight` or `class_weight` where there are exactly
		          one element per model output.

		  Raises:
		      ValueError: In case of invalid user-provided argument.
		  """
		  if x_weight is None or len(x_weight) == 0:  # pylint: disable=g-explicit-length-test
		    return [None for _ in output_names]
		  if len(output_names) == 1:
		    if isinstance(x_weight, list) and len(x_weight) == 1:
		      return x_weight
		    if isinstance(x_weight, dict) and output_names[0] in x_weight:
		      return [x_weight[output_names[0]]]
		    else:
		      return [x_weight]
		  if isinstance(x_weight, list):
		    if len(x_weight) != len(output_names):
		      raise ValueError('Provided `' + weight_type + '` was a list of ' +
		                       str(len(x_weight)) + ' elements, but the model has ' +
		                       str(len(output_names)) + ' outputs. '
		                       'You should provide one `' + weight_type + '`'
		                       'array per model output.')
		    return x_weight
		  if isinstance(x_weight, dict):
		    x_weights = []
		    for name in output_names:
		      x_weights.append(x_weight.get(name))
		    return x_weights
		  else:
		    raise TypeError('The model has multiple outputs, so `' + weight_type + '` '
		                    'should be either a list of a dict. '
		                    'Provided `' + weight_type + '` type not understood: ' +
		                    str(x_weight))


		def _standardize_class_weights(class_weight, output_names):
		  return _standardize_sample_or_class_weights(class_weight, output_names,
		                                              'class_weight')


		def _standardize_sample_weights(sample_weight, output_names):
		  return _standardize_sample_or_class_weights(sample_weight, output_names,
		                                              'sample_weight')


		def _check_array_lengths(inputs, targets, weights):
		  """Does user input validation for numpy arrays.

		  Arguments:
		      inputs: list of Numpy arrays of inputs.
		      targets: list of Numpy arrays of targets.
		      weights: list of Numpy arrays of sample weights.

		  Raises:
		      ValueError: in case of incorrectly formatted data.
		  """
		  x_lengths = [x.shape[0] for x in inputs]
		  y_lengths = [y.shape[0] for y in targets]
		  w_lengths = [w.shape[0] for w in weights]
		  set_x = set(x_lengths)
		  if len(set_x) > 1:
		    raise ValueError('All input arrays (x) should have '
		                     'the same number of samples. Got array shapes: ' + str(
		                         [x.shape for x in inputs]))
		  set_y = set(y_lengths)
		  if len(set_y) > 1:
		    raise ValueError('All target arrays (y) should have '
		                     'the same number of samples. Got array shapes: ' + str(
		                         [y.shape for y in targets]))
		  set_w = set(w_lengths)
		  if len(set_w) > 1:
		    raise ValueError('All sample_weight arrays should have '
		                     'the same number of samples. Got array shapes: ' + str(
		                         [w.shape for w in weights]))
		  if set_x and set_y and list(set_x)[0] != list(set_y)[0]:
		    raise ValueError('Input arrays should have '
		                     'the same number of samples as target arrays. '
		                     'Found ' + str(list(set_x)[0]) + ' input samples '
		                     'and ' + str(list(set_y)[0]) + ' target samples.')
		  if set_y and set_w and list(set_y)[0] != list(set_w)[0]:
		    raise ValueError('Sample_weight arrays should have '
		                     'the same number of samples as target arrays. Got ' +
		                     str(list(set_y)[0]) + ' input samples and ' +
		                     str(list(set_w)[0]) + ' target samples.')


		def _check_loss_and_target_compatibility(targets, loss_fns, output_shapes):
		  """Does validation on the compatibility of targets and loss functions.

		  This helps prevent users from using loss functions incorrectly.

		  Arguments:
		      targets: list of Numpy arrays of targets.
		      loss_fns: list of loss functions.
		      output_shapes: list of shapes of model outputs.

		  Raises:
		      ValueError: if a loss function or target array
		          is incompatible with an output.
		  """
		  key_losses = {
		      'mean_square_error', 'binary_crossentropy', 'categorical_crossentropy'
		  }
		  for y, loss, shape in zip(targets, loss_fns, output_shapes):
		    if loss is None:
		      continue
		    if loss.__name__ == 'categorical_crossentropy':
		      if y.shape[-1] == 1:
		        raise ValueError('You are passing a target array of shape ' + str(
		            y.shape) + ' while using as loss `categorical_crossentropy`. '
		                         '`categorical_crossentropy` expects '
		                         'targets to be binary matrices (1s and 0s) '
		                         'of shape (samples, classes). '
		                         'If your targets are integer classes, '
		                         'you can convert them to the expected format via:\n'
		                         '```\n'
		                         'from keras.utils.np_utils import to_categorical\n'
		                         'y_binary = to_categorical(y_int)\n'
		                         '```\n'
		                         '\n'
		                         'Alternatively, you can use the loss function '
		                         '`sparse_categorical_crossentropy` instead, '
		                         'which does expect integer targets.')
		    if loss.__name__ in key_losses:
		      for target_dim, out_dim in zip(y.shape[1:], shape[1:]):
		        if out_dim is not None and target_dim != out_dim:
		          raise ValueError('A target array with shape ' + str(y.shape) +
		                           ' was passed for an output of shape ' + str(shape) +
		                           ' while using as loss `' + loss.__name__ + '`. '
		                           'This loss expects '
		                           'targets to have the same shape '
		                           'as the output.')


		def _collect_metrics(metrics, output_names):
		  """Maps metric functions to model outputs.

		  Arguments:
		      metrics: a list or dict of metric functions.
		      output_names: a list of the names (strings) of model outputs.

		  Returns:
		      A list (one entry per model output) of lists of metric functions.
		      For instance, if the model has 2 outputs, and for the first output
		      we want to compute "binary_accuracy" and "binary_crossentropy",
		      and just "binary_accuracy" for the second output,
		      the list would look like:
		          `[[binary_accuracy, binary_crossentropy], [binary_accuracy]]`

		  Raises:
		      TypeError: if an incorrect type is passed for the `metrics` argument.
		  """
		  if not metrics:
		    return [[] for _ in output_names]
		  if isinstance(metrics, list):
		    # we then apply all metrics to all outputs.
		    return [copy.copy(metrics) for _ in output_names]
		  elif isinstance(metrics, dict):
		    nested_metrics = []
		    for name in output_names:
		      output_metrics = metrics.get(name, [])
		      if not isinstance(output_metrics, list):
		        output_metrics = [output_metrics]
		      nested_metrics.append(output_metrics)
		    return nested_metrics
		  else:
		    raise TypeError('Type of `metrics` argument not understood. '
		                    'Expected a list or dictionary, found: ' + str(metrics))


		def _batch_shuffle(index_array, batch_size):
		  """Shuffles an array in a batch-wise fashion.

		  Useful for shuffling HDF5 arrays
		  (where one cannot access arbitrary indices).

		  Arguments:
		      index_array: array of indices to be shuffled.
		      batch_size: integer.

		  Returns:
		      The `index_array` array, shuffled in a batch-wise fashion.
		  """
		  batch_count = int(len(index_array) / batch_size)
		  # to reshape we need to be cleanly divisible by batch size
		  # we stash extra items and reappend them after shuffling
		  last_batch = index_array[batch_count * batch_size:]
		  index_array = index_array[:batch_count * batch_size]
		  index_array = index_array.reshape((batch_count, batch_size))
		  np.random.shuffle(index_array)
		  index_array = index_array.flatten()
		  return np.append(index_array, last_batch)


		def _make_batches(size, batch_size):
		  """Returns a list of batch indices (tuples of indices).

		  Arguments:
		      size: Integer, total size of the data to slice into batches.
		      batch_size: Integer, batch size.

		  Returns:
		      A list of tuples of array indices.
		  """
		  num_batches = int(np.ceil(size / float(batch_size)))
		  return [(i * batch_size, min(size, (i + 1) * batch_size))
		          for i in range(0, num_batches)]


		def _slice_arrays(arrays, start=None, stop=None):
		  """Slice an array or list of arrays.

		  This takes an array-like, or a list of
		  array-likes, and outputs:
		      - arrays[start:stop] if `arrays` is an array-like
		      - [x[start:stop] for x in arrays] if `arrays` is a list

		  Can also work on list/array of indices: `_slice_arrays(x, indices)`

		  Arguments:
		      arrays: Single array or list of arrays.
		      start: can be an integer index (start index)
		          or a list/array of indices
		      stop: integer (stop index); should be None if
		          `start` was a list.

		  Returns:
		      A slice of the array(s).
		  """
		  if isinstance(arrays, list):
		    if hasattr(start, '__len__'):
		      # hdf5 datasets only support list objects as indices
		      if hasattr(start, 'shape'):
		        start = start.tolist()
		      return [x[start] for x in arrays]
		    else:
		      return [x[start:stop] for x in arrays]
		  else:
		    if hasattr(start, '__len__'):
		      if hasattr(start, 'shape'):
		        start = start.tolist()
		      return arrays[start]
		    else:
		      return arrays[start:stop]


		def _weighted_masked_objective(fn):
		  """Adds support for masking and sample-weighting to an objective function.

		  It transforms an objective function `fn(y_true, y_pred)`
		  into a sample-weighted, cost-masked objective function
		  `fn(y_true, y_pred, weights, mask)`.

		  Arguments:
		      fn: The objective function to wrap,
		          with signature `fn(y_true, y_pred)`.

		  Returns:
		      A function with signature `fn(y_true, y_pred, weights, mask)`.
		  """
		  if fn is None:
		    return None

		  def weighted(y_true, y_pred, weights, mask=None):
		    """Wrapper function.

		    Arguments:
		        y_true: `y_true` argument of `fn`.
		        y_pred: `y_pred` argument of `fn`.
		        weights: Weights tensor.
		        mask: Mask tensor.

		    Returns:
		        Scalar tensor.
		    """
		    # score_array has ndim >= 2
		    score_array = fn(y_true, y_pred)
		    if mask is not None:
		      mask = K.cast(mask, K.floatx())
		      # mask should have the same shape as score_array
		      score_array *= mask
		      #  the loss per batch should be proportional
		      #  to the number of unmasked samples.
		      score_array /= K.mean(mask)

		    # reduce score_array to same ndim as weight array
		    ndim = K.ndim(score_array)
		    weight_ndim = K.ndim(weights)
		    score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))

		    # apply sample weighting
		    if weights is not None:
		      score_array *= weights
		      score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
		    return K.mean(score_array)

		  return weighted


		def _masked_objective(fn):
		  """Adds support for masking to an objective function.

		  It transforms an objective function `fn(y_true, y_pred)`
		  into a cost-masked objective function
		  `fn(y_true, y_pred, mask)`.

		  Arguments:
		      fn: The objective function to wrap,
		          with signature `fn(y_true, y_pred)`.

		  Returns:
		      A function with signature `fn(y_true, y_pred, mask)`.
		  """

		  def masked(y_true, y_pred, mask=None):
		    """Wrapper function.

		    Arguments:
		        y_true: `y_true` argument of `fn`.
		        y_pred: `y_pred` argument of `fn`.
		        mask: Mask tensor.

		    Returns:
		        Scalar tensor.
		    """
		    # score_array has ndim >= 2
		    score_array = fn(y_true, y_pred)
		    if mask is not None:
		      mask = K.cast(mask, K.floatx())
		      # mask should have the same shape as score_array
		      score_array *= mask
		      #  the loss per batch should be proportional
		      #  to the number of unmasked samples.
		      score_array /= K.mean(mask)

		    return K.mean(score_array)

		  return masked


		def _standardize_weights(y,
		                         sample_weight=None,
		                         class_weight=None,
		                         sample_weight_mode=None):
		  """Performs sample weight validation and standardization.

		  Everything gets normalized to a single sample-wise (or timestep-wise)
		  weight array.

		  Arguments:
		      y: Numpy array of model targets to be weighted.
		      sample_weight: User-provided `sample_weight` argument.
		      class_weight: User-provided `class_weight` argument.
		      sample_weight_mode: One of `None` or `"temporal"`.
		          `"temporal"` indicated that we expect 2D weight data
		          that will be applied to the last 2 dimensions of
		          the targets (i.e. we are weighting timesteps, not samples).

		  Returns:
		      A numpy array of target weights, one entry per sample to weight.

		  Raises:
		      ValueError: In case of invalid user-provided arguments.
		  """
		  if sample_weight_mode is not None:
		    if sample_weight_mode != 'temporal':
		      raise ValueError('"sample_weight_mode '
		                       'should be None or "temporal". '
		                       'Found: ' + str(sample_weight_mode))
		    if len(y.shape) < 3:
		      raise ValueError('Found a sample_weight array for '
		                       'an input with shape ' + str(y.shape) + '. '
		                       'Timestep-wise sample weighting (use of '
		                       'sample_weight_mode="temporal") is restricted to '
		                       'outputs that are at least 3D, i.e. that have '
		                       'a time dimension.')
		    if sample_weight is not None and len(sample_weight.shape) != 2:
		      raise ValueError('Found a sample_weight array with shape ' +
		                       str(sample_weight.shape) + '. '
		                       'In order to use timestep-wise sample weighting, '
		                       'you should pass a 2D sample_weight array.')
		  else:
		    if sample_weight is not None and len(sample_weight.shape) != 1:
		      raise ValueError('Found a sample_weight array with shape ' +
		                       str(sample_weight.shape) + '. '
		                       'In order to use timestep-wise sample weights, '
		                       'you should specify '
		                       'sample_weight_mode="temporal" '
		                       'in compile(). If you just mean to use '
		                       'sample-wise weights, make sure your '
		                       'sample_weight array is 1D.')

		  if sample_weight is not None:
		    if len(sample_weight.shape) > len(y.shape):
		      raise ValueError('Found a sample_weight with shape' +
		                       str(sample_weight.shape) + '.'
		                       'Expected sample_weight with rank '
		                       'less than or equal to ' + str(len(y.shape)))

		    if y.shape[:sample_weight.ndim] != sample_weight.shape:
		      raise ValueError('Found a sample_weight array with shape ' +
		                       str(sample_weight.shape) + ' for an input with shape ' +
		                       str(y.shape) + '. '
		                       'sample_weight cannot be broadcast.')
		    return sample_weight
		  elif isinstance(class_weight, dict):
		    if len(y.shape) > 2:
		      raise ValueError('class_weight not supported for '
		                       '3+ dimensional targets.')
		    if y.shape[1] > 1:
		      y_classes = y.argmax(axis=1)
		    elif y.shape[1] == 1:
		      y_classes = np.reshape(y, y.shape[0])
		    else:
		      y_classes = y
		    weights = np.asarray([class_weight[cls] for cls in y_classes])
		    return weights
		  else:
		    if sample_weight_mode is None:
		      return np.ones((y.shape[0],), dtype=K.floatx())
		    else:
		      return np.ones((y.shape[0], y.shape[1]), dtype=K.floatx())


		class GeneratorEnqueuer(object):
		  """Builds a queue out of a data generator.

		  Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

		  Arguments:
		      generator: a generator function which endlessly yields data
		      pickle_safe: use multiprocessing if True, otherwise threading
		  """

		  def __init__(self, generator, pickle_safe=False):
		    self._generator = generator
		    self._pickle_safe = pickle_safe
		    self._threads = []
		    self._stop_event = None
		    self.queue = None

		  def start(self, workers=1, max_q_size=10, wait_time=0.05):
		    """Kicks off threads which add data from the generator into the queue.

		    Arguments:
		        workers: number of worker threads
		        max_q_size: queue size (when full, threads could block on put())
		        wait_time: time to sleep in-between calls to put()
		    """

		    def data_generator_task():
		      while not self._stop_event.is_set():
		        try:
		          if self._pickle_safe or self.queue.qsize() < max_q_size:
		            generator_output = next(self._generator)
		            self.queue.put(generator_output)
		          else:
		            time.sleep(wait_time)
		        except Exception:
		          self._stop_event.set()
		          raise

		    try:
		      if self._pickle_safe:
		        self.queue = multiprocessing.Queue(maxsize=max_q_size)
		        self._stop_event = multiprocessing.Event()
		      else:
		        self.queue = queue.Queue()
		        self._stop_event = threading.Event()

		      for _ in range(workers):
		        if self._pickle_safe:
		          # Reset random seed else all children processes
		          # share the same seed
		          np.random.seed()
		          thread = multiprocessing.Process(target=data_generator_task)
		          thread.daemon = True
		        else:
		          thread = threading.Thread(target=data_generator_task)
		        self._threads.append(thread)
		        thread.start()
		    except:
		      self.stop()
		      raise

		  def is_running(self):
		    return self._stop_event is not None and not self._stop_event.is_set()

		  def stop(self, timeout=None):
		    """Stop running threads and wait for them to exit, if necessary.

		    Should be called by the same thread which called start().

		    Arguments:
		        timeout: maximum time to wait on thread.join()
		    """
		    if self.is_running():
		      self._stop_event.set()

		    for thread in self._threads:
		      if thread.is_alive():
		        if self._pickle_safe:
		          thread.terminate()
		        else:
		          thread.join(timeout)

		    if self._pickle_safe:
		      if self.queue is not None:
		        self.queue.close()

		    self._threads = []
		    self._stop_event = None
		    self.queue = None

		# 1. Model as subclass of Container, has no __init__ its own
		# 2. to instantiate Model, is just to run Container.__init__()
		class Model(Container):
		  """The `Model` class adds training & evaluation routines to a `Container`.
		  """

		  def compile(self,
		              optimizer,
		              loss,
		              metrics=None,
		              loss_weights=None,
		              sample_weight_mode=None,
		              **kwargs):
		    """Configures the model for training.

		    Arguments:
		        optimizer: str (name of optimizer) or optimizer object.
		            See [optimizers](/optimizers).
		        loss: str (name of objective function) or objective function.
		            See [losses](/losses).
		            If the model has multiple outputs, you can use a different loss
		            on each output by passing a dictionary or a list of losses.
		            The loss value that will be minimized by the model
		            will then be the sum of all individual losses.
		        metrics: list of metrics to be evaluated by the model
		            during training and testing.
		            Typically you will use `metrics=['accuracy']`.
		            To specify different metrics for different outputs of a
		            multi-output model, you could also pass a dictionary,
		            such as `metrics={'output_a': 'accuracy'}`.
		        loss_weights: Optional list or dictionary specifying scalar
		            coefficients (Python floats) to weight the loss contributions
		            of different model outputs.
		            The loss value that will be minimized by the model
		            will then be the *weighted sum* of all individual losses,
		            weighted by the `loss_weights` coefficients.
		            If a list, it is expected to have a 1:1 mapping
		            to the model's outputs. If a tensor, it is expected to map
		            output names (strings) to scalar coefficients.
		        sample_weight_mode: if you need to do timestep-wise
		            sample weighting (2D weights), set this to `"temporal"`.
		            `None` defaults to sample-wise weights (1D).
		            If the model has multiple outputs, you can use a different
		            `sample_weight_mode` on each output by passing a
		            dictionary or a list of modes.
		        **kwargs: Additional arguments passed to `tf.Session.run`.

		    Raises:
		        ValueError: In case of invalid arguments for
		            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
		        RuntimeError: If the model has no loss to optimize.
		    """
		    loss = loss or {}
		    self.optimizer = optimizers.get(optimizer)
		    self.sample_weight_mode = sample_weight_mode
		    self.loss = loss
		    self.loss_weights = loss_weights

		    # Prepare loss functions.
		    if isinstance(loss, dict):
		      for name in loss:
		        if name not in self.output_names:
		          raise ValueError('Unknown entry in loss '
		                           'dictionary: "' + name + '". '
		                           'Only expected the following keys: ' +
		                           str(self.output_names))
		      loss_functions = []
		      for name in self.output_names:
		        if name not in loss:
		          logging.warning(
		              'Output "' + name + '" missing from loss dictionary. '
		              'We assume this was done on purpose, '
		              'and we will not be expecting '
		              'any data to be passed to "' + name + '" during training.',
		              stacklevel=2)
		        loss_functions.append(losses.get(loss.get(name)))
		    elif isinstance(loss, list):
		      if len(loss) != len(self.outputs):
		        raise ValueError('When passing a list as loss, '
		                         'it should have one entry per model outputs. '
		                         'The model has ' + str(len(self.outputs)) +
		                         ' outputs, but you passed loss=' + str(loss))
		      loss_functions = [losses.get(l) for l in loss]
		    else:
		      loss_function = losses.get(loss)
		      loss_functions = [loss_function for _ in range(len(self.outputs))]
		    self.loss_functions = loss_functions
		    weighted_losses = [_weighted_masked_objective(fn) for fn in loss_functions]
		    skip_indices = []
		    self._feed_outputs = []
		    self._feed_output_names = []
		    self._feed_output_shapes = []
		    self._feed_loss_fns = []
		    for i in range(len(weighted_losses)):
		      if weighted_losses[i] is None:
		        skip_indices.append(i)
		      else:
		        self._feed_outputs.append(self.outputs[i])
		        self._feed_output_names.append(self.output_names[i])
		        self._feed_output_shapes.append(self.internal_output_shapes[i])
		        self._feed_loss_fns.append(self.loss_functions[i])

		    # Prepare output masks.
		    masks = self.compute_mask(self.inputs, mask=None)
		    if masks is None:
		      masks = [None for _ in self.outputs]
		    if not isinstance(masks, list):
		      masks = [masks]

		    # Prepare loss weights.
		    if loss_weights is None:
		      loss_weights_list = [1. for _ in range(len(self.outputs))]
		    elif isinstance(loss_weights, dict):
		      for name in loss_weights:
		        if name not in self.output_names:
		          raise ValueError('Unknown entry in loss_weights '
		                           'dictionary: "' + name + '". '
		                           'Only expected the following keys: ' +
		                           str(self.output_names))
		      loss_weights_list = []
		      for name in self.output_names:
		        loss_weights_list.append(loss_weights.get(name, 1.))
		    elif isinstance(loss_weights, list):
		      if len(loss_weights) != len(self.outputs):
		        raise ValueError('When passing a list as loss_weights, '
		                         'it should have one entry per model outputs. '
		                         'The model has ' + str(len(self.outputs)) +
		                         ' outputs, but you passed loss_weights=' +
		                         str(loss_weights))
		      loss_weights_list = loss_weights
		    else:
		      raise TypeError('Could not interpret loss_weights argument: ' +
		                      str(loss_weights) + ' - expected a list of dicts.')

		    # Prepare sample weights.
		    sample_weights = []
		    sample_weight_modes = []
		    if isinstance(sample_weight_mode, dict):
		      for name in sample_weight_mode:
		        if name not in self.output_names:
		          raise ValueError('Unknown entry in '
		                           'sample_weight_mode dictionary: "' + name + '". '
		                           'Only expected the following keys: ' +
		                           str(self.output_names))
		      for i, name in enumerate(self.output_names):
		        if i in skip_indices:
		          weight = None
		          sample_weight_modes.append(None)
		        else:
		          if name not in sample_weight_mode:
		            raise ValueError('Output "' + name +
		                             '" missing from sample_weight_modes '
		                             'dictionary')
		          if sample_weight_mode.get(name) == 'temporal':
		            weight = K.placeholder(ndim=2, name=name + '_sample_weights')
		            sample_weight_modes.append('temporal')
		          else:
		            weight = K.placeholder(ndim=1, name=name + '_sample_weights')
		            sample_weight_modes.append(None)
		        sample_weights.append(weight)
		    elif isinstance(sample_weight_mode, list):
		      if len(sample_weight_mode) != len(self.outputs):
		        raise ValueError('When passing a list as sample_weight_mode, '
		                         'it should have one entry per model outputs. '
		                         'The model has ' + str(len(self.outputs)) +
		                         ' outputs, but you passed '
		                         'sample_weight_mode=' + str(sample_weight_mode))
		      for i in range(len(self.output_names)):
		        if i in skip_indices:
		          weight = None
		          sample_weight_modes.append(None)
		        else:
		          mode = sample_weight_mode[i]
		          name = self.output_names[i]
		          if mode == 'temporal':
		            weight = K.placeholder(ndim=2, name=name + '_sample_weights')
		            sample_weight_modes.append('temporal')
		          else:
		            weight = K.placeholder(ndim=1, name=name + '_sample_weights')
		            sample_weight_modes.append(None)
		        sample_weights.append(weight)
		    else:
		      for i, name in enumerate(self.output_names):
		        if i in skip_indices:
		          sample_weight_modes.append(None)
		          sample_weights.append(None)
		        else:
		          if sample_weight_mode == 'temporal':
		            sample_weights.append(
		                K.placeholder(ndim=2, name=name + '_sample_weights'))
		            sample_weight_modes.append('temporal')
		          else:
		            sample_weights.append(
		                K.placeholder(ndim=1, name=name + '_sample_weights'))
		            sample_weight_modes.append(None)
		    self.sample_weight_modes = sample_weight_modes
		    self._feed_sample_weight_modes = []
		    for i in range(len(self.outputs)):
		      if i not in skip_indices:
		        self._feed_sample_weight_modes.append(self.sample_weight_modes[i])

		    # Prepare targets of model.
		    self.targets = []
		    self._feed_targets = []
		    for i in range(len(self.outputs)):
		      if i in skip_indices:
		        self.targets.append(None)
		      else:
		        shape = self.internal_output_shapes[i]
		        name = self.output_names[i]
		        target = K.placeholder(
		            ndim=len(shape),
		            name=name + '_target',
		            sparse=K.is_sparse(self.outputs[i]),
		            dtype=K.dtype(self.outputs[i]))
		        self.targets.append(target)
		        self._feed_targets.append(target)

		    # Prepare metrics.
		    self.metrics = metrics
		    self.metrics_names = ['loss']
		    self.metrics_tensors = []

		    # Compute total loss.
		    total_loss = None
		    for i in range(len(self.outputs)):
		      if i in skip_indices:
		        continue
		      y_true = self.targets[i]
		      y_pred = self.outputs[i]
		      weighted_loss = weighted_losses[i]
		      sample_weight = sample_weights[i]
		      mask = masks[i]
		      loss_weight = loss_weights_list[i]
		      output_loss = weighted_loss(y_true, y_pred, sample_weight, mask)
		      if len(self.outputs) > 1:
		        self.metrics_tensors.append(output_loss)
		        self.metrics_names.append(self.output_names[i] + '_loss')
		      if total_loss is None:
		        total_loss = loss_weight * output_loss
		      else:
		        total_loss += loss_weight * output_loss
		    if total_loss is None:
		      if not self.losses:
		        raise RuntimeError('The model cannot be compiled '
		                           'because it has no loss to optimize.')
		      else:
		        total_loss = 0.

		    # Add regularization penalties
		    # and other layer-specific losses.
		    for loss_tensor in self.losses:
		      total_loss += loss_tensor

		    # List of same size as output_names.
		    # contains tuples (metrics for output, names of metrics).
		    nested_metrics = _collect_metrics(metrics, self.output_names)

		    def append_metric(layer_num, metric_name, metric_tensor):
		      """Helper function used in loop below."""
		      if len(self.output_names) > 1:
		        metric_name = self.output_layers[layer_num].name + '_' + metric_name
		      self.metrics_names.append(metric_name)
		      self.metrics_tensors.append(metric_tensor)

		    for i in range(len(self.outputs)):
		      if i in skip_indices:
		        continue
		      y_true = self.targets[i]
		      y_pred = self.outputs[i]
		      output_metrics = nested_metrics[i]
		      for metric in output_metrics:
		        if metric == 'accuracy' or metric == 'acc':
		          # custom handling of accuracy
		          # (because of class mode duality)
		          output_shape = self.internal_output_shapes[i]
		          acc_fn = None
		          if (output_shape[-1] == 1 or
		              self.loss_functions[i] == losses.binary_crossentropy):
		            # case: binary accuracy
		            acc_fn = metrics_module.binary_accuracy
		          elif self.loss_functions[i] == losses.sparse_categorical_crossentropy:
		            # case: categorical accuracy with sparse targets
		            acc_fn = metrics_module.sparse_categorical_accuracy
		          else:
		            acc_fn = metrics_module.categorical_accuracy

		          masked_fn = _masked_objective(acc_fn)
		          append_metric(i, 'acc', masked_fn(y_true, y_pred, mask=masks[i]))
		        else:
		          metric_fn = metrics_module.get(metric)
		          masked_metric_fn = _masked_objective(metric_fn)
		          metric_result = masked_metric_fn(y_true, y_pred, mask=masks[i])
		          metric_result = {metric_fn.__name__: metric_result}
		          for name, tensor in six.iteritems(metric_result):
		            append_metric(i, name, tensor)

		    # Prepare gradient updates and state updates.
		    self.total_loss = total_loss
		    self.sample_weights = sample_weights
		    self._feed_sample_weights = []
		    for i in range(len(self.sample_weights)):
		      if i not in skip_indices:
		        self._feed_sample_weights.append(sample_weights[i])

		    # Functions for train, test and predict will
		    # be compiled lazily when required.
		    # This saves time when the user is not using all functions.
		    self.train_function = None
		    self.test_function = None
		    self.predict_function = None
		    self._function_kwargs = kwargs

		    # Collected trainable weights and sort them deterministically.
		    trainable_weights = self.trainable_weights
		    # Sort weights by name.
		    if trainable_weights:
		      trainable_weights.sort(key=lambda x: x.name)
		    self._collected_trainable_weights = trainable_weights

		  def _make_train_function(self):
		    if not hasattr(self, 'train_function'):
		      raise RuntimeError('You must compile your model before using it.')
		    if self.train_function is None:
		      inputs = (
		          self._feed_inputs + self._feed_targets + self._feed_sample_weights)
		      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		        inputs += [K.learning_phase()]

		      training_updates = self.optimizer.get_updates(
		          self._collected_trainable_weights, self.constraints, self.total_loss)
		      updates = self.updates + training_updates
		      # Gets loss and metrics. Updates weights at each call.
		      self.train_function = K.function(
		          inputs, [self.total_loss] + self.metrics_tensors,
		          updates=updates,
		          name='train_function',
		          **self._function_kwargs)

		  def _make_test_function(self):
		    if not hasattr(self, 'test_function'):
		      raise RuntimeError('You must compile your model before using it.')
		    if self.test_function is None:
		      inputs = (
		          self._feed_inputs + self._feed_targets + self._feed_sample_weights)
		      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		        inputs += [K.learning_phase()]
		      # Return loss and metrics, no gradient updates.
		      # Does update the network states.
		      self.test_function = K.function(
		          inputs, [self.total_loss] + self.metrics_tensors,
		          updates=self.state_updates,
		          name='test_function',
		          **self._function_kwargs)

		  def _make_predict_function(self):
		    if not hasattr(self, 'predict_function'):
		      self.predict_function = None
		      self._function_kwargs = {}
		    if self.predict_function is None:
		      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		        inputs = self._feed_inputs + [K.learning_phase()]
		      else:
		        inputs = self._feed_inputs
		      # Gets network outputs. Does not update weights.
		      # Does update the network states.
		      self.predict_function = K.function(
		          inputs,
		          self.outputs,
		          updates=self.state_updates,
		          name='predict_function',
		          **self._function_kwargs)

		  def _fit_loop(self,
		                f,
		                ins,
		                out_labels=None,
		                batch_size=32,
		                epochs=100,
		                verbose=1,
		                callbacks=None,
		                val_f=None,
		                val_ins=None,
		                shuffle=True,
		                callback_metrics=None,
		                initial_epoch=0):
		    """Abstract fit function for `f(ins)`.

		    Assume that f returns a list, labeled by out_labels.

		    Arguments:
		        f: Keras function returning a list of tensors
		        ins: list of tensors to be fed to `f`
		        out_labels: list of strings, display names of
		            the outputs of `f`
		        batch_size: integer batch size
		        epochs: number of times to iterate over the data
		        verbose: verbosity mode, 0, 1 or 2
		        callbacks: list of callbacks to be called during training
		        val_f: Keras function to call for validation
		        val_ins: list of tensors to be fed to `val_f`
		        shuffle: whether to shuffle the data at the beginning of each epoch
		        callback_metrics: list of strings, the display names of the metrics
		            passed to the callbacks. They should be the
		            concatenation of list the display names of the outputs of
		             `f` and the list of display names of the outputs of `f_val`.
		        initial_epoch: epoch at which to start training
		            (useful for resuming a previous training run)

		    Returns:
		        `History` object.
		    """
		    do_validation = False
		    if val_f and val_ins:
		      do_validation = True
		      if verbose:
		        print('Train on %d samples, validate on %d samples' %
		              (ins[0].shape[0], val_ins[0].shape[0]))

		    if ins and hasattr(ins[0], 'shape'):
		      num_train_samples = ins[0].shape[0]
		    else:
		      # May happen if we are running `fit` without Numpy input data,
		      # i.e. if all inputs to the models are data tensors
		      # instead of placeholders.
		      # In that case we will run `fit` over a single batch.
		      num_train_samples = batch_size
		      verbose = 2
		    index_array = np.arange(num_train_samples)

		    self.history = cbks.History()
		    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
		    if verbose:
		      callbacks += [cbks.ProgbarLogger()]
		    callbacks = cbks.CallbackList(callbacks)
		    out_labels = out_labels or []

		    # it's possible to callback a different model than self
		    # (used by Sequential models)
		    if hasattr(self, 'callback_model') and self.callback_model:
		      callback_model = self.callback_model
		    else:
		      callback_model = self

		    callbacks.set_model(callback_model)
		    callbacks.set_params({
		        'batch_size': batch_size,
		        'epochs': epochs,
		        'samples': num_train_samples,
		        'verbose': verbose,
		        'do_validation': do_validation,
		        'metrics': callback_metrics or [],
		    })
		    callbacks.on_train_begin()
		    callback_model.stop_training = False
		    for cbk in callbacks:
		      cbk.validation_data = val_ins

		    for epoch in range(initial_epoch, epochs):
		      callbacks.on_epoch_begin(epoch)
		      if shuffle == 'batch':
		        index_array = _batch_shuffle(index_array, batch_size)
		      elif shuffle:
		        np.random.shuffle(index_array)

		      batches = _make_batches(num_train_samples, batch_size)
		      epoch_logs = {}
		      for batch_index, (batch_start, batch_end) in enumerate(batches):
		        batch_ids = index_array[batch_start:batch_end]
		        try:
		          if isinstance(ins[-1], float):
		            # Do not slice the training phase flag.
		            ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
		          else:
		            ins_batch = _slice_arrays(ins, batch_ids)
		        except TypeError:
		          raise TypeError('TypeError while preparing batch. '
		                          'If using HDF5 input data, '
		                          'pass shuffle="batch".')
		        batch_logs = {}
		        batch_logs['batch'] = batch_index
		        batch_logs['size'] = len(batch_ids)
		        callbacks.on_batch_begin(batch_index, batch_logs)
		        outs = f(ins_batch)
		        if not isinstance(outs, list):
		          outs = [outs]
		        for l, o in zip(out_labels, outs):
		          batch_logs[l] = o

		        callbacks.on_batch_end(batch_index, batch_logs)
		        if callback_model.stop_training:
		          break

		        if batch_index == len(batches) - 1:  # Last batch.
		          if do_validation:
		            val_outs = self._test_loop(
		                val_f, val_ins, batch_size=batch_size, verbose=0)
		            if not isinstance(val_outs, list):
		              val_outs = [val_outs]
		            # Same labels assumed.
		            for l, o in zip(out_labels, val_outs):
		              epoch_logs['val_' + l] = o
		      callbacks.on_epoch_end(epoch, epoch_logs)
		      if callback_model.stop_training:
		        break
		    callbacks.on_train_end()
		    return self.history

		  # 1. ins: list of large array shape (50, 224, 224, 3) as dataset, not iterators
		  # 2. batches = _make_batches(samples, batch_size): split total number of samples into list of batch_ranges like [(0, 32), (32, 50)]
		  # 3. from batch_range to batch of samples, and get predictions for this batch, shape (32, 2)
		  # 4. create a container of 0s for all predictions, shape (50,2)
		  # 5. fill the first batch of predictions into the container by its batch_ranges (0, 32)
		  # 6. fill the second batch of predictions into the container by its batch_range (32, 50)
		  def _predict_loop(self, f, ins, batch_size=32, verbose=0):
		    """Abstract method to loop over some data in batches.

		    Arguments:
		        f: Keras function returning a list of tensors.
		        ins: list of tensors to be fed to `f`.
		        batch_size: integer batch size.
		        verbose: verbosity mode.

		    Returns:
		        Array of predictions (if the model has a single output)
		        or list of arrays of predictions
		        (if the model has multiple outputs).
		    """
		    if ins and hasattr(ins[0], 'shape'):
		      samples = ins[0].shape[0]
		    else:
		      # May happen if we are running `predict` without Numpy input data,
		      # i.e. if all inputs to the models are data tensors
		      # instead of placeholders.
		      # In that case we will run `predict` over a single batch.
		      samples = batch_size
		      verbose = 2
		    outs = []
		    if verbose == 1:
		      progbar = Progbar(target=samples)
		    batches = _make_batches(samples, batch_size)
		    index_array = np.arange(samples)
		    for batch_index, (batch_start, batch_end) in enumerate(batches):
		      batch_ids = index_array[batch_start:batch_end]
		      if ins and isinstance(ins[-1], float):
		        # Do not slice the training phase flag.
		        ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
		      else:
		        ins_batch = _slice_arrays(ins, batch_ids)

		      batch_outs = f(ins_batch)
		      if not isinstance(batch_outs, list):
		        batch_outs = [batch_outs]
		      if batch_index == 0:
		        for batch_out in batch_outs:
		          shape = (samples,) + batch_out.shape[1:]
		          outs.append(np.zeros(shape, dtype=batch_out.dtype))

		      for i, batch_out in enumerate(batch_outs):
		        outs[i][batch_start:batch_end] = batch_out
		      if verbose == 1:
		        progbar.update(batch_end)
		    if len(outs) == 1:
		      return outs[0]
		    return outs

		  def _test_loop(self, f, ins, batch_size=32, verbose=0):
		    """Abstract method to loop over some data in batches.

		    Arguments:
		        f: Keras function returning a list of tensors.
		        ins: list of tensors to be fed to `f`.
		        batch_size: integer batch size.
		        verbose: verbosity mode.

		    Returns:
		        Scalar loss (if the model has a single output and no metrics)
		        or list of scalars (if the model has multiple outputs
		        and/or metrics). The attribute `model.metrics_names` will give you
		        the display labels for the scalar outputs.
		    """
		    if ins and hasattr(ins[0], 'shape'):
		      samples = ins[0].shape[0]
		    else:
		      # May happen if we are running `evaluate` without Numpy input data,
		      # i.e. if all inputs to the models are data tensors
		      # instead of placeholders.
		      # In that case we will run `evaluate` over a single batch.
		      samples = batch_size
		      verbose = 2

		    outs = []
		    if verbose == 1:
		      progbar = Progbar(target=samples)
		    batches = _make_batches(samples, batch_size)
		    index_array = np.arange(samples)
		    for batch_index, (batch_start, batch_end) in enumerate(batches):
		      batch_ids = index_array[batch_start:batch_end]
		      if isinstance(ins[-1], float):
		        # Do not slice the training phase flag.
		        ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
		      else:
		        ins_batch = _slice_arrays(ins, batch_ids)

		      batch_outs = f(ins_batch)
		      if isinstance(batch_outs, list):
		        if batch_index == 0:
		          for batch_out in enumerate(batch_outs):
		            outs.append(0.)
		        for i, batch_out in enumerate(batch_outs):
		          outs[i] += batch_out * len(batch_ids)
		      else:
		        if batch_index == 0:
		          outs.append(0.)
		        outs[0] += batch_outs * len(batch_ids)

		      if verbose == 1:
		        progbar.update(batch_end)
		    for i in range(len(outs)):
		      outs[i] /= samples
		    if len(outs) == 1:
		      return outs[0]
		    return outs

		  def _standardize_user_data(self,
		                             x,
		                             y,
		                             sample_weight=None,
		                             class_weight=None,
		                             check_batch_axis=True,
		                             batch_size=None):
		    if not hasattr(self, 'optimizer'):
		      raise RuntimeError('You must compile a model before '
		                         'training/testing. '
		                         'Use `model.compile(optimizer, loss)`.')

		    output_shapes = []
		    for output_shape, loss_fn in zip(self._feed_output_shapes,
		                                     self._feed_loss_fns):
		      if loss_fn.__name__ == 'sparse_categorical_crossentropy':
		        output_shapes.append(output_shape[:-1] + (1,))
		      elif getattr(losses, loss_fn.__name__, None) is None:
		        output_shapes.append(None)
		      else:
		        output_shapes.append(output_shape)
		    x = _standardize_input_data(
		        x,
		        self._feed_input_names,
		        self._feed_input_shapes,
		        check_batch_axis=False,
		        exception_prefix='input')
		    y = _standardize_input_data(
		        y,
		        self._feed_output_names,
		        output_shapes,
		        check_batch_axis=False,
		        exception_prefix='target')
		    sample_weights = _standardize_sample_weights(sample_weight,
		                                                 self._feed_output_names)
		    class_weights = _standardize_class_weights(class_weight,
		                                               self._feed_output_names)
		    sample_weights = [
		        _standardize_weights(ref, sw, cw, mode)
		        for (ref, sw, cw, mode) in zip(y, sample_weights, class_weights,
		                                       self._feed_sample_weight_modes)
		    ]
		    _check_array_lengths(x, y, sample_weights)
		    _check_loss_and_target_compatibility(y, self._feed_loss_fns,
		                                         self._feed_output_shapes)
		    if self.stateful and batch_size:
		      if x[0].shape[0] % batch_size != 0:
		        raise ValueError('In a stateful network, '
		                         'you should only pass inputs with '
		                         'a number of samples that can be '
		                         'divided by the batch size. Found: ' +
		                         str(x[0].shape[0]) + ' samples')
		    return x, y, sample_weights

		  def _get_deduped_metrics_names(self):
		    out_labels = self.metrics_names

		    # Rename duplicated metrics name
		    # (can happen with an output layer shared among multiple dataflows).
		    deduped_out_labels = []
		    for i, label in enumerate(out_labels):
		      new_label = label
		      if out_labels.count(label) > 1:
		        dup_idx = out_labels[:i].count(label)
		        new_label += '_' + str(dup_idx + 1)
		      deduped_out_labels.append(new_label)
		    return deduped_out_labels

		  def fit(self,
		          x=None,
		          y=None,
		          batch_size=32,
		          epochs=1,
		          verbose=1,
		          callbacks=None,
		          validation_split=0.,
		          validation_data=None,
		          shuffle=True,
		          class_weight=None,
		          sample_weight=None,
		          initial_epoch=0):
		    """Trains the model for a fixed number of epochs (iterations on a dataset).

		    Arguments:
		        x: Numpy array of training data,
		            or list of Numpy arrays if the model has multiple inputs.
		            If all inputs in the model are named,
		            you can also pass a dictionary
		            mapping input names to Numpy arrays.
		        y: Numpy array of target data,
		            or list of Numpy arrays if the model has multiple outputs.
		            If all outputs in the model are named,
		            you can also pass a dictionary
		            mapping output names to Numpy arrays.
		        batch_size: integer. Number of samples per gradient update.
		        epochs: integer, the number of times to iterate
		            over the training data arrays.
		        verbose: 0, 1, or 2. Verbosity mode.
		            0 = silent, 1 = verbose, 2 = one log line per epoch.
		        callbacks: list of callbacks to be called during training.
		            See [callbacks](/callbacks).
		        validation_split: float between 0 and 1:
		            fraction of the training data to be used as validation data.
		            The model will set apart this fraction of the training data,
		            will not train on it, and will evaluate
		            the loss and any model metrics
		            on this data at the end of each epoch.
		        validation_data: data on which to evaluate
		            the loss and any model metrics
		            at the end of each epoch. The model will not
		            be trained on this data.
		            This could be a tuple (x_val, y_val)
		            or a tuple (x_val, y_val, val_sample_weights).
		        shuffle: boolean, whether to shuffle the training data
		            before each epoch.
		        class_weight: optional dictionary mapping
		            class indices (integers) to
		            a weight (float) to apply to the model's loss for the samples
		            from this class during training.
		            This can be useful to tell the model to "pay more attention" to
		            samples from an under-represented class.
		        sample_weight: optional array of the same length as x, containing
		            weights to apply to the model's loss for each sample.
		            In the case of temporal data, you can pass a 2D array
		            with shape (samples, sequence_length),
		            to apply a different weight to every timestep of every sample.
		            In this case you should make sure to specify
		            sample_weight_mode="temporal" in compile().
		        initial_epoch: epoch at which to start training
		            (useful for resuming a previous training run)

		    Returns:
		        A `History` instance. Its `history` attribute contains
		        all information collected during training.

		    Raises:
		        ValueError: In case of mismatch between the provided input data
		            and what the model expects.
		    """
		    # Validate user data.
		    x, y, sample_weights = self._standardize_user_data(
		        x,
		        y,
		        sample_weight=sample_weight,
		        class_weight=class_weight,
		        check_batch_axis=False,
		        batch_size=batch_size)
		    # Prepare validation data.
		    if validation_data:
		      do_validation = True
		      if len(validation_data) == 2:
		        val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
		        val_sample_weight = None
		      elif len(validation_data) == 3:
		        val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
		      else:
		        raise ValueError(
		            'When passing validation_data, '
		            'it must contain 2 (x_val, y_val) '
		            'or 3 (x_val, y_val, val_sample_weights) '
		            'items, however it contains %d items' % len(validation_data))

		      val_x, val_y, val_sample_weights = self._standardize_user_data(
		          val_x,
		          val_y,
		          sample_weight=val_sample_weight,
		          check_batch_axis=False,
		          batch_size=batch_size)
		      self._make_test_function()
		      val_f = self.test_function
		      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		        val_ins = val_x + val_y + val_sample_weights + [0.]
		      else:
		        val_ins = val_x + val_y + val_sample_weights

		    elif validation_split and 0. < validation_split < 1.:
		      do_validation = True
		      split_at = int(len(x[0]) * (1. - validation_split))
		      x, val_x = (_slice_arrays(x, 0, split_at), _slice_arrays(x, split_at))
		      y, val_y = (_slice_arrays(y, 0, split_at), _slice_arrays(y, split_at))
		      sample_weights, val_sample_weights = (_slice_arrays(
		          sample_weights, 0, split_at), _slice_arrays(sample_weights, split_at))
		      self._make_test_function()
		      val_f = self.test_function
		      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		        val_ins = val_x + val_y + val_sample_weights + [0.]
		      else:
		        val_ins = val_x + val_y + val_sample_weights
		    else:
		      do_validation = False
		      val_f = None
		      val_ins = None

		    # Prepare input arrays and training function.
		    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		      ins = x + y + sample_weights + [1.]
		    else:
		      ins = x + y + sample_weights
		    self._make_train_function()
		    f = self.train_function

		    # Prepare display labels.
		    out_labels = self._get_deduped_metrics_names()

		    if do_validation:
		      callback_metrics = copy.copy(out_labels) + [
		          'val_' + n for n in out_labels
		      ]
		    else:
		      callback_metrics = copy.copy(out_labels)

		    # Delegate logic to `_fit_loop`.
		    return self._fit_loop(
		        f,
		        ins,
		        out_labels=out_labels,
		        batch_size=batch_size,
		        epochs=epochs,
		        verbose=verbose,
		        callbacks=callbacks,
		        val_f=val_f,
		        val_ins=val_ins,
		        shuffle=shuffle,
		        callback_metrics=callback_metrics,
		        initial_epoch=initial_epoch)

		  def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
		    """Returns the loss value & metrics values for the model in test mode.

		    Computation is done in batches.

		    Arguments:
		        x: Numpy array of test data,
		            or list of Numpy arrays if the model has multiple inputs.
		            If all inputs in the model are named,
		            you can also pass a dictionary
		            mapping input names to Numpy arrays.
		        y: Numpy array of target data,
		            or list of Numpy arrays if the model has multiple outputs.
		            If all outputs in the model are named,
		            you can also pass a dictionary
		            mapping output names to Numpy arrays.
		        batch_size: integer. Number of samples per gradient update.
		        verbose: verbosity mode, 0 or 1.
		        sample_weight: Array of weights to weight the contribution
		            of different samples to the loss and metrics.

		    Returns:
		        Scalar test loss (if the model has a single output and no metrics)
		        or list of scalars (if the model has multiple outputs
		        and/or metrics). The attribute `model.metrics_names` will give you
		        the display labels for the scalar outputs.
		    """
		    # Validate user data.
		    x, y, sample_weights = self._standardize_user_data(
		        x,
		        y,
		        sample_weight=sample_weight,
		        check_batch_axis=False,
		        batch_size=batch_size)
		    # Prepare inputs, delegate logic to `_test_loop`.
		    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		      ins = x + y + sample_weights + [0.]
		    else:
		      ins = x + y + sample_weights
		    self._make_test_function()
		    f = self.test_function
		    return self._test_loop(f, ins, batch_size=batch_size, verbose=verbose)

		  def predict(self, x, batch_size=32, verbose=0):
		    """Generates output predictions for the input samples.

		    Computation is done in batches.

		    Arguments:
		        x: the input data, as a Numpy array
		            (or list of Numpy arrays if the model has multiple outputs).
		        batch_size: integer.
		        verbose: verbosity mode, 0 or 1.

		    Returns:
		        Numpy array(s) of predictions.

		    Raises:
		        ValueError: In case of mismatch between the provided
		            input data and the model's expectations,
		            or in case a stateful model receives a number of samples
		            that is not a multiple of the batch size.
		    """
		    # Validate user data: x[0].shape = (all_samples, width, height, channels)
		    x = _standardize_input_data(
		        x,
		        self._feed_input_names,
		        self._feed_input_shapes,
		        check_batch_axis=False)
		    if self.stateful:
		      if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
		        raise ValueError('In a stateful network, '
		                         'you should only pass inputs with '
		                         'a number of samples that can be '
		                         'divided by the batch size. Found: ' +
		                         str(x[0].shape[0]) + ' samples. '
		                         'Batch size: ' + str(batch_size) + '.')

		    # Prepare inputs, delegate logic to `_predict_loop`.
		    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		      ins = x + [0.]
		    else:
		      ins = x
		    self._make_predict_function()
		    f = self.predict_function
		    return self._predict_loop(f, ins, batch_size=batch_size, verbose=verbose)

		  def train_on_batch(self, x, y, sample_weight=None, class_weight=None):
		    """Runs a single gradient update on a single batch of data.

		    Arguments:
		        x: Numpy array of training data,
		            or list of Numpy arrays if the model has multiple inputs.
		            If all inputs in the model are named,
		            you can also pass a dictionary
		            mapping input names to Numpy arrays.
		        y: Numpy array of target data,
		            or list of Numpy arrays if the model has multiple outputs.
		            If all outputs in the model are named,
		            you can also pass a dictionary
		            mapping output names to Numpy arrays.
		        sample_weight: optional array of the same length as x, containing
		            weights to apply to the model's loss for each sample.
		            In the case of temporal data, you can pass a 2D array
		            with shape (samples, sequence_length),
		            to apply a different weight to every timestep of every sample.
		            In this case you should make sure to specify
		            sample_weight_mode="temporal" in compile().
		        class_weight: optional dictionary mapping
		            class indices (integers) to
		            a weight (float) to apply to the model's loss for the samples
		            from this class during training.
		            This can be useful to tell the model to "pay more attention" to
		            samples from an under-represented class.

		    Returns:
		        Scalar training loss
		        (if the model has a single output and no metrics)
		        or list of scalars (if the model has multiple outputs
		        and/or metrics). The attribute `model.metrics_names` will give you
		        the display labels for the scalar outputs.
		    """
		    x, y, sample_weights = self._standardize_user_data(
		        x,
		        y,
		        sample_weight=sample_weight,
		        class_weight=class_weight,
		        check_batch_axis=True)
		    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		      ins = x + y + sample_weights + [1.]
		    else:
		      ins = x + y + sample_weights
		    self._make_train_function()
		    outputs = self.train_function(ins)
		    if len(outputs) == 1:
		      return outputs[0]
		    return outputs

		  def test_on_batch(self, x, y, sample_weight=None):
		    """Test the model on a single batch of samples.

		    Arguments:
		        x: Numpy array of test data,
		            or list of Numpy arrays if the model has multiple inputs.
		            If all inputs in the model are named,
		            you can also pass a dictionary
		            mapping input names to Numpy arrays.
		        y: Numpy array of target data,
		            or list of Numpy arrays if the model has multiple outputs.
		            If all outputs in the model are named,
		            you can also pass a dictionary
		            mapping output names to Numpy arrays.
		        sample_weight: optional array of the same length as x, containing
		            weights to apply to the model's loss for each sample.
		            In the case of temporal data, you can pass a 2D array
		            with shape (samples, sequence_length),
		            to apply a different weight to every timestep of every sample.
		            In this case you should make sure to specify
		            sample_weight_mode="temporal" in compile().

		    Returns:
		        Scalar test loss (if the model has a single output and no metrics)
		        or list of scalars (if the model has multiple outputs
		        and/or metrics). The attribute `model.metrics_names` will give you
		        the display labels for the scalar outputs.
		    """
		    x, y, sample_weights = self._standardize_user_data(
		        x, y, sample_weight=sample_weight, check_batch_axis=True)
		    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		      ins = x + y + sample_weights + [0.]
		    else:
		      ins = x + y + sample_weights
		    self._make_test_function()
		    outputs = self.test_function(ins)
		    if len(outputs) == 1:
		      return outputs[0]
		    return outputs

		  def predict_on_batch(self, x):
		    """Returns predictions for a single batch of samples.

		    Arguments:
		        x: Input samples, as a Numpy array.

		    Returns:
		        Numpy array(s) of predictions.
		    """
		    x = _standardize_input_data(x, self._feed_input_names,
		                                self._feed_input_shapes)
		    if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		      ins = x + [0.]
		    else:
		      ins = x
		    self._make_predict_function()
		    outputs = self.predict_function(ins)
		    if len(outputs) == 1:
		      return outputs[0]
		    return outputs

		  def fit_generator(self,
		                    generator,
		                    steps_per_epoch,
		                    epochs=1,
		                    verbose=1,
		                    callbacks=None,
		                    validation_data=None,
		                    validation_steps=None,
		                    class_weight=None,
		                    max_q_size=10,
		                    workers=1,
		                    pickle_safe=False,
		                    initial_epoch=0):
		    """Fits the model on data yielded batch-by-batch by a Python generator.

		    The generator is run in parallel to the model, for efficiency.
		    For instance, this allows you to do real-time data augmentation
		    on images on CPU in parallel to training your model on GPU.

		    Arguments:
		        generator: a generator.
		            The output of the generator must be either
		            - a tuple (inputs, targets)
		            - a tuple (inputs, targets, sample_weights).
		            All arrays should contain the same number of samples.
		            The generator is expected to loop over its data
		            indefinitely. An epoch finishes when `steps_per_epoch`
		            batches have been seen by the model.
		        steps_per_epoch: Total number of steps (batches of samples)
		            to yield from `generator` before declaring one epoch
		            finished and starting the next epoch. It should typically
		            be equal to the number of unique samples if your dataset
		            divided by the batch size.
		        epochs: integer, total number of iterations on the data.
		        verbose: verbosity mode, 0, 1, or 2.
		        callbacks: list of callbacks to be called during training.
		        validation_data: this can be either
		            - a generator for the validation data
		            - a tuple (inputs, targets)
		            - a tuple (inputs, targets, sample_weights).
		        validation_steps: Only relevant if `validation_data`
		            is a generator. Total number of steps (batches of samples)
		            to yield from `generator` before stopping.
		        class_weight: dictionary mapping class indices to a weight
		            for the class.
		        max_q_size: maximum size for the generator queue
		        workers: maximum number of processes to spin up
		            when using process based threading
		        pickle_safe: if True, use process based threading.
		            Note that because
		            this implementation relies on multiprocessing,
		            you should not pass
		            non picklable arguments to the generator
		            as they can't be passed
		            easily to children processes.
		        initial_epoch: epoch at which to start training
		            (useful for resuming a previous training run)

		    Returns:
		        A `History` object.

		    Example:

		    ```python
		        def generate_arrays_from_file(path):
		            while 1:
		                f = open(path)
		                for line in f:
		                    # create numpy arrays of input data
		                    # and labels, from each line in the file
		                    x1, x2, y = process_line(line)
		                    yield ({'input_1': x1, 'input_2': x2}, {'output': y})
		                f.close()

		        model.fit_generator(generate_arrays_from_file('/my_file.txt'),
		                            steps_per_epoch=10000, epochs=10)
		    ```

		    Raises:
		        ValueError: In case the generator yields
		            data in an invalid format.
		    """
		    wait_time = 0.01  # in seconds
		    epoch = initial_epoch

			# prepare self.train_function and self.test_function
		    do_validation = bool(validation_data)
		    self._make_train_function()
		    if do_validation:
		      self._make_test_function()

			# when validation_data is a generator, make sure validation_steps is set
		    # python 2 has 'next', 3 has '__next__'
		    # avoid any explicit version checks
		    val_gen = (hasattr(validation_data, 'next') or
		               hasattr(validation_data, '__next__'))
		    if val_gen and not validation_steps:
		      raise ValueError('When using a generator for validation data, '
		                       'you must specify a value for '
		                       '`validation_steps`.')

		    # Prepare display labels: e.g. loss, acc, val_loss, val_acc
		    out_labels = self._get_deduped_metrics_names()
		    callback_metrics = out_labels + ['val_' + n for n in out_labels]

		    # prepare callbacks:
			# 1. create a keras.callbacks.History object
			# 2. prepare a number of callback objects
			# 3. store this group of callback objects into a single object
		    self.history = cbks.History()
		    callbacks = [cbks.BaseLogger()] + (callbacks or []) + [self.history]
		    if verbose:
		      callbacks += [cbks.ProgbarLogger(count_mode='steps')]
		    callbacks = cbks.CallbackList(callbacks)

		    # it's possible to callback a different model than self:
			# our model for training may have a different model or a callback_model
		    if hasattr(self, 'callback_model') and self.callback_model:
		      callback_model = self.callback_model
		    else:
		      callback_model = self
			# give our model to every callback objects as their attribute
		    callbacks.set_model(callback_model)
			# give our args or params to every callback objects as their attribute
		    callbacks.set_params({
		        'epochs': epochs,
		        'steps': steps_per_epoch,
		        'verbose': verbose,
		        'do_validation': do_validation,
		        'metrics': callback_metrics,
		    })
			# ? this callbacks precedures above will be run at the start of training ?
		    callbacks.on_train_begin()

		    if do_validation and not val_gen:
		      if len(validation_data) == 2:
		        val_x, val_y = validation_data  # pylint: disable=unpacking-non-sequence
		        val_sample_weight = None
		      elif len(validation_data) == 3:
		        val_x, val_y, val_sample_weight = validation_data  # pylint: disable=unpacking-non-sequence
		      else:
		        raise ValueError('validation_data should be a tuple '
		                         '`(val_x, val_y, val_sample_weight)` '
		                         'or `(val_x, val_y)`. Found: ' + str(validation_data))
		      val_x, val_y, val_sample_weights = self._standardize_user_data(
		          val_x, val_y, val_sample_weight)
		      val_data = val_x + val_y + val_sample_weights
		      if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
		        val_data += [0.]
		      for cbk in callbacks:
		        cbk.validation_data = val_data
		    enqueuer = None

		    try:
			  # builds a queue out of a data generator
		      enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
			  # Kicks off threads which add data from the generator into the queue
		      enqueuer.start(max_q_size=max_q_size, workers=workers)

		      callback_model.stop_training = False
		      while epoch < epochs:
		        callbacks.on_epoch_begin(epoch)
		        steps_done = 0
		        batch_index = 0
		        while steps_done < steps_per_epoch:
		          generator_output = None
		          while enqueuer.is_running():
		            if not enqueuer.queue.empty():
		              generator_output = enqueuer.queue.get()
		              break
		            else:
		              time.sleep(wait_time)

		          if not hasattr(generator_output, '__len__'):
		            raise ValueError('output of generator should be '
		                             'a tuple `(x, y, sample_weight)` '
		                             'or `(x, y)`. Found: ' + str(generator_output))
		          if len(generator_output) == 2:
		            x, y = generator_output  # pylint: disable=unpacking-non-sequence
		            sample_weight = None
		          elif len(generator_output) == 3:
		            x, y, sample_weight = generator_output  # pylint: disable=unpacking-non-sequence
		          else:
		            raise ValueError('output of generator should be '
		                             'a tuple `(x, y, sample_weight)` '
		                             'or `(x, y)`. Found: ' + str(generator_output))
		          # build batch logs
		          batch_logs = {}
		          if isinstance(x, list):
		            batch_size = x[0].shape[0]
		          elif isinstance(x, dict):
		            batch_size = list(x.values())[0].shape[0]
		          else:
		            batch_size = x.shape[0]
		          batch_logs['batch'] = batch_index
		          batch_logs['size'] = batch_size
		          callbacks.on_batch_begin(batch_index, batch_logs)

		          outs = self.train_on_batch(
		              x, y, sample_weight=sample_weight, class_weight=class_weight)

		          if not isinstance(outs, list):
		            outs = [outs]
		          for l, o in zip(out_labels, outs):
		            batch_logs[l] = o

		          callbacks.on_batch_end(batch_index, batch_logs)

		          # Construct epoch logs.
		          epoch_logs = {}
		          batch_index += 1
		          steps_done += 1

		          # Epoch finished.
		          if steps_done >= steps_per_epoch and do_validation:
		            if val_gen:
		              val_outs = self.evaluate_generator(
		                  validation_data,
		                  validation_steps,
		                  max_q_size=max_q_size,
		                  workers=workers,
		                  pickle_safe=pickle_safe)
		            else:
		              # No need for try/except because
		              # data has already been validated.
		              val_outs = self.evaluate(
		                  val_x,
		                  val_y,
		                  batch_size=batch_size,
		                  sample_weight=val_sample_weights,
		                  verbose=0)
		            if not isinstance(val_outs, list):
		              val_outs = [val_outs]
		            # Same labels assumed.
		            for l, o in zip(out_labels, val_outs):
		              epoch_logs['val_' + l] = o

		        callbacks.on_epoch_end(epoch, epoch_logs)
		        epoch += 1
		        if callback_model.stop_training:
		          break

		    finally:
		      if enqueuer is not None:
		        enqueuer.stop()

		    callbacks.on_train_end()
		    return self.history

		  def evaluate_generator(self,
		                         generator,
		                         steps,
		                         max_q_size=10,
		                         workers=1,
		                         pickle_safe=False):
		    """Evaluates the model on a data generator.

		    The generator should return the same kind of data
		    as accepted by `test_on_batch`.

		    Arguments:
		        generator: Generator yielding tuples (inputs, targets)
		            or (inputs, targets, sample_weights)
		        steps: Total number of steps (batches of samples)
		            to yield from `generator` before stopping.
		        max_q_size: maximum size for the generator queue
		        workers: maximum number of processes to spin up
		            when using process based threading
		        pickle_safe: if True, use process based threading.
		            Note that because
		            this implementation relies on multiprocessing,
		            you should not pass
		            non picklable arguments to the generator
		            as they can't be passed
		            easily to children processes.

		    Returns:
		        Scalar test loss (if the model has a single output and no metrics)
		        or list of scalars (if the model has multiple outputs
		        and/or metrics). The attribute `model.metrics_names` will give you
		        the display labels for the scalar outputs.

		    Raises:
		        ValueError: In case the generator yields
		            data in an invalid format.
		    """
		    self._make_test_function()

		    steps_done = 0
		    wait_time = 0.01
		    all_outs = []
		    batch_sizes = []
		    enqueuer = None

		    try:
		      enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
		      enqueuer.start(workers=workers, max_q_size=max_q_size)

		      while steps_done < steps:
		        generator_output = None
		        while enqueuer.is_running():
		          if not enqueuer.queue.empty():
		            generator_output = enqueuer.queue.get()
		            break
		          else:
		            time.sleep(wait_time)

		        if not hasattr(generator_output, '__len__'):
		          raise ValueError('output of generator should be a tuple '
		                           '(x, y, sample_weight) '
		                           'or (x, y). Found: ' + str(generator_output))
		        if len(generator_output) == 2:
		          x, y = generator_output  # pylint: disable=unpacking-non-sequence
		          sample_weight = None
		        elif len(generator_output) == 3:
		          x, y, sample_weight = generator_output  # pylint: disable=unpacking-non-sequence
		        else:
		          raise ValueError('output of generator should be a tuple '
		                           '(x, y, sample_weight) '
		                           'or (x, y). Found: ' + str(generator_output))
		        outs = self.test_on_batch(x, y, sample_weight=sample_weight)

		        if isinstance(x, list):
		          batch_size = len(x[0])
		        elif isinstance(x, dict):
		          batch_size = len(list(x.values())[0])
		        else:
		          batch_size = len(x)
		        all_outs.append(outs)

		        steps_done += 1
		        batch_sizes.append(batch_size)

		    finally:
		      if enqueuer is not None:
		        enqueuer.stop()

		    if not isinstance(outs, list):
		      return np.average(np.asarray(all_outs), weights=batch_sizes)
		    else:
		      averages = []
		      for i in range(len(outs)):
		        averages.append(
		            np.average([out[i] for out in all_outs], weights=batch_sizes))
		      return averages

		  # 1. if steps = 3, generator's batch_sizes = 32, a full epoch = 21 then total samples predicted is 21*3 = 63; if a full epoch = 33, then total samples = 32*3 = 96
		  # 2. generator_output = enqueuer.queue.get() provide a batch of samples;
		  # 3. outs = self.predict_on_batch(x), turn batch_samples to predictions
		  # 4. return np.concatenate(all_outs[0]), to get a number of batches of predictions arrays, into a single array of predictions
		  def predict_generator(self,
		                        generator,
		                        steps,
		                        max_q_size=10,
		                        workers=1,
		                        pickle_safe=False,
		                        verbose=0):
		    """Generates predictions for the input samples from a data generator.

		    The generator should return the same kind of data as accepted by
		    `predict_on_batch`.

		    Arguments:
		        generator: Generator yielding batches of input samples.
		        steps: Total number of steps (batches of samples)
		            to yield from `generator` before stopping.
		        max_q_size: Maximum size for the generator queue.
		        workers: Maximum number of processes to spin up
		            when using process based threading
		        pickle_safe: If `True`, use process based threading.
		            Note that because
		            this implementation relies on multiprocessing,
		            you should not pass
		            non picklable arguments to the generator
		            as they can't be passed
		            easily to children processes.
		        verbose: verbosity mode, 0 or 1.

		    Returns:
		        Numpy array(s) of predictions.

		    Raises:
		        ValueError: In case the generator yields
		            data in an invalid format.
		    """
		    self._make_predict_function()

		    steps_done = 0
		    wait_time = 0.01
		    all_outs = []
		    enqueuer = None

		    try:
		      enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
		      enqueuer.start(workers=workers, max_q_size=max_q_size)

		      if verbose == 1:
		        progbar = Progbar(target=steps)

		      while steps_done < steps:
		        generator_output = None
		        while enqueuer.is_running():
		          if not enqueuer.queue.empty():
		            generator_output = enqueuer.queue.get()
		            break
		          else:
		            time.sleep(wait_time)

		        if isinstance(generator_output, tuple):
		          # Compatibility with the generators
		          # used for training.
		          if len(generator_output) == 2:
		            x, _ = generator_output  # pylint: disable=unpacking-non-sequence
		          elif len(generator_output) == 3:
		            x, _, _ = generator_output  # pylint: disable=unpacking-non-sequence
		          else:
		            raise ValueError('output of generator should be '
		                             'a tuple `(x, y, sample_weight)` '
		                             'or `(x, y)`. Found: ' + str(generator_output))
		        else:
		          # Assumes a generator that only
		          # yields inputs (not targets and sample weights).
		          x = generator_output

		        outs = self.predict_on_batch(x)
		        if not isinstance(outs, list):
		          outs = [outs]

		        if not all_outs:
		          for out in outs:
		            all_outs.append([])

		        for i, out in enumerate(outs):
		          all_outs[i].append(out)
		        steps_done += 1
		        if verbose == 1:
		          progbar.update(steps_done)

		    finally:
		      if enqueuer is not None:
		        enqueuer.stop()

		    if len(all_outs) == 1:
		      if steps_done == 1:
		        return all_outs[0][0]
		      else:
		        return np.concatenate(all_outs[0])
		    if steps_done == 1:
		      return [out for out in all_outs]
		    else:
		      return [np.concatenate(out) for out in all_outs]

# datasets
class datasets_folder:

	class __init__py:

		"""Keras datasets: utilities for downloading and pre-processing common datasets.
		"""
		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		from tensorflow.contrib.keras.python.keras.datasets import boston_housing
		from tensorflow.contrib.keras.python.keras.datasets import cifar10
		from tensorflow.contrib.keras.python.keras.datasets import cifar100
		from tensorflow.contrib.keras.python.keras.datasets import imdb
		from tensorflow.contrib.keras.python.keras.datasets import mnist
		from tensorflow.contrib.keras.python.keras.datasets import reuters

	class boston_housing_py:

		"""Boston housing price regression dataset.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np

			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		def load_data(path='boston_housing.npz', seed=113, test_split=0.2):
		  """Loads the Boston Housing dataset.

		  Arguments:
		      path: path where to cache the dataset locally
		          (relative to ~/.keras/datasets).
		      seed: Random seed for shuffling the data
		          before computing the test split.
		      test_split: fraction of the data to reserve as test set.

		  Returns:
		      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
		  """
		  assert 0 <= test_split < 1
		  fh = 'f553886a1f8d56431e820c5b82552d9d95cfcb96d1e678153f8839538947dff5'
		  path = get_file(
		      path,
		      origin='https://s3.amazonaws.com/keras-datasets/boston_housing.npz',
		      file_hash=fh)
		  f = np.load(path)
		  x = f['x']
		  y = f['y']
		  f.close()

		  np.random.seed(seed)
		  np.random.shuffle(x)
		  np.random.seed(seed)
		  np.random.shuffle(y)

		  x_train = np.array(x[:int(len(x) * (1 - test_split))])
		  y_train = np.array(y[:int(len(x) * (1 - test_split))])
		  x_test = np.array(x[int(len(x) * (1 - test_split)):])
		  y_test = np.array(y[int(len(x) * (1 - test_split)):])
		  return (x_train, y_train), (x_test, y_test)

	class cifar_py:

		"""Utilities used by the CIFAR10 and CIFAR100 datasets.
		"""

		from __future__ import absolute_import
		from __future__ import division
		from __future__ import print_function

		import sys

		from six.moves import cPickle


		def load_batch(fpath, label_key='labels'):
		  """Internal utility for parsing CIFAR data.

		  Arguments:
		      fpath: path the file to parse.
		      label_key: key for label data in the retrieve
		          dictionary.

		  Returns:
		      A tuple `(data, labels)`.
		  """
		  f = open(fpath, 'rb')
		  if sys.version_info < (3,):
		    d = cPickle.load(f)
		  else:
		    d = cPickle.load(f, encoding='bytes')
		    # decode utf8
		    d_decoded = {}
		    for k, v in d.items():
		      d_decoded[k.decode('utf8')] = v
		    d = d_decoded
		  f.close()
		  data = d['data']
		  labels = d[label_key]

		  data = data.reshape(data.shape[0], 3, 32, 32)
		  return data, labels

	class cifar10_py:

		"""CIFAR10 small image classification dataset.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import os

			import numpy as np

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		def load_data():
		  """Loads CIFAR10 dataset.

		  Returns:
		      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
		  """
		  dirname = 'cifar-10-batches-py'
		  origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
		  path = get_file(dirname, origin=origin, untar=True)

		  num_train_samples = 50000

		  x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
		  y_train = np.zeros((num_train_samples,), dtype='uint8')

		  for i in range(1, 6):
		    fpath = os.path.join(path, 'data_batch_' + str(i))
		    data, labels = load_batch(fpath)
		    x_train[(i - 1) * 10000:i * 10000, :, :, :] = data
		    y_train[(i - 1) * 10000:i * 10000] = labels

		  fpath = os.path.join(path, 'test_batch')
		  x_test, y_test = load_batch(fpath)

		  y_train = np.reshape(y_train, (len(y_train), 1))
		  y_test = np.reshape(y_test, (len(y_test), 1))

		  if K.image_data_format() == 'channels_last':
		    x_train = x_train.transpose(0, 2, 3, 1)
		    x_test = x_test.transpose(0, 2, 3, 1)

		  return (x_train, y_train), (x_test, y_test)

	class cifar100_py:

		"""CIFAR100 small image classification dataset.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import os

			import numpy as np

			from tensorflow.contrib.keras.python.keras import backend as K
			from tensorflow.contrib.keras.python.keras.datasets.cifar import load_batch
			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		def load_data(label_mode='fine'):
		  """Loads CIFAR100 dataset.

		  Arguments:
		      label_mode: one of "fine", "coarse".

		  Returns:
		      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

		  Raises:
		      ValueError: in case of invalid `label_mode`.
		  """
		  if label_mode not in ['fine', 'coarse']:
		    raise ValueError('label_mode must be one of "fine" "coarse".')

		  dirname = 'cifar-100-python'
		  origin = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
		  path = get_file(dirname, origin=origin, untar=True)

		  fpath = os.path.join(path, 'train')
		  x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

		  fpath = os.path.join(path, 'test')
		  x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

		  y_train = np.reshape(y_train, (len(y_train), 1))
		  y_test = np.reshape(y_test, (len(y_test), 1))

		  if K.image_data_format() == 'channels_last':
		    x_train = x_train.transpose(0, 2, 3, 1)
		    x_test = x_test.transpose(0, 2, 3, 1)

		  return (x_train, y_train), (x_test, y_test)

	class imdb_py:
		"""IMDB movie review sentiment classification dataset.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import json

			import numpy as np
			from six.moves import zip  # pylint: disable=redefined-builtin

			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		def load_data(path='imdb.npz',
		              num_words=None,
		              skip_top=0,
		              maxlen=None,
		              seed=113,
		              start_char=1,
		              oov_char=2,
		              index_from=3):
		  """Loads the IMDB dataset.

		  Arguments:
		      path: where to cache the data (relative to `~/.keras/dataset`).
		      num_words: max number of words to include. Words are ranked
		          by how often they occur (in the training set) and only
		          the most frequent words are kept
		      skip_top: skip the top N most frequently occurring words
		          (which may not be informative).
		      maxlen: truncate sequences after this length.
		      seed: random seed for sample shuffling.
		      start_char: The start of a sequence will be marked with this character.
		          Set to 1 because 0 is usually the padding character.
		      oov_char: words that were cut out because of the `num_words`
		          or `skip_top` limit will be replaced with this character.
		      index_from: index actual words with this index and higher.

		  Returns:
		      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

		  Raises:
		      ValueError: in case `maxlen` is so low
		          that no input sequence could be kept.

		  Note that the 'out of vocabulary' character is only used for
		  words that were present in the training set but are not included
		  because they're not making the `num_words` cut here.
		  Words that were not seen in the training set but are in the test set
		  have simply been skipped.
		  """
		  path = get_file(
		      path, origin='https://s3.amazonaws.com/text-datasets/imdb.npz')
		  f = np.load(path)
		  x_train = f['x_train']
		  labels_train = f['y_train']
		  x_test = f['x_test']
		  labels_test = f['y_test']
		  f.close()

		  np.random.seed(seed)
		  np.random.shuffle(x_train)
		  np.random.seed(seed)
		  np.random.shuffle(labels_train)

		  np.random.seed(seed * 2)
		  np.random.shuffle(x_test)
		  np.random.seed(seed * 2)
		  np.random.shuffle(labels_test)

		  xs = np.concatenate([x_train, x_test])
		  labels = np.concatenate([labels_train, labels_test])

		  if start_char is not None:
		    xs = [[start_char] + [w + index_from for w in x] for x in xs]
		  elif index_from:
		    xs = [[w + index_from for w in x] for x in xs]

		  if maxlen:
		    new_xs = []
		    new_labels = []
		    for x, y in zip(xs, labels):
		      if len(x) < maxlen:
		        new_xs.append(x)
		        new_labels.append(y)
		    xs = new_xs
		    labels = new_labels
		    if not xs:
		      raise ValueError('After filtering for sequences shorter than maxlen=' +
		                       str(maxlen) + ', no sequence was kept. '
		                       'Increase maxlen.')
		  if not num_words:
		    num_words = max([max(x) for x in xs])

		  # by convention, use 2 as OOV word
		  # reserve 'index_from' (=3 by default) characters:
		  # 0 (padding), 1 (start), 2 (OOV)
		  if oov_char is not None:
		    xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x]
		          for x in xs]
		  else:
		    new_xs = []
		    for x in xs:
		      nx = []
		      for w in x:
		        if skip_top <= w < num_words:
		          nx.append(w)
		      new_xs.append(nx)
		    xs = new_xs

		  x_train = np.array(xs[:len(x_train)])
		  y_train = np.array(labels[:len(x_train)])

		  x_test = np.array(xs[len(x_train):])
		  y_test = np.array(labels[len(x_train):])

		  return (x_train, y_train), (x_test, y_test)


		def get_word_index(path='imdb_word_index.json'):
		  """Retrieves the dictionary mapping word indices back to words.

		  Arguments:
		      path: where to cache the data (relative to `~/.keras/dataset`).

		  Returns:
		      The word index dictionary.
		  """
		  path = get_file(
		      path,
		      origin='https://s3.amazonaws.com/text-datasets/imdb_word_index.json')
		  f = open(path)
		  data = json.load(f)
		  f.close()
		  return data

	class mnist_py:
		"""MNIST handwritten digits classification dataset.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import numpy as np

			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		def load_data(path='mnist.npz'):
		  """Loads the MNIST dataset.

		  Arguments:
		      path: path where to cache the dataset locally
		          (relative to ~/.keras/datasets).

		  Returns:
		      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
		  """
		  path = get_file(
		      path, origin='https://s3.amazonaws.com/img-datasets/mnist.npz')
		  f = np.load(path)
		  x_train = f['x_train']
		  y_train = f['y_train']
		  x_test = f['x_test']
		  y_test = f['y_test']
		  f.close()
		  return (x_train, y_train), (x_test, y_test)

	class reuters_py:

		"""Reuters newswire topic classification dataset.
		"""
		def import_libs():
			from __future__ import absolute_import
			from __future__ import division
			from __future__ import print_function

			import json

			import numpy as np
			from six.moves import zip  # pylint: disable=redefined-builtin

			from tensorflow.contrib.keras.python.keras.utils.data_utils import get_file


		def load_data(path='reuters.npz',
		              num_words=None,
		              skip_top=0,
		              maxlen=None,
		              test_split=0.2,
		              seed=113,
		              start_char=1,
		              oov_char=2,
		              index_from=3):
		  """Loads the Reuters newswire classification dataset.

		  Arguments:
		      path: where to cache the data (relative to `~/.keras/dataset`).
		      num_words: max number of words to include. Words are ranked
		          by how often they occur (in the training set) and only
		          the most frequent words are kept
		      skip_top: skip the top N most frequently occurring words
		          (which may not be informative).
		      maxlen: truncate sequences after this length.
		      test_split: Fraction of the dataset to be used as test data.
		      seed: random seed for sample shuffling.
		      start_char: The start of a sequence will be marked with this character.
		          Set to 1 because 0 is usually the padding character.
		      oov_char: words that were cut out because of the `num_words`
		          or `skip_top` limit will be replaced with this character.
		      index_from: index actual words with this index and higher.

		  Returns:
		      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

		  Note that the 'out of vocabulary' character is only used for
		  words that were present in the training set but are not included
		  because they're not making the `num_words` cut here.
		  Words that were not seen in the training set but are in the test set
		  have simply been skipped.
		  """
		  path = get_file(
		      path, origin='https://s3.amazonaws.com/text-datasets/reuters.npz')
		  npzfile = np.load(path)
		  xs = npzfile['x']
		  labels = npzfile['y']
		  npzfile.close()

		  np.random.seed(seed)
		  np.random.shuffle(xs)
		  np.random.seed(seed)
		  np.random.shuffle(labels)

		  if start_char is not None:
		    xs = [[start_char] + [w + index_from for w in x] for x in xs]
		  elif index_from:
		    xs = [[w + index_from for w in x] for x in xs]

		  if maxlen:
		    new_xs = []
		    new_labels = []
		    for x, y in zip(xs, labels):
		      if len(x) < maxlen:
		        new_xs.append(x)
		        new_labels.append(y)
		    xs = new_xs
		    labels = new_labels

		  if not num_words:
		    num_words = max([max(x) for x in xs])

		  # by convention, use 2 as OOV word
		  # reserve 'index_from' (=3 by default) characters:
		  # 0 (padding), 1 (start), 2 (OOV)
		  if oov_char is not None:
		    xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x]
		          for x in xs]
		  else:
		    new_xs = []
		    for x in xs:
		      nx = []
		      for w in x:
		        if skip_top <= w < num_words:
		          nx.append(w)
		      new_xs.append(nx)
		    xs = new_xs

		  x_train = np.array(xs[:int(len(xs) * (1 - test_split))])
		  y_train = np.array(labels[:int(len(xs) * (1 - test_split))])

		  x_test = np.array(xs[int(len(xs) * (1 - test_split)):])
		  y_test = np.array(labels[int(len(xs) * (1 - test_split)):])

		  return (x_train, y_train), (x_test, y_test)


		def get_word_index(path='reuters_word_index.json'):
		  """Retrieves the dictionary mapping word indices back to words.

		  Arguments:
		      path: where to cache the data (relative to `~/.keras/dataset`).

		  Returns:
		      The word index dictionary.
		  """
		  path = get_file(
		      path,
		      origin='https://s3.amazonaws.com/text-datasets/reuters_word_index.json')
		  f = open(path)
		  data = json.load(f)
		  f.close()
		  return data
