
class InputLayer(Layer):
  """
  Layer to be used as an entry point into a graph.

  Return a layer

	# Importance
  It can either wrap an existing tensor (pass an `input_tensor` argument)
  or create its a placeholder tensor (pass arguments `input_shape`
  or `batch_input_shape` as well as `dtype`).

  Arguments:
      input_shape: Shape tuple, not including the batch axis.
      batch_size: Optional input batch size (integer or None).

	  # batch_input_shape = (batch, input_shape)
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

	# set layer name
    if not name:
      prefix = 'input'
      name = prefix + '_' + str(K.get_uid(prefix))

	# set layer dtype
    if not dtype:
      if input_tensor is None:
        dtype = K.floatx()
      else:
        dtype = K.dtype(input_tensor)

	# Initialize InputLayer's parent (Layer) class, use two args 
    super(InputLayer, self).__init__(dtype=dtype, name=name)

	# set built true, sparse false
    self.built = True
    self.sparse = sparse

	# input_shape and batch_input_shape can't be available at the same time
    if input_shape and batch_input_shape:
      raise ValueError('Only provide the input_shape OR '
                       'batch_input_shape argument to '
                       'InputLayer, not both at the same time.')

	# when input_tensor is given, do automatical shape inference
    if input_tensor is not None:

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

	# set batch_input_shape whether or not batch_input_shape is there or not
    if not batch_input_shape:
      if not input_shape:
        raise ValueError('An Input layer should be passed either '
                         'a `batch_input_shape` or an `input_shape`.')
      else:
        batch_input_shape = (batch_size,) + tuple(input_shape)
    else:
      batch_input_shape = tuple(batch_input_shape)
    self.batch_input_shape = batch_input_shape

	# create input_tensor as placeholder when input_tensor not given
    if input_tensor is None:
      self.is_placeholder = True
      input_tensor = K.placeholder(
          shape=batch_input_shape,
          dtype=dtype,
          sparse=self.sparse,
          name=self.name)

	# if input_tensor is given, set is_placeholder False
    else:
      self.is_placeholder = False

    input_tensor._uses_learning_phase = False

	# and set output_tensors' _keras_history.
    input_tensor._keras_history = (self, 0, 0)

	# Create an input node to add to self.outbound_node
    Node(
        self,
        inbound_layers=[],
        node_indices=[],
        tensor_indices=[],
        input_tensors=[input_tensor],
        output_tensors=[input_tensor],
        input_masks=[None],
        output_masks=[None])

    # to get configuration values for batch_input_shape, dtype, sparse, name
  def get_config(self):
    config = {
        'batch_input_shape': self.batch_input_shape,
        'dtype': self.dtype,
        'sparse': self.sparse,
        'name': self.name
    }
    return config
