
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
    # initialize core TF layers: tf_base_layers.Layer
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
