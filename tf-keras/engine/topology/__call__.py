# self is eg. Dense object
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
	  # doc getattr

  # Optionally load weight values that were specified at layer instantiation.
  if hasattr(self, '_initial_weights') and self._initial_weights is not None:
	self.set_weights(self._initial_weights)
	del self._initial_weights
  return output
