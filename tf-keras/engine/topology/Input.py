
def Input(  # pylint: disable=invalid-name
    shape=None,
    batch_shape=None,
    name=None,
    dtype=K.floatx(),
    sparse=False,
    tensor=None):
  """`Input()` is used to instantiate a Keras tensor.

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
  # when batch_shape and tensor are not given, then must have shape ready
  if not batch_shape and tensor is None:
    assert shape, ('Please provide to Input either a `shape`'
                   ' or a `batch_shape` argument. Note that '
                   '`shape` does not include the batch '
                   'dimension.')

  # ues shape to define batch_shape:
  if shape and not batch_shape:
    batch_shape = (None,) + tuple(shape)

  ##  create an input_layer:
  # check with dr input_layer and dt input_layer
  input_layer = InputLayer(
      batch_input_shape=batch_shape, # (None, 224, 224, 3)
      name=name,
      dtype=dtype, # float32
      sparse=sparse, # False
      input_tensor=tensor) # None
  # Return tensor including `_keras_history`.
  # Note that in this case train_output and test_output are the same pointer.


## access output tensor from InputLayer, return output tensor
  # input_layer.inbound_nodes: a list of Nodes included
  # check inside a Node: dr input_layer.inbound_nodes[0]
  outputs = input_layer.inbound_nodes[0].output_tensors
  if len(outputs) == 1:
    return outputs[0]
  else:
    return outputs