from tensorflow.contrib.keras.python.keras.backend import arange


stop_None = arange(start = -5, stop=None, step=1, dtype='int32')
# return nothing

# step and dtype are consistent, return tensor in dtype int32
stop_10 = arange(start = -5, stop=10, step=1, dtype='int32')

# step and dtype are inconsistent, return tensor in dtype float32, not int32
stop_10 = arange(start = -5, stop=10, step=0.1, dtype='int32')

# dtype float32 will set for dtype, even though step = 1
stop_float = arange(start = -5, stop=10, step=1, dtype='float32')

#
# -------
#
# Signature: arange(start, stop=None, step=1, dtype='int32')
# Source:
# def arange(start, stop=None, step=1, dtype='int32'):
#   """Creates a 1D tensor containing a sequence of integers.
#
#   The function arguments use the same convention as
#   Theano's arange: if only one argument is provided,
#   it is in fact the "stop" argument.
#
#   The default type of the returned tensor is `'int32'` to
#   match TensorFlow's default.
#
#   Arguments:
#       start: Start value.
#       stop: Stop value.
#       step: Difference between two successive values.
#       dtype: Integer dtype to use.
#
#   Returns:
#       An integer tensor.
#
#   """
#   # Match the behavior of numpy and Theano by returning an empty seqence.
#   if stop is None and start < 0:
#     start = 0
#   result = math_ops.range(start, limit=stop, delta=step, name='arange')
#   if dtype != 'int32':
#     result = cast(result, dtype)
#   return result
# File:      ~/miniconda2/envs/kur/lib/python3.6/site-packages/tensorflow/contrib/keras/python/keras/backend.py
# Type:      function
