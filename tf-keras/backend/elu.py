from tensorflow.contrib.keras.python.keras.backend import elu, arange
import tensorflow as tf
import numpy as np

np_int = np.arange(-5, 10)


elu_try = elu()

----
see source code in ipython nicely


In [21]: ?? tf.contrib.keras.backend.elu
Signature: tf.contrib.keras.backend.elu(x, alpha=1.0)
Source:
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
File:      ~/miniconda2/envs/kur/lib/python3.6/site-packages/tensorflow/contrib/keras/python/keras/backend.py
Type:      function
