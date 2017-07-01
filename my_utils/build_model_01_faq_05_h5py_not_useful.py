"""
### How can I use HDF5 inputs with Keras?

No need for HDF5Matrix, if I just want to access output tensors and weights

You can use the `HDF5Matrix` class from `keras.utils.io_utils`. See [the HDF5Matrix documentation](/utils/#hdf5matrix) for details.

You can also directly use a HDF5 dataset:
"""

from tensorflow.contrib.keras.python.keras.utils.io_utils import HDF5Matrix

# not working, as it is hard to know the field names such as `data` or other field names

X_data = HDF5Matrix('to_delete_weights.h5', 'weights')
model.predict(X_data)


from tensorflow.contrib.keras.python.keras.utils import io_utils
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)


### can HDF5 save keras objects? like iterator objects?
