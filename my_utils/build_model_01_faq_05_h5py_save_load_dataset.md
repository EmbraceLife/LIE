
### How can I use HDF5 inputs with Keras?

You can use the `HDF5Matrix` class from `keras.utils.io_utils`. See [the HDF5Matrix documentation](/utils/#hdf5matrix) for details.

You can also directly use a HDF5 dataset:

```python
from tensorflow.contrib.keras.python.keras.utils import io_utils
import h5py
with h5py.File('input/file.hdf5', 'r') as f:
    x_data = f['x_data']
    model.predict(x_data)
```

### can HDF5 save keras objects? like iterator objects?
