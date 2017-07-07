"""
- access keras configuration file

- access keras dataset folder
"""


### access keras configure file at
 `$HOME/.keras/keras.json`

```
{
    "image_data_format": "channels_last", # channels_last or channels_first

    "epsilon": 1e-07, # numerical fuzz factor to be used to prevent division by zero in some operations

    "floatx": "float32", # default float data type
    "backend": "tensorflow" # default backend
}
```

### access keras datasets
- cached dataset file path can be accessed by `utils.get_file()`
- stored under `$HOME/.keras/datasets/`
