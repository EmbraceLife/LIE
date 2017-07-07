
"""
### Save and Load models and weights
- model.save("model_name.h5")
- model.save_weights("model_weights_name.h5")
- model1 = load_model("mode_name.h5")
- model_in_json = model.to_json()
- model_from_json = model_from_json(model_in_json)
- model_from_json.load_weights("model_weights_name.h5")
- model_from_json.load_weights("model_weights_name.h5", by_name=True) # when model_from_json has a few layers differ from model
"""


"""

*It is not recommended to use pickle or cPickle to save a Keras model.*

You can use `model.save(filepath)` to save a Keras model into a single HDF5 file which will contain:

- the architecture of the model, allowing to re-create the model
- the weights of the model
- the training configuration (loss, optimizer)
- the state of the optimizer, allowing to resume training exactly where you left off.

You can then use `keras.models.load_model(filepath)` to reinstantiate your model.
`load_model` will also take care of compiling the model using the saved training configuration
(unless the model was never compiled in the first place).

Example:
"""


from tensorflow.contrib.keras.python.keras.models import load_model, Model, Sequential
from tensorflow.contrib.keras.python.keras.models import model_from_json, model_from_yaml
from tensorflow.contrib.keras.python.keras.layers import Input, Dense

input_tensor = Input(shape=(100,))
output_tensor = Dense(2)(input_tensor)
model = Model(input_tensor, output_tensor)

model.save('to_delete.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('to_delete.h5')

"""
If you only need to save the **architecture of a model**,
and not its weights or its training configuration,

you can do:
"""

# save as JSON
json_string = model.to_json()

# save as YAML
yaml_string = model.to_yaml()

"""
The generated JSON / YAML files are human-readable and can be manually edited if needed.

You can then build a fresh model from this data:
"""

# model reconstruction from JSON:
model_json = model_from_json(json_string)

# model reconstruction from YAML
model_yaml = model_from_yaml(yaml_string)

"""
If you need to save the **weights of a model**, you can do so in HDF5 with the code below.

Note that you will first need to install HDF5 and the Python library h5py, which do not come bundled with Keras.
"""

model.save_weights('to_delete_weights.h5')

"""
Assuming you have code for instantiating your model, you can then load the weights you saved into a model with the *same* architecture:
"""

model_json.load_weights('to_delete_weights.h5')

"""
If you need to load weights into a *different* architecture (with some layers in common), for instance for fine-tuning or transfer-learning, you can load weights by *layer name*:
"""
model_yaml.load_weights('to_delete_weights.h5', by_name=True)

# make sure they share the same weights: total 202 parameters
(model_json.get_weights()[0] == model_yaml.get_weights()[0]).sum()
(model_json.get_weights()[1] == model_yaml.get_weights()[1]).sum()

"""
For example
Assume original model looks like this:
"""
model1 = Sequential()
model1.add(Dense(2, input_dim=3, name='dense_1'))
model1.add(Dense(3, name='dense_2'))
model1.save_weights("weights1.h5")

# check out the weights
model1.get_weights()

# new model
model2 = Sequential()
model2.add(Dense(2, input_dim=3, name='dense_1'))  # will be loaded
model2.add(Dense(10, name='new_dense'))  # will not be loaded

# load weights from first model; will only affect the first layer, dense_1.
model2.load_weights("weights1.h5", by_name=True)

# check out the weights
model2.get_weights()
