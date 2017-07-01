"""
### How can I "freeze" Keras layers?

To "freeze" a layer means to exclude it from training, i.e. its weights will never be updated. This is useful in the context of fine-tuning a model, or using fixed embeddings for a text input.

You can pass a `trainable` argument (boolean) to a layer constructor to set a layer to be non-trainable:

```python
frozen_layer = Dense(32, trainable=False)
```

Additionally, you can set the `trainable` property of a layer to `True` or `False` after instantiation. For this to take effect, you will need to call `compile()` on your model after modifying the `trainable` property. Here's an example:
"""
from tensorflow.contrib.keras.python.keras.layers import Input, Dense
from tensorflow.contrib.keras.python.keras.models import Model
import numpy as np

input_tensor = Input(shape=(32,))
layer = Dense(5)
layer.trainable = False
tensor_no_train = layer(input_tensor)

frozen_model = Model(input_tensor, tensor_no_train)
# in the model below, the weights of `layer` will not be updated during training
frozen_model.compile(optimizer='rmsprop', loss='mse')

layer.trainable = True
tensor_do_train = layer(input_tensor)
trainable_model = Model(input_tensor, tensor_do_train)
# with this model the weights of the layer will be updated during training
# (which will also affect the above model since it uses the same layer instance)
trainable_model.compile(optimizer='rmsprop', loss='mse')

# create dataset
data = np.random.random((100, 32)) # check on source
labels = np.random.randint(5, size=(100, 5)) # check source

frozen_model.fit(data, labels, epochs=5) # no validation_set, then no val_loss
frozen_model.fit(data, labels, validation_split=0.3, epochs=5)  # this does NOT update the weights of `layer`, therefore, loss stays the same

trainable_model.fit(data, labels, validation_split=0.3, epochs=5)  # this updates the weights of `layer`, therefore, loss keeps changing
