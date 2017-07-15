# activation_function: how to choose?
sigmoid_distribution

## for hidden units
- most time use `relu`

## for final output
depend on what you need to get
- scalar numeric values: use `linear`
- binary classification: use `sigmoid`
- multiple classification: use `softmax`
- write your own activation function for final output: `relu_limited`

## Examples
- search `relu_limited` in this repo
```python
def relu_limited(x, alpha=0., max_value=1.):
    return K.relu(x, alpha=alpha, max_value=max_value)

get_custom_objects().update({'custom_activation': Activation(relu_limited)})
```

## distribution of sigmoid
```python
import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
# X = np.random.normal(scale=0.5, size=1000)
X = np.random.normal(scale=3, size=1000)
plt.hist(X)
plt.show()

x_tensor = K.constant(X)
x_sig = K.sigmoid(x_tensor)
sess = tf.Session()
x_sig_array = sess.run(x_sig)
plt.hist(x_sig_array)
plt.show()

```
