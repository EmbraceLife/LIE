# activation_function: how to choose?

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
