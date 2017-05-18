
#### Access layer outputs directly from kur for keras_backend
- kurfile setup:
	- model:
		- sink: yes, name: conv_layer1
		- sink: yes, name: conv_layer2
	- plot_weights:
		- layer_index: [conv_layer1, conv_layer2]
- keras_backend.py:
	- line 589: comment out codes to match each output layer to each loss
	- conv_layer1 and conv_layer2 do not need loss so the following code seems a problem in this case, should I simply comment them out or do something else about it? (create a kur issue)
	```python
	if len(loss) != len(model.outputs):
		raise ValueError('Model has {} outputs, but only {} loss '
			'functions were specified.'
			.format(len(model.outputs), len(loss)))
	```
- now inside `run_batch()` of `keras_backend.py` to see how many items are produced out of `K.function`
- inside `compile()` of `keras_backend.py` create `K.function()` to access all inter layer outputs
- inside `run_training_hooks` of `executor.py` to store all output layers inside `info['inter_layers_outputs']`
- inside `plot_weights_hook.py`, take all layers outputs from `info['inter_layers_outputs']`
- get a single data's layer outputs and plot them

```python
img_dim = (1,) + conv_layer1_output1.shape
conv_layer1_output1 = conv_layer1_output1.reshape(img_dim)
plot_conv_layer(conv_layer1_output1, self.layer_idx[0])
```

#### Access layer outputs directly from kur for pytorch_backend
- pytorch_backend.py, comment out 269 for layers match not losses

- `kur/backend/pytorch/modules.py`, handle loss not match with num of layers and outputs:
	- solution: leave out conv_layer2 and conv_layer1 without run loss check
```python
losses = []
for output_name, P in zip(self.outputs, predictions):
	for loss_name in loss:
		if output_name == loss_name:
			losses.append(get_loss(loss[output_name], P))
```

- if loss_name not match with output_name, then don't get loss function, go inside `pytorch_backend.py`, find `test()`, replace with the following code for `metrics`

```python
torch_model = model.compiled['test']['model']
losses = model.compiled['test']['loss']

# added code: get loss_name from losses
loss_name = None
for loss in losses:
	loss_name = loss

predictions, losses = torch_model.test(data, losses)

# # loss_name and output_name should match
# metrics = {
# 	k : loss.data.cpu().numpy().squeeze(-1)
# 	for k, loss in zip(model.outputs, losses)
# }
# added code
metrics = {}
for k in model.outputs:
	if k == loss_name:
		metrics[k] = losses[0].data.cpu().numpy().squeeze(-1)
```
