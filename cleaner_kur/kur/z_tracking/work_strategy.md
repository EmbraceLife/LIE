
# Goals
- Making kur my primary experiment tool of deep learning

## Experiment kur inside out
- pytorch minimum (done)
- keras minimum (....)
	- how to make a keras.Tensor?
- experiment on demos (working on)
- basic features in separate files
	- logger, examiner, pytorch import (done)
	- every layer output codes (done)
		- use pytorch to access each layer's output (done)
		- see Adam's suggestions (asked, and waiting)
	- plot_images (done)
	- plot_weights_hooks (done)
	- adding new activations:
		- how args of activation is passed from kurfile to source code 
		- leaky_relu (done)
	- Understand the workflow of MnistSupplier supplier and BatchProvider
		- write down the workflow
		- make catsdogs dataset using this style


## Kur official upcoming feature list
- more activations
- build multiple models for gans
	- asked for more suggestions
-

## Write a 100 new kur examples
- fast.ai course
- keras, pytorch examples
- siraj examples
- paper implementation

### Experiment now
How to utilize an activation from model?
- build on `kur build` or in the end of a batch training inside `wrapped_train`
- access `model.compiled['raw'].layers[3]`
- get a tensor or variable from pytorch
- try it out
- but `pytorch` version `model.compiled['raw']` does not exist and `['train']` does not have `pytorch.layer or activation` objects ready

```python
# experiment start #######################
sample = None
for batch in provider:
	# get a single image data
	sample = batch['images'][0]
	break

act = self.model.compiled['train']
```
