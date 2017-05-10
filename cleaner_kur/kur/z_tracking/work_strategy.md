
# Goals
- match kurfile to keras, pytorch, tensorflow seamlessly
	- tutorials, examples dissected
	- write them with kur
	- write flexible techniques of keras, pytorch, tensorflow into kur

## Features and techniques written into kur
- work done stored in kur/z_tracking/added_features
- plot_layers in kur (working)



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
