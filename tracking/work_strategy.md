
# Goals
- kur inside out
- write 100 examples in kur


## Added Features and techniques written into kur
- see 'kur/z_tracking/added_features'


## Write a 100 new kur examples for 2017
- fast.ai course
- keras, pytorch examples
- siraj examples
- paper implementation

```python
# experiment start #######################
sample = None
for batch in provider:
	# get a single image data
	sample = batch['images'][0]
	break

act = self.model.compiled['train']
```

### work to do now
- dive into kur inner workings through cifar deom

### Kur inner working exploration
- access not only input, weights, activations, but also output layer ?
- every n epoch, use latest model run on the same data sample
	- restore the current model and apply a new data, like `kur evaluate|test` on the same data every n epoch
