
# Goals
- match kurfile to keras, pytorch, tensorflow seamlessly
	- tutorials, examples dissected
	- write them with kur
	- write flexible techniques of keras, pytorch, tensorflow into kur
	- visualize neuralnet with p5 and flask

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
- matplotlib
	- datacamp https://www.datacamp.com/courses/introduction-to-data-visualization-with-python
	- morvan: https://www.youtube.com/watch?v=4Y7f0znUT6E&index=3&list=PLXO45tsB95cKiBRXYqNNCw8AUo6tYen3l
	- boken: https://www.datacamp.com/courses/interactive-data-visualization-with-bokeh
- dive into kur inner workings through cifar deom

### Kur inner working exploration
- access not only input, weights, activations, but also output layer ?
- every n epoch, use latest model run on the same data sample
	- restore the current model and apply a new data, like `kur evaluate|test` on the same data every n epoch
