
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

### work to do now
- LIE_demo: cifar (to be cleaned for better)
- Writing Morvan's tutorials (keras) to kur
- note-taking for shiffman AI
- update new features to official kur (continue)

```python
# experiment start #######################
sample = None
for batch in provider:
	# get a single image data
	sample = batch['images'][0]
	break

act = self.model.compiled['train']
```
