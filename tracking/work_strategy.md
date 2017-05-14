
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
- layer animation
	- plot layers for the same image
	- 2 files to learn start with each_layer...md here
	- using plot_layer_same_image branch in kur


- improve my PR (...)
- new PR (leakyrelu, plot_convol_layer,
- access output of intermediate layer directly in kur: (...)
- LIE_demo: cifar (to be cleaned for better): (done today)
- update new features to official kur (continue)
- Writing Morvan's tutorials (keras) to kur (continue)
- note-taking for shiffman AI (continue)
