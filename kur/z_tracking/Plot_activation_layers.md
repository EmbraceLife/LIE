# How to plot every activation layer?



## What I know:
- model(weights, biases) at each epoch or even each batch can be assessed
- I don't know how to access each activation tensors directly
	- it seems kur use keras (not sure about pytorch) directly offer prediction and loss through `keras_backend.run_batch`
	- I don't see an obvious way to access each activation values directly
- But indirectly, activation values can be calculated: using a sample, weights&biases, activation functions
	- Can I use kur.layer.activation object to perform activation on numpy.array?  

## Logic flow: get output of each layer
- when do I need output from each layer?
	- at the end of a batch or an epoch
- how do I get the output of each layer?
	- backward: from trained weights + inputs + (activation) ==> layer output
- how to get input?
	- through batch_provider, throw out batch one at a time
- how to get weights?
	- after a batch or epoch of training, or certain number of them
	- create a temp folder and save model weights into it
	- select a particular layer's weights + bias files from the folder and open with `idx.open`, to get numpy array of weigths and biases
- how to get activation?
	- through numpy functions to create activation operator (more likely chosen), e.g., tanh: np.tanh; relu: np.relu (not exist)
	- use keras.activation operator if possible? (not sure)
		- How to use relu activation directly from keras
			- `from keras import backend as K`
			- `K.relu(x)`
	- use pytorch activation operator if possible? (more likely to be the easier solution found here)
	- best solution:
		- apply numpy.arrays to kur.layer.Activation objects, such as `softmax`, `leakyrelu`
		- so that, no need to convert numpy array to Theano.tensor or Tensorflow tensor, or pytorch tensor, and back

## Experiment built upon `kur build` source code
- get data ready
- get model.compiled['raw'].layers[1] object
	- e.g., <keras.layers.core.Flatten object at 0x124423320>
	- I can't apply a single sample to this object and flatten the sample, or can I?
	- but output out of flatten object is a tensor not a numpy.array
	- I have no idea how to convert from the tensor to numpy array.  
- get weights ready:
	- weights are in numpy.array
- convert the output to numpy array and reshape it to dot operation on weights + biases
- then apply activation functions
	- but again,activation function may only take and output tensors
	- tensors have to be converted to numpy arrays

## I can't get around of tensor operation and conversion backforth between tensors and numpy arrays 
