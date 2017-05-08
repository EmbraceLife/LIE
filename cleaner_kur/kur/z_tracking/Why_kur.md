# Kur

**Why so many backend?**
- Keras backend:
	- Theano (first emerged, but now overshadowed by tensorflow)
	- Tensorflow
- pytorch backend:
	- raising star
- keep up with these two backends, so that kur can always ride on the front edge of deep learning libraries
- no worry about one library will become obsolete

**Easy experiment**
- all functionalities are built into the source (__main__.py, kurfile.py, other source folders)
- use `set.trace()` to dive inside to see the mechanism
- when create new functionality, I can build upon the existing blocks

**strength of kurfile**
- detailed description of working flow
	- hyperparameters
	- model description
	- loss func, optimizer func
	- train, validate, evaluate, test
		- how it wants dataset
		- where to load and save weights
		- do some plotting or other actions
- kur source code takes care of this working flow infrastructure

**strength of api**
api coding block (see example below) intends to work with 'kurfile.yml'
- use api to create block functionality like `kur data|build|train|evaluate|test`
- simple `kur evaluate` api example below:
	- no need to run in console through `main(args)`
	- no need to get provider
	- ... cut unwanted part from kur source code
- once experiment is done, write the api block into a `kur evaluate` block in `__main__.py`
- now experiment using api, upon existing blocks like `kur build|train|evaluate`

**Example of API**
```python
import numpy
from kur import Kurfile
from kur.engine import JinjaEngine

# Load the Kurfile and parse it.
kurfile = Kurfile('Kurfile.yml', JinjaEngine())
kurfile.parse()

# Get the model and assemble it.
model = kurfile.get_model()
model.backend.compile(model)

# If we have existing weights, let's load them into the model.
model.restore('weights.kur')

# Now we can use the model for evaluating data.
pdf, metrics = model.backend.evaluate(
    model,
    data={
        'X' : numpy.array([1, 2, 3]),
        'Y' : numpy.array([4, 5, 6]),
    }
)
```
