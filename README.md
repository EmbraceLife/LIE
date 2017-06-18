# LIE: Learning Is Experimenting

Do you believe in LIE?

Step by step, this repo's history (see history commits) will testify LIE.

**LIE**
Learning how to use NeuralNets by primarily diving in working examples and source codes, not reading blogs or papers.


# transfer learning based on VGG16
- learn source codes of tf with example on VGG16
	- minimum comment on source codes
- Notebook to show process [example by fast.ai](https://www.youtube.com/watch?v=V2h3IOBDvrA&t=1529s)
	- process of image data going through each filter or layer
	- process of weights update through running each sample
	- process of effects of small and large learning rate
- core tf.contrib.keras building VGG16

# pandas and finance
- https://www.datacamp.com/community/tutorials/finance-python-trading#gs.vBkZrk0
- http://training.tpq.io/

# Understanding the nature of transfer learning
- uniqueness about imagenet datasets for this task
	- one central object in each image
	- rest is background or environment
- nature of the problem:
	- inputs:lots of images, each contain a central object
	- return: an object-name of 1000 object-names
- VGG16 use many layers of filters to screen the same image with different features (of different depths and focus)
- weights and layers are to capture the correlation between inputs and output

# how to use VGG16 to transfer learning
### borrow on VGG16
when you don't have large images datasets and huge GPU computing power
- prepare data
- build model
- load weights
### make it work for you
when you have a small datasets of your own (correctly labeled for training)
- finetune
- train
- predict
### interactive visualization of CNN model
what if there is no readily labeled datasets, but unlabeled images and videos
- find useful weights or layers (e.g. weights or layers to detect vertically flying object)
- build model using those specific weights and layers
- use model to filter out interesting images

----
# workflow

## prepare data
- how to prepare images into train, valid, test, sample folders
[notebook](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/nbs/dogs_cats_folder_organise.ipynb)

- convert train|valid|test|sample folder into batch_iterator objects [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_iterator_from_directory.py)

- convert images in folders to batches then to a large array: [source](vgg16_02_array_from_batches_from_folder.py)

----

## build model
- how to create vgg16 model instance from `tf.contrib.keras` [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_tf_kr.py)

----

## Fine_tuning vgg16 model
- fine_tune1: just replace the last layer: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_finetune.py)

- fine_tune2: replace the last layer and add an additional dense layer before the last layer [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_finetune2.py)


----

## train fine-tuned model
- train or fit vgg16 model: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_fit_fit_generator.py)

- save-load model+weights, save-load model, save-load weights, load old and make new model, load weights to a new model: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_save_load_models_weights.py)

- load, train_again, and save vgg16 model: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_train_again.py)

- save large arrays with maximum memory efficiency: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/save_load_large_array.py)
	- for the same large array to be saved, `bcolz` can shrink its size by 4 times smaller than file saved by `numpy`, `pickle`, `torch`, `kur.idx`

----

## predict or test models
- load vgg16 model, predict with test batches, and save preds: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_test_predict.py)

- `np.clip`, `log-loss`, `ids-pred` array:  [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/np_clip_dogscats.py)
	- why and how to clip predictions
	- extract image ids and cbind ids with predictions
	- how log-loss or cross-entropy behave against predictions


----
- build model with Sequential: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/sequential_keras.py)

- vgg16 model decode preds for 1000 classes: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_decode_prediction.py)


- process image dataset for vgg16? [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_preprocess_input.py)

----
1. how to call a keras function manually to test it? [forum](http://forums.fast.ai/t/lesson-10-discussion/1807/17?u=daniel)



1. (todo) how to plot [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_visualize_preds.py)
	- A few correct labels at random
	- A few incorrect labels at random
	- The most correct labels of each class (ie those with highest probability that are correct)
	- The most incorrect labels of each class (ie those with highest probability that are incorrect)
	- The most uncertain labels (ie those with probability closest to 0.5).
	- confusion matrix
----

1. (kr1.2) how to finetune vgg16 with new set of classes [fast.ai source](https://hyp.is/3ADedEmxEeeVW2OZ6Z3UbQ/github.com/fastai/courses/blob/master/deeplearning1/nbs/vgg16.py)



### fast.ai Lesson notes
#### Lesson1: vgg16 on dogscats
1. most noble goal of fast ai course: [fast.ai.wiki](https://hyp.is/nHYjZEmhEeeYtDv9sRdakg/wiki.fast.ai/index.php/Lesson_1_Notes)

1. why VGG16 preprocess images that way: [fast.ai.forum](https://hyp.is/ytuVFEmbEeeHzVsjpaDKoQ/forums.fast.ai/t/why-reshape-3-224-224-in-vgg16-py/812)
	- mean and order of RGB are prefixed

1. how to apply VGG to catsdogs problem: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=5679)

1. why do batch training: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=5462)

1. (kr) how to switch from theano to tf backend, and switch cpu and gpu in theano: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=5235)

1. why study state-of-art models like VGG: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=4515)

1. 7 steps to recognize basic images using vgg16: [fast.ai](https://hyp.is/Ug37DkmqEeerDXOvDvoDZQ/wiki.fast.ai/index.php/Lesson_1_Notes)

1. how to see limit and bias of a pretrained model like vgg16: [fast.ai](https://hyp.is/rY8HNkmoEeeVVXvlY3irxg/wiki.fast.ai/index.php/Lesson_1_Notes)

1. What is finetune to a pretrained model like vgg16? [fast.ai](http://wiki.fast.ai/index.php/Fine_tuning)

1. Why create a sample/train,valid,test folders from full dataset? [fast.ai](https://hyp.is/B91f6EmlEee5c79f2cNwEQ/wiki.fast.ai/index.php/Lesson_1_Notes)

1. how to count number of files in a folder: `ls folder/ | wc -l`

1. how to unzip a zip file: `unzip -q data.zip`

1. how to check python version: `which python`

1. how to organize dataset before training: [source]()
	- train, test, sample folder
	- sample: train, valid, test subfolders
	- experiment codes on sample dataset

1. how to check the first few files of a folder: `ls folder/ | head`

1. how to submit to kaggle: [fast.ai](https://youtu.be/e3aM6XTekJc?t=1386)

1. how to use kaggle cli to download and submit #@F: [source](http://wiki.fast.ai/index.php/Kaggle_CLI)

----

#### Lesson2: vgg16 on dogscats

1. make a todo list before working on any project or problem

1. how to prepare data after downloaded from kaggle? [fast.ai.wiki](http://wiki.fast.ai/index.php/Lesson_2_Notes#Preparing_the_Data)

1. why we need save weights other than save model?
	- Answer: with save, load empty model, save and load only weights, we can load weights to the same or even a different model [keras doc](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)

1. how to submit to kaggle using `keras.predict_generator` and `FileLink`: [wiki](http://wiki.fast.ai/index.php/Lesson_2_Notes#Submitting_Results)

1. why clip the final predictions for better log loss measurement? [wiki](http://wiki.fast.ai/index.php/Lesson_2_Notes#Dealing_with_Log_Loss)

1. what does CNNs learn [fast.ai video](https://youtu.be/e3aM6XTekJc?t=3283)
	- beautifully explained
	- it seems very useful to view all layers and weights of vgg16
		- (todo) check forum for similar questions
		- (todo) do it myself with my own prewritten functions

1. how deep learning work in excel? [fast.ai video](https://youtu.be/e3aM6XTekJc?t=3810)
	- (todo) display in excel how input layer and weights layer create output | activation layer
	- (todo) display how to do Axier initialization on weights

1. how to visualize SGD gradually optimize weights to plot the correct line? [fast.ai video](https://youtu.be/e3aM6XTekJc?t=4448)
	- (todo) rewrite to numpy code
	- (todo) keras on linear model with SGD

1. how to efficient save and load large arrays; save, load, plot from images files in folders; do one_hot encoding [fast.ai video](https://youtu.be/e3aM6XTekJc?t=4448)
	- (todo) use `bcolz` to efficiently save and load predictions array
	- (todo) get batches of images loaded from folders into proper arrays
	- (todo) save, load, plot those arrays into images
	- (todo) do one_hot encoding

1. how to use a single dense network with Vgg16 model's power? [fast.ai video](https://youtu.be/e3aM6XTekJc?t=4448)
	- (todo) use keras Sequential, Dense and vgg16's outputs

1. power of activation layer - non-linear function [fast.ai video](https://youtu.be/e3aM6XTekJc?t=6252)
	- input dot weights == linear model
	- activation function == non-linear model
	- activation func make deep learning powerful

1. (todo) how to visualize model as dots: [forum](http://forums.fast.ai/t/model-and-training-visualization/234)
	- [source]([source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_pydot_viz.py))
		- not useful at all, `model.summary()` can do a better job, seemingly

1. challenges:
	- (todo) dissect every line of `vgg16()`

----

#### Lesson 3 fast.ai part1
1. how to experiment the notebook [video](https://youtu.be/6kwQEBMandw?t=10)
	- write the source code
	- experiment for 30 minutes,, check questions and answers on wiki and forum
	- then ask questions on forum

1. better explain the usage of deep viz toolbox [video](https://youtu.be/6kwQEBMandw?t=528)
	- (todo) install and try as the video do
	- (todo) can this apply to RNN? and other neuralnets?
	- (todo later) there are vis tools on RNN too

1. understand cnn with spreadsheet and notebook: [video](https://youtu.be/6kwQEBMandw?t=791)
	- (todo) spreadsheet: compare Dense with Conv on create activation layer
	- (todo) notebook: explain Conv, Pad, and Pool in steps
	- (todo) convert notebook to spreadshee: spreadsheet is more intuitive displaying

1. review VGG: [video](https://youtu.be/6kwQEBMandw?t=1594)
	- (todo) demo the above todo task with vgg16 model example

1. maxpooling, padding: [video](https://youtu.be/6kwQEBMandw?t=1813)
	- (todo) spreadsheet and notebook: maxpool lose pixel, and layer goes deep, filter shrink size further, but activation layer becomes more high concept recognisable images?
	- how to make sense of it all

1. softmax with spreadshee: [video](https://youtu.be/6kwQEBMandw?t=2640)
	- (todo)

1. review SGD notebook: [video](https://youtu.be/6kwQEBMandw?t=3071)
	- (todo) intuition: how sgd or derivative help update weight, bias to the optimal weight and bias of target
	- (todo) convert notebook to spreadsheet?

1. how to decide on filter size and num_fitlers: [video](https://youtu.be/6kwQEBMandw?t=3321)
	- practically, filter size (3,3)
	- num_fitlers: not yet finalized
	- how to deal with much larger images:
		- not yet finalized: help from attention with LSTM to mimic how actual eye work?

1. how to retrain vgg16 with more layers? [video](https://youtu.be/6kwQEBMandw?t=3839)
	- (todo) how many layers to make trainable?
		- intuition: knowing which layer does what through weights vis
		- experiment to see what works better
		- vgg16 retain 1 dense layer for dogscats, retain 3 dense layers for statefarm dataset, as CNN layers are positional invariance oriented.

1. understand under and overfitting: [wiki](http://wiki.fast.ai/index.php/Lesson_3_Notes#Training_a_Better_Model)
	- underfitting: use linear model to do vgg16's job, use far too less parameters to take on more complex problems, in the end error for both training and validation are still very high
	- overfitting: use too many parameters to take on relatively simpler problems, in the end training loss is very low, but val_loss is higher

1. what dropout layer do: [video](https://youtu.be/6kwQEBMandw?t=4524) and [wiki](http://wiki.fast.ai/index.php/Lesson_3_Notes#Dropout)
	- drop out 50% neurons, model won't get overfit even though model has a huge number of neurons
	- in vgg16 dogscats case, 50% is too much, and made model underfitting, so drop 50% to 10% maybe can avoid underfitting
	- dropout: like ensemble, is to create many smaller but different models

1. split vgg16 into 2 models: [video](https://youtu.be/6kwQEBMandw?t=4876)
	- (todo) split to a cnn model
		- don't change anything in cnn model, as it is expensive to train cnn model weights
	- (todo) split to a dense model: do change dense model as you like, it is cheaper to train

1. how to do data augumentation: [video](https://youtu.be/6kwQEBMandw?t=5523)
	- help reduce overfitting
	- how to rotate or augumentation on your dataset
	- (todo) make it your own

1. how to do batch normalization: [video](https://youtu.be/6kwQEBMandw?t=6018)
	- why normalize inputs:
	 	- cos if inputs have different scales, it is harder to train and loss can mess up and too high, so ALWAYS normalize your inputs
	- why do batch normalization:
		- 10 faster
		- reduce overfitting
	- what is batch normalization:
		- normalize not inputs but intermediate layers
		- apply two trainable parameters to each layer: arbitrary std and mean
	- (todo) cnn inputs + dense model + batch_normalization and train this new model

1. (todo) end-to-end model building process with mnist: [video](https://youtu.be/6kwQEBMandw?t=6530)
	- load mnist dataset
	- onehot label
	- normalize inputs
	- single dense model, 1-hidden dense model on mnist
	- vgg-style simple cnn model [video](https://youtu.be/6kwQEBMandw?t=6811)
	- make sure the model is capable of some overfitting, then try to reduce overfitting
	- data augmentation, batch_normalization on every layer(do understand the source), dropout
	- ensembling

1. challenge: statefarm solutions



----

## Stanford Tensorflow for Deep Learning Research

1. (tf) how to use math ops: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_tf_math_ops.py)

1. how to create histogram plot: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_plot_histogram_random.py)

1. (tf) how to create random dataset: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_create_random.py)

1. (tf) how to make a sequence with `tf.linspace`, `tf.range`: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_make_sequence.py)
	- `np.linspace` and `range` can do iterations
	- not for `tf.linspace`, `tf.range`

1. (tf) how to use `tf.ones`, `tf.ones_like`, `tf.zeros`, `tf.zeros_like`, `tf.fill`: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_tensor_fills.py)

1. (tf) how to use `tf.constant` and tricks: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_tf.constant.py)

1. (tf) how to use `sess`, `graph`, to display ops in `tensorboard`: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_sess_graph_tensorboard.py)

----

## matplotlib

1. how to do subplots without space in between: [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt16.3_subplots_no_space_between.py)

1. how to quick-plot stock csv (use full csv path, plot close and volume right away): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt16.2_plt_plotfile.py)

1. how to stock csv with date (csv to vec object, date formatter): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt16.1_date_index_formatter.py)

1. how to reverse a list or numpy array?
	- `list(reversed(your_list))`
	- `your_array[::-1]`

1. how to gridsubplot (stock chart like subplots): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt16_grid_subplot.py)

1. how to subplot (4 equal subplots, 1 large + 3 small subplots): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt15_subplot.py)

1. how to plot images (array shape for image, interpolation, cmap, origin studied): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt13_image.py)

1. how to plot contour map (not studied): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt12_contours.py)

1. how to plot bars (set facecolor, edgecolor, text loc for each bar, ticks, xlim): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt11_bar.py)

1. how to do scatter plot (set size, color, alpha, xlim, ignore ticks): [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt10_scatter.py)

1. how to set x,y ticks labels fontsize, color, alpha: [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt9_tick_visibility.py)

1. how to add annotation or text: [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt8_annotation.py)

1. how to add labels to lines and change labels when setting legend: [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt7_legend.py)

1. how to reposition x,y axis anywhere with `plt.gca()`, `ax.spines`, `ax.xaxis`: [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt6_ax_setting2.py)

1. how to set line params, xlim, xticks: [source](https://github.com/EmbraceLife/tutorials/blob/my_project/matplotlibTUT/plt5_ax_setting1.py)

1. how to plot subplots of 4 activation lines: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/subplots_4_activationlines.py)

----

## Pytorch

1. torch activations [sources](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/activation.py)

1. torch Variables: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/variable.py)

1. torch.tensor vs numpy: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/torch_numpy.py)

1. read stocks csv:

1. train cnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/train_cnn.py)

1. train rnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/train_rnn.py)

1. build rnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/build_rnn.py)

1. build cnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/build_cnn.py)

1. prepare_mnist: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/prepare_mnist.py)

----

## Generally useful

1. how to do folding in atom: [source](http://flight-manual.atom.io/using-atom/sections/folding/)
	- `option+cmd+shift+[` and `]`: to fold and unfold all
	- `option+cmd+ [` and `]` : to fold and unfold where mouse is at

1. how to make alias in `.bash_profile` and `.pdbrc`: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/bash_profile-pdbrc.md)

1. how to use pdb: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/152528672306f2868568d7b65dfefb1da6900986/tutorial-contents/pdb.md)

1. most used conda command: [source](https://github.com/EmbraceLife/LIE/blob/master/conda_commands.md)

1. often used git commands: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/git_tools.md)
	- how to make multiple PR: build multiple branches for each PR

1. make gif out of images: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/img2gif.py)

1. relationship between keras and tensorflow: [google I/O](https://youtu.be/UeheTiBJ0Io?t=133)
	- best practices are set as default in keras [same video](https://youtu.be/UeheTiBJ0Io?t=820)

1. how to best access library internals:
	- not through ipython `keras.` and tab, but some submodules and methods somehow are hidden
	- run code in ipython: `%doctest_mode`
	- best: use `pdb`, `pdbpp`, and alias to access all internals

1. how to install ipython for python2 and python3: [doc](https://ipython.readthedocs.io/en/stable/install/kernel_install.html#kernels-for-python-2-and-3)

1. how to install ealier version of keras: `pip install keras==1.2`

1. how to update tensorflow to the latest release:
	- download nightly binary whl from [tf repo](https://github.com/tensorflow/tensorflow#installation)
	- install to upgrade `sudo pip install --upgrade tensorflow-1.2.0rc1-py3-none-any.whl`
	- or simply `sudo pip install tensorflow-1.2.0rc2-py3-none-any.whl`
	- try both if one is not working

1. how to install keras from source:
	- fork keras and add remote official url
	- go inside keras folder
	- `python setup.py install`

1. how to install tenforflow from source: [see stackoverflow](https://stackoverflow.com/questions/43364264/how-to-installing-tensorflow-from-source-for-mac-solved/44299779#44299779)

1. how to install pytorch from source:
	- https://github.com/pytorch/pytorch#from-source
	- `export CMAKE_PREFIX_PATH=[anaconda root directory]`: my case miniconda root path: /Users/Natsume/miniconda2
	- `conda install numpy pyyaml setuptools cmake cffi`
	- `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install`

----
## Numpy
1. 100 exercises to numpy: [source](https://github.com/rougier/numpy-100/blob/master/100%20Numpy%20exercises.md)
1. pandas exercises: [source](https://github.com/guipsamora/pandas_exercises)

1. how to save and load numpy arrays with numpy: [doc example](https://docs.scipy.org/doc/numpy/reference/generated/numpy.save.html)


## Visualize convolution

1. regression on fake 1d dataset:     

![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_up301.gif?raw=true)

2. classification on fake 1d dataset:     

![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_up302.gif?raw=true)

3. cnn on mnist:    

![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_down401.gif?raw=true)

4. rnn on mnist:
need tensorboard to see model structure
![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_down402.gif?raw=true)
