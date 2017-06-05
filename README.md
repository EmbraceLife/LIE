# LIE: Learning Is Experimenting

Do you believe in LIE?

Step by step, this repo's history (see history commits) will testify LIE.

**LIE**
Learning how to use NeuralNets by primarily diving in working examples and source codes, not reading blogs or papers.

## Road Map
- fast.ai: apply advanced | useful models like VGG using keras, theano
- to understand and implement VGG model, low details is unavoidable
- how to build vgg from scratch in keras raw, tf.keras.application, tf.slim?

## Examples
High speed gif can help see the changes of weights and layers during training

1. read stocks csv: [source]()
![]()

1. train cnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/train_cnn.py)

1. train rnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/train_rnn.py)

1. build rnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/build_rnn.py)

1. build cnn net: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/build_cnn.py)

1. 100 exercises to numpy: [source](https://github.com/rougier/numpy-100/blob/master/100%20Numpy%20exercises.md)
pandas exercises: [source](https://github.com/guipsamora/pandas_exercises)

1. prepare_mnist: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/prepare_mnist.py)

1. most noble goal of fast ai course: [fast.ai.wiki](https://hyp.is/nHYjZEmhEeeYtDv9sRdakg/wiki.fast.ai/index.php/Lesson_1_Notes)

1. how vgg16 decode prediction percentage: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_decode_prediction.py)

1. how to process image dataset for vgg16? [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/vgg16_preprocess_input.py)

1. why VGG16 preprocess images that way: [fast.ai.forum](https://hyp.is/ytuVFEmbEeeHzVsjpaDKoQ/forums.fast.ai/t/why-reshape-3-224-224-in-vgg16-py/812)
	- mean and order of RGB are prefixed

1. `tf.contrib.keras.applications.vgg16.VGG16()` internals: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/tf_kr_vgg16.py)

1. `keras.models.Sequential` internals: [source](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/keras_internals/sequential_keras.py)

1. how to apply VGG to catsdogs problem: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=5679)

1. why do batch training: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=5462)

1. (kr) how to switch from theano to tf backend, and switch cpu and gpu in theano: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=5235)

1. why study state-of-art models like VGG: [fast.ai](https://youtu.be/Th_ckFbc6bI?t=4515)

1. 7 steps to recognize basic images using vgg16: [fast.ai](https://hyp.is/Ug37DkmqEeerDXOvDvoDZQ/wiki.fast.ai/index.php/Lesson_1_Notes)

1. how to see limit and bias of a pretrained model like vgg16: [fast.ai](https://hyp.is/rY8HNkmoEeeVVXvlY3irxg/wiki.fast.ai/index.php/Lesson_1_Notes)

1. What is finetune to a pretrained model like vgg16? [fast.ai](http://wiki.fast.ai/index.php/Fine_tuning)

1. Why create a sample/train,valid,test folders from full dataset? [fast.ai](https://hyp.is/B91f6EmlEee5c79f2cNwEQ/wiki.fast.ai/index.php/Lesson_1_Notes)

1. how to organize catsdogs dataset in folders for experiment, train, test: [notebook](https://github.com/EmbraceLife/courses/blob/my_progress/deeplearning1/nbs/dogs_cats_folder_organise.ipynb)

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

1. (tf) how to use math ops: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_tf_math_ops.py)

1. how to create histogram plot: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_plot_histogram_random.py)

1. (tf) how to create random dataset: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_create_random.py)

1. (tf) how to make a sequence with `tf.linspace`, `tf.range`: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_make_sequence.py)
	- `np.linspace` and `range` can do iterations
	- not for `tf.linspace`, `tf.range`

1. (tf) how to use `tf.ones`, `tf.ones_like`, `tf.zeros`, `tf.zeros_like`, `tf.fill`: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_tensor_fills.py)

1. (tf) how to use `tf.constant` and tricks: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_tf.constant.py)

1. (tf) how to use `sess`, `graph`, to display ops in `tensorboard`: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_sess_graph_tensorboard.py)

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

1. torch activations [sources](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/activation.py)

1. torch Variables: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/variable.py)

1. torch.tensor vs numpy: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/torch_numpy.py)

1. how to make alias in `.bash_profile` and `.pdbrc`: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/bash_profile-pdbrc.md)

1. how to use pdb: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/152528672306f2868568d7b65dfefb1da6900986/tutorial-contents/pdb.md)

1. most used conda command: [source](https://github.com/EmbraceLife/LIE/blob/master/conda_commands.md)

1. often used git commands: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/git_tools.md)

1. make gif out of images: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/img2gif.py)

1. relationship between keras and tensorflow: [google I/O](https://youtu.be/UeheTiBJ0Io?t=133)
	- best practices are set as default in keras [same video](https://youtu.be/UeheTiBJ0Io?t=820)

1. how to best access library internals:
	- not through ipython `keras.` and tab, but some submodules and methods somehow are hidden
	- best: use `pdb`, `pdbpp`, and alias to access all internals

1. how to install ealier version of keras: `pip install keras==1.2`

1. how to update tensorflow to the latest release:
	- download nightly binary whl from [tf repo](https://github.com/tensorflow/tensorflow#installation)
	- install to upgrade `pip install --upgrade tensorflow-1.2.0rc1-py3-none-any.whl`

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

1. regression on fake 1d dataset:     

![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_up301.gif?raw=true)

2. classification on fake 1d dataset:     

![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_up302.gif?raw=true)

3. cnn on mnist:    

![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_down401.gif?raw=true)

4. rnn on mnist:
need tensorboard to see model structure
![](https://github.com/EmbraceLife/LIE/blob/master/gifs/out_down402.gif?raw=true)
