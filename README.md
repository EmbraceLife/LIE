# LIE: Learning Is Experimenting

Do you believe in LIE?

Step by step, this repo's history (see history commits) will testify LIE.

**LIE**
Learning how to use NeuralNets by primarily diving in working examples and source codes, not reading blogs or papers.

## Road Map
1. experiment on `kur`:
	- use of `pdb`, `inspect`
	- read and contribute to source code of library
	- create visualization features for kur
	- [visit my source](https://github.com/EmbraceLife/kur/tree/dive_source_kur)
2. apply my learning in `kur` to Morvan's Pytorch tutorials
	- apply `kur` style of `argparse`
	- save, load, re-train models
	- visualize every step of training
		- it does help understanding
	- [visit my source](https://github.com/EmbraceLife/PyTorch-Tutorial/tree/my_progress)

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

1. (tf) how to use `sess`, `graph`, `tensorboard` on basics: [source](https://github.com/EmbraceLife/tf-stanford-tutorials/blob/my_progress/examples/01_sess_graph_tensorboard.py)

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


1. often used git commands: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/git_tools.md)


1. make gif out of images: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/img2gif.py)

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
