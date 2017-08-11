
## Generally useful

1. how to install talib: [link](https://mrjbq7.github.io/ta-lib/install.html)

1. how to use `pdb` and `atom` to experiment and write up codes?
	- use `pdb` to test out codes and refresh usage of classes and functions
	- add codes onto `atom`, and `c` or `r` to continue and refresh `codes` in `pdb` mode
	- this way, there is no need to exit and re-enter `pdb` mode

1. how to do folding in atom: [source](http://flight-manual.atom.io/using-atom/sections/folding/)
	- `option+cmd+shift+[` and `]`: to fold and unfold all
	- `option+cmd+ [` and `]` : to fold and unfold where mouse is at


1. how to write math notations in atom
	- install `markdown preview plus` and `mathjax wrapper`
	- `ctrl + shift + m` to toggle markdown view
	- `ctrl + shift + x` to toggle math notation view
	- resources
		- math expressions http://www.rapidtables.com/math/symbols/Basic_Math_Symbols.htm#annotations:w0yVRvONEeaRnMP1yvzSZg
		- math latex http://csrgxtu.github.io/2015/03/20/Writing-Mathematic-Fomulars-in-Markdown/
		
1. how to make alias in `.bash_profile` and `.pdbrc`: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/my_progress/tutorial-contents/bash_profile-pdbrc.md)

1. how to use pdb: [source](https://github.com/EmbraceLife/PyTorch-Tutorial/blob/152528672306f2868568d7b65dfefb1da6900986/tutorial-contents/pdb.md)

1. most used conda command: [source](https://github.com/EmbraceLife/LIE/most_used/conda_commands.md)

1. often used git commands: [source](https://github.com/EmbraceLife/LIE/most_used/git_tools.md)
	- how to make multiple PR: build multiple branches for each PR

1. make gif out of images: [source](https://github.com/EmbraceLife/LIE/most_used/img2gif.py)

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
	- fork keras and add remote official url `git remote add upstream official-url-git`
	- go inside keras folder `git checkout master`
	- `git pull upstream master` and then `git checkout keras_trade`
	- `git merge master` and `git commit -m 'sync'`
	- `python setup.py install`

1. how to install tenforflow from source: [see stackoverflow](https://stackoverflow.com/questions/43364264/how-to-installing-tensorflow-from-source-for-mac-solved/44299779#44299779)

1. how to install pytorch from source:
	- https://github.com/pytorch/pytorch#from-source
	- `export CMAKE_PREFIX_PATH=[anaconda root directory]`: my case miniconda root path: /Users/Natsume/miniconda2
	- `conda install numpy pyyaml setuptools cmake cffi`
	- `MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py install`


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
