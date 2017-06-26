## What are the most practical strategies for common people to make the most out of advanced deep learning models?
**Who are “common people” here?**
People like me:
- love to fantasize how to use deep learning models to solve real-life problems (not necessarily very complex ones)
- not clever, nor super hard-working, even fear reading papers (may change in the future)
- is spending much of time experimenting and tweaking working basic examples and source codes of advanced models

**What triggered this question?**
- Even completing DLND very fast, still don't feel I can do anything real with deep learning;
- Google Cloud AI is offering deep learning easy-to-use API for app developers to utilize deep learning functionalities for doing a number of useful things;
- If those functionalities are still limited at this moment, then Algorithmia (online marketplace for over 3000 algorithms, make it even much easier to utilize deep learning algorithms for app developers and others) can potentially make deep learning very accessible to 18 million non-deep-learning programmers. App programmers without knowing anything at all about deep learning may be able to combine different deep learning functionalities to create novel features...

All these forced me to question myself whether I spent my time wisely on learning basics of deep learning (source code, playing models): If I can't learn deep learning in depth and have a thorough understanding of it to become a proper deep learning practitioner within a foreseeable future, then I have to make myself useful somehow, to app development for example, I should at least know more or learn more on all the possible functionalities or utilities google or algorithmia api can offer app developers. This way, when app developers tell me what kind of features they think of, I can help them combine different deep learning functionalities to create a novel for them. Then again, why couldn't app developers do it themselves, what special can I offer? Can I do it faster and better than them?

**Not without hope**
App developers need me to do it for the following reasons:
- they can use deep learning api, but the source code of algos on algorithmia is hidden. app developers can't really play it to really grasp the real capacity of deep learning model at play;
- they can't tweak the api or model under the hood to do something slightly different
- therefore, they can't really be creative to combine different deep learning utilities to create a real novel feature
- They can't do it, but I could? why?
- I dissected many (dynamic) advanced and useful models (so far only two models)
- I experimented and tweaked a lot of models
- therefore, I really understand what the models are capable and how to make them do slightly new things
So far, everything sounds good, my hope comes back. But as a common individual, I have to take into account two factors: 1. I don't have much GPU computation power; 2. I don't have much dataset.

**What are the most practical strategies for common people to explore existing and novel utilities of advanced models?**
Where are we now:
- I don't have much GPU computing power, nor huge dataset
- I don't read papers
- I do dissect, experiment and tweak examples or source codes of advanced models
- I want to explore all utilities and potential new features of advanced models like VGG16

**Is transfer learning the best strategy?**
- transfer learning is the only thing come to my mind, as it works great in applying VGG16 from imagenet to catsdogs;
- transfer learning in catsdogs case:
	- I don't need to huge dataset, small dataset of cats and dogs can be easily found and is enough for the task;
	- I don't need to lots of GPU to train for days (pre-trained weights is given), just need a CPU for a night to train 1-2 layers;
	- I don't need to stretch my hair for novel innovation, just apply similar functionalities to slightly new areas and I can get real result back.
	- these are enough to satisfy my heart for now.
- this is why I ask whether transfer learning can be applied to all other advanced models
- If VGG16 and catsdogs problem prove transfer learning is the most practical strategies for CNN models, could this be for RNN and others?
- if not, any other practical strategies?


## keep an eye on Wechat
1. check latest on https://medium.com/@yelin.qiu, https://mp.weixin.qq.com/debug/wxadoc/dev/

## focus on
1. tf.keras  
1. advanced models transferring


1. digest: viz_01 and viz_02 to digest
1. try out every layer and activation functions with vgg16_catsdogs example
	- dense layer and relu activation

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
