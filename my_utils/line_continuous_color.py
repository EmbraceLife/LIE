"""
line_continuous_color

key answer is found here
https://stackoverflow.com/questions/17240694/python-how-to-plot-one-line-in-different-colors

# I have two datasets, one is array with shape (30,), named line_data; the other one is array (1, 30), named color_data.
# I have used line_data to plot a line, use color_data to plot a color bar like image.
# I want to fill the color of the line with the color bar's continuous color change.
# Could anyone show me how to do it in python libraries, for example matplotlib or else?

"""



import numpy as np
import matplotlib.pyplot as plt

line_data = np.array([  0.89917704,   1.89812886,   2.89733245,   3.87308733,
         4.79016642,   4.8327078 ,   5.81535641,   5.81631461,
         5.81652544,   5.81652555,   5.81652639,   5.81652663,
         5.93220416,   6.74091009,   7.61425993,   7.66313944,
         8.60456767,   8.65866624,   9.5472393 ,   9.63912952,
         9.84010958,  10.83984404,  11.83848397,  11.83959435,
        12.1176459 ,  12.39335136,  12.39511715,  13.20027627,
        14.00576137,  14.07948385])

color_data = np.array([[  8.99476647e-01,   2.99607753e-04,   9.99251425e-01,
          4.78358124e-05,   9.75802720e-01,   5.87236322e-02,
          1.61822475e-02,   9.98830855e-01,   9.99789059e-01,
          9.99999881e-01,   1.00000000e+00,   9.99999166e-01,
          9.99999404e-01,   8.84321868e-01,   7.56159425e-02,
          9.48965788e-01,   9.97845292e-01,   5.64170629e-02,
          2.31849123e-03,   8.90891552e-01,   7.99001336e-01,
          9.99981403e-01,   2.46947806e-04,   9.98886883e-01,
          9.99997258e-01,   7.21945703e-01,   9.97651160e-01,
          9.99416947e-01,   1.94257826e-01,   9.99742925e-01]])

# plt.imshow(color_data, cmap='binary')
# plt.show()
#
# plt.plot(line_data, c='blue')
# plt.show()

color_data = color_data.transpose((1,0))
line_data = line_data.reshape((-1,1))

def uniqueish_color(color_data):
    """There're better ways to generate unique colors, but this isn't awful."""
    # return plt.cm.gist_ncar(color_data)
    return plt.cm.binary(color_data)

# xy = (np.random.random((10, 2)) - 0.5).cumsum(axis=0)
X = np.arange(len(line_data)).reshape((-1,1))
y = line_data
xy = np.concatenate((X,y), axis=1)

fig, ax = plt.subplots()
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax.plot(x, y, color=uniqueish_color(col[0]))

plt.show()


"""
# real thing I want to solve


X = np.arange(len(line_data)).reshape((-1,1))
y = line_data
xy = np.concatenate((X,y), axis=1)

fig, ax = plt.subplots()
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax.plot(x, y, color=uniqueish_color(col))
plt.show()

"""



####################
# how to fill a line with a continous 2 colors
# how to plot a color legend for this continuous 2 colors
# https://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots


import matplotlib as mpl
import matplotlib.pyplot as plt

min_c = 0.000000000000001 # color_data.min()
max_c = 1.0 # color_data.max()
step = 1

# Setting up a colormap that's a simple transtion
mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',['blue', 'yellow','red'])

# Using contourf to provide my colorbar info, then clearing the figure
Z = [[0,0],[0,0]]
levels = range(int(min_c*100),int(max_c*100),step)
CS3 = plt.contourf(Z, levels, cmap=mymap)
plt.clf()

for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    # setting rgb color based on z normalized to my range
    r = (col[0]-min_c)/(max_c-min_c)
    g = 0
    b = 1-r

    x, y = zip(start, stop)
    plt.plot(x, y, color=(r,g,b))
plt.colorbar(CS3) # using the colorbar info I got from contourf
plt.show()
