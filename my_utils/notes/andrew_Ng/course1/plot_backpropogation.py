from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm
import numpy as np
import math


x1 = 1
x2 = -1
# 碗形
y1 = 0.5
y2 = 0.5
# 其他形状 ？？？
# y1 = -0.5
# y2 = 0.5

X = [x1, x2]
Y = [y1, y2]
w = np.linspace(-1,1,21)
b = np.linspace(-1,1,21)

w, b = np.meshgrid(w, b)
Cb = 0
Cw = 0
for x,y in zip(X, Y):
	a = w*x + b
	f = 1/(1+np.exp(-a))

	#### 我的设想
	# 1. 用loglikelihood 损失函数， 更容易找到唯一最小值，容易生成碗形
	l_bowl = -(y*np.log(f)+(1-y)*np.log(1-f))
	Cb += l_bowl
	# 2. 用Squred Error 损失函数， 容易产生多个局部最小值，或多个最小值，容易生成波浪形

	#### 目前的现实：
	## 但问题是，Squred Error 损失函数，似乎跟loglikelihood 损失函数做出的图形没有区别，都是碗形
	l_wave = (1/2)*(f-y)**2
	Cw += l_wave

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 5),
                        subplot_kw={'projection': '3d'})

ax1.plot_surface(w, b, Cb, rstride=1, cstride=1, cmap='rainbow')
ax1.set_title("bowl shape, single minimum value")

ax2.plot_surface(w, b, Cw, rstride=1, cstride=1, cmap='rainbow')
ax2.set_title("wave shape, multiple minimum value")

plt.show()
