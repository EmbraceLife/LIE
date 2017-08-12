# from mpl_toolkits.mplot3d.axes3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np
# import math
#
#
# fig, ax1 = plt.subplots(1, 1, figsize=(8, 5), subplot_kw={'projection': '3d'})
#
# # Get the test data
# x1 = 1
# x2 = 1
# y = 0.8
# w = np.linspace(-10,10,100)
# # w = np.random.random(100)
# wl = np.linspace(-10,10,100)
# # wl = np.random.random(100)
# w1 = np.ones((100,100))
# w2 = np.ones((100,100))
# for idx in range(100):
# 	w1[idx] = w1[idx]*w
# 	w2[:,idx] = w2[:,idx]*wl
#
# L = []
# for i in range(w1.shape[0]):
# 	for j in range(w1.shape[1]):
# 		a = w1[i,j]*x1 + w2[i,j]
# 		f = 1/(1+math.exp(-a))
# 		l = -(y*math.log(f)+(1-y)*math.log(1-f))
# 		# l = (1/2)*(f-y)**2
# 		L.append(l)
# l = np.array(L).reshape(w1.shape)
#
#
# ax1.plot_wireframe(w1,w2,l)
# ax1.set_title("plot backpropogation")
#
#
# plt.tight_layout()
# plt.show()

###################

# from mpl_toolkits.mplot3d.axes3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# import numpy as np
# import math
#
#
# fig, ax1 = plt.subplots(1, 1, figsize=(8, 5), subplot_kw={'projection': '3d'})
#
# # Get the test data
# x = 0.5
# y = 0.5
# w = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
#
# LL = np.ones((21,21))
#
# for i in range(21):
# 	for j in range(21):
# 		a = w[i]*x + b[j]
# 		f = 1/(1+math.exp(-a))
# 		l = -(y*math.log(f)+(1-y)*math.log(1-f))
# 		LL[i,j] = l
# 		# L.append(l)
#
# ax1.plot_wireframe(w,b,LL)
# ax1.set_title("plot backpropogation")
#
# plt.tight_layout()
# plt.show()

################################
#
# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# figure = plt.figure()
# ax = Axes3D(figure)
#
# from matplotlib import cm
# import numpy as np
# import math
#
#
# # Get the test data
# x1 = -1
# x2 = 1
# x3 = 0.5
# x4 = -0.5
# y = 0.5
# w = np.linspace(-1,1,21)
# b = np.linspace(-1,1,21)
#
# L = np.ones((21,21))
# LL = np.ones((21,21))
# LLL = np.ones((21,21))
# LLLL = np.ones((21,21))
#
#
# for i in range(21):
# 	for j in range(21):
# 		a = w[i]*x1 + b[j]
# 		f = 1/(1+math.exp(-a))
# 		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
# 		l = (1/2)*(f-y)**2
# 		L[i,j] = l
# 		# L.append(l)
#
# for i in range(21):
# 	for j in range(21):
# 		a = w[i]*x2 + b[j]
# 		f = 1/(1+math.exp(-a))
# 		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
# 		l = (1/2)*(f-y)**2
# 		LL[i,j] = l
#
# for i in range(21):
# 	for j in range(21):
# 		a = w[i]*x3 + b[j]
# 		f = 1/(1+math.exp(-a))
# 		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
# 		l = (1/2)*(f-y)**2
# 		LLL[i,j] = l
#
# for i in range(21):
# 	for j in range(21):
# 		a = w[i]*x4 + b[j]
# 		f = 1/(1+math.exp(-a))
# 		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
# 		l = (1/2)*(f-y)**2
# 		LLLL[i,j] = l
#
# LLL = (L+LL+LLL+LLLL)/4
#
# w, b = np.meshgrid(w, b)
# ax.plot_surface(w, b, LLL, rstride=1, cstride=1, cmap='rainbow')
#
# plt.show()

####################


from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
figure = plt.figure()
ax = Axes3D(figure)

from matplotlib import cm
import numpy as np
import math


# fig, ax1 = plt.subplots(1, 1, figsize=(8, 5), subplot_kw={'projection': '3d'})

# Get the test data
x1 = -1
x2 = 1
x3 = 0.1
x4 = -0.8
x = 0.5
y = 0.5
y1 = -1
y2 = 1
y3 = 0.1
y4 = -0.7
w = np.linspace(-1,1,21)
b = np.linspace(-1,1,21)

L = np.ones((21,21))
LL = np.ones((21,21))
LLL = np.ones((21,21))
LLLL = np.ones((21,21))


w, b = np.meshgrid(w, b)


for i in range(w.shape[0]):
	for j in range(w.shape[1]):
		a = w[i,j]*x1 + b[i,j]
		f = 1/(1+math.exp(-a))
		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
		l = (1/2)*(f-y)**2
		L[i,j] = l
		# L.append(l)

for i in range(w.shape[0]):
	for j in range(w.shape[1]):
		a = w[i,j]*x2 + b[i,j]
		f = 1/(1+math.exp(-a))
		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
		l = (1/2)*(f-y)**2
		LL[i,j] = l
		# L.append(l)

for i in range(w.shape[0]):
	for j in range(w.shape[1]):
		a = w[i,j]*x3 + b[i,j]
		f = 1/(1+math.exp(-a))
		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
		l = (1/2)*(f-y)**2
		LLL[i,j] = l
		# L.append(l)

for i in range(w.shape[0]):
	for j in range(w.shape[1]):
		a = w[i,j]*x4 + b[i,j]
		f = 1/(1+math.exp(-a))
		# l = -(y*math.log(f)+(1-y)*math.log(1-f))
		l = (1/2)*(f-y)**2
		LLLL[i,j] = l
		# L.append(l)


LL = (L + LL + LLL + LLLL)/4

ax.plot_surface(w, b, LL, rstride=1, cstride=1, cmap='rainbow')

# ax1.plot_wireframe(w,b,LLL)
# ax1.set_title("plot backpropogation")
#
# plt.tight_layout()
plt.show()
