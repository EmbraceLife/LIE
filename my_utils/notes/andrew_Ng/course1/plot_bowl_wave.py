from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 5),
                        subplot_kw={'projection': '3d'})

alpha = 10
r = np.linspace(-alpha,alpha,100)
X,Y= np.meshgrid(r,r)

# 只要由平方出现，就能产生碗形
R = np.sqrt(X**2+Y**2)

# 只要由cos或则sin求值，就能产生波浪
l = np.sin(R)

# 碗形
ax1.plot_surface(X, Y, R, rstride=1, cstride=1, cmap='rainbow')
ax1.set_title("bowl shape, single minimum value")
# 波浪形
ax2.plot_surface(X, Y, l, rstride=1, cstride=1, cmap='rainbow')
ax2.set_title("wave shape, multiple minimum value")


plt.show()
