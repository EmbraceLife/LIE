from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np

fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 12), subplot_kw={'projection': '3d'})

# Get the test data
X, Y, Z = axes3d.get_test_data(0.05)
x = X[0]
y = Y[0]
z = Z[0]

logit =


ax1.plot_wireframe(X,Y,Z)
ax1.set_title("plot backpropogation")

# Give the second plot only wireframes of the type x = c
ax2.plot_wireframe(x,y,z, rstride=0, cstride=10)
ax2.set_title("Row (y) stride set to 0")

plt.tight_layout()
plt.show()
