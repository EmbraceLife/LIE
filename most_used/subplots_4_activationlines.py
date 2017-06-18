import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# check functions and attr
x = torch.linspace(-5, 5, 200)

# look through all attr and methods
x = Variable(x)

# only variables can be applied with activations
y_relu = F.relu(x)
y_sigmoid = F.sigmoid(x)
y_tanh = F.tanh(x)
y_softplus = F.softplus(x)

# F.softmax only take 2d variable
x1 = x.view((-1,5))
y_softmax = F.softmax(x1)

"""
# checkout all functions availabe in F
dt F
# check out documents of F.softmax
sources torch.nn.Softmax
"""

#######################################
## subplotting

x_np = x.data.numpy()

# doc
plt.figure(1, figsize=(5, 3))

# dr, dt plt,
# doc plt.subplot, very clearly explained
plt.subplot(221)

# plot relu
plt.plot(x_np, y_relu.data.numpy(), c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')
# plot sigmoid
plt.subplot(222)
plt.plot(x_np, y_sigmoid.data.numpy(), c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
# plot tanh
plt.subplot(223)
plt.plot(x_np, y_tanh.data.numpy(), c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
# plot softplus
plt.subplot(224)
plt.plot(x_np, y_softplus.data.numpy(), c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

# plt.show()

#################################

# dr fig, fig.__class__
fig = plt.figure(2, figsize=(5, 3))

# plot relu
fig.add_subplot(221)
plt.plot(x_np, y_relu.data.numpy(), c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')
# plot sigmoid
fig.add_subplot(222)
plt.plot(x_np, y_sigmoid.data.numpy(), c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')
# plot tanh
fig.add_subplot(223)
plt.plot(x_np, y_tanh.data.numpy(), c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')
# plot softplus
fig.add_subplot(224)
plt.plot(x_np, y_softplus.data.numpy(), c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')



plt.show()
