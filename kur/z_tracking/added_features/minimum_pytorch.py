
"""
## pytorch minimum necessaries
"""

import matplotlib.pyplot as plt
import numpy as np
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues


###################################################################
# **How to create numpy.array and convert back forth with torch.Tensor**

import torch
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

####################################
# **List -> Tensor ->variable -> numpy**

import torch
from torch.autograd import Variable
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)
np_var = variable.data.numpy()

############################################################
# Tensor | Variable ops -> backward -> grad
t_out = torch.mean(tensor*tensor)       # x^2
v_out = torch.mean(variable*variable)   # x^2

v_out.backward()
print(variable.grad)

###############################################################
# **reshape + shape on Tensor | Variable ==> view() + size()**

import torch
t = torch.ones((2, 3, 4))
t.size()
# torch.Size([2, 3, 4])
t.view(-1, 12).size()
# torch.Size([2, 12])

# create a tensor with random number and particular shape
sample_tensor = torch.randn(1, 28, 28)
sample_tensor.size()

# reshape to (28, 28)
sample_reshaped = sample_tensor.view(28, 28)
sample_reshaped = sample_tensor.view(-1, 1)
sample_reshaped = sample_tensor.view(-1, )
sample_reshaped = sample_tensor.view(-1)
sample_reshaped.size()


####################################
# apply activation to Tensor
import torch
import torch.nn.functional as F
from torch.autograd import Variable

sample_tensor = torch.randn(1,5,5)
# from torch tensor to Variable
sample_var = Variable(sample_tensor)

# access a single value in the Variable
sample_var[0][0][0]

# F.relu only takes in Variable not tensor
y_relu_var = F.relu(sample_var).data.numpy()

y_sigmoid = F.sigmoid(sample_var).data.numpy()
y_tanh = F.tanh(sample_var).data.numpy()
y_softplus = F.softplus(sample_var).data.numpy()
y_softmax = F.softmax(sample_var).data.numpy()
