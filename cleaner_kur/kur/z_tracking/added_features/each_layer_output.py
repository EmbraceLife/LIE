########################################################################
"""
# Experiment: get outcome of every layer
"""
#######################################################
# a batch is ready from above
# get weights from a particular weight file

# create a single data sample
sample = None
for batch in provider:
	sample = batch['images'][0]
	break

# make weights and biases available as numpy arrays
import kur.utils.idx as idx
# shape (10, 784)
dense_w = idx.load("../../Hvass_tutorial1_folders/mnist.best.valid.w/layer___dense_0+weight.kur")
# shape (10,)
dense_b = idx.load("../../Hvass_tutorial1_folders/mnist.best.valid.w/layer___dense_0+bias.kur")
# sample (1, 784), weight(784, 10) + bias (10,)

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# convert data sample, weights, biases to torch.Variable
# get numpy into Tensor, then to Variable
sample_tensor = torch.from_numpy(sample)
sample_var = Variable(sample_tensor)
weight_tensor = torch.from_numpy(dense_w)
weight_var = Variable(weight_tensor)
bias_tensor = torch.from_numpy(dense_b)
bias_var = Variable(bias_tensor)

# flatten layer
# reshape sample_var
sample_flat = sample_var.view(1, -1)

# hidden 10 node dense layer
# transpose a tensor or variable
weight_var_t = torch.t(weight_var)
# matmul: torch.mm
sample_w = torch.mm(sample_flat, weight_var_t)
# add: torch.add
# hidden layer output
sample_wb = torch.add(sample_w, bias_var)
# size(): see shape (1, 10)
# sample_wb.size()


# activation layer output
# apply F.softmax
output = F.softmax(sample_wb)
# test F.softmax
# output.sum()
# torch.sum(output)

################################
