################################
# create temporal folders 
import tempfile

################################
# prepare logger functionality
import logging
import matplotlib.pyplot as plt
import numpy as np
logger = logging.getLogger(__name__)
from .utils import DisableLogging
# with DisableLogging(): how to disable logging for a function
# if logger.isEnabledFor(logging.WARNING): work for pprint(object.__dict__)

################################
# prepare examine tools
from pdb import set_trace
from pprint import pprint
from inspect import getdoc, getmembers, getsourcelines, getmodule, getfullargspec, getargvalues
# to write multiple lines inside pdb
# !import code; code.interact(local=vars())

################################
# use pytorch to do tensor ops, layers, activations
import torch
from torch.autograd import Variable
import torch.nn.functional as F
