# why and how to clip predictions
# extract image ids and cbind with predictions
# how log-loss or cross-entropy behave against predictions 


#############################################
# load sample prediction array using bcolz
from save_load_large_array import bz_load_array

trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
preds = bz_load_array(trained_model_path+"/preds_bz")

# If second column is 1, it's a dog, otherwise cat
isdog = preds[:,1]
print("Raw Predictions: " + str(isdog[:5]))
print("Mid Predictions: {}".format(isdog[(isdog < .6) & (isdog > .4)]))
# %d to display a number, {} can display anything like array or number
print("Edge Predictions: {}".format((isdog[(isdog == 1) | (isdog == 0)])))

###############################################
#Swap all ones with .95 and all zeros with .05
isdog = isdog.clip(min=0.05, max=0.95) # not 0.95, display as 0.9499999
isdog = isdog.round(2) # not help at all


###############################################
# get test folder images into batches and access it
from vgg16_iterator_from_directory import test_batches
import numpy as np
# Extract imageIds from the filenames in our test/unknown directory
filenames = test_batches.filenames
names=[]
for f in filenames:
	names.append(int(f[8:f.find('.')]))

# cbind ids and preds together into a single array
ids = np.array(names)
subm = np.stack([ids,isdog], axis=1)

###############################################
# why clip predictions?

# Log Loss doesn't support probability values of 0 or 1--they are undefined (and we have many). Fortunately, Kaggle helps us by offsetting our 0s and 1s by a very small value. So if we upload our submission now we will have lots of .99999999 and .000000001 values. This seems good, right?

# Not so. There is an additional twist due to how log loss is calculated--log loss rewards predictions that are confident and correct (p=.9999,label=1), but it PUNISH predictions that are confident and wrong FAR MORE (p=.0001,label=1). See visualization below.

# clip is to avoid prediction values to be extremely too large or too small

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

x = [i*.0001 for i in range(1,10000)]

# y_true : array-like or label indicator matrix, Ground truth (correct) labels for n_samples samples
y_true = np.ones(9999).reshape(-1,1)
y_pred1 = np.array([[i*.0001] for i in range(1,10000,1)])
y_pred2 = np.array([[1-(i*.0001)] for i in range(1,10000,1)])

ll1 = []
ll2 = []
for i in range(9999):
	ll1.append(log_loss(y_true[i],y_pred1[i],eps=1e-15, labels=[1.0, 0.0]))
	ll2.append(log_loss(y_true[i],y_pred2[i],eps=1e-15, labels=[1.0, 0.0]))
# log_loss: return a float, Note: use of labels, read source please

plt.plot(x, ll1, c='red', label="pred_increase")
plt.plot(x, ll2, c='blue', label="pred_decrease")
plt.axis([-.05, 1.1, -.8, 10])
plt.title("Log Loss when true label = 1")
plt.xlabel("predicted probability")
plt.ylabel("log loss")
plt.legend(loc="best")

plt.show()
