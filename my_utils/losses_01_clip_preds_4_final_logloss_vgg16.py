"""
1. loss can be overly rewarded or punished, see through viz of loss func
2. if certain range of predictions can overly reward or punish loss, we need to clip predictions
"""


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


#####################################
from prep_data_98_funcs_save_load_large_arrays import bz_load_array
trained_model_path = "/Users/Natsume/Downloads/data_for_all/dogscats/results"
# clip predictions
# load test_predictions on training set total 200 samples
preds_test = bz_load_array(trained_model_path+"/preds_test")
# If second column is 1, it's a dog, otherwise cat
isdog = preds_test[:,1]
print("Raw Predictions: " + str(isdog[:5]))
print("Mid Predictions: {}".format(isdog[(isdog < .6) & (isdog > .4)]))
# %d to display a number, {} can display anything like array or number
print("Edge Predictions: {}".format((isdog[(isdog == 1) | (isdog == 0)])))

###############################################
# clip every value above 0.95 to 0.95, and every value below 0.05 to 0.05
isdog = isdog.clip(min=0.05, max=0.95) # not 0.95, display as 0.9499999
isdog = isdog.round(2) # not help at all
