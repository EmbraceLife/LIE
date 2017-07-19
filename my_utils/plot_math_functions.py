"""
plot_math_functions
"""


import matplotlib.pyplot as plt
import numpy as np
import keras.backend as K
# a function between two variables (y_true, y_pred)
def f(y_true,y_pred):
    return np.exp(-y_true)*np.sin(y_pred-y_true)

y_true = np.linspace(0,5,3001)
y_pred = np.arange(0,40000,4000)

for tval in y_pred:
    plt.plot(y_true, f(y_true, tval))
plt.show()


"""
logloss function

1. loss can be overly rewarded or punished, see through viz of loss func
2. if certain range of predictions can overly reward or punish loss, we need to clip predictions
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import log_loss

x = [i*.0001 for i in range(1,10000)]

# y_true : array-like or label indicator matrix, Ground truth (correct) labels for n_samples samples
y_true1 = np.ones(9999).reshape(-1,1) # all 1s: when all targets are buy
y_true2 = np.zeros(9999).reshape(-1,1)
y_true3 = np.zeros(9999).reshape(-1,1)
y_true = np.concatenate((y_true1, y_true2, y_true3), axis=1)

y_pred1 = np.array([[i*.0001] for i in range(1,10000)]) # gradually becomes buy features
y_pred2 = np.zeros((9999,1))
y_pred3 = np.zeros((9999,1))
y_pred = np.concatenate((y_pred1, y_pred2, y_pred3), axis=1)

ll1 = []
ll2 = []
ll = []
for i in range(9999):
	# ll1.append(log_loss(y_true[i],y_pred1[i],eps=1e-15, labels=[1.0, 0.0]))
	# ll2.append(log_loss(y_true[i],y_pred2[i],eps=1e-15, labels=[1.0, 0.0]))
	ll.append(log_loss(y_true[i],y_pred[i],eps=1e-15, labels=[1.0, 0.0]))

plt.plot(x, ll, c='blue', label="pred_decrease")
# plt.axis([-.05, 1.1, -.8, 10])
plt.title("How loss move when preds[buy] move from 0.0 to 1.0")
plt.xlabel("preds[buy] grow from 0.0 to 1.0")
plt.ylabel("loss")
plt.legend(loc="best")

plt.show()



#
# import matplotlib.pyplot as plt
# import numpy as np
# import keras.backend as K
#
# y_pred = np.random.random(100)
# y_true = np.linspace(0,5,3001)
#
# classes = K.categorical_crossentropy(y_pred, y_true)
# print(classes)
