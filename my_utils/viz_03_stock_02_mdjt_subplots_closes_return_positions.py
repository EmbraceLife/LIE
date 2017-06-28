"""
### Summary
- combine training, valid, test predictions arrays into one array
- get cumulative return curve
- do subplot close price, return curve and positions bar
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# get mdjt closes array
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays
mdjt_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices/mdjt_prices.csv"
_,_,_,_, closes, _ = csv_df_arrays(mdjt_path)

# get mdjt preds_target array
from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
mdjt_preds_target = get_stock_preds_target(mdjt_path)


### get return curve dataset
target_closes = closes/closes[0]# normalized price
daily_capital=[]
# first day's capital
daily_capital.append(1. * (1. + mdjt_preds_target[0,0]*mdjt_preds_target[0,1]))

## see the picture the author drew to refresh the logic

# from second day onward
for idx in range(1, len(mdjt_preds_target)):
	daily_capital.append(daily_capital[idx-1]*(1+mdjt_preds_target[idx,0]*mdjt_preds_target[idx,1]))
accum_profit = np.array(daily_capital)-1


### plot return curve
# todo: plot the max and latest value on graph
plt.figure()
ax1 = plt.subplot2grid((7, 3), (0, 0), colspan=3, rowspan=3)  # stands for axes
ax1.plot(target_closes, c='blue', label='close_price')
ax1.set_title('close_prices')
ax2 = plt.subplot2grid((7, 3), (3, 0), colspan=3, rowspan=3)
ax2.plot(accum_profit, c='red', label='train_profit')
ax3 = plt.subplot2grid((7, 3), (6, 0), colspan=3)
X = np.arange(len(mdjt_preds_target))
ax3.bar(X, mdjt_preds_target[:,0], facecolor='#9999ff', edgecolor='blue')


plt.tight_layout()
plt.show()
