"""
### Summary
- combine training, valid, test predictions arrays into one array
- get cumulative return curve
- do subplot close price, return curve and positions bar
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# get closes array
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays
index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index50.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index300.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index500.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index100.csv"
date,_,_,_, closes, _ = csv_df_arrays(index_path)


# get mdjt preds_target array
from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
index_preds_target = get_stock_preds_target(index_path)

# zoom in and out
closes = closes[-700:]
index_preds_target = index_preds_target[-700:]

### get return curve dataset
target_closes = closes/closes[0]-1# normalized price to price changes
daily_capital=[]
# first day's capital
daily_capital.append(1. * (1. + index_preds_target[0,0]*index_preds_target[0,1]))

## see the picture the author drew to refresh the logic

# from second day onward
for idx in range(1, len(index_preds_target)):
	# # situation one:
	# # everyday's accumulated capital, ignore trading cost
	# accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
	# daily_capital.append(accum_capital)

	# situation two:
	# everyday's accumulated capital, consider trading cost
	if index_preds_target[idx-1,0] == index_preds_target[idx,0] == 1.0 or index_preds_target[idx-1,0] == index_preds_target[idx,0] == 0.0:
		# no trade, no trading cost today
		accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
		daily_capital.append(accum_capital)
	elif index_preds_target[idx-1,0] == index_preds_target[idx,0] and index_preds_target[idx-1,1] == index_preds_target[idx,1] == 0.0:
		# no trade, no trading cost today
		accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
		daily_capital.append(accum_capital)

	else:
		# cost = (today's holding position capital - yesterday's holding position capital)*0.001
		cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
		# today's accum_capital = today's accum_capital - cost
		accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
		daily_capital.append(accum_capital)

accum_profit = np.array(daily_capital)-1
print("final date:", date[-1])
print("final_return:", accum_profit[-1])

##### accumulation of transaction percentage
preds = index_preds_target[:,0]
changes_preds = np.abs(preds[1:] - preds[:-1])
accum_change_capital = np.cumsum(changes_preds)

### plot return curve
# todo: plot the max and latest value on graph
plt.figure()
ax1 = plt.subplot2grid((8, 3), (0, 0), colspan=3, rowspan=4)  # stands for axes
ax1.plot(target_closes, c='blue', label='index000001') # change index name
ax1.plot(accum_profit, c='red', label='accum_profit')
ax1.legend(loc='best')
ax1.set_title('from %s to %s return: %04f' % (date[0], date[-1], accum_profit[-1]))

ax2 = plt.subplot2grid((8, 3), (4, 0), colspan=3, rowspan=2)
ax2.plot(accum_change_capital, c='red', label='accum_change_capital')
ax2.legend(loc='best')
ax2.set_title("accumulated total transactions of full capital: %02f" % accum_change_capital[-1])

ax3 = plt.subplot2grid((8, 3), (6, 0), colspan=3, rowspan=2)
X = np.arange(len(index_preds_target))
# ax3.bar(X, index_preds_target[:,0], facecolor='#9999ff', edgecolor='blue')
ax3.plot(index_preds_target[:,0], c="green")
ax3.set_title('model2 with lr0.001 no cost') # change model name

plt.tight_layout()
plt.show()
# plt.savefig("/Users/Natsume/Downloads/data_for_all/stocks/model_performance/index000001_position.png")
