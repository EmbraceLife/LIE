"""
Uses: run this file
1. use an index csv
2. to plot its last 700 days of return curve with postion, close_prices


"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices/000001.csv"

# load 000001.csv closes
from prep_data_03_stock_01_csv_2_objects_2_arrays_DOHLCV import read_csv_2_arrays
_, _, _, _, _, closes, _ = read_csv_2_arrays(index_path)

# convert index csv to array of preds_target
from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
index_preds_target = get_stock_preds_target(index_path)

# only plot the latest 700 days
target_pos_price_change = index_preds_target[-700:]
target_closes = closes[-700:]
target_closes = target_closes/target_closes[0]# normalized price

# calc return curve
daily_capital = []
daily_capital.append(1 * (1+target_pos_price_change[0,0]*target_pos_price_change[0,1]))

for idx in range(1, len(target_pos_price_change)):
	daily_capital.append(daily_capital[idx-1]*(1+target_pos_price_change[idx,0]*target_pos_price_change[idx,1]))
accum_profit = np.array(daily_capital)-1


# todo: plot the max and latest value on graph
plt.figure()
ax1 = plt.subplot2grid((7, 3), (0, 0), colspan=3, rowspan=3)  # stands for axes
ax1.plot(target_closes, c='blue', label='close_price')
ax1.set_title('close_prices')
ax2 = plt.subplot2grid((7, 3), (3, 0), colspan=3, rowspan=3)
ax2.plot(accum_profit, c='red', label='train_profit')
ax3 = plt.subplot2grid((7, 3), (6, 0), colspan=3)
X = np.arange(len(target_pos_price_change))
ax3.bar(X, target_pos_price_change[:,0], facecolor='#9999ff', edgecolor='blue')


plt.tight_layout()
plt.show()
