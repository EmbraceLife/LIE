"""
plot_price_return_num_trades
- get predictions of a stock
- calc cumulative return
- calc num_trades
- do subplots

- reduce_trade_frequency
- continuous_color_reference
http://matplotlib.org/examples/color/colormaps_reference.html

Datasets: ETF 300 and ETF 50
- use the simple method to cut down trade frequency

"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# 获取标的收盘价和日期序列
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF50.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF500.csv"
index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF300.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index50.csv"

date,open_prices,_,_, closes, _ = csv_df_arrays(index_path)


from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
index_preds_target = get_stock_preds_target(index_path)
"""
index_preds_target 中包含2组数据：
index_preds_target[:, 0]:预测值，即下一日的持仓的总资产占比
index_preds_target[:, 1]:下一日的当天价格变化
"""

#####################3
# plot with different time span
####################

# 30 days
# 90 days
time_span = 700  # 从今天回溯700 days
# time_span = 90  # 从今天回溯90 days
# time_span = 30  # 从今天回溯30 days
# time_span = 1  # 从今天回溯1 days

# time_span = 6  # 昨天开始交易，到今天收盘，交易开始两天了



# zoom in and out for the last 700 trading days
open_prices = open_prices[-time_span:]
closes = closes[-time_span:]
index_preds_target = index_preds_target[-time_span:]


################################################################
# The latest algo to cut down trade frequency
################################################################
init_capital = 1000000 # 初始总现金
y_pred = index_preds_target[:,0] # 预测值，当日持仓市值占比
y_true = index_preds_target[:,1] # 相邻两天收盘价变化率
open_prices = open_prices # 每日开盘价，标准化处理 (受否必须再减1？)
closes = closes # 每日收盘价，标准化处理
daily_shares_pos = [] # 用于收集每日的持股数，让实际交易便捷
daily_capital = []



buy_threshold=0.99 # 0.9 for ETF50, 0.99 for ETF 300
sell_threshold=0.01 # 0.1 for ETF50, 0.01 for ETF 300


# add a 1.0 to the beginning of y_pred array
y_pred = np.concatenate((np.array([1]), y_pred),0)
daily_shares_pos = [0.0] # 用于收集每日的持股数，让实际交易便捷
daily_capital = [init_capital]

# start from the second value of y_pred and end before the last y_pred
for idx in range(1, len(y_pred)-1):
	if y_pred[idx] > buy_threshold:
		y_pred[idx] = 1.0
	elif y_pred[idx] < sell_threshold:
		y_pred[idx] = 0.0
	else:
		y_pred[idx] = 0.5

# 从第二个y_pred开始

shares_pos = 0.0
for idx in range(1, len(y_pred)-1):
	if idx == 1:
		if y_pred[idx] == 1.0: # 如果是当下y_pred是1.0
			shares_pos = np.round(1000000/open_prices[idx], -2) # 全仓买入，100股为一手
		elif y_pred[idx] == 0.0: # 如果当下是0.0；维持空仓
			shares_pos = 0.0
		else:
			shares_pos = 0.0  # 如果是0.5，维持空仓
		daily_shares_pos.append(shares_pos) # 总持仓
		cost = shares_pos*open_prices[idx]*0.001
		cash_left = 1000000 - shares_pos*open_prices[idx] - cost
		end_day_capital = shares_pos*closes[idx] + cash_left
		daily_capital.append(end_day_capital)

	# 从第三个y_pred开始
	else:
		if y_pred[idx-1] == 1.0: # 如果上一个y_pred是1.0
			if y_pred[idx] == 1.0: # 如果上一个y_pred是1.0，维持仓位
				shares_pos = daily_shares_pos[idx-1]
			elif y_pred[idx] == 0.0: # 如果上一个y_pred是0.0，那么全仓买入
				shares_pos = 0.0
			else:
				shares_pos = daily_shares_pos[idx-1]


		elif y_pred[idx-1] == 0.0: # 如果上一个y_pred是1.0
			if y_pred[idx] == 1.0: # 如果上一个y_pred是1.0，维持仓位
				shares_pos = np.round(daily_capital[idx-1]/open_prices[idx], -2)
			elif y_pred[idx] == 0.0: # 如果上一个y_pred是0.0，那么全仓买入
				shares_pos = 0.0
			else:
				shares_pos = 0.0


		elif y_pred[idx-1] == 0.5 and daily_shares_pos[idx-1]>0.0: # 如果上一个y_pred是0.5,同时是满仓
			if y_pred[idx] == 1.0: # 如果上一个y_pred是1.0，维持仓位
				shares_pos = daily_shares_pos[idx-1]
			elif y_pred[idx] == 0.0: # 如果上一个y_pred是0.0，那么全仓买入
				shares_pos = 0.0
			else:
				shares_pos = daily_shares_pos[idx-1]


		elif y_pred[idx-1] == 0.5 and daily_shares_pos[idx-1] == 0.0: # 如果上一个y_pred是0.5,同时是满仓
			if y_pred[idx] == 1.0: # 如果上一个y_pred是1.0，维持仓位
				shares_pos = np.round(daily_capital[idx-1]/open_prices[idx], -2)
			elif y_pred[idx] == 0.0: # 如果上一个y_pred是0.0，那么全仓买入
				shares_pos = 0.0
			else:
				shares_pos = 0.0

		daily_shares_pos.append(shares_pos) # 总持仓
		cost = np.abs(np.array(shares_pos) - np.array(daily_shares_pos[idx-1]))*open_prices[idx]*0.001
		cash_left = daily_capital[idx-1] - shares_pos*open_prices[idx] - cost
		end_day_capital = shares_pos*closes[idx] + cash_left
		daily_capital.append(end_day_capital)

# 计算累积总资金曲线
accum_profit = (np.array(daily_capital)/daily_capital[0])-1# 累积总资金曲线， 减去初始资金 1，获得收益累积曲线
print("final date:", date[-1]) # 最近日期
print("final_return:", accum_profit[-1]) # 累积总收益

changes_pos_rate = np.abs(np.array(daily_shares_pos[1:]) - np.array(daily_shares_pos[:-1]))/np.array(daily_shares_pos).max() # 相邻两日的持仓占比之差（变化）
turnover_rate = np.cumsum(changes_pos_rate) # 累积差值，获得总资产进出市场的次数

## color data for close prices： 经过阀值调节过的预测值
color_data = y_pred
## 用于画收盘价曲线的数据
target_closes = closes/closes[0]
################################################################
# price_curve_prediction_continuous_color_simple
# plot close price curve and fill prediction array as price curve color
################################################################
# different ways of setting color data

line_data = target_closes.reshape((-1,1))-1 # close_price normalized to start from 1 onward, so minus 1 to bring it down to 0 original for plotting

def uniqueish_color(color_data):
    """There're better ways to generate unique colors, but this isn't awful."""
    # return plt.cm.gist_ncar(color_data)
    # return plt.cm.binary(color_data)
    return plt.cm.bwr(color_data)

X = np.arange(len(line_data)).reshape((-1,1))
y = line_data
xy = np.concatenate((X,y), axis=1)



plt.figure()
ax1 = plt.subplot2grid((8, 3), (0, 0), colspan=3, rowspan=4)
#############
### plot close_price curve and fill predictions as continuous color ######
#############
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax1.plot(x, y, color=uniqueish_color(col))

ax1.plot(accum_profit, c='gray', alpha=0.5, label='accum_profit')
ax1.legend(loc='best')
ax1.set_title('ETF300(>0.99, <0.01) from %s to %s return: %04f' % (date[0], date[-1], accum_profit[-1]))

#############
### plot 换手率
#############
ax2 = plt.subplot2grid((8, 3), (4, 0), colspan=3, rowspan=2)
ax2.plot(turnover_rate, c='red', label='turnover_rate')
ax2.legend(loc='best')
ax2.set_title("TurnOver Rate: %02f" % turnover_rate[-1])

### plot daily_shares_pos curve
ax3 = plt.subplot2grid((8, 3), (6, 0), colspan=3, rowspan=2)
init_shares_full = init_capital/open_prices[0]
daily_shares_rate = np.array(daily_shares_pos)/init_shares_full
ax3.plot(daily_shares_rate, c='k', label='daily_shares_rate')
ax3.legend(loc='best')
ax3.set_title('init_share_number: %d, latest_share_number: %d' % (init_shares_full, daily_shares_pos[-1])) # change model name

plt.tight_layout()
plt.show()


# img = index_preds_target[:,0].reshape((-1,1)).transpose((1,0))
# imgs = np.concatenate((img, img, img, img, img, img),axis=0)
# plt.imshow(imgs, cmap='binary')
# plt.title("pos as color array")
# plt.show()


# plt.savefig("/Users/Natsume/Downloads/data_for_all/stocks/model_performance/ETF300_model4_addcost.png")
