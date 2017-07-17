"""
plot_price_return_num_trades
- get predictions of a stock
- calc cumulative return
- calc num_trades
- do subplots

- continuous_color_reference
http://matplotlib.org/examples/color/colormaps_reference.html

### Summary
- combine training, valid, test predictions arrays into one array
- get cumulative return curve
- do subplot close price, return curve and positions bar
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm

# 获取标的收盘价和日期序列
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays
index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF50.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF500.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF300.csv"
# index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/ETF100.csv"

date,_,_,_, closes, _ = csv_df_arrays(index_path)


from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target
index_preds_target = get_stock_preds_target(index_path)
"""
index_preds_target 中包含2组数据：
index_preds_target[:, 0]:预测值，即下一日的持仓的总资产占比
index_preds_target[:, 1]:下一日的当天价格变化
"""



# zoom in and out for the last 700 trading days
closes = closes[-700:]
index_preds_target = index_preds_target[-700:]

### get return curve dataset
target_closes = closes/closes[0]-1# normalized price to price changes
daily_capital=[]

######################################################
# 收益计算 _ v1.0, 对我的脑力要求太高，can't do it!
######################################################
	# 设置阀值，筛选买入，持有和卖出的时间点
	# 买入： 在第n个（pos>0.95）出现时，买入; 更改持仓状态
	# 持有： 0.05 < pos < 0.95时，均持有;
	# 卖出： 在第n个 （pos < 0.05）出现时，卖出; 更改持仓状态
	# 持仓状态： has_pos = True or False
# buy_signal = 0.9
# sell_signal = 0.1
# patience = 3 # 1,2,3
# pos = index_preds_target[:,0]
# price_ch = index_preds_target[:,1]
# pos_state = False
# buy_signal_count = 1
# sell_signal_count = 1
# init_capital = 100000
# daily_capital_list = []
#
# ### 搜索最佳组合
# for (buy_signal, sell_signal) in [(0.9,0.1), (0.8, 0.2), (0.7, 0.3)]:
# 	for patience in [1,2,3]:
#
# 		daily_capital=[]
# 		for index in range(len(pos)):
# 			# 买入条件： 没有持仓，买入信号出现 buy_signal_count 次
# 			if pos[index] > buy_signal and buy_signal_count == patience and pos_state == False:
# 				pos_state = True # 设置持仓状态
# 				if len(daily_capital) == 0:
# 					daily_capital.append(init_capital * (1 + price_ch[index])) # 计算当日收益，收集
# 				else:
# 					daily_capital.append(daily_capital[index-1] * (1 + price_ch[index])) # 计算当日收益，收集
# 				buy_signal_count += 1
# 				sell_signal_count = 1
#
# 			# 卖出条件： 有持仓，卖出信号出现 sell_signal_count 次
# 			elif pos[index] < sell_signal and buy_signal_count == patience and pos_state == True:
# 				pos_state = False # 设置持仓状态
# 				sell_signal_count+=1
# 				buy_signal_count = 1
# 				daily_capital.append(daily_capital[index-1]) # 计算当日收益，收集
#
#
# 			else:
# 				# 持有状态下, 计算当天市值，收集
# 				if pos_state == True and index > 0:
# 					daily_capital.append(daily_capital[index-1] * (1 + price_ch[index]))
# 				# 空仓状态下，计算当天市值，收集
# 				else:
# 					if len(daily_capital) == 0:
# 						daily_capital.append(init_capital)
# 					else:
# 						daily_capital.append(daily_capital[index-1])
# 				# 依旧累积买入和卖出信号发生次数
# 				if pos[index] > buy_signal:
# 					buy_signal_count+=1
# 				if pos[index] < sell_signal:
# 					sell_signal_count+=1
#
# 		daily_capital_list.append(np.array(daily_capital))

# for captial in daily_capital_list:
# 	plt.plot(capital)
# plt.show()

######################################################
# 收益计算v0
######################################################
"""
### 如何计算累积的每日的总资产（包括计算交易成本）？代码在下面可见

daily_capital.append(1. * (1. + index_preds_target[0,0]*index_preds_target[0,1]))
- 上面第一个 '1' : 初始本金
- 1 * index_preds_target[0,0]*index_preds_target[0,1]: 持仓市值产生的利润 = 1 * 当日开盘时所持仓资产占总资产的比例（即前一日收盘时作出的预测值）* 当日价格的变化值

accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
daily_capital.append(accum_capital)
- daily_capital[idx-1]: 昨天的总资产
- daily_capital[idx-1]*index_preds_target[idx,0]*index_preds_target[idx,1]:今天持仓市值产生的利润 = 昨日收盘总资产 * 当日开盘时所持仓资产占总资产的比例（即昨日收盘时作出的预测值）* 当日价格的变化值

daily_capital.append(accum_capital): 将累积的每天的总资产收入daily_capital这个list中

### 如何在计算每日总资产的同时，减去交易成本？
if index_preds_target[idx-1,0] == index_preds_target[idx,0] == 1.0 or index_preds_target[idx-1,0] == index_preds_target[idx,0] == 0.0:
- 如果昨天的持仓占比于今天的持仓占比都是100%， 那么不存在交易，也就没有交易成本；
- 如果昨天的持仓占比于今天的持仓占比都是0%， 那么不存在交易，也就没有交易成本；

elif index_preds_target[idx-1,0] == index_preds_target[idx,0] and index_preds_target[idx-1,1] == index_preds_target[idx,1] == 0.0:
- 如果昨天的持仓占比 == 今天的持仓占比， 而且昨日价格变化和今日价格变化都是0， 那么没有交易，也就没有交易成本

else:
	# cost = (today's holding position capital - yesterday's holding position capital)*0.001
	cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
	# today's accum_capital = today's accum_capital - cost
	accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
	daily_capital.append(accum_capital)

那么其他情况，全部存在交易成本问题：

cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
- 今日交易成本 = abs(今日持仓资产 - 昨日持仓资产）* 交易成本占比 = abs(昨日总资产*今日持仓的资产占比 - 前日总资产*昨日持仓的资产占比)*交易成本占比

accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
- 今日总资产 = 昨日总资产 + 昨日总资产*今日持仓的总资产占比*今日价格变化 - 今日交易成本

daily_capital.append(accum_capital)
- 将累积的每天的总资产收入daily_capital这个list中
"""


## see the picture the author drew to refresh the logic

# from second day onward
for idx in range(len(index_preds_target)):

	# 第一天的市值： 第一天开盘买入，当天结束时的市值
	if idx == 0:
		daily_capital.append(1. * (1. + index_preds_target[0,0]*index_preds_target[0,1]))

	else:
		# 情况1:
		# 完全忽略交易成本
		accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
		daily_capital.append(accum_capital)

		# # 情况2：
		# # 计算实际交易成本
		# if index_preds_target[idx-1,0] == index_preds_target[idx,0] == 1.0 or index_preds_target[idx-1,0] == index_preds_target[idx,0] == 0.0:
		# 	# no trade, no trading cost today
		# 	accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
		# 	daily_capital.append(accum_capital)
		# elif index_preds_target[idx-1,0] == index_preds_target[idx,0] and index_preds_target[idx-1,1] == index_preds_target[idx,1] == 0.0:
		# 	# no trade, no trading cost today
		# 	accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1])
		# 	daily_capital.append(accum_capital)
		#
		# else:
		# 	# cost = (today's holding position capital - yesterday's holding position capital)*0.001
		# 	cost = np.abs((daily_capital[idx-1]*index_preds_target[idx,0]- daily_capital[idx-2]*index_preds_target[idx-1,0])*0.001)
		# 	# today's accum_capital = today's accum_capital - cost
		# 	accum_capital = daily_capital[idx-1]*(1+index_preds_target[idx,0]*index_preds_target[idx,1]) - cost
		# 	daily_capital.append(accum_capital)

accum_profit = np.array(daily_capital)-1 # 累积总资产减去初始资产 = 累积收益
print("final date:", date[-1])
print("final_return:", accum_profit[-1])

#################################### 收益计算_v0 end


####################################
##### accumulation of transaction percentage，即换手率
####################################
preds = index_preds_target[:,0] # 每日持仓的总资产占比
changes_preds = np.abs(preds[1:] - preds[:-1]) # 相邻两日的持仓占比之差（变化）
accum_change_capital = np.cumsum(changes_preds) # 累积差值，获得总资产进出市场的次数

################################################################
# price_curve_prediction_continuous_color_simple
# plot close price curve and fill prediction array as price curve color
################################################################
color_data = index_preds_target[:,0]
line_data = target_closes.reshape((-1,1))

def uniqueish_color(color_data):
    """There're better ways to generate unique colors, but this isn't awful."""
    # return plt.cm.gist_ncar(color_data)
    # return plt.cm.binary(color_data)
    return plt.cm.bwr(color_data)

X = np.arange(len(line_data)).reshape((-1,1))
y = line_data
xy = np.concatenate((X,y), axis=1)

plt.figure()
ax1 = plt.subplot2grid((6, 3), (0, 0), colspan=3, rowspan=4)
#############
### plot close_price curve and fill predictions as continuous color ######
#############
for start, stop, col in zip(xy[:-1], xy[1:], color_data):
    x, y = zip(start, stop)
    ax1.plot(x, y, color=uniqueish_color(col))
ax1.plot(accum_profit, c='gray', alpha=0.5, label='accum_profit')
ax1.legend(loc='best')
ax1.set_title('from %s to %s return: %04f' % (date[0], date[-1], accum_profit[-1]))

#############
### plot 换手率
#############
ax2 = plt.subplot2grid((6, 3), (4, 0), colspan=3, rowspan=2)
ax2.plot(accum_change_capital, c='red', label='accum_change_capital')
ax2.legend(loc='best')
ax2.set_title("ETF50 TurnOver Rate: %02f" % accum_change_capital[-1])

### plot pos as bars or as color map
# ax3 = plt.subplot2grid((10, 3), (6, 0), colspan=3, rowspan=4)
# X = np.arange(len(index_preds_target))
# ax3.bar(X, index_preds_target[:,0], facecolor='#9999ff', edgecolor='blue')
# ax3.set_title('pos as bars') # change model name

plt.tight_layout()
plt.show()


# img = index_preds_target[:,0].reshape((-1,1)).transpose((1,0))
# imgs = np.concatenate((img, img, img, img, img, img),axis=0)
# plt.imshow(imgs, cmap='binary')
# plt.title("pos as color array")
# plt.show()


# plt.savefig("/Users/Natsume/Downloads/data_for_all/stocks/model_performance/ETF300_model4_addcost.png")
