import pandas as pd
import numpy as np
import datetime
import bcolz

# filename='E:/Deep Learning/Comm/Datas/A.txt'
filename = "/Users/Natsume/Downloads/data_for_all/stocks/commodities/A.txt"
A = pd.read_csv(filename)
A.columns = ['total_data']
A = A['total_data'].str.split(' ', expand=True)
A.columns = ['dates','open','high','low','close','volume']
A = A.set_index('dates')
A['close']=A['close'].astype(float)
A['pct_chg']=A['close']/A['close'].shift(1)-1

A['MA10']=pd.rolling_mean(A['close'],window=10)
# print(A)

train_arr = A.loc['2005-01-01':'2009-12-31',['close','pct_chg','MA10']]
valid_arr = A.loc['2010-01-01':'2011-12-31',['close','pct_chg','MA10']]
train_arr = train_arr.values
valid_arr = valid_arr.values
# print(valid_arr.shape)


p = 0
window=30
train_features = []
train_targets = []
while p + window < train_arr.shape[0]:  # start from 30th day, loop for each everyday onward
    x = train_arr[p:p + window,:]  # take all indicators for 30 days range, start from day 1, then start from day 2, and so on
    # use percent of change as label
    # x = x.flatten("F")  # from (61, 30) to (61*30, )
    y = train_arr[p + window,1]
    train_features.append(np.nan_to_num(x))  # 1. convert nan to 0.0; 2. store x a 30_days_61_indicators as a long vector into list moving_features
    train_targets.append(y)
    p += 1

p = 0
valid_features = []
valid_targets = []
while p + window < valid_arr.shape[0]:  # start from 30th day, loop for each everyday onward
    x = valid_arr[p:p + window,:]  # take all indicators for 30 days range, start from day 1, then start from day 2, and so on
    # use percent of change as label
    # x = x.flatten("F")  # from (61, 30) to (61*30, )
    y = valid_arr[p + window,1]
    valid_features.append(np.nan_to_num(x))  # 1. convert nan to 0.0; 2. store x a 30_days_61_indicators as a long vector into list moving_features
    valid_targets.append(y)
    p += 1


train_features=np.asarray(train_features)
train_targets=np.asarray(train_targets)
valid_features=np.asarray(valid_features)
valid_targets=np.asarray(valid_targets)

print(valid_features.shape)

# train_features_path = "E:/Deep Learning/Comm/OutPut/train_features_path"
# train_targets_path = "E:/Deep Learning/Comm/OutPut/train_targets_path"
# valid_features_path = "E:/Deep Learning/Comm/OutPut/valid_features_path"
# valid_targets_path = "E:/Deep Learning/Comm/OutPut/valid_targets_path"
train_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets_commodities/train_features_path"
train_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets_commodities/train_targets_path"
valid_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets_commodities/valid_features_path"
valid_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets_commodities/valid_targets_path"


# test_features_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets_commodities/test_features_path"
# test_targets_path = "/Users/Natsume/Downloads/data_for_all/stocks/features_targets_commodities/test_targets_path"



c=bcolz.carray(train_features, rootdir=train_features_path, mode='w')
c.flush()
c=bcolz.carray(train_targets, rootdir=train_targets_path, mode='w')
c.flush()
c=bcolz.carray(valid_features, rootdir=valid_features_path, mode='w')
c.flush()
c=bcolz.carray(valid_targets, rootdir=valid_targets_path, mode='w')
c.flush()
