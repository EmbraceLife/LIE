"""
extract_features_target_from_csv
- from a stock csv file, extract features array, target array
- eatures array has 61 indicators
- target array has price_change

Uses: extract_feature()
1. with OHLCV arrays and many indicators, create features array and target array
2. example using it

Inputs:
1. OHLCV arrays (embedded within the function)
2. file_path
3. list of indicator names
4. window or steps

Return:
1. feature array (?, 61, 30): 61 different indicators
2. target array (?,): price_change compared with next day

Note:
1. features created with day_[0,..., 29] and shift one day forward after
2. target created with day_[29, 30] and shift one day forward after

"""

import talib
import math
import numpy as np
import numpy


class IndicatorCreator(object):
    """
    1. self.selector: user selected indicators (only applicable if matched with internally supported indicators)
    2. self.supported: show internal supported indicators
    3. self.feature: empty list to store converted indicator data later
    """

    def __init__(self, selector):
        self.selector = selector
        self.supported = {"ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"}
        self.feature = []

    """
    1. convert a list of 61 arrays (6464, ) to a single array (61, 6464)
    2. convert (61, 6464) to a list of 6434 arrays (61, 30)
    3. convert the above list to array (6434, 61*30)
    4. create targe array (6434, ) of price change start from 30th day onward
    5. return feature array (6434, 61*30), target array (6434, )
    """
    def moving_extract(self, window=30, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                       volumes=None, with_label=True, flatten=True):
        self.extract(open_prices=open_prices, close_prices=close_prices, high_prices=high_prices, low_prices=low_prices,
                     volumes=volumes) # self.feature is a list with 61 indicators
        feature_arr = np.asarray(self.feature) # list to array, shape (61, 6464) = (num_indicators, num_days)
        p = 0
        rows = feature_arr.shape[0]
        print("feature dimension: %s" % rows)
        if with_label:
            moving_features = []
            moving_labels = []
            while p + window < feature_arr.shape[1]: # start from 30th day, loop for each everyday onward
                x = feature_arr[:, p:p + window] # take all indicators for 30 days range, start from day 1, then start from day 2, and so on
                # p_change: (the day-30th-price - day_29th_price)/day_29th_price
                p_change = (close_prices[p + window] - close_prices[p + window - 1]) / close_prices[p + window - 1]
                # use percent of change as label
                y = p_change
                if flatten:
                    x = x.flatten("F") # from (61, 30) to (61*30, )
                moving_features.append(np.nan_to_num(x)) # 1. convert nan to 0.0; 2. store x a 30_days_61_indicators as a long vector into list moving_features
                moving_labels.append(y)
                p += 1
			# now moving_features is a list of 6434 == 6464 - 30
            return np.asarray(moving_features), np.asarray(moving_labels)
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(np.nan_to_num(x))
                p += 1
            return moving_features


    """
    1. check every user provided indicators, see whether match with internally supported indicators
    2. if so, self.extract_by_type() will create each indicator array from arrays of OHLCV
    3. save all indicator arrays into ChartFeature.feature (list)
    """

    def extract(self, open_prices=None, close_prices=None, high_prices=None, low_prices=None, volumes=None):
        self.feature = []
        for feature_type in self.selector:
            if feature_type in self.supported:
                print("extracting feature : %s" % feature_type)
                self.extract_by_type(feature_type, open_prices=open_prices, close_prices=close_prices,
                                     high_prices=high_prices, low_prices=low_prices, volumes=volumes)
            else:
                print("feature type not supported: %s" % feature_type)
        return self.feature

    """
    1. given an user selected indicator name, and arrays of OHLCV
    2. if the indicator match with internal supported indicators, make the indicator array (or arrays)
    3. then append the indicator array(s) into self.feature (list)
    """
    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                        volumes=None):
        if feature_type == 'ROCP':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP':
            orocp = talib.ROCP(open_prices, timeperiod=1)
            self.feature.append(orocp)
        if feature_type == 'HROCP':
            hrocp = talib.ROCP(high_prices, timeperiod=1)
            self.feature.append(hrocp)
        if feature_type == 'LROCP':
            lrocp = talib.ROCP(low_prices, timeperiod=1)
            self.feature.append(lrocp)
        if feature_type == 'MACD':
            macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            norm_signal = numpy.minimum(numpy.maximum(numpy.nan_to_num(signal), -1), 1) # no peeking into the future
            norm_hist = numpy.minimum(numpy.maximum(numpy.nan_to_num(hist), -1), 1)
            norm_macd = numpy.minimum(numpy.maximum(numpy.nan_to_num(macd), -1), 1)

            zero = numpy.asarray([0])
            macdrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(macd)))), -1), 1) # no peeking into future
            signalrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(signal)))), -1), 1)
            histrocp = numpy.minimum(numpy.maximum(numpy.concatenate((zero, numpy.diff(numpy.nan_to_num(hist)))), -1), 1)

            self.feature.append(norm_macd)
            self.feature.append(norm_signal)
            self.feature.append(norm_hist)

            self.feature.append(macdrocp)
            self.feature.append(signalrocp)
            self.feature.append(histrocp)
        if feature_type == 'RSI':
            rsi6 = talib.RSI(close_prices, timeperiod=6)
            rsi12 = talib.RSI(close_prices, timeperiod=12)
            rsi24 = talib.RSI(close_prices, timeperiod=24)
            rsi6rocp = talib.ROCP(rsi6 + 100., timeperiod=1)
            rsi12rocp = talib.ROCP(rsi12 + 100., timeperiod=1)
            rsi24rocp = talib.ROCP(rsi24 + 100., timeperiod=1)
            self.feature.append(rsi6 / 100.0 - 0.5)
            self.feature.append(rsi12 / 100.0 - 0.5)
            self.feature.append(rsi24 / 100.0 - 0.5)
            # self.feature.append(numpy.maximum(rsi6 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi12 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.maximum(rsi24 / 100.0 - 0.8, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.minimum(rsi6 / 100.0 - 0.2, 0))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            # self.feature.append(numpy.maximum(numpy.minimum(rsi6 / 100.0 - 0.5, 0.3), -0.3))
            self.feature.append(rsi6rocp)
            self.feature.append(rsi12rocp)
            self.feature.append(rsi24rocp)
        if feature_type == 'VROCP':
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1))) # no peeking into the future
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            # self.feature.append(norm_volumes)
            self.feature.append(vrocp)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)
        if feature_type == 'MA':
            ma5 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=5))
            ma10 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=10))
            ma20 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=20))
            ma30 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=30))
            ma60 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=60))
            ma90 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=90))
            ma120 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=120))
            ma180 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=180))
            ma360 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=360))
            ma720 = numpy.nan_to_num(talib.MA(close_prices, timeperiod=720))
            ma5rocp = talib.ROCP(ma5, timeperiod=1)
            ma10rocp = talib.ROCP(ma10, timeperiod=1)
            ma20rocp = talib.ROCP(ma20, timeperiod=1)
            ma30rocp = talib.ROCP(ma30, timeperiod=1)
            ma60rocp = talib.ROCP(ma60, timeperiod=1)
            ma90rocp = talib.ROCP(ma90, timeperiod=1)
            ma120rocp = talib.ROCP(ma120, timeperiod=1)
            ma180rocp = talib.ROCP(ma180, timeperiod=1)
            ma360rocp = talib.ROCP(ma360, timeperiod=1)
            ma720rocp = talib.ROCP(ma720, timeperiod=1)
			# add OHLC for qcg_model_2000 only, for other models must comment out
            # self.feature.append((open_prices-open_prices[0])/open_prices[0])
            # self.feature.append((high_prices-high_prices[0])/high_prices[0])
            # self.feature.append((low_prices-low_prices[0])/low_prices[0])
            # self.feature.append((close_prices-close_prices[0])/close_prices[0])

            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append((ma5 - close_prices) / close_prices)
            self.feature.append((ma10 - close_prices) / close_prices)
            self.feature.append((ma20 - close_prices) / close_prices)
            self.feature.append((ma30 - close_prices) / close_prices)
            self.feature.append((ma60 - close_prices) / close_prices)
            self.feature.append((ma90 - close_prices) / close_prices)
            self.feature.append((ma120 - close_prices) / close_prices)
            self.feature.append((ma180 - close_prices) / close_prices)
            self.feature.append((ma360 - close_prices) / close_prices)
            self.feature.append((ma720 - close_prices) / close_prices)
        if feature_type == 'VMA':
            ma5 = talib.MA(volumes, timeperiod=5)
            ma10 = talib.MA(volumes, timeperiod=10)
            ma20 = talib.MA(volumes, timeperiod=20)
            ma30 = talib.MA(volumes, timeperiod=30)
            ma60 = talib.MA(volumes, timeperiod=60)
            ma90 = talib.MA(volumes, timeperiod=90)
            ma120 = talib.MA(volumes, timeperiod=120)
            ma180 = talib.MA(volumes, timeperiod=180)
            ma360 = talib.MA(volumes, timeperiod=360)
            ma720 = talib.MA(volumes, timeperiod=720)
            ma5rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma5, timeperiod=1)))
            ma10rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma10, timeperiod=1)))
            ma20rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma20, timeperiod=1)))
            ma30rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma30, timeperiod=1)))
            ma60rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma60, timeperiod=1)))
            ma90rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma90, timeperiod=1)))
            ma120rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma120, timeperiod=1)))
            ma180rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma180, timeperiod=1)))
            ma360rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma360, timeperiod=1)))
            ma720rocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(ma720, timeperiod=1)))
            self.feature.append(ma5rocp)
            self.feature.append(ma10rocp)
            self.feature.append(ma20rocp)
            self.feature.append(ma30rocp)
            self.feature.append(ma60rocp)
            self.feature.append(ma90rocp)
            self.feature.append(ma120rocp)
            self.feature.append(ma180rocp)
            self.feature.append(ma360rocp)
            self.feature.append(ma720rocp)
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma5 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma10 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma20 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma30 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma60 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma90 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma120 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma180 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma360 - volumes) / (volumes + 1))))
            self.feature.append(numpy.arctan(numpy.nan_to_num((ma720 - volumes) / (volumes + 1))))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            # norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            # vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            vrocp = numpy.arctan(numpy.nan_to_num(talib.ROCP(numpy.maximum(volumes, 1), timeperiod=1)))
            pv = rocp * vrocp
            self.feature.append(pv)


"""
- given OHLCV arrays imported as global variables
- 30 days as window, user_selected_indicators
- use indicators.moving_extract() to get indicator_features array, target array
	- user select or define 61 indicators
	- 5 arrays OHLCV become single array (61, 6464)
	- use 30 days window to split this array above, we get a list of 6434 arrays of shape (61, 30)
	- convert list to array shape (6434, 61, 30), now we got features array
	- and flatten 61*30 for saving convenience
	- use closes array, with 30 days window, start from 30th day, we get daily price change pct as targets array (6434, )
	- finally, return (moving_features_array, moving_targets_array)

"""

from stock_csv_pandas_array import csv_df_arrays
# from stock_csv_object_array import read_csv_2_arrays

def extract_feature(selector, file_path, window=30, with_label=True, flatten=False):
	# selector: user_selected_indicators to create IndicatorCreator object
    indicators = IndicatorCreator(selector)

	# differentiate csv files from indices and from individual stocks
    # if file_path.find("prices") > -1 or file_path.find("index")>-1 or file_path.find("ETF")> -1:
    dates, opens, highs, lows, closes, volumes = csv_df_arrays(file_path)
    # else: # to be commnet out
    	# _, dates, opens, highs, lows, closes, volumes = read_csv_2_arrays(file_path)

    if with_label:
        moving_features, moving_labels = indicators.moving_extract(window=window, open_prices=opens, close_prices=closes, high_prices=highs, low_prices=lows, volumes=volumes, with_label=with_label, flatten=flatten)
		# extract_feature: func outside ChartFeature class
		# dive into ChartFeature.moving_extract():
		# then into ChartFeature.extract(): extract one feature group at a time
		# ChartFeature.feature is a list has 61 indicators in total
        return moving_features, moving_labels
    else:
        moving_features = indicators.moving_extract(window=window, open_prices=opens, close_prices=closes,
                                                       high_prices=highs, low_prices=lows, volumes=volumes,
                                                       with_label=with_label, flatten=flatten)
        return moving_features

"""
### dataset example 1
# try dataset from csv to objets to arrays
# from prep_data_03_stock_01_csv_2_objects_2_arrays_DOHLCV import opens, highs, lows, closes, volumes


### dataset example 2
# try dataset from csv to pandas to arrays
from prep_data_03_stock_01_csv_2_pandas_2_arrays_DOHLCV import csv_df_arrays

stock_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices/mdjt_prices.csv"
dates, opens, highs, lows, closes, volumes = csv_df_arrays(stock_path)

### get features and targets
# all internal supported indicators are selected here
user_indicators = ["ROCP", "OROCP", "HROCP", "LROCP", "MACD", "RSI", "VROCP", "BOLL", "MA", "VMA", "PRICE_VOLUME"]

# get features array and target array
moving_indicators_features, moving_real_price_changes = extract_feature(selector=user_indicators, file_path=stock_path)

moving_real_price_changes.shape
"""
