# source from https://github.com/happynoom/DeepTrade_keras

"""
### workflow of funcs and class to produce features and target arrays

extract_feature():
	0. ChartFeature.__init__:
	1. raw_data: a list of objects converted from stock.csv; each object is a day's stock info (date, OHLCV)
	2. raw_data converted to 5 arrays corresponding to OHLCV data
	3. chart_feature.moving_extract() to create features and target
		1. chart_feature.extract:
			- 6464 days OHLCV, converted to array of 61 features, shape (61, 6464);
			- extract_by_type: convert OHLCV to indicator array one at a time
		2. split (61, 6464) into a list of 6434 of (61, 30);
		3. then convert it to array shape (6434, 61*30);
		4. but why not (6434, 61, 30)
		5. return moving_features (6434, 1830), moving_labels (6434, )
	4. return moving_features (6434, 1830), moving_labels (6434, )
"""


import numpy
import talib
import math


class ChartFeature(object):
    """
    1. instantiate ChartFeature object with a user selected indicator names
    2. self.supported: internal supported indicators
    3. self.feature: empty list to store converted indicator data
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
        feature_arr = numpy.asarray(self.feature) # list to array, shape (61, 6464) = (num_indicators, num_days)
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
                moving_features.append(numpy.nan_to_num(x)) # 1. convert nan to 0.0; 2. store x a 30_days_61_indicators as a long vector into list moving_features
                moving_labels.append(y)
                p += 1
			# now moving_features is a list of 6434 == 6464 - 30
            return numpy.asarray(moving_features), numpy.asarray(moving_labels)
        else:
            moving_features = []
            while p + window <= feature_arr.shape[1]:
                x = feature_arr[:, p:p + window]
                if flatten:
                    x = x.flatten("F")
                moving_features.append(numpy.nan_to_num(x))
                p += 1
            return moving_features


    """
    1. if user provided indicator names are supported
    2. self.extract_by_type(): create indicator arrays based on stock prices arrays (OHLCV arrays)
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
    1. given a indicator name group, create an indicator array from OHLCV data
    2. store indicator array inside ChartFeature.feature (list)
    3. if all indicators names are supported, then there will be 61 features in total
    """
    def extract_by_type(self, feature_type, open_prices=None, close_prices=None, high_prices=None, low_prices=None,
                        volumes=None):
        if feature_type == 'ROCP': # price change pct daily basis on close
            rocp = talib.ROCP(close_prices, timeperiod=1)
            self.feature.append(rocp)
        if feature_type == 'OROCP': # price change pct daily basis on open
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
            norm_macd = numpy.nan_to_num(macd) / math.sqrt(numpy.var(numpy.nan_to_num(macd)))
            norm_signal = numpy.nan_to_num(signal) / math.sqrt(numpy.var(numpy.nan_to_num(signal)))
            norm_hist = numpy.nan_to_num(hist) / math.sqrt(numpy.var(numpy.nan_to_num(hist)))
            macdrocp = talib.ROCP(norm_macd + numpy.max(norm_macd) - numpy.min(norm_macd), timeperiod=1)
            signalrocp = talib.ROCP(norm_signal + numpy.max(norm_signal) - numpy.min(norm_signal), timeperiod=1)
            histrocp = talib.ROCP(norm_hist + numpy.max(norm_hist) - numpy.min(norm_hist), timeperiod=1)
            # self.feature.append(macd / 100.0)
            # self.feature.append(signal / 100.0)
            # self.feature.append(hist / 100.0)
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
            # vrocp = talib.ROCP(volumes, timeperiod=1)
            norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            # self.feature.append(norm_volumes)
            self.feature.append(vrocp)
        if feature_type == 'BOLL':
            upperband, middleband, lowerband = talib.BBANDS(close_prices, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            self.feature.append((upperband - close_prices) / close_prices)
            self.feature.append((middleband - close_prices) / close_prices)
            self.feature.append((lowerband - close_prices) / close_prices)
        if feature_type == 'MA':
            ma5 = talib.MA(close_prices, timeperiod=5)
            ma10 = talib.MA(close_prices, timeperiod=10)
            ma20 = talib.MA(close_prices, timeperiod=20)
            ma30 = talib.MA(close_prices, timeperiod=30)
            ma60 = talib.MA(close_prices, timeperiod=60)
            ma90 = talib.MA(close_prices, timeperiod=90)
            ma120 = talib.MA(close_prices, timeperiod=120)
            ma180 = talib.MA(close_prices, timeperiod=180)
            ma360 = talib.MA(close_prices, timeperiod=360)
            ma720 = talib.MA(close_prices, timeperiod=720)
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
            self.feature.append((ma5 - volumes) / (volumes + 1))
            self.feature.append((ma10 - volumes) / (volumes + 1))
            self.feature.append((ma20 - volumes) / (volumes + 1))
            self.feature.append((ma30 - volumes) / (volumes + 1))
            self.feature.append((ma60 - volumes) / (volumes + 1))
            self.feature.append((ma90 - volumes) / (volumes + 1))
            self.feature.append((ma120 - volumes) / (volumes + 1))
            self.feature.append((ma180 - volumes) / (volumes + 1))
            self.feature.append((ma360 - volumes) / (volumes + 1))
            self.feature.append((ma720 - volumes) / (volumes + 1))
        if feature_type == 'PRICE_VOLUME':
            rocp = talib.ROCP(close_prices, timeperiod=1)
            norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            # vrocp = talib.ROCP(volumes, timeperiod=1)
            pv = rocp * vrocp * 100.
            self.feature.append(pv)

    """
    1. raw_data: a list of objects converted from stock.csv; each object is a day's stock info (date, OHLCV)
    2. closes, opens, highs, etc: arrays to contain close price, open price, etc
    3. chart_feature.moving_extract():
        1. 6464 days OHLCV, converted to array of 61 features, shape (61, 6464);
        2. split (61, 6464) into a list of 6434 of (61, 30);
        3. then convert it to array shape (6434, 61*30);
        4. but why not (6434, 61, 30)
    4. return moving_features (6434, 1830), moving_labels (6434, )
    """
def extract_feature(raw_data, selector, window=30, with_label=True, flatten=True):
    chart_feature = ChartFeature(selector) # only add 3 simple attributes
    sorted_data = sorted(raw_data, key=lambda x:x.date) # from past to now
    closes = []
    opens = []
    highs = []
    lows = []
    volumes = []
    for item in sorted_data: # for each day's object (date, OHLCV)
        closes.append(item.close) # append its close price
        opens.append(item.open)
        highs.append(item.high)
        lows.append(item.low)
        volumes.append(float(item.volume))
    closes = numpy.asarray(closes) # convert list to array
    opens = numpy.asarray(opens)
    highs = numpy.asarray(highs)
    lows = numpy.asarray(lows)
    volumes = numpy.asarray(volumes)
    if with_label:
        moving_features, moving_labels = chart_feature.moving_extract(window=window, open_prices=opens, close_prices=closes, high_prices=highs, low_prices=lows, volumes=volumes, with_label=with_label, flatten=flatten)
		# extract_feature: func outside ChartFeature class
		# dive into ChartFeature.moving_extract():
		# then into ChartFeature.extract(): extract one feature group at a time
		# ChartFeature.feature is a list has 61 indicators in total
        return moving_features, moving_labels
    else:
        moving_features = chart_feature.moving_extract(window=window, open_prices=opens, close_prices=closes,
                                                       high_prices=highs, low_prices=lows, volumes=volumes,
                                                       with_label=with_label, flatten=flatten)
        return moving_features
