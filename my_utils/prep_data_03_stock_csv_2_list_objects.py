# source from https://github.com/happynoom/DeepTrade_keras
"""
### workflow

1. we got stock.csv
2. open csv, each line is a day's stock info (date, OHLCV), split it to a list
3. convert the list of strings above to an object contains attributes of date, OHLCV with proper dtypes
4. store all objects above into a list, sort the order by date
5. return the list of objects
"""


"""
class RawData is to create object to contain a day's OHLCV and date info
1. user provide date, OHLCV inputs
2. RawData.__init__ return an object contains date, OHLCV as attributes
"""
class RawData(object):
    def __init__(self, date, open, high, close, low, volume):
        self.date = date
        self.open = open
        self.high = high
        self.close = close
        self.low = low
        self.volume = volume

"""
read_sample_data(path):
1. open csv or txt file
2. read the file one line at a time
3. split each line into a list of 6 items: date, OHLCV
4. give proper dtype to date and OHLCV, make them an object of RawData
5. store everyday object into list raw_data, and sort the list order by date from past to current
6. return the list of objects
"""
def read_sample_data(path):
    print("reading histories...")
    raw_data = []
    separator = "\t"
    with open(path, "r") as fp: # open csv or txt file
        for line in fp: # read the file one line at a time
            if line.startswith("date"):  # ignore label or title line
                continue
            l = line[:-1] # leave the last string item
            fields = l.split(separator) # split into 6 items date, OHLCV
            if len(fields) > 5: # make sure we have date, OHLCV
                raw_data.append(RawData(fields[0], float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]))) # 1. use each line's date and OHLCV to create an object of RawData; 2. store everyday object into list raw_data
    sorted_data = sorted(raw_data, key=lambda x: x.date) # sort order of the list by date, from past to current
    print("got %s records." % len(sorted_data))
    return sorted_data
