"""
Uses: run this file
1. convert an index csv file to an array of preds_target
"""


from predict_model_03_stock_01_predict_stock_best_model_save_result import get_stock_preds_target

index_path = "/Users/Natsume/Downloads/data_for_all/stocks/indices_predict/index000001.csv"
#
index_preds_target = get_stock_preds_target(index_path)
