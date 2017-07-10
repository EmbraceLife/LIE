# deep_trader_guide? 如何使用深度交易者代码？

所有代码在https://github.com/EmbraceLife/LIE/tree/master/my_utils

## turn csv into features and targets 将CSV数据转化为特征序列和目标序列
```
python -m pdb /Users/Natsume/Downloads/LIE/my_utils/prep_data_03_stock_03_multiple_csv_2_features_targets_arrays_saved.py
```
- set up features, see file below 特征序列和目标序列设定，查看一下文档
```
python /Users/Natsume/Downloads/LIE/my_utils/prep_data_03_stock_02_OHLCV_arrays_2_features_targets_arrays.py # 在MA项目中查看，OHLC作为特征值是如何设置的
```

## checkout features and targets 查看生成的特征序列和目标序列
```
python -m pdb /Users/Natsume/Downloads/LIE/my_utils/prep_data_03_stock_04_load_train_valid_test_features_targets_arrays_from_files_for_train.py
```

## setup parameters and instantiate a model before training 调设参数，构建模型
```
python -m pdb /Users/Natsume/Downloads/LIE/my_utils/build_model_03_stock_02_build_compile_model_with_WindPuller_init.py
```
- set up input_shape based on number of features 例如调设输入值（特征值）的维度

```python
input_shape = (30, 24) # (30, 61) original or (30, 24) as custom need
```

## training the model

```
python -m pdb /Users/Natsume/Downloads/LIE/my_utils/build_model_03_stock_03_train_evaluate_save_best_model_in_training.py
```

## plot return curve 绘制收益曲线，换手率曲线
```
python -m pdb /Users/Natsume/Downloads/LIE/my_utils/viz_03_stock_02_ETF_predict_return_plots.py
```

## look inside the model structure
```
python -m pdb build_model_03_stock_02_build_compile_model_with_WindPuller_init.py
```
