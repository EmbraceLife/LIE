# Why y_pred[idx], open_prices[idx], closes[idx] are on the same day

y_pred_open_close_on_same_day

## model vs features and target
- features covers day0 to day29
- target covers day29 and day30
- prediction is on day30
- we use new data on day30 to get prediction on day30
- therefore, y_pred, close, open_prices are used on the same day 
