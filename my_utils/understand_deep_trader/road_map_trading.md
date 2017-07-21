# a road map to deep trading

## what I learnt so far
- create theory and intuition on deep learning help innovation
- sigmoid as last output layer produces a model with only buy and sell two options
- design trade frequency reduction algorithms based on real trade procedures

## how to start trade with deep learning constructively?
- what more do I need to do before trading?
- what do I need to record and learn from everyday's trading?

## what the model can do for trading?
**目前模型能为交易提供什么**
- plotting 走势图
![](https://lh3.googleusercontent.com/R0qo5CgK4jUNmcKr0bL17-QXk9ZvLfnqmGjk4HyI6BZq30a9mZ4Qbix8N3RdZe_1fT0OOUdC1NKWGFZ8Oh5cch2k250NM7S-W5HFfDZrwW4eGaPr-jbXV3uUOJEA04jNSaFOQK53XBNykclbygrecQ7nAsj-f6ab_Ur-qkTmhIC-6l6N1VKb8wB_4tdFdIA01N9db-cg3L2H3vD7-jMEzwdsSSLyHNZuPCSmY3Kvzsxzib-DCILd71TF-vxjRKi80UONE6FPTjIOQ96XcXmpKLSkHIv_y7akXw3rSd5fKwuCyaQLyWoz-zLBo7uhMcsNA54DdDHLid9xAm7KMix_dqkuL_XzcyWIh5PzrIo6nXhhm3owwOpXJ5jdk-Bu8OTr05KoB9YPTLHjEgVc2flJwFplxwBk5JMcZ3xzNYfjfw6PwRxEitfqAd9gLyk-6AYCLzNog2cMy-h-vb-l_EmVxfstKnUR-rimHtdCUPi33ty8eJ6ewUNOCIxpeDaiYiaUKzYdQvrYVe4pcBhRYJBkOuiDrdyu1QvauWrsnORIWwXk1KD87881wozUP8ZeibhLQgrlkfp4J8ab-VQdIIP89DJcgXFegOTKl6pp4d56GX7XOuLszy1A42qg=w2238-h1136-no)
- prediction output 生成预测值
```
final date: 2017-07-20 # 最近日期
final_return: 1.2657575113 # 累积收益
Now, after today's trading, before tomorrow morning ....
today's close price: 3.801 # 当天收盘价
tomorrow's prediction:  0.999898433685  # 预测明早的市值占比
estimate how many shares to hold tomorrow: 600292.232948 # 预测明早持股数
estimate how many shares to trade tomorrow: -0.785707888892 # 预测明早买卖数量
tomorrow to trade or not:  no trade # 预测明早是否交易
```
- 在开始用模型交易前，必须做的事项还有哪些呢？
- 开始真实模型交易时，需要记录哪些数据来帮助改进？
