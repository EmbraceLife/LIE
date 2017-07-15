## re-organise my code
- keywords
- one purpose one file (stock is done)
- deep learning book (simple version)
- how far can I go without reading math equations?
- build models from simple to complex

## focus on
- all examples made with keras
- especially all advanced models or paper implementations with keras
- learn to read papers

Deep Trader 修改版

模型修改内容
1. 取消了模型最后两次对预测值的处理（BatchRenormalization处理，relu_limited处理），让sigmoid处理来代替
2. 损失函数risk_estimation去除了-0.0002的交易成本（根据前面的模型版本，在使用了-0.0002情况下，但交易频率并不低；去除0.0002后，交易频率似乎也没有明显增长；再训练中先忽略交易成本，计算收益曲线时再算进去）

sigmoid处理预测值的分布，会因原始预测值区间大小不同，而改变
1. 虽然预测值处理前的分布接近于正态分布，但sigmoid处理后分布特征也会发生改变，见下图（用tensorboard图）
2. 究其原因，是因为sigmoid函数自身特点所导致的，见下图
	- 如果预测值的正态分布区间比较小，比如（-1.5， 1.5）， 那么sigmoid处理后的预测值区间同样是正态分布， 见（bell1, sig1)
	- 如果预测值的正态分布区间比较大，比如（-10， 10）， 那么sigmoid处理后的预测值区间就不再是正态分布了，而是相反形态, 见（bell2, sig2)

训练损失和验证损失
- 训练损失通常可达到-0.20左右，1000轮训练时
- 如果验证数据周期只有100天，验证损失会在-0.04到-0.05之间
- 如果验证数据周期有700天，验证损失会在-0.14到-0.16之间
