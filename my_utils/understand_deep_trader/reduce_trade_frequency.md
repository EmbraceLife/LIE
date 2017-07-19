# 如何降低交易频率-尝试3

## 不变的
- 沿用模型：输出一个预测值，只做一次sigmoid处理
```
 _________________________________________________________________
Layer (type)                 Output Shape              Param #
 =================================================================
dropout_1 (Dropout)          (None, 30, 61)            0
 _________________________________________________________________
lstm_1 (LSTM)                (None, 16)                4992
 _________________________________________________________________
dense_1 (Dense)              (None, 1)                 17
 _________________________________________________________________
activation_2 (sigmoid)       (None, 1)                 0
 =================================================================
Total params: 5,009
Trainable params: 5,009
Non-trainable params: 0
 _________________________________________________________________
Train on 84332 samples, validate on 17500 samples
```
- 损失函数不变：
```python
def risk_estimation(y_true, y_pred):
    return -100. * K.mean((y_true - 0.0002) * y_pred)
```
- 特征值：（?, 30, 61）
- 目标值： (?, 1)

## 模型训练状况
- 只训练了300 epoch (次)，最好能训练1000次
- 从损失值上判断效果和之前的模型（训练1000次，验证损失最佳值在-0.15左右）没什么太大差距， 见下图
![](https://lh3.googleusercontent.com/CQw-U9vireeGK4-7ljRVw8mv8tIdhiMQCP-Lts4J2siMCCMRXowy4125Zpse5egQi1HaJvj3xOKVndvN1rMLb2zkZ-d9bd6yuGgaCcPDfYzW9hxt6HPEBotAskeONHlGidK151KxhumI5OOU4YJdWaaDiDjquu-bZ5M8LHCMrt19aDD1Hf1k2Fc283xnrGB1TqlITYkIJO4YnArV1eqW0evD2BmYy-hbtAlGymtDHfIeBLGExs4fnx7o1v0V9_v2LjO9V_4Vi8bkmzzv-CtqP9ReL-FhmtfzKIG3icaXBKbbL2gVGDy-r2mmrwQTvtx5ylAytgd7JxIIdB66bf-m7K4yIqIgPvQ7TAGTOEErAEtawCiiD5XkOPb_qlrQa3W7nbOGjxqal3PKOTMiSHGX0PWN83gjt5cO0wXO7QdDRpnk5i_9Tbbvf3upKCatz0On0ROboJTuZj8pa1dVY6sxL3RcMjiDDMu-Conehob52nePQi80mGSCD1MTKe-x3iKyvXlMGDHM5GPxtGAJJrir60lhsMwcYAPxFZ8DzcRdlbJjqqa_8g15OFf-2gLVmvTrFsXGLPCQlYt_a-oqRb7Cfx6LZRzgYtf2NveX1PKGGFeh5HuZmhlyvXBn=w686-h1136-no)
- 模型预测值在被处理前的分布，见下图,(下分图为处理前的分布，上分图为处理后的分布)
![](https://lh3.googleusercontent.com/yjmnE6AlxIsuQrf6AYepbjXIZ0B5v54lhsm1yHAvlVnACRBWoeYJv3GxfeVq61gy66bBLl9affVqBVLU-rAZpEI2vkvhrlOXWipclXWomQx_o_Wys4A5kBA7PfDXSib1o8FFHxtPX5Cvtl4_gUKOTSFsgR-t_l8JGb_X7yjUaTVhELeC98ifkPIdzJN6UxodRkDzCACsXrBjcS2BoMMhQItEkzdIRA9e2qXzwKyRvyikrPOWf1dAZyGPQaMUasZsnU7oPIbSmH5gRgYIykXWeJeCn7cVCuKG8NxccT7eNtMghkhNjSgrpyf_yb69yI-PrFOozJhZbXuXRE5hAOWsK_r2lfNaqbX2ZDe1N1_AhmaarOqj9RPXwQHpXCZEM6rXu6KZ8kjQLbqhCVs9ZYdNn0CP628JwUFn6shramNNdGcm2R4Mqia-VKs4RnxLGtugm1LIbdAGvh3m6uZUJGjsKHK4hy-2yB9-3RvayeVhlcWvmGCN6x-j5GN4W4eSQRYA5f_ObOX4FXTD4IXpkCYML0NlErftM7QEvbAKff7c5Gy9X80nZgfxM8feh7V3_m_8nqBHTreqnAXRs6wLtDksDlrA4gDMsCuTS2gIkRpqwoK_lrqdh9pQxLDI=w612-h1136-no)
- 总而言之，模型本身结构上没有变化，所以也不可能有实质性的改进
- 看看在当前预测值内涵为市值占比的这个概念下，基于最新的（基于MonteCarlo建议)成本和收益计算方法，我们看看模型效果，（上图为无交易成本，下图为有交易成本）
![](https://lh3.googleusercontent.com/HSztx40HJ1Dy_GMn79dGu9-Qura4hcUl76lhvSXFFchoMoC4EYPvXujdLxD2_bDg2RwrEIbpnI8oVGgdC8UW6S9bam081hf9ieChuc9F4sbYbqYih6zlpQS3ynJqPuIJv8Kse4uKW0RZjMeFhfVfuEtjqJLYhXafeNpuD1-ALQbKGCCg8tkwETdvKWyfb5g9j7uPNe4NXKDJCXt9PCJ2QfciKwIG6R4w2M9vVuqG_xpwWqzGvjQ0U5UX4en_VrorrbKm1kK9JDzJ7UMlAY9i5bVvnCLtZirakLADT3lel6PufwjN1FsZtXWmQeXT2d-EiYxvHVQVlUOrV055uf9zrOFTks0fgfiP1OPQiNZyf8sLyDsrlCSOi0yZPgUndscUT7CLuFC79_LeyBTGCnvtJcqLho-yRAcxjZ-tnoiCQeXN4fL0Phoxcj9n88ki_lgZlcWoqyIDiUz9Kt7buG35wJ6km8pwL5xKcz4ejPGR0PHT3JRtQwHzeGeTnCtULRauKIIHxEia9C9O7IgVkRlph1Ln4eEcJwItMv-64LNpudxwFlbmZpCzdnVRSzTCmyGB_jXb-ZWEdM5fnMaTEmPp_gz2XNjFxUDHmuiLz5Z9HSj3R_T49Mq3QxPA=w2238-h1136-no)
![](https://lh3.googleusercontent.com/kgapNkoSAG2M4Jhj4f4lovvlTR68xFGO4f9x1a3haHwtrhL0uhq1aAkOsuxs2L7jwIcfSmfndg15OV48bLBteeTga29PZAdNf6iY7UNYK6tC2p8YWuyfKhWbPTofyZ41utLmd2BTSnDPORX45iVayfyizniRUQkWPILSYIbtnfFoi98DgVWBnLMOy_3WPckCOrZ55jvZhbOC46ZVH7mW7Xt-w0-ag2EYJfhWYskJLVayD-szulkzlUlQle2lg0aq8osHSso47daQKE_klS9gE2-PY2uDFcWJ46JJZ0oOIEDgmmjCxWFG7aeNtWE2hp442KTVYJyc7VtLSY6HDm6kg1G2gSrIFt8EOA_0NQHqjV4H-M3oNV52e33N6odDqbcPgrd6_dCdddQe3css6kxT9QCuJgup0Vm4TperaR40QGLNG_NhKrA32AnYA4ddxlYKpje6U9kCPMt7NyEKivg1J2d2YOYJR3w2E2VzkylRP7bSfgmTq92bhXhTLKBqOF22iQMrhdGyj5w735l-K7vIgK6S6b-54DD9v6TBtiqBdFk2yu3CEUP9RB9uZPxmF8tWhOYm11xNO5-UP8VjUgPtPgs0YPH3e_83taKpYDyWuYPHzMjOdRN8RK7G=w2238-h1136-no)

## 改变的
- 原有预测值y_pred内涵的设定：
	- risk_estimation的设计者，将预测值（或y_pred)当作持仓市值在总资产中的占比
	- 但risk_estimation本身并没有限制y_pred的内涵，我们有对其自由但合理定义的权利
- 为什么y_pred作为持仓市值在总资产中的占比，不利于降频？
	- 降频的一个基本思路：当紧邻两天的预测值，没有变化，或者变化小于10%，直觉上和逻辑上我们应该要维持仓位不变，不买也不卖，从而降低交易频率；
	- 但市值占比，这个概念细想一下会发现，只要预测值不是0，不是1，那么哪怕当紧邻两天预测值相同，也会要求会产生交易；因为，相同的市值占比，只要昨天的价格变化率不是0，那么今天开盘前的实际市值占比已经变化了，开盘后今天的预测值要求与昨天的预测值相等的话，显然就不得不交易了
	- 这就导致了更高频率的交易
	- 那么有么有更稳定的预测值定义呢？是否能找一个新预测值定义，来实现一个逻辑： 当昨天预测值 == 今天预测值，那么就不存在交易的可能性，持仓不变
- 预测值新内涵设定：简言之，用股数比率代替市值比率
	- 我们需要的变量：
		- 初始资产：1
		- 初始资产对应的总股数：1
	- 预测值新内涵： 当天实际持股数在总股数的占比，区间（0，1）
	- 当昨天和今天的预测值都是0.8时，逻辑上说昨天我们需要有0.8股持仓，今天我们也需要0.8股持仓，昨天和今天无变化，所以不需要交易
	- OK， 这个概念解决了上面的问题
- 新的内涵定义，带来新的累积市值计算公式，产生新的换手率数值，看看在不（设阀值）降频的情况下，频率是多少？
- 具体如何下降交易频率？（设阀值）
	- 如果昨天今天的预测值之差的波动在正负10%以内，那么持仓保持不变
	- 看看收益率和换手率的变化情况
