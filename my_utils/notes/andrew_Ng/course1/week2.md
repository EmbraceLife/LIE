# Logistic Regression as a Neural Network

## Binary Classification

神经网络的两个小特点
- no for loop on dataset
	- 一次性计算所有数据，而非一个就一个地让样本通过模型
- forward and backward passes
	- 模型训练包含有两个阶段：正向传递和反向传递

用logistic regression来解说神经网络
- logistic regression的用途
	- binary classification 二元分类
- logistic regression 对猫的识别
	- ![][image1]
	- Input: 输入值是图片，一张图片的 tensor shape (64,64,3)
	- Input: 将图片tensor转化为tensor shape（64*64*3, m), m 样本总数
	- Target: 目标值，是猫1.0，不是猫0.0, tensor shape (1, m)
	- ![pay attention to notations][image2]
----

## Logistic Regression
- 详细解说logistic regression 的数学表达
	- ![][image3]
	- Input, Output的范围表达
	- Ouput的内涵以及数学表达
	- Sigmoid的内涵和数学表达
	- `w` has $n_x$ dimensional vectorand `b`(a real number) 是分开表达，但有些地方将`b`作为`w`的第一个值，见图右上红色字体内容

---

## Logistic Regression Cost Function
- ![][image4]
- logistic regression 的数学表达
- logistic regression 损失函数
	- 损失函数是建立在 $\hat{y}$ 和 $y$ 的数学关系，寻求损失值最小化
	- 最直观建议的数学关系可以是 $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$
	- 问题在于，上述关系式，右侧中间波浪曲线，意味着会有多个最小值
- 针对binary classification 的损失函数
	- $L(\hat{y}, y) = -[y*log(\hat{y}) + (1-y)log(1-\hat{y})]$
	- 当`y==1`时，$L(\hat{y}, y) = -log(\hat{y})$
	- 当`y==0`时，$L(\hat{y}, y) = -log(1-\hat{y})$
	- 这两个函数图可以找到最小值？？？
	- ![][image5]
- loss function vs cost Function
	- loss function: 一个样本的损失值
	- cost function: 所有样本的平均损失值
		- $J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}, y)$
	- 在不断最小化损失值的尝试中，`w`和`b`不断得到更新

---

## Gradient Descent
![][image6]
- 公式1: $\hat{y} = \sigma(w^Tx + b)$
- 公式2: $\sigma(z) = \frac{1}{1+e^{-z}}$
- 公式3_a: $L(\hat{y}, y) = -[y*log(\hat{y}) + (1-y)log(1-\hat{y})]$
- 公式3_b: $L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2$
- 公式4: $J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}, y)$

#### 碗型函数和波浪碗形函数是怎么得到的？
- 碗形函数的由来：
	- 逻辑上应该是将`w`, `b`,代入公式1，2，3_a,4 可以得到一个基于 $f(w,b) = w^2 + b^2$ 原型的函数，<span style="color:pink">(能得到吗？我目前不知道如何做这个代入过程的演算）</span>
	- 如果演算成立，那么就可以生成碗形图，如下图
	- ![][image8]
	- 碗型函数，决定了convex或者是碗形特征，会有一个最小值
- 波浪形函数的由来：
	- 如果 将`w`, `b`,通过代入公式1，2，3_b, 4 可以得到一个基于 $f(w,b) = sin(w)cos(b)$ 为原型的函数，<span style="color:pink">(能得到吗？我目前不知道如何做这个代入过程的演算）</span>
	- 如果演算成立，这样的函数会呈现波浪型，没有唯一最小值， 如下图
	- ![][image9]
- 参数`w`和`b`是如何更新的
	- ![][image7]
	- `w` and `b` and `J(w,b)`构建了碗形函数，单一一个变量`w`or `b` and `J(w)` or `J(b)`得到如上图
	- 如何利用当前的 $J(w)^{i}$ 和 $w^{i}$ 来判断 $w^{i}$ 应该的更新方向和量，得到 $w^{i+1}$，从而让下一个损失值 $J(w)^{i+1}$ 更逼近最小值
	- $w^{i}$ 应该的更新方向和量，由 $-\alpha\frac{dJ(w)}{dw}$ 决定
	- 当 $J(w)$ 变成 $J(w,b)$ 时，只需要换个符号就行，$d$ as derivative 变成 $\partial$ as partial derivative
	- `w`更新的公式: $w := w -\alpha\frac{\partial J(w,b)}{\partial w} = w - \alpha*{dw}$
	- `b`更新的公式: $b := b -\alpha\frac{\partial J(w,b)}{\partial b} = b - \alpha*{db}$

---

## Derivatives
- goals
	- 不需要懂太多细节，只需要构建直觉就好了
- 什么是derivative?
	- ![][image10]
	- 函数图形在一个瞬间的变化：量和方向
	- `slope` at a particular `a` value
- 简单：不变的derivative 或 slope
 	- $y = 3a$
	- 上图中的函数的`slope`在任何点上或任何一个瞬间变化上（不论`a`向右或向左变化），都是相等的方向和量
	- `slope` 总是 3， $\frac{dy}{da} = 3 = \frac{\Delta{y}}{\Delta{a}}$
	- $\Delta$ 要求是极致小的变化量，所以称之为瞬间的变化

----

## More Derivative Examples
- goals： 更复杂的derivative 状态
	- ![][image11]
	- 复杂的情况 $y = a^2$, $\frac{dy}{da} = 2a$
	- `slope`在不同的瞬间的变化，或量和方向，可以是不同的
	- 当 $a = 2, \frac{dy}{da} = 4$
	- 当 $a = 5, \frac{dy}{da} = 10$
	- derivative的计算，可以借助书和软件计算，推演过程不重要
	- 更多复杂的函数的derivative 的值
		- ![][image12]

----

## Computation Graph
- goal: show forward pass when optimizing the final output
	- ![][image13]
	- given $J(a,b,c) = 3(a + bc)$ , 拆分画出每一个计算步骤，构成computation graph, 来一步一步求`J` or `loss` or `cost`
	- later, 可以用computation graph画出backward pass来一步一步更新`a,b,c`，实现`cost`不断最小化的目的

----

## Derivatives with computation graph
- goal: show backward pass
	- ![][image14]
- meaning of derivative
	- $\frac{d(FinalOutput)}{dvar} = dvar =$ 描述当`var`变化一个最小单位时，`FinalOutput`如何变化
	- 上图计算了 $\frac{d(FinalOutput)}{da} = \frac{dJ}{dv}\frac{dv}{da}$
- 当`var`，`FinalOutput`相隔较远，用`chain_rule`来计算他们之间的`derivative`
	- ![][image15]
	- 计算 $\frac{d(FinalOutput)}{db} = \frac{dJ}{dv}\frac{dv}{du}\frac{du}{db}$
	- 计算 $\frac{d(FinalOutput)}{dc} = \frac{dJ}{dv}\frac{dv}{du}\frac{du}{dc}$

----

## Logistic Regression Gradient Descent
- goal: 演示`derivative` 如何在最小化损失值的过程中更新`w_1, w_2, b`中发挥作用
	- ![][image16]
- 通过`backward pass`, 推演出$dw_1, dw_2, db$
	- ![][image17]
	- $dw_1 = \frac{dL(a,y)}{dw_1} = \frac{dL(a,y)}{da}\frac{da}{dz}x_1$
	- $dw_2 = \frac{dL(a,y)}{dw_2} = \frac{dL(a,y)}{da}\frac{da}{dz}x_2$
	- $db = \frac{dL(a,y)}{db} = \frac{dL(a,y)}{da}\frac{da}{dz}$
	- $dw_1, dw_2, db$ 是每个参数相对损失值的变化量和方向
- $dw_1, dw_2, db$ 如何帮助更新`w_1, w_2, b`来实现损失值的最小化
	- $\alpha$ 是学习速度，或参数自我更新的步伐的大小
	- $w_1 := w_1 - \alpha dw_1$
	- $w_2 := w_2 - \alpha dw_2$
	- $b := b - \alpha db$
- 以上，是面对一个样本数据，得到的一个损失值，和所得的一次参数更新
---

## Gradient descent on m examples
- goals: m examples, m losses, update m times
	- ![][image18]
	- 上一节，we get 单次损失值和单次更新，单次`derivative`, 即 $dw_1^i = \frac{dL(a^i,y^i)}{dw_1^i}$
	- 这一节，we get 一次性计算多个样本的损失值，更新和`derivatives`
		- $dw_1 = \frac{dJ(w,b)}{dw_1} = \frac{1}{m}\sum_{i=1}^m\frac{dL(a^i,y^i)}{dw_1^i}$
- 计算 $J_{avg}, dw_1, dw_2, db$ 均值，而非单个样本下的值
	- 需要使用2个loop, 和计算平均值, pay attention to the loop coding on the left
	- loop所有样本， loop所有参数`w_1, w_2, ..., w_n, b`
- 使用loop的弊端
	- 计算效率低
	- 用vectorization 代替loops
	- ![][image19]

---

## Vectorization
- vectorization remove loops and much faster
- `np.dot(w,x)+b` 代替 上面的2个loops
	- ![][images20]


[image1]: https://lh3.googleusercontent.com/KqzCHNE4GoH-8Mgqdh7Y6PQkkR0xDcLyFvZbMTHX8cSDTmHB-0efMYrQe2njCjvGaP86tyZ8s2q3XnQ3nsPp9laAt7YgYpCONNkVm__m6mY_fjquRPFbFNn33hyHxu5m_vw1DYhXCWXrVBnjF8Fgdc4f7zOJATLkWnwjOy-2dqrfbc4u20s6L0H5JleMbThY7iZW2QG_PPqDkIgG0qg4F9GdybM-Ku1O_feYYzHFuieCWci4gV4qFjJmuPx4Y9eAq1P7bUx39_ht6BrNIhy02qDDr4vxWSvI7xzoZdR-HVJhkZrqJWidVrwAAIifMITCTLlB-aow1eB6cSSmCrF2227FklG2xUE8Sw2P6CU1qohuFw4hj5IbUVBfmGXPX_2Gmk61CuJYWUx2eFA5iUE3KnhzwPabhl7enmJmQrTR-l6FDa4Kf5g5zw51Ef7MUuKQNf1F_O4ZZmgh-uPgT93AQx0-G8dPp9Bw6uaXbGkw1XfXQ3XuYDN54uD0stA0Jtj_H9hWLDOeunfOBrbeK7XakoGBDQgvs1jGrpfsBlkqHt-LrVIAUnKqMu8mMIqpjRDvWYvqbHVrZ4qsqexRd3lpasfF3F06g1k44pLg1yYNrtU22uWcvulqF3eu=w1900-h1032-no
[image2]: https://lh3.googleusercontent.com/czHGUpQPJZuJYckHZIyBijbxONm0J9UIfwTls8NnYFib8tnD0Dtfz0IHisZ3lGrjUb63PWU1s4DoJhnYFsquI6R5T2ak9wIsI0ALG5xtIwPhY4bYdV20A5W2gAS7gS-3owM99rR2QZ2qzsmIXTr4sldRT-v9cdt6CYQXTkmEo0CecxnuqgWkb-uNP5TJ0JrT5_eEhwRBeHIsB91bNW7g-1tcdB_tCWfBWjrP_exfhJqrp0EluONR7TLaaiFfjhklEQxYSM3MQS9jj9osoULGT48p-sqj9nbuHWBdEZy2XMjjhfxwyL2DjDcJyHyxCw_DVeC_jI9U-9JP_vlgNCO_w4fh9CNfbHFKwmBARehS_DyCDHEJm1najTQVg_woADiVORgZ8W5zP0lnlJKDtvrh35jmctw6CKbBfft_UjqDBb7XWnb_gvIkmHWC_4YU2AKfXb6nF4OJ5-OmKH_gfe6NhQ5AjR9sXI_se7nWKxSNlAdQQGT5bue_-vI5hgm_--Ckqw4jBOY4MVsz4HmJoKuk6xIKpmA1lk6up6YWfRHttCGLtMlC6EezcU8YSPNJCoWXUw2nG9oo6gK1NMQbgCX0TuxHDUCi1WuAAkFUh-d_e_5e8G1gW3GG2C8H=w1760-h978-no
[image3]: https://lh3.googleusercontent.com/U3h9Ymvloex2Y9fXAdSWBjfFjXsLEHFhf189BkYizOryDNsdpLdKRrzLX4ZkWYHBQEoKJvK67cKvEI9MFOnitzYqfTtWOSCWoptYEAI3HgOK6j8HmCVHOw7iciIRrn-1m0HzJTbVb58NFJtYeAnPSo65lXl_PEDZqWf81UpU9B8jyiBQ7vcm0ixTFUE9MF2q7aVJpRn3hnfRJa1b-0-3rmgBH2kfHfrkENa6WuZzIofJp7d50zIOcJfScpDXfowx4IABPRQiax0yK8Ohrhx-Rj4L-9xC3Bk9AsuzlnPu0nBf3S8P-ZKAnzr1XhIDvr7pmzlp0mxUZAnglFTrvlm0dOL_y0pOx0Y3cYBtnwcEHEhD3TAwQPwtPImMKiyE425K3yZ2CAeo7tiGXd5vnfFEufRl-ld831nWoy1Cju0774A_lwcTyH4jcDxV17HCTNNyDfB_NrCiYfNy0spDOUb8VxYVEg2w2nSRKwVpWV8ncy0YHKrBwnJnbTbRL01w6Tpm8KXTeYc6Xxt_cIYSK1THz2WwPx1xBZJzwpn_oZHJghVuf7hnnBZNspARCV00Cpt_N-kelwiPPHvtJznT95ouAhwxXx6QxXxM7UyksCuB0hODF6WNWv29ThoK=w2094-h1224-no
[image4]: https://lh3.googleusercontent.com/DrBLbnF0qSP4XzAd2AMPpZ3eyiNVTNAj0SipdzeR37uy3VDJChbu4Po1nT66_5aDjQpUD0PRVLVmHPL7TIXtD32VE_vwmv4rI2MmSEuTk-upbhnvGERNHZ8oxSMk4wvkYZrJ61TKcGSPup45xemRDQoLSWFH13aBAwdL0ZgkXNa-t0AU7Sr2kNEpcjeYnJ7SvaZCWJmTtY26oapUs1YFbJ4dtb3PU5qcn-PezH6lD-bROu71GLBw_r7e_9mp4pKYBrqYejG99uwAhe-5SOVWV9IhCY2HsTPY_hvoBYSiYAjBEa-2Nd9Q1yeKnEa8hKhgfBDNrmK_oGy2Yylrco0XhFvvBpW2t5xdGWWb87GGSpNuf81sQutFpVkzXk7YvGGYg9h0atKnIv9wA9RZOXfQQlj28p4XCtQ7wMXOnyJLb8AkmWsCs9BmZjj_lTB8fwdNg9uHoQsil17Dajmc0tbFycVHGVCcrRDmmSwVNwbs7O-zae5DIgbDHIuXNHyQCk-6wWcM5vOyHnQxbbGDGFWBl4BrpBNAfyYAx64R1Zj3kY1RwbXrWeLHgIvL-7xDNy4jQXlD3S7mt5ZB3u-xPyfn_GZK5UHvmPZBXIlHMPASlYWmI2DCMBfAL5Yw=w1918-h996-no
[image5]: https://lh3.googleusercontent.com/o7FZdccz5FzTEb_W5kw6oLnq6x-nOz61paXKEqGoZ36r_eCaLIQRjY8o51okupX9vtY8AE0TXGZSmAfN6u3gin35vbEQV-8Ij28ih9mfmvALP6d8mOCF-nlFfaBwx4nuAtmzU_Y3zAeTa5qyrZD7W-PwaBlm9KNou_aLnIyTcJOqySkVP1k41BG97-nQlucHrMDMqeJ21vsRqKb8lmtSOheJOgIND_AN6L1c-nlnaQMUjq16sUnu29MBuqBkFQoOlD75XAU6IofURaVjxgUqN-A3rXuQQEZFS1rI8ISucaA8KmBq0hMeEjZtZj65dZPvmSRYcWVK_zbzxM4r1qBIRYTAFWfRgHKIrupQ0WMAsvWnyVjHNB_EJDDTy831N-PzwIHPJCgB4velciTKdXPiwLYaGBSX95gh7O384OjwU9DRMMPi32KKSa-ebiN36fQps5VSM5mFRiAvvFZJgAGg67iHs0cPm5p8VtZknMeE8WUEfBDRcFJnKv8xPIKk3QvlwVA1HVdyq9fKDfFh_inDZMBinE_2ySQVqIupjfIHR21yZzZVFZ8gmhaoIqupxqSmB06Z7fWz-9rEFwhEiUOiD_0vsZzuFKr28Wq43XQhlb6zQloD3UJj_Oal=w2090-h542-no
[image6]: https://lh3.googleusercontent.com/ZBqjR9tg54EdEw1iF98ujFqsYDy93_H4fc3AhXUSO0E5a8q9ZIXN-exjtkcQXg-HLmY_NxloCC90OMPrq_bsNWj2O11NpxyWOta7GEl9myWZ3vvDC8xvh2TZz73WHfM6WX9W_oVM5R0B7NaO0PWH2mCWTxPcOHHiFxxcy1lv9SXr8rbpc_frK6pXl7ZL7Qp6zeiubUYY2rryBKpFeFwxc-9AC_ULl5t52bt7gGpZcc2vZ0-kKt5ooaqmX6kQG8r2c9nB-i87_xZh1H5cT2pcuFG9DN5dlO8aoNmGFFLPL1A3dflQDbD-nzgBvIGiHnlM7G2ZM-h7_ICpVt2IwPui9pCwlsNqcDwHnaLDZ3gCyf4_2aPONaR4GIDCtSNWGw5VhMgDO7vL_D3XhsBWYlVS2E_qBXlWkGrOJ7XfPOmxiuDwWTRmYrOqPBe0_EEh0G27pO_e95nr7FMAt0AMXq6lK4fMaORARrAp_QXVRDsVoxovA7Cz0o0PvCdjPKdeygY5etqDwf26_6vsROwZ5Ex_vRjOCaB0HBRL4xqs5sa2yJU3GAA3J1y_YI5U_ffXhKEKSucOWmj8m_xKI5afsZOjP0GKRVOy36nmp59H48eA3qgMzsfeWM-QUURZ=w1914-h958-no
[image7]: https://lh3.googleusercontent.com/y_J_yslGvUEhnDzf0RtQ4zFXrlWG427hUzNL5F1b2WCASmkD1I0NhYXKaDwgt4q2lGT3s0l4IRetZiS4sGCP-layErG7Cldf1ZVe54yjbh_r0Stv58g1ndnq1N6bHpVyeqbCReR3BqX7Y0skmdpxhroB-mP1iNxLQ7zroG7lMDE0DR76KjE95FxokVFqF4lMaNkj0vpSJ974rhjRY7h2M47vdmcL4b4AmK-ygwjr6zXiQUaVRjO_en8GL3AXQtuOr4Sj6iRGqGiYl8uD7KI--t5pqCiudfqnCTzcoXkGIRV1CXkz-8pHEzMfvFCT5BUOodECOEYlRIBWoem9Jxg6Xsh9O08jS6euISCBNS-5jiujN3yWeYscPS-YfZmb_nYidqkTAP8IvIe5-6mAT83FkryJV4o47nnDlrTCb9r91yjqBSsAqO6UhMMsJzTLIUwRZncN9r771dQ2_heASPPQeJzfnC_poENDK3e-mpJ3DV56DSgdMStZOINPM-MnkTCwc7rdNlgoeRQ1BR5P0JH4-1deQtdFG0FjuUpyMeCIyINkvIjpvBn4aXvZuhrasVZOfkYSjkQ_NInBUHzzUTwdEiJYLBAketVYYFx5h-mWy7nmEeyKcN0AYy3a=w584-h322-no
[image8]: https://lh3.googleusercontent.com/RZEdyoheelD-a0NgUbb3ZcAHJqdTz_KXa8aGskYHXjLCKSF8wuM2Jbr6mreNStM3h72pxrtMagNlvGp1leTWD8zquLU_5HiP6lUcLc78RlpR5cgTLi2qIVZ8OAp69PiLdUaY5ng62XsHY9XxjWRMSCrsyWzhY3FMouCI6OVT3dmdwR0bG5EZJqdE-cJq2CnKqnO1C0DfA_MaV_4sfh_edVIEgx8BSPbHqZv7mNssh6MFE7qQ8IwaClrytFzklVM5tT9GZ7fZysWwhoLNoGb0COVwZu7pAZWXH5TKTqfCEIWeajZdQzySNIIZYDz_3zk5Sd9TQGNhj6B3UqAEQKAP5mwbeOaP7fh2Myf8yZFWDVLcqwla7jKoXhdJcZF9MiGuU8L9ry0X3Vq1pfe0Xa8hNFA3jYLq_usqinBXArWZTtsSt6-oPDlwLLWIwlQWKgc7q9haB3sBkW7MesyQPo3v4_QPhGMzLKhei505VwMioEXibA7RjrtB60rn3oGNsOXBJE_SgOlF_vsoQ-vRis7a-JMPBK--rd9CAF_OI6Y5yrayRB5uSXvOEktRLFjhGsxexWao9S8zcbi6hqESD6gesztK70xfrER_NzFIkOwM3GG9O81j9k08=w1094-h946-no
[image9]: https://lh3.googleusercontent.com/atytBnhFGoh80MrvLtXVpJ3QOQlEgrLS03nM7cOGR3aKAi7a4-77xqaAp0WNA_0MjsqSw-NtsKQyvH735rLELyHwUttCYIGxY_RonThTM7nLddkRfjC_7N_PCShgNc9yv1UWTHHtG06moqfwQyuL8zg37f9SJy7NeK3aziGU-mzEy7qIpB2jGHW_4hMdZXuNcbIEZJfH3b3eCT_OYGYrdVEX6xsDvnOLQ2TpBR1Qd-iMp1EP15Ew1ToVBbSTlm9II9p7yCzWgk-K-b3MG6oWq1XIV75NOt6nK74Sh5_36tzNwipHIc3JMtxriedzYq9_1tgI9q9CMyYcuuDoe1FubFO1eGpHsxaaYGgrJSs_cKlK3lldKTmTX24Hrp2jj0EK9WDiMMgkRdWa0kbqmjJuLUNpEPx9aZZ_GrXNX_5jn4tu_sB3ZPGyPoNLqf5F9s9GH0qyO6-Uf0TpBAtzSvv1m_6nO97uNKHSs5jrks37wMjDbiEhlSsJDL3mbmmKyHd10YFEf-1hKyCRSro-o1IhWF0L3WBP4sDVkWpaXreWbqyuVrfJoc2z8E-awzB1nowL8m3OHub589PZU6Zga-4qxkAuUzAAdYfquH3rEs3ohbiR8qytSsqm=w1056-h842-no
[image10]: https://lh3.googleusercontent.com/bUgjlsjYY_eOrxlXmc4jgNDiG70b6h2Uw3etKkeZTiRenLPcDEwfcnAx6EbrrgNOlL9g6bmO2TyPP50o-OB1Jq3S5dWU4wxHehyjNol_ydIK9NpaWY3eMhwqJ09smmSGME_2oEU_gTF_x4g48SNI-fwpn8Ba35f0Sq9uVKwBqlBL5baPfRp-oltV_7ae9dDTKv1YXqQreSbCVI_Vkg7XZKUSRFDE_u9Hl8M25CTe3Dimay1BbjmL8nH4mky8HYxcq8vHtd6s05ybQDL_tQ97xWmc-XwGVj6xx2nU9khY571qB-Bm8lMqXx_Pv_Z9R3sMqpytX-N-gdd1C7UbMRa2O-0VttI-1_z7AlhoTlt2I0WUjogGDC8hIy0vqlDE6w5aTVNqXv7ezS_qqs3ZedCAP-2utbqUE_kSjdZdO6PPtiH3dTzvNUER0_Vci0xLcxDMYsJX24TfKOKGC7s2P9R866lYA1AXY2AfYdMQ_CHcF0YwiuP-9lcHRo-F-cqTLYuf9dqz4bdMB0jrJQuddZUEdsVmYvsTmy_H-Xyx_r-ksnDpjU1iKOm36KQgVUDlpSiq72WTCZjuZEAvhw0s-zTVND5fgtG71to-V6Dq9bVhkkYfuk4jY1tq=w2280-h1224-no
[image11]: https://lh3.googleusercontent.com/IW1Dk4SvsbW6P-VLvvOx6GTA7wD5bjHlKFqB6vISEzH3y0wnRbqfxU5ZU3jYmTSuBKjjGtdrni8JPEZ1ZYGDwyrb87eQofJrO9wEXjvq_am4GajnEdUxhVRNQIK5_tFSv6FyLQNZUjylwHouLhFUx6cQPMg5hR5XNo55bImp1dTVZ2i47SqF11d5ntT2-qvYRK7SfIKhbO73lgN3p0UgGQnNCQQWPYiUV8ke5kzJqmC8d2dg6O_qQOWwWTyIWjVkJyCC9TW_LXuDjMmkftDpVc2plXLTAYdxOkomyEt3CBMB3kCNmboKHPtxW4jzyMrTwdK9fFkWMCZNZoVjXfvRjeW75zKLOOLadForilKZASohGNB6MW9zPq72DYbTfKFnzipUEN0IL-9_-pBzVOTBaxbAREFxs4faRfBv6lFU7RiQgQG3-5hEdpC9S-XIuKlTaE6JpjB1GRvE3dNcQhk4UB2iSw-fy5HWPckqQk33y22HevMH2DbvqLeTnORnO4PA8jzP-zx-bSrP0xIhTWnqF-GWH7Y7UcY54gl__lbe1EQQC-Vt78ruacV2wFyx_dkCbd8fo-VbZf26MqnQ-1H1_6Q0WyKfFzkNuwn-KA8PGjgbiRzp8v1K=w1840-h948-no

[image12]: https://lh3.googleusercontent.com/DVg0vjXhDN09OHeW4BhQPDmaqLxLsy0IXsw3IhIFVhw0IEk662iVLN1-jWqdNxJzGeOp75rCndPvW8MRICTPYP2dpLLz6f6j3ZAYOSvRbyy6q7N8ZjLkQecz_eoYKZz4sj2YZi-3RbKXQE0HKQ4mDbRl_TEkUULPEk7tQ34fKd3alKyAtZMX0iqNcPH-D74Vnbp2fG4_Y68mVKe3V2rV5JKv6wA5AkGEHQS1q0qFO3x5Bj_J_DrMASFfS5IdTidXz-WYBTKVjtwzp1AyS-nO28wm9KDeicOjQz4AuYczx9puNfQF6eHAdRSBHDOEe8ZxLoWYWzR4r5bo7qIK1SCFspL_dH5YYVeVyc2epvLZwHekRtM_nn-9-fG_6PX14f-guIkwnqXv_dgKIW3zLkCUkmS85_negXXzVN6T3fMaBUkJOctVKiBQd-9A2B2A8ZcEtPSsa6ReWrBdbYiY3ZvySF446UWCG03NDu5YBr05Zzg_gqJE8qyrUk1ECs5nd3kdZQo_1_2e6NG4ytOytcsVeXozjf9qWp1rFq9nLtDS9-4YiLNJ6cp6D5ryeG390zqkFwVQ1tVSaMQZiszSVfzvIYfGsKy4tbq8ApTGYnQ5KJs4ui2c2A8X=w1788-h908-no

[image13]: https://lh3.googleusercontent.com/wkcF_rqZDt29URuC2unwhj_WgE3h8QRCCnNFGYB7lbj94wwUAVqzUfuOcKDZbGMmJzk9EP2KtmwbVaG3-LwW73Uqz3O_C9smIIYicRtLcKVVIexCWhSd8BMoeAIHoPii0OkCwxbuF1gtBV3gOrRtdYlEJv5u0ydS2wNnoYB7i4zSkgbeRZsTjCqM05DpIMGIaXM0nKVjYjCuDfDyjDYFxpAbfKCtfPq1uboSbwy5W1ZR2aoGePmVZPNGqKCj6dE8vSI1yzUF9GoBdfcV7scaV8JxZLX2NlWhd79H_odDJfI-eOrAZ2A7rxPz9Tw8DifCxdx4wMHBmVXs5_9TVDR3YKxTpYV6zAe49X7N7QiO33KtrhcgbgvIP6WxQ4GPYLFFCsLbKDWs22cL-mGdp2IvvX0yuH6AnI8K1G5dP1ebnvrqY2CBrDj03UU07YTNThFuUxzQ2mgKdCIWpVtUsZzLfY8gdRK1pyNMV8DVhlF_yL1FgwQEz-a-2wx_41K2tF0o24xZgAtCTjf7OdbMrHViy2xdwq7p-vfCXpV3OIC6eh0MSVEBwNzs0REOorFPtfhfBDpRhepVMYNy4F99Q7y4DE4rzn72DaTIiz6G419UJgE2Qowv-Z0-=w1730-h884-no

[image14]: https://lh3.googleusercontent.com/N9U_-aXjjF186oIZIlRYWAtZLZhMqcikpD7gD6tHprVS-8l4Xh3otZ7dHprpQ_S-68rXv6s0oh0LkfKaRqVJvbK93qdRxeCEnBquY3a-p8R7hbYkBxsNiiYGPBGoNsZYbRopithcmljQqRFFBQvyHe7yrXFqYMsH_-GXhOe-MvHpuGXAkh9msRbUkIX_5izFqLgDrzNDoKiBwKVlYCkTj_hZQgjT7qy1vh3IFqyjeuzBvs81rlMZgpNCYXcZsDT_id2UVfszvMgFeKTr2pNsVMH65pEFbCspgF_Og-YGHG6XS6PTRgxqovl9QripqXLxnIJZDz3fC4Fp6HULWqZqP6S098VM0hax7UvcUo6gAmAjhQNuk5btwlLivj6hMOUr6KL_Xcks2cQKAj7MHS__iIc00crzEt0suNJT6kkRlX_b_ndja150MNaaWy50ZRK95oD-l3bqIINd2Hh48ImyChRKU3e-9NZS2ne4qoKtWWnWUQ2rPpnUD29GvI6mW21OSmrjM5l_kw_K8u03AeAnbDOj69y9OfI0i-zdYrAlZCnwVFrXBE8zvOJ2ss6ywZMVqSfTaovyfAoJHRCThmEd5iIC9uYOSnAYQnqW1AEyj02s3gBn0AnD=w2316-h1224-no

[image15]: https://lh3.googleusercontent.com/RNApLOV7bZfK6Hr3I_ckiGI5DEPj5vfqe9wN2xgI7O_1aGmiRUhVct6eLkyCXBGaQyhy6ub5H58P1DQ28SxQe_gFhrjMTZ1HLLTZHTNAJn4ynu25ceszbd-KRQmilUi4hpl4xnCSQrLfagLbRUub0zOh0CuUeAswKi9zZpnOF2_oZv3IqwgY5yll5ustI9AAK9l4AGOhTsfHQ3RcqT5NIdxAXr2yj6ZEQJkrqcPxWicihcpSyywKMBMkiGDUJf6Smt2BQrs2s8hwIeNDHTYF5ivEVZOUKQMD4LmUHbLGfiKko5HO7G69Vy5y5AQKKJFZTr7xyC9yb2fbeOl-cpdRplIoi8RTQMWvO8QtfEAgijdORngeMQ2L0JUyPOS_KutYdqX48rPKdKi8bcMooh0qoY1_qot_vpJlXxycRVfo0zc5z1Z4Xa34uvKZ7AQAYmoexl2snDGq-P_hFqCyNVCvMYDNE9uEBOkCkwZFBPau53L_o-SpCNjbuJRU8AJ5rdqw_EHfD4BCc1DJmoQIxe0hqXkQS1aGOdWq5t8Xcg4KXKdYpreuiyKSzq0mNb4vtNYr0_zrS8gKG0ODKVH9dQUTOL_2b4uPYw39B1vbO8rDyrTmvn0MxMJM=w1812-h930-no

[image16]: https://lh3.googleusercontent.com/EHN2To4V7l0eh3Rja1-IBxyU-t5e4RMGTVivik9SXhliZPsgNNhrYWHbWWNwwu-SDHNtNxf7593vY1bgqrefqU3R9ydHY2iSi2r-xv24gkJaOE8LDSbUgMovlQwMaBrT-yvnS5I_v9s-tW6d7mYuO640VRPB_ro4-3hEYhfCv-5q-WCofXqrlQvNADJ7mFqhna0M7bRKti9YiDAd9bR0xcbVWXtFEywrCOMYi8Aid9EFAWJcLztb_2yh9IlF3h2E6INsWgPOtiygooySoEma73OeThx4qOfbpoNh0rzaVth5W8fZWY5MAC1eoC4m9jkRmPE5rbsHngQpHnpP-oKBYlPWIXd-vvXe47ym747APLk-ReSUfSw-n9z-V_1DuCoYHw3orAq4Rd8IX0J-W3iHLi6Sr7ZJC0WZmbB5qMHxjziaIXvy93IoEjub7ox2DsPH-TvDZz9SGMuC1DJMEpOOT2guu-cjFaMLu0vxLtVwe5bHWla0PEsp9L2MkO0IAClS0jLXbKrtzJ2tywgCGNd13ygY-mzVPz-_4_XxMsLcVyTdma_mIbLShAEOHEbhSfvoRIahRB1Z4vJgk8PLQ4PD724UNFtWviNvZA5Iw4s5IsM3-MpJvjzc=w1646-h942-no

[image17]: https://lh3.googleusercontent.com/S79dP_KJsqN9-XOkXJVUFY-439VsGa3N4F9DXSAsWM1AyuUmhmBDKCSNEWrUYLy5Q1LQVHVbo3LTHv3im68j0e3R2cQNF-KPAd6tzMGudQn5HPqZVVj4xboCt1XoQsULJMiEQH9gWH6Uj76Jvr76-THucBeQdJGFeJLcVr9cKW4yrmiiOeliG9_0eC9075olQ-fWlC29h5dZMV7u0evxLWr5SX5gf1WRrF1jAKTmx32jBWLfA0S-0ZQPJ9RpA0m4voK5dS2po_lTeVyCNCeQUqXSlWLmgJV3gOFyMjNOnqJx8qo4eyXCQn4Ki-ZN0LtOrZSoUdKoQrMUJ0cPMCYItxh0_xgxbeB8tC9myi5NUydmCLqlFToSwT-CeDN_yiiwJLoN1Jfb3gVhINPyc2tVNagw2o-WjFZp9i-onjT-qg5qwFEsveAx-5SmJDGh2jojBOdGHe7aZ4Mrs0AceROASe4TZChhAO-8UBWXBBVNIcdN99Gu5UDlTaRIXgJ2oGc6noxVWsdisozQ3MPsXL_4QfpAsNc1op-yyQ-04glcAOb3hzK-7cusZBqeybR1qKErO2W7CskaYqjG0jdtWhcRx9bZd6_-MUrb1nsZ3rj8V7Fzne5WGF5B=w1866-h930-no

[image18]: https://lh3.googleusercontent.com/Tw4H0JSDUsspF7jGyyqWzqGY_toH4XqnYorSNwsFdgI57rXaTqwR79shf3DDVGbt3Hp6IFQCHrk5jysn88G7n-thxctT1HwBVlYl3F1FkF8HosWyoI6tR0I8bTJXE4dv9iV_rJpNg0S4fc8ObvvDrpcDeekJYGjwYmxn7U2nKird33es8ekmb8J9DA4x6TSlqn1cJxGJH5rqD5c-EXkYO-XeFscDwDIr0mxpb7joiM5MCkSTojK0qnbPqYIyx-JBCSVXsKFQO3OHdAWA50sNQo0aDbTgdgiePN7ZEkoyFHQbiiVz4t3Whg2SS69wff5f8IzJu0O5S69t-5i-ibbYmOLN6Ra0ww1i9zcUUmKrnJrQKa64LV7cwH5IjNeHaclJzNvAs4JMc2bEo3s-y-MrQ5dlGPq0C9_qnhym8Ak0hmbqtGIQHPJ97eBJkGj5MdBE_D871AIEX23SWUJODYj0D-sqouZO5f8PGkacTPNS9k2keifYCzQxAzYwjBfuy78rwykir6dNxU3tW0eXGOq1cIKEBw6AhpJLpA1eybtI53QnzJLAEKqUSAAzLhqD-bzDB9bdDMoYa6X9Vd4e6RP_CWiMz_WlxY-AH9WpDtamhxvir_QQsIA2=w1888-h786-no

[image19]: https://lh3.googleusercontent.com/oeFaKKc7oxihPTQnlQhqB6cWqdIXFd3uFiUQ7JzeyFUDhn0bN2p-2aZiQMmFqEj86RD_k97Xv6p8e6LXWu03DIUY1YJsC5r5RTNL4Bc7EZWSzpBD35k85YceL4LCwBTDHGhLKq4PcefPgbV6hvRpQrh0NUINNQ7-VU8kJtzJxq2eZ4zF6z4Rgrsm5CtP71X7V1SWJOuWFuYtBE1ztUBF_qTaFmUBFbDYn9Ap9jib6Jaj7uMVgea1NO3GmLbq-a7tVtz984huCoevSP5UsctUJ-T9NmLTdUYbH7gZVyJ4HgBkHQSIdO0m4rhTNVzixy9NRzAbJaaAy03SWJNizwkdIdtS9xbzcHA7-mIfili53pTIE2CoEXQfPpcXdwvRnGCtz_O5QbPsGG5UpxEUaOQPw0CMpozPIcWRaR3wXeoOkcUO2i0-s9Zfkm-O8v4a3Y8e189ZYEplAIRJhPlUnp4L8E4B_KTr_YPlB0eigOe6Kx6pJaBZJ46AxcnMwIqijQRGDO7fX8a9L035GmlUxvdNLf66zbJys5QF4U-135CmUxAVHIx_7VeAhJ6ZHwKybdXRYFrKdz76l5KNT85Pt1pRJd0SL_tN_Bj2hE6h83gHhcGuw3rg6GnR=w2066-h1224-no

[images20]: https://lh3.googleusercontent.com/N0gi0s-ACbES3rxQvgxHz828ntGJqcS7tWpgtCTXCEZY467UJ2Tvu7k3U-f71Yu72SieE0yqX0OYKcCZFulDWiGwQzmrN0sfKnSzO43ZesLbcYczq_Te5KLPPo3Q6b6aP8qrEPEJ2HisdJGCKMWk0FFqytFAIEd5RrHt5BsrZZ242f6Ujbk1_mx9iCdIM7MrO-3dI4nNx3nZpJJOr2wu4T4kEf_2j7rAP_aEKSYDj9KEuMRFSjpNgAg6pt0Wqjq6qBsabTeVhYtMYUwBQbp3Ts-bADxg-48CoatoKwLH-p7DPK40zYwUm-vxIaYaJa7g7bmmM-_yaag5dGmI2xNDTihZiFJuyvYzmp3_JHw-XARNTFhNFC5jahou1xzTFLxHRd-pmgys55-7SDQ87Ka4U_EdjUGrCR72bjx8yi-JiKe95k6Udz_bqqlUA6mp2WPv_kcwWgXr2f-7a-UVJO-pw-inDoKC4-ffYLOmr7llb_LQWxal8_p88J7p4WjekFR6p1VsV_Tn7FBKcsEWYNmVHTOl15Y0io3fS1H9gBQYZKcVmsF8vucua9Ub9yP4_FrXa8jRJsQbTSSPtvE39Pee0zKEIfs6x4wbYtKHgpGBo8qFRsj_VMJz=w1720-h806-no
