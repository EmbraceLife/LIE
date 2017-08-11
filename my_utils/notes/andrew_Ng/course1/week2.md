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
- ![][image6]
- 碗型函数是怎么得到的？肯定不是由 $J(w,b) = \frac{1}{m}\sum_{i=1}^mL(\hat{y}, y)$ 画的；一个长得怎样的函数才能画出这样的碗型图呢？
- 碗型函数，决定了convex特征，会有一个最小值；不会是波浪型，没有最小值
- ![][image7]
- `w` and `b` and `J(w,b)`构建了碗形函数，单一一个变量`w`or `b` and `J(w)` or `J(b)`得到如上图
- 如何利用当前的 $J(w)^{i}$ 和 $w^{i}$ 来判断 $w^{i}$ 应该的更新方向和量，得到 $w^{i+1}$，从而让下一个损失值 $J(w)^{i+1}$ 更逼近最小值
- $w^{i}$ 应该的更新方向和量，由 $-\alpha\frac{dJ(w)}{dw}$ 决定
- 当 $J(w)$ 变成 $J(w,b)$ 时，只需要换个符号就行，$d$ as derivative 变成 $\partial$ as partial derivative
- `w`更新的公式: $w := w -\alpha\frac{\partial J(w,b)}{\partial w} = w - \alpha*{dw}$
- `b`更新的公式: $b := b -\alpha\frac{\partial J(w,b)}{\partial b} = b - \alpha*{db}$

---

## 


[image1]: https://lh3.googleusercontent.com/KqzCHNE4GoH-8Mgqdh7Y6PQkkR0xDcLyFvZbMTHX8cSDTmHB-0efMYrQe2njCjvGaP86tyZ8s2q3XnQ3nsPp9laAt7YgYpCONNkVm__m6mY_fjquRPFbFNn33hyHxu5m_vw1DYhXCWXrVBnjF8Fgdc4f7zOJATLkWnwjOy-2dqrfbc4u20s6L0H5JleMbThY7iZW2QG_PPqDkIgG0qg4F9GdybM-Ku1O_feYYzHFuieCWci4gV4qFjJmuPx4Y9eAq1P7bUx39_ht6BrNIhy02qDDr4vxWSvI7xzoZdR-HVJhkZrqJWidVrwAAIifMITCTLlB-aow1eB6cSSmCrF2227FklG2xUE8Sw2P6CU1qohuFw4hj5IbUVBfmGXPX_2Gmk61CuJYWUx2eFA5iUE3KnhzwPabhl7enmJmQrTR-l6FDa4Kf5g5zw51Ef7MUuKQNf1F_O4ZZmgh-uPgT93AQx0-G8dPp9Bw6uaXbGkw1XfXQ3XuYDN54uD0stA0Jtj_H9hWLDOeunfOBrbeK7XakoGBDQgvs1jGrpfsBlkqHt-LrVIAUnKqMu8mMIqpjRDvWYvqbHVrZ4qsqexRd3lpasfF3F06g1k44pLg1yYNrtU22uWcvulqF3eu=w1900-h1032-no
[image2]: https://lh3.googleusercontent.com/czHGUpQPJZuJYckHZIyBijbxONm0J9UIfwTls8NnYFib8tnD0Dtfz0IHisZ3lGrjUb63PWU1s4DoJhnYFsquI6R5T2ak9wIsI0ALG5xtIwPhY4bYdV20A5W2gAS7gS-3owM99rR2QZ2qzsmIXTr4sldRT-v9cdt6CYQXTkmEo0CecxnuqgWkb-uNP5TJ0JrT5_eEhwRBeHIsB91bNW7g-1tcdB_tCWfBWjrP_exfhJqrp0EluONR7TLaaiFfjhklEQxYSM3MQS9jj9osoULGT48p-sqj9nbuHWBdEZy2XMjjhfxwyL2DjDcJyHyxCw_DVeC_jI9U-9JP_vlgNCO_w4fh9CNfbHFKwmBARehS_DyCDHEJm1najTQVg_woADiVORgZ8W5zP0lnlJKDtvrh35jmctw6CKbBfft_UjqDBb7XWnb_gvIkmHWC_4YU2AKfXb6nF4OJ5-OmKH_gfe6NhQ5AjR9sXI_se7nWKxSNlAdQQGT5bue_-vI5hgm_--Ckqw4jBOY4MVsz4HmJoKuk6xIKpmA1lk6up6YWfRHttCGLtMlC6EezcU8YSPNJCoWXUw2nG9oo6gK1NMQbgCX0TuxHDUCi1WuAAkFUh-d_e_5e8G1gW3GG2C8H=w1760-h978-no
[image3]: https://lh3.googleusercontent.com/U3h9Ymvloex2Y9fXAdSWBjfFjXsLEHFhf189BkYizOryDNsdpLdKRrzLX4ZkWYHBQEoKJvK67cKvEI9MFOnitzYqfTtWOSCWoptYEAI3HgOK6j8HmCVHOw7iciIRrn-1m0HzJTbVb58NFJtYeAnPSo65lXl_PEDZqWf81UpU9B8jyiBQ7vcm0ixTFUE9MF2q7aVJpRn3hnfRJa1b-0-3rmgBH2kfHfrkENa6WuZzIofJp7d50zIOcJfScpDXfowx4IABPRQiax0yK8Ohrhx-Rj4L-9xC3Bk9AsuzlnPu0nBf3S8P-ZKAnzr1XhIDvr7pmzlp0mxUZAnglFTrvlm0dOL_y0pOx0Y3cYBtnwcEHEhD3TAwQPwtPImMKiyE425K3yZ2CAeo7tiGXd5vnfFEufRl-ld831nWoy1Cju0774A_lwcTyH4jcDxV17HCTNNyDfB_NrCiYfNy0spDOUb8VxYVEg2w2nSRKwVpWV8ncy0YHKrBwnJnbTbRL01w6Tpm8KXTeYc6Xxt_cIYSK1THz2WwPx1xBZJzwpn_oZHJghVuf7hnnBZNspARCV00Cpt_N-kelwiPPHvtJznT95ouAhwxXx6QxXxM7UyksCuB0hODF6WNWv29ThoK=w2094-h1224-no
[image4]: https://lh3.googleusercontent.com/DrBLbnF0qSP4XzAd2AMPpZ3eyiNVTNAj0SipdzeR37uy3VDJChbu4Po1nT66_5aDjQpUD0PRVLVmHPL7TIXtD32VE_vwmv4rI2MmSEuTk-upbhnvGERNHZ8oxSMk4wvkYZrJ61TKcGSPup45xemRDQoLSWFH13aBAwdL0ZgkXNa-t0AU7Sr2kNEpcjeYnJ7SvaZCWJmTtY26oapUs1YFbJ4dtb3PU5qcn-PezH6lD-bROu71GLBw_r7e_9mp4pKYBrqYejG99uwAhe-5SOVWV9IhCY2HsTPY_hvoBYSiYAjBEa-2Nd9Q1yeKnEa8hKhgfBDNrmK_oGy2Yylrco0XhFvvBpW2t5xdGWWb87GGSpNuf81sQutFpVkzXk7YvGGYg9h0atKnIv9wA9RZOXfQQlj28p4XCtQ7wMXOnyJLb8AkmWsCs9BmZjj_lTB8fwdNg9uHoQsil17Dajmc0tbFycVHGVCcrRDmmSwVNwbs7O-zae5DIgbDHIuXNHyQCk-6wWcM5vOyHnQxbbGDGFWBl4BrpBNAfyYAx64R1Zj3kY1RwbXrWeLHgIvL-7xDNy4jQXlD3S7mt5ZB3u-xPyfn_GZK5UHvmPZBXIlHMPASlYWmI2DCMBfAL5Yw=w1918-h996-no
[image5]: https://lh3.googleusercontent.com/o7FZdccz5FzTEb_W5kw6oLnq6x-nOz61paXKEqGoZ36r_eCaLIQRjY8o51okupX9vtY8AE0TXGZSmAfN6u3gin35vbEQV-8Ij28ih9mfmvALP6d8mOCF-nlFfaBwx4nuAtmzU_Y3zAeTa5qyrZD7W-PwaBlm9KNou_aLnIyTcJOqySkVP1k41BG97-nQlucHrMDMqeJ21vsRqKb8lmtSOheJOgIND_AN6L1c-nlnaQMUjq16sUnu29MBuqBkFQoOlD75XAU6IofURaVjxgUqN-A3rXuQQEZFS1rI8ISucaA8KmBq0hMeEjZtZj65dZPvmSRYcWVK_zbzxM4r1qBIRYTAFWfRgHKIrupQ0WMAsvWnyVjHNB_EJDDTy831N-PzwIHPJCgB4velciTKdXPiwLYaGBSX95gh7O384OjwU9DRMMPi32KKSa-ebiN36fQps5VSM5mFRiAvvFZJgAGg67iHs0cPm5p8VtZknMeE8WUEfBDRcFJnKv8xPIKk3QvlwVA1HVdyq9fKDfFh_inDZMBinE_2ySQVqIupjfIHR21yZzZVFZ8gmhaoIqupxqSmB06Z7fWz-9rEFwhEiUOiD_0vsZzuFKr28Wq43XQhlb6zQloD3UJj_Oal=w2090-h542-no
[image6]: https://lh3.googleusercontent.com/ZBqjR9tg54EdEw1iF98ujFqsYDy93_H4fc3AhXUSO0E5a8q9ZIXN-exjtkcQXg-HLmY_NxloCC90OMPrq_bsNWj2O11NpxyWOta7GEl9myWZ3vvDC8xvh2TZz73WHfM6WX9W_oVM5R0B7NaO0PWH2mCWTxPcOHHiFxxcy1lv9SXr8rbpc_frK6pXl7ZL7Qp6zeiubUYY2rryBKpFeFwxc-9AC_ULl5t52bt7gGpZcc2vZ0-kKt5ooaqmX6kQG8r2c9nB-i87_xZh1H5cT2pcuFG9DN5dlO8aoNmGFFLPL1A3dflQDbD-nzgBvIGiHnlM7G2ZM-h7_ICpVt2IwPui9pCwlsNqcDwHnaLDZ3gCyf4_2aPONaR4GIDCtSNWGw5VhMgDO7vL_D3XhsBWYlVS2E_qBXlWkGrOJ7XfPOmxiuDwWTRmYrOqPBe0_EEh0G27pO_e95nr7FMAt0AMXq6lK4fMaORARrAp_QXVRDsVoxovA7Cz0o0PvCdjPKdeygY5etqDwf26_6vsROwZ5Ex_vRjOCaB0HBRL4xqs5sa2yJU3GAA3J1y_YI5U_ffXhKEKSucOWmj8m_xKI5afsZOjP0GKRVOy36nmp59H48eA3qgMzsfeWM-QUURZ=w1914-h958-no
[image7]: https://lh3.googleusercontent.com/y_J_yslGvUEhnDzf0RtQ4zFXrlWG427hUzNL5F1b2WCASmkD1I0NhYXKaDwgt4q2lGT3s0l4IRetZiS4sGCP-layErG7Cldf1ZVe54yjbh_r0Stv58g1ndnq1N6bHpVyeqbCReR3BqX7Y0skmdpxhroB-mP1iNxLQ7zroG7lMDE0DR76KjE95FxokVFqF4lMaNkj0vpSJ974rhjRY7h2M47vdmcL4b4AmK-ygwjr6zXiQUaVRjO_en8GL3AXQtuOr4Sj6iRGqGiYl8uD7KI--t5pqCiudfqnCTzcoXkGIRV1CXkz-8pHEzMfvFCT5BUOodECOEYlRIBWoem9Jxg6Xsh9O08jS6euISCBNS-5jiujN3yWeYscPS-YfZmb_nYidqkTAP8IvIe5-6mAT83FkryJV4o47nnDlrTCb9r91yjqBSsAqO6UhMMsJzTLIUwRZncN9r771dQ2_heASPPQeJzfnC_poENDK3e-mpJ3DV56DSgdMStZOINPM-MnkTCwc7rdNlgoeRQ1BR5P0JH4-1deQtdFG0FjuUpyMeCIyINkvIjpvBn4aXvZuhrasVZOfkYSjkQ_NInBUHzzUTwdEiJYLBAketVYYFx5h-mWy7nmEeyKcN0AYy3a=w584-h322-no
