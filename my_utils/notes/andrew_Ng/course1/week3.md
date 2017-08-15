# Shallow neural networks
- objectives:
	- Learn to build a neural network with one hidden layer, using forward propagation and backpropagation.
	- Understand hidden units and hidden layers
	- Be able to apply a variety of activation functions in a neural network.
	- Build your first forward and backward propagation with a hidden layer
	- Apply random initialization to your neural network
	- Become fluent with Deep Learning notations and Neural Network Representations
	- Build and train a neural network with one hidden layer.
---

## Neural Network Overview
- 从一层单个神经元，到2层多个神经元
	- ![][image1]
	- Note:
		- 最简单的结构，只有输入层（3个neurons，如图）和输出层（1个neuron）
		- 复杂一点的结构，有一个隐藏层
			- 隐藏层有3个neurons
			- 输出层只有1个neuron
			- 每一层，都要做一次linear combination 和 一次 activation
				- $Z = w^TX + b$
				- $A = \sigma(Z)$
---

## Neural Network Representation
- ![][image2]
	- 2-layer-NN:
		- layer0:
			- Input layer: X (3,1)
			- activation $a^{[0]}$
			- no w, b
		- layer1:
			- Hidden layer,
			- activation $a^{[1]}$
			- $w^{[1]}$: (4,3), $b^{[1]}$: (4,1)
		- layer2:
			- Output layer,
			- activation $a^{[2]}$
			- $w^{[2]}$: (1,4), $b^{[2]}$: (1,1)
		- <span style="color:pink">不清楚shape和T在什么情况下使用？？</span>

---

## Computing a neural network output
- ![][image3]
	- a neuron 有2步计算：
		- linear_combination: $z = w^TX + b$
		- activation: $a = \sigma(z)$
- ![][image4]
	- vectorization on $w^{[1]}$
	- $X$: (3, 1)
	- $w^{[1]}$ shape (3, 4)
	- $w^{[1]T}$ shape (4, 3)
	- $W^{[1]} = w^{[1]T}$ shape (4, 3)
	- $b^{[1]}$ shape (4,1)
	- $z = w^{[1]T}X + b^{[1]} = W^{[1]}X + b^{[1]}$
		- z shape (4,1)
	- $a = \sigma(z)$, a shape (4,1)
- ![][image5]
	- $a^{[1]}$ shape (4,1)
	- $w^{[2]}$ shape (4,1)
	- $W^{[2]} = w^{[2]T}$ shape (1,4)
	- $b^{[2]}$ shape (1,1)
	- $z^{[2]} = w^{[2]T}a^{[1]} + b^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$
		- $z^{[2]}$ shape (1,1)
	- $a^{[2]} = \sigma(z^{[2]})$, a shape (1,1)
---

## Vectorizing across multiple examples
- ![][image6]
	- 以上图和案例，只有一个样本，多样本怎么办？
	- 单一样本处理 + Loop `for i in range(m):`
	- sample1 $X^{(1)}$, sample2 $X^{(2)}$
	- $z^{[1](1)}, a^{[1](1)}, z^{[2](2)}, a^{[2](2)}$
	- $^{[1]}$ 代表 layer1, $^{(1)}$ 代表样本1， $^{[2]}$ 代表 layer2, $^{(2)}$ 代表样本2
- ![][image7]
	- vectorization remove for_loop `for i in range(m):`
	- $X$`.shape` from $(n_x, 1)$ to $(n_x, m)$
	- $Z^{[1]}$`.shape` from $(4, 1)$ to $(4, m)$
	- $A^{[1]}$`.shape` from $(4, 1)$ to $(4, m)$
	- $Z^{[2]}$`.shape` from $(1, 1)$ to $(1, m)$
	- $A^{[2]}$`.shape` from $(1, 1)$ to $(1, m)$

---

## Explanation for vectorized implementation
- 画图理解，如何将for-loop去掉，使用vectorization
	- ![][image8]
	- 这张图中三色数列，很贴切
- 数学表达上图的内容
	- ![][image9]
	- 记住： $^{[2]}$ 代表layers $^{(1)}$ 代表样本

---

## Activation functions
- one hidden layer neural network
- ![][image10]
	- every layer has linear combination and activation function
	- 隐藏层 hidden layers，tanh 总是优于 sigmoid
		- Input layer 输入层，不用linear combination or activation
		- Output layer 输出层，对于二元分类问题binary classification，选sigmoid，因为取值范围0到1，更合适
		- tanh: $g(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$
			- 实质就是 $\sigma(z)$ 向下平移，在1到-1取值
	- `tanh and sigmoid`的弊端
		- 当`z`稍微比较大或比较小时，如大于2， 小于-2，根据两个函数曲线来看，他们的`slope`都很小很小接近0.0，`slope`快成一条水平线
		- 意味着，`z`很多时候都导致`slope`接近0
		- 导致结果是，梯度消失，最早前的隐藏层的参数会更新得特别特别小
	- `RELU`出来取代`tanh`的地位
		- relu: $relu(z) = \max(0, z)$
			- 当`z`小于0， `slope`为0
			- 当`z`大于0， `slope`为1
		- leaky_relu:
			- 当`z`小于0， `slope`大于0，但较小
			- 当`z`大于0， `slope`为1
		- 多数情况下，`z`都会 `> 0`, 所以`slope`都会 `=1`
		- 因此，`relu` 是 `default` 选择，虽然`leaky_relu`看起来更高级
- ![][image11]
	- tanh is always better than sigmoind for hidden layer activation
	- relu is default activation for hidden layer, if you don't know better
	- `leaky_relu`: $= \max(0.01z, z)$
- practical tips
	- 虽然有default 选择，但不同的问题，不同的activation function表现不同，
	- 逐个尝试一遍，比较一下，才知道谁最合适解决当前的问题

---

## Why do you need non-linear activation functions?
- within domain of 1 hidden layer neural network
- ![][image12]
	- non-linear activation: `relu, sigmoid, tanh, leaky_relu`
	- linear activation: `linear`, 即, $g(z) = z$
	- 为什么hidden layer 必须用non-linear activation func?
		- if hidden layer and output layer both use linear activation function,
		- 处理后，就是一个新的linear function or linear combination
		- meaning, hidden layer, output layer 失去存在的意义
		- 其意义在于，通过non-linear activation 创造任何形态的函数
	- 在预测股价或价格变化率时，可以用linear activation function 在output layer
---

## Derivatives of activation functions
- 求解sigmoid函数的`slope`
	- ![][image13]
	- $a = g(z) = \sigma(z) = \frac{1}{1+e^{-z}}$
	- $g^{\prime}(z) = a(1-a)$
	- sanity check:
		- 当‘z = 10’（比较大时），`a`接近1， `1-a`接近0， $g^{\prime}(z)$ 接近0
		- 当‘z = -10’（比较小时），`a`接近0， `1-a`接近1， $g^{\prime}(z)$ 接近0
		- 当‘z = 0’（中间），`a=1／2`， `1-a=1/2`， $g^{\prime}(z)=1/4$
- 求解tanh函数`slope`
	- ![][image14]
	- 具体的细节见图
- 求解relu函数的`slope`
	- ![][image15]
	- 具体的细节见图
		- `z = 0`情况很少很少见，所以可以归并在 `z >= 0`的情况下


[image1]: https://lh3.googleusercontent.com/a7C1XrUZCxjqgVE85HgrHvh0sdlfXHt5zrK47eopwIoz3h9tS1AjL-0H0rcxIi2eaivXPIVGV7MFvnLHclltWnrnhr9L4ipS-QZaybmWEPTk71C9e8QFtE2JwFpWEhLuplzOfjFL2L0t9W5eGwoDjXNMFU1kZML1cXjZVz1d0ztFUGqQlI6w0RXObFq8SQGrcxov08XJLpYy1DyfS_kVREAuwKKwBFwzuxeT7VugNn-CwjJlqedV7S2FZPXHnVMIRbac1qxq4nF_NHpb5F200tyDRaoINSc33g0aLgl7qKsRjeKQbCtwOD0zpNbgtvgm-Wg_XRFMqFDtK7BBFQQe1HlMqHY44lTZQyn2832VdezUCTLlXKn0Oyj6lgx0FBmGDKqHqJQCBniv9AFZSCf-zY3M0VWm4o1P4eQodK2CUh2CLovqRMTT_fooiA_D_BTfSWAi_9XDW1gvo4eXy5jr6nZ70iNwex2byAk74qk6AcrNd9iTq02sml32jCUGNmK0PQfRqgW9PePFMkdARG6TqQoq3GxIZ1r4hUtGRf_TJZdkMdZJks0D4Wh7iGBUHruHlfDEpCZ2NUEBEb_fCzhYaz8RoqrxrByLhJt0dKZjN9qfH64lyUJZ=w2014-h1136-no

[image2]: https://lh3.googleusercontent.com/Ev4_Mukf6BBI0qefa6NVOl6Y3qPykyO2gCCKC_S9gk0hw6T94cpjmxfg21lRpBXG7shnHMKhi8D5_KB-Z900H1yBBs6EXkxWhCo7y86zHgA__hZvXxWDWboE9xTnOZ_qaLsP-CiK-N8h42UBhHWLJ8mJk16rvXC2DLHKWBHNnV-LF59SE0TGm1aIadAciC4tT2wOCuAxbjMNa-tEI2IigrJOgA1hrKpHguXNQR3q6lK-Wz1xkpJ6udRHRH-y-IZWZRAbtfJk5pfgfA5O1mDBmnGHQrSvdWKGq_aoig-3ybOatnoOmgXR16qTtGwMRHnAetJmXMxrIR8WnUYa6tylvFKXArwK_9RfGWRJm3uzTngzDXnw6rl7YerVGdt5S4EG52ohJtUL8WdAHASEDruLptNBgjtcgnOYFhWlt1TCXrOrXL_p6LrAg4CIEjw59EBjQRidkNuBZCMQpSJ8kB15fhU2kgKMro0drXsFc_hlG2S91YUNtdLb-H2OafID0iRUKbcxiyIul8Cnxw4Tj_p_xAv2EQyc4zeI3iKXjChhHDWR6U2DSuleuN8mooQmuHrER47MKZxTwW7MOQsPBtgnRLfWxfaGX1PTU9AFTK2HJzHPk5bxIopA=w1914-h994-no

[image3]: https://lh3.googleusercontent.com/q7hDZyGjzi3Dn-6d09amRfcx4PMhosvE3pSapTXfsycov-f2j0D8XapuIWnY4iGwAHX4-odyGNSlW2J5WN-SOpjJ3SmTyWjLHaYRBfCAO8aIlyHobDOYoOnytQ9KODuzLN5PsGZj5K3QpqtrNCItQODlkkc0U370x3Vy0Q6u-kIqvo9XEk-LDaHyMyrMgKb6FZG5r9Nbn6TZ7ONYsJeMec2IGLeqgHvXhzbfuZBUtKMLaT5FCT-NNYt52wRPBL5WXkStCs07yaHaMDna5V69wC9Wysvt1GwPt77i9dZVE_OZL8jMABYnJ6GUkSxNt2-zizqO7Iqa4JYBXlAYcjs_W7E-rhU6hLsVsxsz2Oz1PAw3jg_tkLxxIx7VA4tNi40x81wAkSfET-8GoFEQIiDXWJJYEE-jUSkJGgTUzJKV_L0Fe9M7DngqFSCMAMG486K1PosGUCYSmEZ_-zRH5WnMMC0-ZSoQik06ZcOo5jsAsfHUT6yQp_baGBOAKxTRn57T9zSDLU7ztKUWwWQVsv-651b-ke_S0Qt-57wdXW0NuwD-1ntqAnG6QdeOVwXdahfbpaN7YhOdoOuXQGk99EKidsuRqni320Jtovbw-zvY5vmXOX8y4mFR=w1906-h980-no

[image4]: https://lh3.googleusercontent.com/2rILt7u6oST4LNLONqLQMJZ0U2ltjW5eWKIHr5OrWybpSqC6hGt_g9qURP1lJayYEy5dERxSaxC0P4xYIZLG5UyqvJ8bW0lrFrbSdpDZZxvRq46y2sLD-LtvnKn6TSrOUcW4m4LQAH4zWYhRJ-RFd3hlsm9Dp5dZQ-ZuVvVmRT4-pQolDfQKIHPSnFhcx6r7YTpMpjv4SRRCYxEoWpUFEvl7sX5AjSgHZ_BAiWC6yDg1_KPSMsPkI-C-MfSrHYFJjPgO351AisDaL8RL2b1lFBhSp1qO52nwbObFxEEvpt3FafDXax89fI_P9PfzaTQycFP4tQrJSUgeMhdqNaB4PwosP7RC0IkmbzEDMzXleInyzH337jjZlM7IR_D2AqL5ZQfvPm_3eBBS7jgesP9_C_klkjqy_2zHbHjRPu3yJzFyZIs08x2WXa4c0Dbu5xJrk3beMNaHbHMpo2I5YiF9yqZg7lLpHJ9_CGaXnfCF3YdvM7SRpBbfTGVJZTgVRz-mFPeIKGDJL5bfZjfTQ8GUHA_Mh_bg7WOpRFchUWjPpB4h94JGjloRvxQT0SiwRJT_33IbU1ftY2coiopzR_YHXzD4Jw0omIcTkgPATXkVqqnqTmYzFaES=w2036-h1136-no

[image5]: https://lh3.googleusercontent.com/1x4j2y6-QhtP_ezD5rzrWWz2r1OT-IaVPX0DFgtGBj5C9iYZBCWZVDAwWJRZMZbH4MvTW2GNqLTqhytquPvXiYvs_TKO4eUS92TqEFGbW_TD0duet7yNT7feSS1MPkXQ_cco7iSxUm-JvNRVsh5Bg-WxRLkOx9kNJKWvms3mzKOoQRigtsBdHMxU-rCiD8mn_lxkAiesKxjs1z36kQSGBN9szZPXncDxVlusk2GutwkdlfGEm_4zC8q4frO4ASTlkWI7XezsiOlCG4Hd4fZ5KvDPwph7ZAKTfM_Fs_vlNtTBtyymXLo4iouz5M1stGj3kdcu7LvkoL_iLYc3YBEg2FJeKkafvWjmiw8U-0Z8HknkkPIFLtKebbsXjGP0tnhHw46Bb5nLk3FqCRPjkX2jNRB8_59XhllLtxp1GCRLOH667eMDma_LS7T_m54mwM2Dem5NVJMdNNuk1Az7PiQ6KfbCv8dBb1LiN4AcGQ-_P-e1gfdKElqtUbSXwFT2E3d3kDClSXzdDKC1K6fMlEHdIAxln2VjY2p8KtDIrlLuZvlD_2HzOVOyMTxapwaYqNhTxke9YNcsC5pR5WGhuoDfATYKaOcY37uMMFoSi4MNWrVJid11i_VN=w1796-h912-no

[image6]: https://lh3.googleusercontent.com/AsMK8hpqAH8zIcYlnMKSUR0b61ubw6r-u0Lyn9CRLqmMQyYT8emYpF_PU5Ty_tcs8aRTmLsO28cBLAdK3bNle53onyE0PGljzk9-7p1VPlzYE6lrxP5_ZWoriNZFzAoHSEFO6hww9Qs7WG7dIttRpMyCJ6CifEW3KhsYbsFXFr-Kh8bpsFsPMZuC_q4TiZiNi4YEoxlIaVfaMpbwzEjQxRPNdAmphHxcaXLyu-IpCHE71JCvKF3Dpu7X-z8rq3sbZ-nl6deeUVtjQY9vZPqAlAoFia-Fz1T5i9WKRY2TASyIgg2bSg7CeiqU_rIOrRycNgkn5fyXdCEZLVX93kxhxfO8ZKPfGFWSosA2w6A0HOLnwpZYEiMJ2l2_LaQXWwZJKWLtblleOa_i_YYers4Z1Qn3rTLdKsgPRN9o7EYbI-I_4OD4cB36wsFaQ7Br2YMCQMmKSIpC8aLYVgCoNYG1_t7f3dH0JnBeZvb5_--tE6fESN49gRS6w2u4Do53us-Tf9Sb7XMCsEp1Uj5eTZHr2iJdzjgQWEEcGVRXYmgAJbVlaVqsdu61VWKNdG1d3WMKnA1lhKMxCOC1k2Fh943khGzTU7DUUF_J7V3yfVznHr8hwBNHD5Rq=w1910-h1136-no

[image7]: https://lh3.googleusercontent.com/aVMVShNYuHO0eCdPEuwBgok7g6O9dWvUD0_BWVw0NLAZYnpImjnTWCMnFqSiPqGDAk2KphpXd0E22BMVrsFwo4DRYXtlpUH6hM7PuJ6PDkJUhoqadxgpgWakLtISOVi85b8sYTkzo6EtUzd8IwixaMvRY3z4hVW4HKJr9xU2wmoi1U6wD-rJj8KaKmTH8lBHHqo91D3BSnbeMqeTqSZp4ryXHysARU4dxDK9yFp2tpBIjcancctQG79fKO6_ZlzxMJ1-kzaK4Rw5p_JDUy-ebZ4c-zL_KJ8QhKehPeVbqNfumtoblkKXxtvlGURp7gmAT_Gh2H80LU3zAg7HFRU1pKCij411EXzfQrbz0ZzX6WsFLIsZCvbDahe88VNFZHF9qJGnXr0xTbQPJi0zMFtrSf6dL8tzcWiFttBMxm_qO8QoTsppGyxzRjZZFr1p4zuHarB6msDxMTwWcyMJCe0X-6hlvIwzoCtkDB6NDrs_g-7UPvorV40r33KJysZJbjP6IzNLwiaFtwPro8zUafXmHF6vmEOWwMBFrp90ZIRV2U3afRoetUsGQqAYuxkH0tst8FsIAoCdG7ju84hZDPTH2yyaQfeB7CWB9YzW2LoKZ-emD9-lt4dq=w2084-h1136-no

[image8]:https://lh3.googleusercontent.com/-XSh0ZPu-iQ4WdYOD2g7w9THYiWYCfQYVFh5JLtyFk_JZtFyM22024xeTOnYm60SD_ZZIqdsVFegWPXHAnZkd16cdW1uZwGMVChI7DBpC6qDJzVUpllgPWWalEBkOKXd15IhJKQsdkzHI8jESWoVYqds_X09ox6yx8mYfgciEO6tT29MO6howBNAuiDH50KOlGarX76la05zMk8p3e4X4mmM_oiuLzMUODD-KKQxMwzCRaIRAcH1x8ICTbkTqWFdJyraDJSGUqxEVwhDR1JsPpayMfs-1xTrlV-uMEq4Bj6Eveiu3FtIeoIauf6xY6bcII0tp-ISQuOutE9YU9daoZ6BpB_mgT8Oo2vbmB6mEUAKKIvsbDyBMx818MSSVRNFKfeO9FaardxlcEJnxNOvuIML8tlKiqYHe9FujTRVe8RE4lliZdvMlmSHzSIpzDXTq7ubk7_wnYMupKGUQU-qZXqJiVEUtc_ZtkVxV4lojbMahQZdSPyHtJVavE_kNHHb48ArhJtLDXKQLcS_I0eEVhavtUVdMWFITfhzSm1AnuRBe3Fa2Q5J1QU-7WqeKvEQ41GiRdPfTjIbsDxVV5cvrN7OyDSp-PxzwdMySZ-YcPXElI8x9o0X=w2082-h1136-no

[image9]: https://lh3.googleusercontent.com/WT_DfTZ4S9Kz59BtQYvokQ1tL7TcKdgCNusH28M-3zQJXQB5VqMrcSkBLbaqWhY8Md8b0YmHXqOkFxD69NQsA0_UPSdITUTHtGFjTzbtpjo1gFaxUs-n3h_8kguhvz6LhE8F0Bj0hRylFtzXRbPTboGnJXeYtPwQ34Eajj_KxMr3ehvuvrg0Oe0ARQ-cLu2IcRxAdtPR8SNuNN7n61jEcYx0jljPDut_nhXNnEkFzT_ph4uENvKuIv3kPntD2w0676kkXPq9ptpS9chAVx-J6ux-dw_cfFdD7DirJhZKzVk7OxogUysrZglphmi57a5PCkwy3eDUN0qv5aChLXu0cmJ2v5hOmyL-3JRNj0cDhX7EioNwqxBQ3YQh0_l-VTjOGwVnvwVEM2awyuut0oWNgPnYOozxIcTKb_ymk4qZ9LjWJQGMmgMyULj5VvALmNRG7SohyFr12zK9wt9pEi5uRDaIcOe7tG0rFSKbjMGnq2TaYxiis7_oj8rXDx0Ha0rq99TN5wqCxVKrpxza0cG_wgmfTK7zNp-enYcaB-SV0i-OFEIcwE_K5zB70UuXrRz55Dnqoi_V5y0rQBxxSHVBdEDTiiKmAp-VpEKABXKszrDGNL5GJhur=w2106-h1136-no

[image10]: https://lh3.googleusercontent.com/K9zMOy9vvVv0bWR-PfV7ALhkH81I5w-30Rw94CDyHIVeGCJTIjS_YLQO7SDPxbm-Aj1nw54hCBLaCWEzHJXzcwjIrq_SJRMxPvdH-pSJfB1KFn2N7yDEWXG9D8su8cOm7ivJI57zqm5iUwcuYtduCT4OiQhQttTKMCMf40piwSbutmL_HFg1wvspDzhqevAStLnQftN1V4tC6dQRpya9cs0qu_7z61bcarIBB5tFBvMoONe7blRAYtbhUZhLOGvgM52MA3a2u5igqaGaClRtFce3mGOWe-_pIMzvUsO6dsI71lCKTE7zeXsFRiq5V7xj9NpWZ7NJQtQNvovEzn5eQm7ftfV84FaFfG1-Dw41GIRhwp-RNTOoKhKRCkg4jaNmdxF1xPZ2lgw0pXny_Z1cdwdfkWKNvBNjkB7zVURyeTwg_BARZSW8tdOcs0_C8iQ5XGwJAaAo-eF3EfHfUVLrO3TKVtoyMvYUDsp3MwZBdHXDoOM0M19CGprenRtdXqnh3F0wumPymcE_LhtlS9w6Ootd5E3OG3ViLvtMKOVkRMSN7-TVvL6cj4tOpcVrPeeYr4trxDJ18g6G19IQIOw3nxDeWootto0WfYjPd0G3aTqHpIuZOQHI=w2016-h1136-no

[image11]: https://lh3.googleusercontent.com/rteQ9oUfYwQnPwwR8uC_3hMRT9E44nHUuV1cOJVuzP1gXlSnUiRfeXfPBiTKZqq_wsc8Fnbrz754oLPWBHSrjo1iPIRQEwOPIIH0gTFeRC0QNSZNd6T1Jkj0gJ5pk8LmduXJHmGREbcj-KERlyvALIJKBr7p9ZKmsuml-IYgCL6Hno33B4R7z5TbC4iflZ14ibmKEa_aVJp3lYpw-QALqSsWtckOMwIgWoHZEaReLE0zI1xqJ8L4Xw8YerCgTgY4sNLytIexLVtaQ1oWY9fHUXyzXmErnan9UmfiLoophsAWHYmbag0k8CdMqDtILxe0F74DOi3HU6qMAhe5LZjC-wwDyOXB0Q44uuTplmnbivJZ7kXA6PIw6kL8vSc_WGuEFDF0f9Wv6Fg_wSRJX0p2Mrud49nKQFtCxsQ49Izhi-4xrYxFO-XVIeM_P4JiHAix4MrqmmapLwJ0nm8R-kbyAdUo0NMv2M6HSWKQCJyQfv6Sgs1bHt-XQxrJ_p8z8ZBR7IgmqNEYGzCgo4y-tfSj0vFZzlg2oKIPDp7Rwh2OJ-c0m75ahdesBv4tJudHwUTnknvly4LapUrjbNTzd34uFOzMhFMrxiITEzYv5VOBX4XMuBUZtyFY=w2040-h1136-no

[image12]: https://lh3.googleusercontent.com/gGeRcprsw4s0ec2u3EfGa81Pen3nHGsTSDSmVXsILc9cNy6POPN5MSYruMS2K7TeRJu9MogXEf0GS1ylQI16iytCijrMaJL_WRzi_uucOGO6Pkp2gPn42-l-qwDsCoCvKujzRqThz0jffzOyQAWnrY8TvjGlVC6_1lXWelpbl9B1rcS1mgTduPGL7_4RyzhJCNPP-yJY22sOUHnQ49XK0orTL__xlh4YBLADbPpgM6bH3WYbnJ6TibmDs5kBGEgK2-1rKvSVQeyUpgdbfErIBJ2_2H5fIjhHtzq9Lz5pp8tJJvz991VsF-TtibLNR49-Vb8NPZNYvY9MmoGlaK4DkrjVtw4-BQ9KJphPffh6Lr6x3RA9wgH-HusjOzITiV_D3CXuUX_w0t-J7ZPWTw4OWKEK_wblsDE0dIbqvPrvuZOQuqpK9j_pnzk_1r0y62plgfcXj5aSpNB79oDCbFyPOO6HIXtae7AmEmgnvvj2x4ahZ2eqMQ0kSMJAP1dx9qX6tlweqzhSdDBjlzPDB17aqy6BgiTOrDaIEhqIYtxtZ6JrJyOG8Jx0TJWE0AxWAFrt8BXNP2oHucKsS_4q547HkRMO5gD6wgJe-1lFAG9WqV9P6QPuJqpr=w2138-h1136-no

[image13]: https://lh3.googleusercontent.com/O4C0P1UVlubTgYASIrawHQ2pZBuDpyzaYw87pz4ZrnD1W8PV7sJBJINvCwjaxlWGR4NMNVJH9daGiLQSOs2sQPyq2jvXE9DKO_N4Z3ZIEBvQpGWi1I-N-ao35JlyiT4tBNqIkT8zDmt1rsgLTU2VjIGwiCNztidq9ZBOdwXF5GYU_ZdNgCfqpMNLMga99jGhKaQ6phAinx-B7CSlE4Ms6TJERC-gfLICY9zUtgsTxngO11tG9N95IlLThm4263r2lNpnYarHsM5aFEizGhXushmkYbaRwT-Pk3k3iSWBES7d6ftMargI69enH19ePLoABf2tZY3Z2xMMt9uycnDmC6XO6O4BZgKY6-RdxxFIq5epRmqb8CSS8TcmcTmyTop2h0HHuBVpdGshOHAn6hhGUw3ReDQYuabxyog9M74xK3kh43EhBD7kH8XHLQK4joOTVL_wNxwPpJL7Am1eif-vOqMsheXj-RI-ALzk87HE9-S9It2GVxRMc3iYip6RoyGVM9LVU33CGxzFXZ92OEE_hP8D42yTxyduC2yFdoUY0Yy6jfEFeh2qIXmddquBZrLu40ljchrPW6Wx7oK0fMReckJiEVrB1fajWHMa6NhUY3NT0VgRDPeW=w2110-h1136-no

[image14]: https://lh3.googleusercontent.com/8lN3usB7YZCjuQfGVMTLMEbcu3wnOV8DBbucQc1OdF63AJkzgXnR0vVC9v-ilQDcwsECF1jxs2SrQd9fY0xY1J2l9G_YrcM7A5_WTTEI91a5EzjhQFhxxvof4uGmFD5P5pnzo20V2BvWQkAmETrCebtTava7bizpue-oL2efWRkLMLbL7bHZAIqed66q8XiJX8fGPXijiYLbiifIk6OH3W5FAiQzRyMUemF3g8NieBrcNy-LSq4JwY5eyy53snesMwXKgUDUGz1u8TIORiY10k2yjapQRnEJ4cqg9HZaa7mMWpayqN5X1xjGO9k3rh2SlRwis6HS8lHukDp9TpClvHn5asoTTJrLwv1R7swojLhRGLpZBGGpglyfng86kU4m_vE8srMa6UgJ9nRL1VLVK3VMwSiBm6BEjgoY5h4oN8xxSBe_HwyHxBCUHa84ecYuMd0t1mcgEpAHxmItXYHnEudWM8QHqgdAyMcU1nkCT_fXWaBwEAaTGEJTjI3WEZr37ErXSdWQvg9CM2yrBIscOxQCDM8Q9k-5HzgvmVsbeTsMBE05j3lBmmgdoam97KdDvTcECwqo7SH3Tli8fJOveSgVeka-OyedAJYWcb1bggqDoD03uUI0=w2062-h1136-no

[image15]: https://lh3.googleusercontent.com/_XhcoKNRAM3wZhckyQOPOSHqbNxAFGy-V6h1FzlDVrcF80zqvHwCqdoMy-MIUhodzW76n5sczTwfj9pn4w7hTMsXWdbA_IhjBY07H3qtIX7dGFwZuVVyn28VGF8XjHzctG6NZ82dgh-qCWX_2ltUjabEInuecYHqqejAm_6y_T4Uh7qW3qsqULg9cVBs3-ySq1_Cyc3go72lm9u3-tSwTjXCwHAmyfRR3XW1DGo3aUNe-lcYrz3GGIWWqI8uZqLU9JtarR7XoIzTGmBfCecbDq_l_IbtzOxnOiiFIitmVDjyyWzcO-nNwClWr3qaB6-cmK8ZcuU2rVzArkTKDIHX2T0OM7oIn5rdUqyDbyJpmFmQgEjKyqo9kOWdkJ9ompZD7Sx8h71kPJO1aZGEyAsz2y_DWt8Cg-oSs8jS8A5PObB6m1-sO9ON1A5-mDSGEi08QXNSA4Y5T_2diP1jhHl_C7rCEyrnGmbtQr2UlVkmzieBxgDrlwxqh3byuG0RDG8GIX-TtNuOtZCsSgJWMWjVTQJNThlsNZzoPKeJSIfSdnWr-x-WlDgPrRMxM0EPfeClz_9M-3OTle5d_NiOBSxEw1gbNl4KKZ469sTzebwsVlsowaRL-OPL=w2066-h1136-no
