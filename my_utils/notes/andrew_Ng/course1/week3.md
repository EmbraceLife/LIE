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

[image1]: https://lh3.googleusercontent.com/a7C1XrUZCxjqgVE85HgrHvh0sdlfXHt5zrK47eopwIoz3h9tS1AjL-0H0rcxIi2eaivXPIVGV7MFvnLHclltWnrnhr9L4ipS-QZaybmWEPTk71C9e8QFtE2JwFpWEhLuplzOfjFL2L0t9W5eGwoDjXNMFU1kZML1cXjZVz1d0ztFUGqQlI6w0RXObFq8SQGrcxov08XJLpYy1DyfS_kVREAuwKKwBFwzuxeT7VugNn-CwjJlqedV7S2FZPXHnVMIRbac1qxq4nF_NHpb5F200tyDRaoINSc33g0aLgl7qKsRjeKQbCtwOD0zpNbgtvgm-Wg_XRFMqFDtK7BBFQQe1HlMqHY44lTZQyn2832VdezUCTLlXKn0Oyj6lgx0FBmGDKqHqJQCBniv9AFZSCf-zY3M0VWm4o1P4eQodK2CUh2CLovqRMTT_fooiA_D_BTfSWAi_9XDW1gvo4eXy5jr6nZ70iNwex2byAk74qk6AcrNd9iTq02sml32jCUGNmK0PQfRqgW9PePFMkdARG6TqQoq3GxIZ1r4hUtGRf_TJZdkMdZJks0D4Wh7iGBUHruHlfDEpCZ2NUEBEb_fCzhYaz8RoqrxrByLhJt0dKZjN9qfH64lyUJZ=w2014-h1136-no

[image2]: https://lh3.googleusercontent.com/Ev4_Mukf6BBI0qefa6NVOl6Y3qPykyO2gCCKC_S9gk0hw6T94cpjmxfg21lRpBXG7shnHMKhi8D5_KB-Z900H1yBBs6EXkxWhCo7y86zHgA__hZvXxWDWboE9xTnOZ_qaLsP-CiK-N8h42UBhHWLJ8mJk16rvXC2DLHKWBHNnV-LF59SE0TGm1aIadAciC4tT2wOCuAxbjMNa-tEI2IigrJOgA1hrKpHguXNQR3q6lK-Wz1xkpJ6udRHRH-y-IZWZRAbtfJk5pfgfA5O1mDBmnGHQrSvdWKGq_aoig-3ybOatnoOmgXR16qTtGwMRHnAetJmXMxrIR8WnUYa6tylvFKXArwK_9RfGWRJm3uzTngzDXnw6rl7YerVGdt5S4EG52ohJtUL8WdAHASEDruLptNBgjtcgnOYFhWlt1TCXrOrXL_p6LrAg4CIEjw59EBjQRidkNuBZCMQpSJ8kB15fhU2kgKMro0drXsFc_hlG2S91YUNtdLb-H2OafID0iRUKbcxiyIul8Cnxw4Tj_p_xAv2EQyc4zeI3iKXjChhHDWR6U2DSuleuN8mooQmuHrER47MKZxTwW7MOQsPBtgnRLfWxfaGX1PTU9AFTK2HJzHPk5bxIopA=w1914-h994-no

[image3]: https://lh3.googleusercontent.com/q7hDZyGjzi3Dn-6d09amRfcx4PMhosvE3pSapTXfsycov-f2j0D8XapuIWnY4iGwAHX4-odyGNSlW2J5WN-SOpjJ3SmTyWjLHaYRBfCAO8aIlyHobDOYoOnytQ9KODuzLN5PsGZj5K3QpqtrNCItQODlkkc0U370x3Vy0Q6u-kIqvo9XEk-LDaHyMyrMgKb6FZG5r9Nbn6TZ7ONYsJeMec2IGLeqgHvXhzbfuZBUtKMLaT5FCT-NNYt52wRPBL5WXkStCs07yaHaMDna5V69wC9Wysvt1GwPt77i9dZVE_OZL8jMABYnJ6GUkSxNt2-zizqO7Iqa4JYBXlAYcjs_W7E-rhU6hLsVsxsz2Oz1PAw3jg_tkLxxIx7VA4tNi40x81wAkSfET-8GoFEQIiDXWJJYEE-jUSkJGgTUzJKV_L0Fe9M7DngqFSCMAMG486K1PosGUCYSmEZ_-zRH5WnMMC0-ZSoQik06ZcOo5jsAsfHUT6yQp_baGBOAKxTRn57T9zSDLU7ztKUWwWQVsv-651b-ke_S0Qt-57wdXW0NuwD-1ntqAnG6QdeOVwXdahfbpaN7YhOdoOuXQGk99EKidsuRqni320Jtovbw-zvY5vmXOX8y4mFR=w1906-h980-no

[image4]: https://lh3.googleusercontent.com/2rILt7u6oST4LNLONqLQMJZ0U2ltjW5eWKIHr5OrWybpSqC6hGt_g9qURP1lJayYEy5dERxSaxC0P4xYIZLG5UyqvJ8bW0lrFrbSdpDZZxvRq46y2sLD-LtvnKn6TSrOUcW4m4LQAH4zWYhRJ-RFd3hlsm9Dp5dZQ-ZuVvVmRT4-pQolDfQKIHPSnFhcx6r7YTpMpjv4SRRCYxEoWpUFEvl7sX5AjSgHZ_BAiWC6yDg1_KPSMsPkI-C-MfSrHYFJjPgO351AisDaL8RL2b1lFBhSp1qO52nwbObFxEEvpt3FafDXax89fI_P9PfzaTQycFP4tQrJSUgeMhdqNaB4PwosP7RC0IkmbzEDMzXleInyzH337jjZlM7IR_D2AqL5ZQfvPm_3eBBS7jgesP9_C_klkjqy_2zHbHjRPu3yJzFyZIs08x2WXa4c0Dbu5xJrk3beMNaHbHMpo2I5YiF9yqZg7lLpHJ9_CGaXnfCF3YdvM7SRpBbfTGVJZTgVRz-mFPeIKGDJL5bfZjfTQ8GUHA_Mh_bg7WOpRFchUWjPpB4h94JGjloRvxQT0SiwRJT_33IbU1ftY2coiopzR_YHXzD4Jw0omIcTkgPATXkVqqnqTmYzFaES=w2036-h1136-no

[image5]: https://lh3.googleusercontent.com/1x4j2y6-QhtP_ezD5rzrWWz2r1OT-IaVPX0DFgtGBj5C9iYZBCWZVDAwWJRZMZbH4MvTW2GNqLTqhytquPvXiYvs_TKO4eUS92TqEFGbW_TD0duet7yNT7feSS1MPkXQ_cco7iSxUm-JvNRVsh5Bg-WxRLkOx9kNJKWvms3mzKOoQRigtsBdHMxU-rCiD8mn_lxkAiesKxjs1z36kQSGBN9szZPXncDxVlusk2GutwkdlfGEm_4zC8q4frO4ASTlkWI7XezsiOlCG4Hd4fZ5KvDPwph7ZAKTfM_Fs_vlNtTBtyymXLo4iouz5M1stGj3kdcu7LvkoL_iLYc3YBEg2FJeKkafvWjmiw8U-0Z8HknkkPIFLtKebbsXjGP0tnhHw46Bb5nLk3FqCRPjkX2jNRB8_59XhllLtxp1GCRLOH667eMDma_LS7T_m54mwM2Dem5NVJMdNNuk1Az7PiQ6KfbCv8dBb1LiN4AcGQ-_P-e1gfdKElqtUbSXwFT2E3d3kDClSXzdDKC1K6fMlEHdIAxln2VjY2p8KtDIrlLuZvlD_2HzOVOyMTxapwaYqNhTxke9YNcsC5pR5WGhuoDfATYKaOcY37uMMFoSi4MNWrVJid11i_VN=w1796-h912-no

[image6]: https://lh3.googleusercontent.com/AsMK8hpqAH8zIcYlnMKSUR0b61ubw6r-u0Lyn9CRLqmMQyYT8emYpF_PU5Ty_tcs8aRTmLsO28cBLAdK3bNle53onyE0PGljzk9-7p1VPlzYE6lrxP5_ZWoriNZFzAoHSEFO6hww9Qs7WG7dIttRpMyCJ6CifEW3KhsYbsFXFr-Kh8bpsFsPMZuC_q4TiZiNi4YEoxlIaVfaMpbwzEjQxRPNdAmphHxcaXLyu-IpCHE71JCvKF3Dpu7X-z8rq3sbZ-nl6deeUVtjQY9vZPqAlAoFia-Fz1T5i9WKRY2TASyIgg2bSg7CeiqU_rIOrRycNgkn5fyXdCEZLVX93kxhxfO8ZKPfGFWSosA2w6A0HOLnwpZYEiMJ2l2_LaQXWwZJKWLtblleOa_i_YYers4Z1Qn3rTLdKsgPRN9o7EYbI-I_4OD4cB36wsFaQ7Br2YMCQMmKSIpC8aLYVgCoNYG1_t7f3dH0JnBeZvb5_--tE6fESN49gRS6w2u4Do53us-Tf9Sb7XMCsEp1Uj5eTZHr2iJdzjgQWEEcGVRXYmgAJbVlaVqsdu61VWKNdG1d3WMKnA1lhKMxCOC1k2Fh943khGzTU7DUUF_J7V3yfVznHr8hwBNHD5Rq=w1910-h1136-no

[image7]: https://lh3.googleusercontent.com/aVMVShNYuHO0eCdPEuwBgok7g6O9dWvUD0_BWVw0NLAZYnpImjnTWCMnFqSiPqGDAk2KphpXd0E22BMVrsFwo4DRYXtlpUH6hM7PuJ6PDkJUhoqadxgpgWakLtISOVi85b8sYTkzo6EtUzd8IwixaMvRY3z4hVW4HKJr9xU2wmoi1U6wD-rJj8KaKmTH8lBHHqo91D3BSnbeMqeTqSZp4ryXHysARU4dxDK9yFp2tpBIjcancctQG79fKO6_ZlzxMJ1-kzaK4Rw5p_JDUy-ebZ4c-zL_KJ8QhKehPeVbqNfumtoblkKXxtvlGURp7gmAT_Gh2H80LU3zAg7HFRU1pKCij411EXzfQrbz0ZzX6WsFLIsZCvbDahe88VNFZHF9qJGnXr0xTbQPJi0zMFtrSf6dL8tzcWiFttBMxm_qO8QoTsppGyxzRjZZFr1p4zuHarB6msDxMTwWcyMJCe0X-6hlvIwzoCtkDB6NDrs_g-7UPvorV40r33KJysZJbjP6IzNLwiaFtwPro8zUafXmHF6vmEOWwMBFrp90ZIRV2U3afRoetUsGQqAYuxkH0tst8FsIAoCdG7ju84hZDPTH2yyaQfeB7CWB9YzW2LoKZ-emD9-lt4dq=w2084-h1136-no
