# How to read correlation between input and output through weights?
I have built a simple neuralNet to build a model to do xor. The neural structure is in the drawing below.     
![](https://lh3.googleusercontent.com/w3dqSSnbXDrQEXHbrfOi93Xh4c7ZJVKKgeaLANf2gv2YI-4qL_7_gEKZyXO4k9gaZ5E0uwJebPWf59vTulZ_YzusSzM756LJVOc9qX9Pi8OsB3wirMo4SZIzgB-1kVH-1VHUHdZYiL32C6XIyfin61NjtaK9NkhQaXGOGX3ph4qxaEft96MnJTbtZdkazdxAk6mYd7TdVkUDS7dZEewvS071_hYQkQpUNnpVLl4KVEOs3JcojmbrwLzlYfQZSpKYi88HZh8pNpWwbRGVnfrsD1Ofky3Lbd_X4XYq0sR-6MNfqIEtAnV_teWbbblptzcK4lLiMMTJT8WThpwHVfeMrtCNbwx06_P4q4sTjLmfVm17AJl-FUMn4iTOojeZz4HUHH5PmTDi6_jCNOpUFhPIdvnz8K8zh3K5CCz9ObRLmxwC86wKKwchvbCuyXbShCZRhI8-zhMAVtE9iWxsaEMZTw6PSgocDOsJ4NZHG2oiZAH6QHZpnycE926Jw5LBA3q7139aiDfHhQ7dMsAyd7JHe1IiMt_BdwnCuD-LvRSdiNB42jGzROjKF1Q1qBBC7hIuJUpQ62BeH2d2cJOe8oSCkSNgdaMGUBTKxSKnUHKNZYFTQt-24oNHeWxg=w1732-h1224-no)

plotting from NeuralNet without relu on hidden layer and no sigmoid on final output     
![](https://lh3.googleusercontent.com/qpMGGdWc61mJ417s4wa0bqIM0RsXMj0FN5VFKkVdV-JrtfSyCwF81swSok70XUR6LOJeu7FnqPWD_pA293j4heurJtV_hkOhLEjO8Y5nNhLNuFYInUs8QwT_S-1o6jUNkS6oEhNTLegzo2zQcLGhbmdmCQOQ6uHSsN-sNHfFyMuOpoFmtCV3nTDC8qzOWok41uUAntlJmN-UxxluGvmDbHfxrJWXyhA9MnMagGMapMcvC3Kr6DXhfBjaaC-ZBAIeUfNC7S85XsxuKO4sCSUkH9sNZMdjzwD3rXZF00MV49cEomI_Rn06Km0SHbDMmJrlxaMCUcx4HToJqvawx-OwvAx_LTYIPOls-K8wpZ1O7t06-6cYA7HpV2_zXtOPo1LpgJ029O1dJ5LxTOaMhm5xWA5k8wOq8rr2FwtvSQ_0wpWVW_Fubr0bOCH6XXbhMquF-RDM33cOofpeHOKaY5bU5H8InkmyqDozOoKZud96SE5D9TRa2qSr0lXuNEmSh4ySrmtu3HiQFuT1ynB5TtswHxIg2qygJlIrx3atb4FSKNCCTDNq5oQHJNpo3e3k0mWcG0u0zUMVkn2TJcRwCe3IN0GYbFxYUUxA8uho9CTjHtnzpe2zV8Ozo6kw=w2452-h1224-no)

plotting from NeuralNet without relu on hidden layer but with sigmoid on final output    
![](https://lh3.googleusercontent.com/Scz7ELOHXOAgKjnE-ciR208UHfy3DECH_jpl9MhpPmLU0ZeuL6Wvq-9nWAsWuh2FEbhSpfOMTS-L2etmv5KDtG-9L3wXwJ6Z81wSKvvtGEmmia_y1Ei0neGfYyUEVyt2OWRO9Lv2ZXY_5EvBYVnXwiu4YpbSoiFZ2nuePnimZQSBVKZIu0jtfM8v9u-UQhtFL4frupwGxF7unaiBHVyIN8JXIVSxdCAESfdreQq42NyFK0veuuKDLfSRKn1r1NeT-M7DEDWdz5J78ZEWd_uiJN9GgjJ7dYNlaMajvKPdwbvIt96mq9qdqbM2cODvqqy2LrR4ZPK8zBmTc7IDOEUtR_P8CcjcQRdOrBW3qXUDup8x7MfUcbUOQuYnQ0G2KawHS4FM5XC3A46IbyMluyrchVIx1lZm31IYCNKz8ouYkynamiHK_PHWCHxUi4zSNW1Awh8CZ9qGaBdMEySR3wn1vugxuWxwGlQYEgLBSTRMt_TEqp_AGU6k6Z9VtO8BHAoloylrGDdnOkvAmxAsnyvbm4tqQxrVYMgnsKsgNK9aMyXDkG00B4fTM1cnmlSmxXW4nu-EPCTCk1nojJZ2mFfAz8oiOoHpWEjD0vuYcsJZ41iLaolVtk4pDbkR=w2416-h1224-no)

plotting from NeuralNet with relu on hidden layer but without sigmoid on final output    
![](https://lh3.googleusercontent.com/bo4G958sg89Wh-4MzSj_PGSwMkPfU-ryXu_Ol-FympecCMW-HxP0EolhzG5_i5Gx2rN5yUd_7OoUIU0RIuTMnWFgPpjfIOEbUaZSJ1VAp3nxNt4OM8Cp6nFkDPu3GgORHMRyvaXnE26tBT1JS5JZ7JRmoQBUPo6RooI2Wx7YJTLJVw03PZCQsJx2hNCLTVs7c5fD4C5ubL9SUoINS9_1grUocwoCuhZC5cswOeIXXOgAsz63FjosGJVTwJqmW0aziGvarZ4-9VG-OZhzI5QLEDjVqtTyj8Vg4XB32DVqCryNnOviU5jzBpmil0OI-GgDNeE1-kgW3t3y3IEqmMYAdc2x8Ar6BLggNtyLnMuIA6nhCHcpIQxsz2CKMHiofjogc-QLRbHlY6JOov4KxnxaxziPeT2ZUV8r44UmJ1ym0xA82Vh6cHAjhqR7f2TGy03-BsZn9axRh72TfQzPdOr830ziUwshOOnmVcmBWjbQfOxfnjkzw4DpbV6jWmv7PCk_yMSlYtpU-xIDr1QMTL9HSZCViHToRqkLQ3gBcE_pahypuOiwwoKRC5FS2Fb2rA25wPKkH-xtP7enDhKiBJljMiwv8nFiRT4oPtq_M_Sy0TzEOMEcLuMc06oD=w2410-h1224-no)

plotting from NeuralNet with relu on hidden layer and with sigmoid on final output    
![](https://lh3.googleusercontent.com/r9xx3zaHzxD87-cBAsU2-U9wserUHsr2QqBsUHyNcVnsOnk7NT2VGuaGF2Qmt7Ghua8H30JD8macb6ZqUTXAFxQnWk5wJQClyacPU1AfMQUJxmFhYgixAbRp4Z3mqVNM4tiUrJhveXtW-hl2RF6IphbkYYbfkaUWJNTiTTpGl4dmDidcVuNIezBa6GDGMt9wmXCGaVWYpGUjVaCygSDC1hdapzk_EKRcet51Oq13pC6YyEIHRvW69PvTx-E29Qcs2mSWSxyu71dOLEHG6R5APcYzqFtWCwP3qb62EiSUfDZhaXCG2OkdCR3TtPzBX_NwVL27p1V-mlo0W1io4ABIbHmhhO0LDIB5tsFP4nqij6Kpot-ase4P2yn3ZlYB8SqScXKmpAJlgdjcOUsvuzqN_oyyPCDuVx9la1CZP2FDIOi72Efn3TQysbJq_26ijLd9t4_9G6F869VYlJypvt2EqFg60vQybHy_qKJ6kcEgfqGxkHBy5hdDPKs7aAChZZmkL7-mjmxmj38QabV2nb1PYgJt7kqCZ0tGnh4n9Oj-YaEcH0Gd7aXx-V7QVoeabgpihKAujqe183AekIW0oA061Sa6gt6z4_m2i3amh9ZQq6tUXorpBCecj-l5=w2428-h1224-no)

## Basic understanding
- neuralNet is to capture the correlation between inputs and outputs
- if neuralNet is successful, then the pattern or correlation between inputs and outputs will be stored inside weights
- usually in real world, it is hard to directly discover correlation between inputs and outputs
- in this case, we need to build intermediate_outputs between inputs and outputs
- we use a neuralNet (multiple inputs to multiple outputs) to capture conditional correlation between inputs and intermediate_outputs with the help of non-linear activation function
- then we can use another neuralNet (multiple input, one output) to easily find direct correlation between intermediate_outputs and final outputs (no non-linear activation function is needed, in this case, sigmoid is used but it is not for discovering correlation, but for nice format)
- without non-linear activation function to help find conditional correlation between inputs and intermediate_outputs, there won't be direct correlation between intermediate_outputs and final outputs

## Questions: Are the following statements correct?     
- based on the plottings, we can see that without relu activation, our neuralNet won't learn much useful, prediction is merely guessing;
- with relu, we can easily achieve 100% accuracy with about 50 epochs and loss is reduced close to 0;
- we can say that without relu, our neuralNet does not find correlation between inputs and outputs, not find correlation between intermediate_outputs and final outputs;
- with relu, conditional correlation is found between our intermediate_outputs and inputs; and direct correlation is found between intermediate_outputs and final outputs;

## Questions: How to use weights to tell correlation?
Based on the plotting     
- how can I tell whether the weights between inputs and intermediate_outputs find no correlation between them?
- how can I tell whether the weights between inputs and intermediate_outputs find conditional correlation between them?
- how can I tell whether the weights between intermediate_outputs find no correlation between them?
- how can I tell whether the weights between intermediate_outputs find direct correlation between them?
