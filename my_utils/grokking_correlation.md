# How to read correlation between input and output through weights?
grokking_deep_learning_correlation

I found chapter 6 of grokking deep learning is not as crystal clear as previous chapters. In order to have a better understanding, I attempted the following example.

I have built a simple neuralNet to imitate a function of doing xor. The neural structure is in the drawing below.     
![](https://lh3.googleusercontent.com/w3dqSSnbXDrQEXHbrfOi93Xh4c7ZJVKKgeaLANf2gv2YI-4qL_7_gEKZyXO4k9gaZ5E0uwJebPWf59vTulZ_YzusSzM756LJVOc9qX9Pi8OsB3wirMo4SZIzgB-1kVH-1VHUHdZYiL32C6XIyfin61NjtaK9NkhQaXGOGX3ph4qxaEft96MnJTbtZdkazdxAk6mYd7TdVkUDS7dZEewvS071_hYQkQpUNnpVLl4KVEOs3JcojmbrwLzlYfQZSpKYi88HZh8pNpWwbRGVnfrsD1Ofky3Lbd_X4XYq0sR-6MNfqIEtAnV_teWbbblptzcK4lLiMMTJT8WThpwHVfeMrtCNbwx06_P4q4sTjLmfVm17AJl-FUMn4iTOojeZz4HUHH5PmTDi6_jCNOpUFhPIdvnz8K8zh3K5CCz9ObRLmxwC86wKKwchvbCuyXbShCZRhI8-zhMAVtE9iWxsaEMZTw6PSgocDOsJ4NZHG2oiZAH6QHZpnycE926Jw5LBA3q7139aiDfHhQ7dMsAyd7JHe1IiMt_BdwnCuD-LvRSdiNB42jGzROjKF1Q1qBBC7hIuJUpQ62BeH2d2cJOe8oSCkSNgdaMGUBTKxSKnUHKNZYFTQt-24oNHeWxg=w1732-h1224-no)

plotting from NeuralNet without relu on hidden layer and no sigmoid on final output     
![](https://lh3.googleusercontent.com/qpMGGdWc61mJ417s4wa0bqIM0RsXMj0FN5VFKkVdV-JrtfSyCwF81swSok70XUR6LOJeu7FnqPWD_pA293j4heurJtV_hkOhLEjO8Y5nNhLNuFYInUs8QwT_S-1o6jUNkS6oEhNTLegzo2zQcLGhbmdmCQOQ6uHSsN-sNHfFyMuOpoFmtCV3nTDC8qzOWok41uUAntlJmN-UxxluGvmDbHfxrJWXyhA9MnMagGMapMcvC3Kr6DXhfBjaaC-ZBAIeUfNC7S85XsxuKO4sCSUkH9sNZMdjzwD3rXZF00MV49cEomI_Rn06Km0SHbDMmJrlxaMCUcx4HToJqvawx-OwvAx_LTYIPOls-K8wpZ1O7t06-6cYA7HpV2_zXtOPo1LpgJ029O1dJ5LxTOaMhm5xWA5k8wOq8rr2FwtvSQ_0wpWVW_Fubr0bOCH6XXbhMquF-RDM33cOofpeHOKaY5bU5H8InkmyqDozOoKZud96SE5D9TRa2qSr0lXuNEmSh4ySrmtu3HiQFuT1ynB5TtswHxIg2qygJlIrx3atb4FSKNCCTDNq5oQHJNpo3e3k0mWcG0u0zUMVkn2TJcRwCe3IN0GYbFxYUUxA8uho9CTjHtnzpe2zV8Ozo6kw=w2452-h1224-no)

```python
print("features:", X)
print("targets:", y)
print("predictions:", model.predict(X))

features:
[[0 0]
 [0 1]
 [1 0]
 [1 1]]
targets: [0 1 1 0]
predictions:
[[ 0.50276363]
 [ 0.50332505]
 [ 0.50126481]
 [ 0.50182629]]
```

plotting from NeuralNet without relu on hidden layer but with sigmoid on final output    
![](https://lh3.googleusercontent.com/Scz7ELOHXOAgKjnE-ciR208UHfy3DECH_jpl9MhpPmLU0ZeuL6Wvq-9nWAsWuh2FEbhSpfOMTS-L2etmv5KDtG-9L3wXwJ6Z81wSKvvtGEmmia_y1Ei0neGfYyUEVyt2OWRO9Lv2ZXY_5EvBYVnXwiu4YpbSoiFZ2nuePnimZQSBVKZIu0jtfM8v9u-UQhtFL4frupwGxF7unaiBHVyIN8JXIVSxdCAESfdreQq42NyFK0veuuKDLfSRKn1r1NeT-M7DEDWdz5J78ZEWd_uiJN9GgjJ7dYNlaMajvKPdwbvIt96mq9qdqbM2cODvqqy2LrR4ZPK8zBmTc7IDOEUtR_P8CcjcQRdOrBW3qXUDup8x7MfUcbUOQuYnQ0G2KawHS4FM5XC3A46IbyMluyrchVIx1lZm31IYCNKz8ouYkynamiHK_PHWCHxUi4zSNW1Awh8CZ9qGaBdMEySR3wn1vugxuWxwGlQYEgLBSTRMt_TEqp_AGU6k6Z9VtO8BHAoloylrGDdnOkvAmxAsnyvbm4tqQxrVYMgnsKsgNK9aMyXDkG00B4fTM1cnmlSmxXW4nu-EPCTCk1nojJZ2mFfAz8oiOoHpWEjD0vuYcsJZ41iLaolVtk4pDbkR=w2416-h1224-no)

```python
print("features:", X)
print("targets:", y)
print("predictions:", model.predict(X))

features:
[[0 0]
 [0 1]
 [1 0]
 [1 1]]
targets: [0 1 1 0]
predictions:
[[ 0.43570971]
 [ 0.43603104]
 [ 0.4360652 ]
 [ 0.43638653]]
```


plotting from NeuralNet with relu on hidden layer but without sigmoid on final output    
![](https://lh3.googleusercontent.com/bo4G958sg89Wh-4MzSj_PGSwMkPfU-ryXu_Ol-FympecCMW-HxP0EolhzG5_i5Gx2rN5yUd_7OoUIU0RIuTMnWFgPpjfIOEbUaZSJ1VAp3nxNt4OM8Cp6nFkDPu3GgORHMRyvaXnE26tBT1JS5JZ7JRmoQBUPo6RooI2Wx7YJTLJVw03PZCQsJx2hNCLTVs7c5fD4C5ubL9SUoINS9_1grUocwoCuhZC5cswOeIXXOgAsz63FjosGJVTwJqmW0aziGvarZ4-9VG-OZhzI5QLEDjVqtTyj8Vg4XB32DVqCryNnOviU5jzBpmil0OI-GgDNeE1-kgW3t3y3IEqmMYAdc2x8Ar6BLggNtyLnMuIA6nhCHcpIQxsz2CKMHiofjogc-QLRbHlY6JOov4KxnxaxziPeT2ZUV8r44UmJ1ym0xA82Vh6cHAjhqR7f2TGy03-BsZn9axRh72TfQzPdOr830ziUwshOOnmVcmBWjbQfOxfnjkzw4DpbV6jWmv7PCk_yMSlYtpU-xIDr1QMTL9HSZCViHToRqkLQ3gBcE_pahypuOiwwoKRC5FS2Fb2rA25wPKkH-xtP7enDhKiBJljMiwv8nFiRT4oPtq_M_Sy0TzEOMEcLuMc06oD=w2410-h1224-no)

```python
print("features:", X)
print("targets:", y)
print("predictions:", model.predict(X))

features:
[[0 0]
 [0 1]
 [1 0]
 [1 1]]
targets: [0 1 1 0]
predictions: [[  7.77083784e-08]
 [  9.99999940e-01]
 [  9.99999940e-01]
 [  1.81037336e-08]]
```


plotting from NeuralNet with relu on hidden layer and with sigmoid on final output    
![](https://lh3.googleusercontent.com/Ljm9YhlPASIew7_R2zm7gFdvA2GiD0-ygu7dG37z1-UxZSXQfQlM2vD4XGxMD6GGnIOAeNhqsDIuwWi0Yvqp3PfsaDCc4vh0n4U8dMSxLQijynban-xfwIA1Y6WPc3iL9yDs822kqk681fK9dBUF_fdScXgba925g7dJCyzSgChQ_MXSB2oil7xqYxikgAHQD9xmKC4b394af0G8c83wyOkpx9WKTAPIFjGcIBSSJSxcxX2BthRhzgUa_QpKghSQInhdcC7mOifwSbsDj83p91tTibDXxbFfXT5LzMMfpkNT3-YRMjYgwOmZlsfQB6lvowQxXs8sM5YRxkcCR8HeYgXthstow-C5NPQuuXSfvgC7uFjowCux103Y4HiTS9Ris17dzHkCfTNCOsDTvoc2llnNx9TNiZtQ_60aSeqTtq0IXbxjsSe7klISPFW1cYssbukefZcCgpu6sC_6agnRVRtI2odDLidqOXKqHbNH9TmU0IND5UKiIdtGIquo3fPYIzV_vSH5fUXeOh9iyNFDWwv0qKCZgCr9t6WbYYO2lTHsMjX6zPi4Hdw0SwAh548OZKkmk7ieQI3W33xpP1BckcdYzQvne0l8LPfjKZqZZ90d9i9UvT-wb4lp=w2412-h1224-no)

```python
print("features:", X)
print("targets:", y)
print("predictions:", model.predict(X))

features: [[0 0]
 [0 1]
 [1 0]
 [1 1]]
targets: [0 1 1 0]
predictions: [[ 0.07140524]
 [ 0.97043109]
 [ 0.97092551]
 [ 0.0201302 ]]

```


## Basic understanding
1. neuralNet is to capture the correlation between inputs and outputs
1. if neuralNet is successful, then the pattern or correlation between inputs and outputs will be stored inside weights
1. usually in real world, it is hard to directly discover correlation between inputs and outputs
1. in this case, we need to build intermediate_outputs between inputs and outputs
1. we use a neuralNet (multiple inputs to multiple outputs) to capture conditional correlation between inputs and intermediate_outputs with the help of non-linear activation function
1. then we can use another neuralNet (multiple input, one output) to easily find direct correlation between intermediate_outputs and final outputs (no non-linear activation function is needed, in this case, sigmoid is used but it is not for discovering correlation, but for nice format)
1. without non-linear activation function to help find conditional correlation between inputs and intermediate_outputs, there won't be direct correlation between intermediate_outputs and final outputs
Are these understanding above correct? if not please correct me

## Questions_A: Are the following statements correct?     
1. based on the plottings, we can see that without relu activation, our neuralNet won't learn much useful, prediction is merely guessing;
2. with relu, we can easily achieve 100% accuracy with about 50 epochs and loss is reduced close to 0;
2. we can say that without relu, our neuralNet does not find correlation between inputs and outputs, not find correlation between intermediate_outputs and final outputs;
2. with relu, conditional correlation is found between our intermediate_outputs and inputs; and direct correlation is found between intermediate_outputs and final outputs;

## Questions_B: How to use weights to tell correlation?
Based on the plotting     
1. how can I tell whether the model is learning or not?
	- if the weight is not going up or down, then we say this weight is not learning, right?
	- weight, either going up or going down, both count as learning, right?
1. how can I tell whether the weights between inputs and intermediate_outputs find no correlation between them?
	- of course, we can tell by observing the loss and accuracy curves, but in the book it seems to suggest we can tell by observe the changes of weights, right?
1. how can I tell whether the weights between inputs and intermediate_outputs find conditional correlation between them?
1. how can I tell whether the weights between intermediate_outputs and final outputs find no correlation between them?
1. how can I tell whether the weights between intermediate_outputs and final outputs find direct correlation between them?

## Questions_C: How to recreate overfitting situations?
- overfitting is model weights remember all samples
- overfitting is model weights learns from all samples and managed to cut loss down to near zero,
- but weights fail to learn the overall pattern, or the weights which corresponding to the most important features of inputs have not really updated enough to capture pattern or correlation between inputs and outputs
- but accidentally the updated weights managed to cut loss very low and stop learning eventually;
- therefore, model weights only remember all training samples not capture pattern behind inputs and outputs, this is overfitting
- we can tell overfitting by comparing training loss and validation loss; if validation loss getting worse later on, but training loss keep improving, then we say model is overfitting
1. we can't tell which input features are important, nor can tell which weights are important, right?
2. how can we avoid overfitting?

## discussion with trask
whether a model is learning or is able to capture the correlation between inputs and outputs or not, can be confirmed by comparing training and validation loss curves.

By contemplating on the book illustrations on pages 111,112,114,115, it makes me wonder does Andrew suggest we can know something useful from observing weights curves. Of course, adding a relu on hidden layer, will change weights from Input layer to hidden layer, but in what does relu change or improve weights, maybe we can see it from weights curves?

weights plotting 1: no relu on hidden layer, no sigmoid on output layer
- obs0: loss stays at 0.25 from 1.5; accuracy stays between 0, 25%, 50%
- obs1: weights between intermediate_outputs and final outputs are 0 most of time;
	- it indicates no matter what intermediate_outputs are, final outputs will stay 0; therefore, there is no correlation between intermediate_outputs and final outputs, right?
- obs2: weights between inputs and intermediate_outputs are all approaching 0
	- no correlation between inputs and intermediate_outputs

weights plotting 2: no relu on hidden layer, with sigmoid on output layer
- obs0: loss stays at 0.255 from 0.275; accuracy stays between 0, 0.25, and 0.5
- obs1:

1. If both loss curves keep approaching 0, then we say this model captures correlation; can we observe this situation from the model's weights?
1. if training loss is approaching 0 but validation loss curve later become worse, we say this is overfitting, this model accidentally updated weights to a state which cut loss near 0 to stop learning but the weights didn’t learn the real correlation; can we observe this situation from the model's weights?
1. if training loss is not approaching 0, or stay on a value up and down a little, we say this model couldn’t move the weights either up or down, so weights are not learning anything. can we observe this situation from the model's weights?
