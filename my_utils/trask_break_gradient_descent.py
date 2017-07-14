"""
gradient_descent
loss_raw_error_weight_delta_alpha
slope_derivative_weight_delta
divergence
learning_rate_alpha

## what we have
- there is just a single input: 0.5
- initial weight: 0.5
- true target: 0.8
- alpha: 1
- loss_function: mse

## what to do
- based on this single sample, move up and down weights based on gradient_descent calculated direction and amount, to update weights, then apply to the same input-output, for 20 times

## what to see
- given input, output fixed, when weight initialized differently
"""

weight= 0.5
goal_pred = 0.8
# input is normal size, alpha is 1
input = 0.5
alpha = 1
# # when input is enlarged, alpha is 1
# input = 2
# alpha = 1
# # when inputis enlarged and alpha is shrinked
# input = 2
# alpha = 0.1

for iteration in range(20):
	prediction = input * weight
	error = (prediction - goal_pred) ** 2
	delta = prediction - goal_pred
	weight_delta = input * delta
	weight = weight - weight_delta * alpha
	print('prediction: %02f, error: %02f, delta: %02f, weight_delta: %02f, weight: %02f' % (prediction, error, delta, weight_delta, weight))

# compare the print output, we can see the reason why all values explode
