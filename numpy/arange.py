import numpy as np

arange1 = np.arange(start=-5, stop=10, step=1, dtype='int32') # int32
arange2 = np.arange(start=-5, stop=10, step=1., dtype='int32') # int32
arange3 = np.arange(start=-5, stop=10, step=1., dtype='float32')
arange4 = np.arange(start=-5, stop=10, step=.8, dtype='float32') # last value beyond 9

print(arange1)
print(arange2)
print(arange3)
print(arange4)


----
Docstring:
arange([start,] stop[, step,], dtype=None)

Return evenly spaced values within a given interval.

Values are generated within the half-open interval ``[start, stop)``

When using a non-integer step, such as 0.1, the results will often not be consistent.
It is better to use ``linspace`` for these cases.

Parameters
----------
start : number, optional
    Start of interval.  The interval includes this value.  The default
    start value is 0.
stop : number
    End of interval.
	The interval does not include this value,
	except in some cases where `step` is not an integer and floating point round-off affects the length of `out`.
step : number, optional
    Spacing between values, ``out[i+1] - out[i]``.
	The default step size is 1.
	If `step` is specified, `start` must also be given.
dtype : dtype
    The type of the output array.  If `dtype` is not given, infer the data
    type from the other input arguments.

Returns
-------
arange : ndarray
    Array of evenly spaced values.

    length of the result is ``ceil((stop - start)/step)``.

	the last element of `out` being greater than `stop`.



Examples
--------
>>> np.arange(3)
array([0, 1, 2])
>>> np.arange(3.0)
array([ 0.,  1.,  2.])
>>> np.arange(3,7)
array([3, 4, 5, 6])
>>> np.arange(3,7,2)
array([3, 5])
Type:      builtin_function_or_method
