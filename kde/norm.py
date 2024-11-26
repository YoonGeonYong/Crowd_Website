import numpy as np


''' norm '''
'''
    L1 :    mahattan/taxicab        sum( |x_i| )                = 절댓값 합
    L2 :    euclidean               root_2( sum( (x_i)^2 ))     = (절댓값) 제곱합 제곱근
    Lp :    minkowski               root_p( sum( |x_i|^p ))     = 절댓값 p제곱합 p제곱근
    L∞ :    chebyshev               max( |x_i| )                = 절댓값 최대값
'''


def l1(x, axis=None): # taxicab or manhattan
    return np.sum(np.abs(x), axis=axis)

def l2(x, axis=None): # euclidean
    return np.sqrt(np.sum(np.square(x), axis=axis))

def lp(x, p, axis=None): # minkowski
    return np.power(np.sum(np.power(np.abs(x), p), axis=axis), 1/p)

def linf(x, axis=None): # chebyshev
    return np.max(np.abs(x), axis=axis)

# additional
def product(x, axis=None):
    return np.prod(x, axis=axis)