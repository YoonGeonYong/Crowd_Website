import numpy as np


# KDE
'''
    kernel desity estimation (KDE)

    kernel estimate (Points based)
    1. kernel
    2. normalization
    3. method
'''
def kernel_density_estimation(A, S, h, kernel, method, normal):
    # data
    m, l = A.shape
    n = S.shape[1]
    A = A.reshape(m, 1, l)
    S = S.reshape(m, n, 1)

    # kernel
    K = None
    if kernel == 'gaussian':
        K = lambda x : (1 / np.sqrt(2 * np.pi)) * np.exp(-np.square(x) / 2)
    elif kernel == 'epanechnikov':
        K = lambda x : np.where(np.abs(x) < np.sqrt(5), 3/(4*np.sqrt(5)) * (1 - (1/5) * np.square(x)), 0)
    elif kernel == 'biweight':
        K = lambda x : np.where(np.abs(x) < 1, (15/16) * np.square(1 - np.square(x)), 0)
    elif kernel == 'triangular':
        K = lambda x : np.where(np.abs(x) < 1, 1 - np.abs(x), 0)
    elif kernel == 'rectangular':
        K = lambda x : np.where(np.abs(x) < 1, 1/2, 0)
    else:
        print('Input correct kernel name')
        return -1
    
    # normalization
    N = None
    if normal == True:
        N = n
    else:
        N = 1
    
    # method
    if method == 'product':
        return 1/(N*h) * np.sum(np.prod(K(A - S) / h, axis=0), axis=0)
    elif method == 'radial':
        return 1/(N*h) * np.sum(K(np.sqrt(np.sum(np.square((A - S) / h), axis=0))), axis=0)
    else:
        print('Input correct method')
        return -1