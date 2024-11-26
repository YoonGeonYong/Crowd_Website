import numpy as np
from .kernel_function import gaussian, epanechnikov, biweight, triweight, tricube, cosine, logistic, sigmoid, silverman, triangular, rectangular
from .norm import l1, l2, linf, lp, product


# KDE
'''
    kernel desity estimation (KDE)

    kernel estimate (Space based)
    1. kernel
    2. metric
    3. normalization
'''


# valid kernels
KERNELS = {
    'gaussian' : gaussian,
    'epanechnikov' : epanechnikov,
    'biweight' : biweight,
    'triweight' : triweight,
    'tricube' : tricube,
    'cosine' : cosine,
    'logistic' : logistic,
    'sigmoid' : sigmoid,
    'silverman' : silverman,
    'triangular' : triangular,
    'rectangular' : rectangular,
}

# valid metrics
METRICS = {
    'l1' : l1,
    'l2' : l2,
    'linf' : linf,
    'product' : product
}


# KDE
class KernelDensityEstimator():

    # set hyper parameter
    def __init__(self, kernel, metric, normal, h):
        self.kernel = kernel
        self.metric = metric
        self.K = KERNELS[self.kernel]
        self.M = METRICS[self.metric]
        self.h = h
        self.normal = normal
        self.S = None

    # set sample data (S)
    def fit(self, S):
        self.S = S
 
    # set arbitrary data (A), calc KDE
    def score(self, A):
        # normalize
        if self.normal == True:
            n = self.S.shape[0]
            N = n
        else:
            N = 1
        
        # calc
        diff = A[:, None, :] - self.S[None, :, :]

        if self.metric == 'product':
            return 1/(N*self.h) * np.sum( self.M( self.K(diff)/self.h, axis=-1), axis=-1)
        else:
            return 1/(N*self.h) * np.sum( self.K( self.M(diff/self.h, axis=-1) ), axis=-1)




def kernel_density_estimation(A, S, h, kernel, metric, normal):
    K = KERNELS[kernel]
    M = METRICS[metric]

    # normalization
    if normal:
        n = S.shape[0]
        N = n
    else:
        N = 1
    
    # calc
    diff = A[:, None, :] - S[None, :, :]

    if metric == 'product':
        return 1/(N*h) * np.sum( M( K(diff)/h, axis=-1), axis=-1)
    else:
        return 1/(N*h) * np.sum( K( M(diff/h, axis=-1) ), axis=-1)
    

# # kde for crowd point (gaussian, radial, normal x)
# def kernel_density_estimation_cp(A, S, h):
#     diff = A[:, None, :] - S[:, :, None]
#     return 1/h * np.sum((1 / np.sqrt(2 * np.pi)) * np.exp(-np.square(np.sqrt(np.sum(np.square((diff) / h), axis=0))) / 2), axis=0)



