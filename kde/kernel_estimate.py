import numpy as np


''' kernel estimate (f_hat) '''
'''
    f_hat(t)
            = 1/nh * Sigma(i=1~n){ K((t - x_i) / h) }                                   1-dim

    f_hat(t1, .., tm) *
            = 1/nh * Sigma(i=1~n) { Pi(j=1~m){ K((t_j - x_ji) / h) } }                  m-dim (product)
            = 1/nh * Sigma(i=1~n) { K( sqrt( Sigma(j=1~m){ ((t_j - x_ji) / h)^2 } ) }   m-dim (radial)

    implement
        1. Point based
            in: vector (indexes of point)
            out: scalar (value of point)

        2. Points(space) based *
            in: matrix (indexes of points)
            out: vector (values of points)
'''

# 1. Point based (m-dim, point)
'''
    input
        a : 임의의 데이터,  [ a_1, a_2, ..., a_m ]      (m,) -> (m,1) 변환      / m=차원수, n=샘플수

        S : 샘플 데이터들,  [[ s_11, s_12, ..., s_1n ]  (m,n)
                         [ s_21, s_22, ..., s_2n ]
                         ...
                         [ s_m1, s_m2, ..., s_mn ]]
        K : 커널 함수
        h : 평활화 계수
    
    output
        f_hat(a) : 추정한 pdf,   f_hat(a)   (scalar)
'''
def kernel_estimate_pt(a, S, K, h):
    m, n = S.shape
    a = a.reshape(m, 1)
    return 1/(n*h) * np.sum(np.prod(K(a - S) / h, axis=0)) # (m,1)-(m,n) = (m,n) =prod=> (n,) =sum=> (scalar)


# 2. Points based (m-dim, points)
'''
    input
        A : 임의의 데이터들,    [[ a_11, a_12, ..., a_1l ]      (m,l) -> (m,1,l) 변환       / m=차원수, n=샘플수, l=데이터수
                            [ a_21, a_22, ..., a_2l ]
                            ...
                            [ a_m1, a_m2, ..., a_ml ]]

        S : 샘플 데이터들,      [ s_11, s_12, ..., s_1n ]       (m,n) -> (m,n,1) 변환
                            [ s_21, s_22, ..., s_2n ]
                            ...
                            [ s_m1, s_m2, ..., s_mn ]
        K : 커널 함수
        h : 평활화 계수
    
    output
        f_hat(A) : 추정한 pdf,   f_hat(A) = [f_hat(a1), f_hat(a2), ..., f_hat(al),]   (l,)

    extention
        m=1 : 1차원
        l=1 : 1개 데이터
'''
def kernel_estimate_pts(A, S, K, h):
    m, l = A.shape
    n = S.shape[1]
    A = A.reshape(m, 1, l)
    S = S.reshape(m, n, 1)
    return 1/(n*h) * np.sum(K(np.sqrt(np.sum(np.square((A - S) / h), axis=0))), axis=0)   # radial
    # return 1/(n*h) * np.sum(np.prod(K(A - S) / h, axis=0), axis=0)                        # product