import numpy as np


''' kernel estimate (f_hat) '''
'''
    f_hat(t)
            = 1/nh * Sigma(i=1~n){ K((t - x_i) / h) }                       1-dim

    f_hat(t1, .., tm) *
            = 1/nh * Sigma(i=1~n) { Pi(j=1~k){ K((t_j - x_ji) / h) } }      k-dim (product)
            = 1/nh * Sigma(i=1~n) { K( Norm((t_j - x_i) / h) }              k-dim (norm)

                p-Norm(x) = ( Sigma(j=1~k) {x^p} )^(1/p)

    implement
        1. Point based
            in: vector  (coord of one point)
            out: scalar (value of point)

        2. Space based *
            in: matrix  (coords of many points)
            out: vector (values of points)
'''

# 1. Point based (m-dim, point)
'''
    input
        a : 임의의 데이터,  [ a_1, ..., a_k ]       (k,) -> (1,k) 변환      / k=차원수, n=샘플수

        S : 샘플 데이터들,  [[ s_11, ..., s_1k ]    (n, k)
                         [ s_21, ..., s_2k ]
                         ...
                         [ s_n1, ..., s_nk ]]
        K : 커널 함수
        h : 평활화 계수
    
    output
        f_hat(a) : 추정한 pdf의 한 점 (scalar)
'''
def kernel_estimate_pt(a, S, K, h):
    n, k = S.shape
    a = a[None, k]
    return 1/(n*h) * np.sum(np.prod(K(a - S) / h, axis=0)) # (k,1)-(k,n) = (k,n) =prod=> (n,) =sum=> (scalar)


# 2. Points based (m-dim, points)
'''
    input
        A : 임의의 데이터들,    [[ a_11, ..., a_1k ]    (m,k) -> (m,1,k) 변환       / k=차원수, n=샘플수, m=데이터수
                            [ a_21, ..., a_2k ]
                            ...
                            [ a_m1, ..., a_mk ]]

        S : 샘플 데이터들,      [ s_11, ..., s_1k ]     (n,k) -> (1,n,k) 변환
                            [ s_21, ..., s_2k ]
                            ...
                            [ s_n1, ..., s_nk ]
        K : 커널 함수
        h : 평활화 계수
    
    output
        f_hat(A) : 추정한 pdf의 공간,   f_hat(A) = [f_hat(a1), f_hat(a2), ..., f_hat(al),]   (m,)

    extention
        k=1 : 1차원
        m=1 : 1개 데이터
'''
def kernel_estimate_pts(A, S, K, h):
    n = S.shape[0]
    A = A[:, None, :]
    S = S[None, :, :]
    return 1/(n*h) * np.sum(K(np.sqrt(np.sum(np.square((A - S) / h), axis=-1))), axis=-1)   # radial
    # return 1/(n*h) * np.sum(np.prod(K(A - S) / h, axis=-1), axis=-1)                        # product