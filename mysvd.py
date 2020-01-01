#!/usr/bin/env python
# coding: utf-8

# In[56]:


import numpy as np
import numpy.lib.scimath as csqrt
import scipy 
from sklearn.utils import check_random_state
from scipy import fftpack
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import image
from scipy import sparse
from cmath import log,sqrt
from functools import partial
from sklearn.utils.extmath import safe_sparse_dot
from util import * 

# In[52]:

import numpy as np
# import numpy.linalg as la
# import scipy.linalg as ls 
import matplotlib.pyplot as plt

def subspace_iter(A, Y0, n_iters):
    """Algorithm 4.4: Randomized subspace iteration (p. 244 of Halko et al).
    Uses a numerically stable subspace iteration algorithm to down-weight
    smaller singular values.
    :param A:       (m x n) matrix.
    :param Y0:      Initial approximate range of A.
    :param n_iters: Number of subspace iterations.
    :return:        Orthonormalized approximate range of A after power
                    iterations.
    """
    # Q = ortho_basis(Y0)
    Q, _ = np.linalg.qr(Y0)
    for _ in range(n_iters):
        Z,_ = np.linalg.qr(A.T @ Q)
        Q,_= np.linalg.qr(A @ Z)
    return Q


def find_Q(A, n_samples, n_subspace_iters=None):
    """Algorithm 4.1: Randomized range finder (p. 240 of Halko et al).
    Given a matrix A and a number of samples, computes an orthonormal matrix
    that approximates the range of A.
    :param A:                (m x n) matrix.
    :param n_samples:        Number of Gaussian random samples.
    :param n_subspace_iters: Number of subspace iterations.
    :return:                 Orthonormal basis for approximate range of A.
    """
    m, n = A.shape
    O = np.random.randn(n, n_samples)
    Y = A @ O

    if n_subspace_iters:
        return subspace_iter(A, Y, n_subspace_iters)
    else:
        Q, _ = np.linalg.qr(Y)
        return Q

def rsvd(A,k,p,n_iter):
    print("in rsvd")
    Q = find_Q(A, k, n_iter)
    # print("QSHAPE",Q.shape)
    B=np.dot(Q.T,A)
    # print("Bshape",B.shape)
    UB,sigmaB,V_TB=la.svd(B,full_matrices=False)
    #A=QQTA -> A=(Q UB) sigmaB V_TB
    U_new=np.dot(Q,UB)
    # print("U_newshape",U_new.shape)
    #recon
    A_new=np.dot((U_new*sigmaB),V_TB)
    
    return A_new,U_new[:,:k], sigmaB[:k], V_TB[:k, :]



def csvd(A,k,p,n_iter):
    print("in CSVD")
    A=np.array(A,dtype=np.float64)
    # print(A.dtype)
    l=k+p
    
    # Q=johnson_lindenstrauss(A,l)#gaussian

    
    # Q=fast_johnson_lindenstrauss(A,l)#gaussian
    Q=sparse_johnson_lindenstrauss(A,l)#gaussian

    if n_iter==0:
        Q=Q.T
#     print("Qshape",Q.shape)
#     Q=np.random.randn(A.shape[1],l)
#     Q=np.dot(A,Q).T
#     print("Qshape",Q.shape)
#     Q=sparse_johnson_lindenstrauss(A,l).T#gaussian
    if n_iter>0:
        Q=perform_subspace_iterations(A, Q).T
#         for it in range(n_iter):
#             Q = np.dot(Q.conj().T, A).conj().T

#             (Q, _) = la.lu(Q, permute_l=True)

#             Q = np.dot(A, Q)

#             if it + 1 < n_iter:
#                 (Q, _) = la.lu(Q, permute_l=True)
#             else:
#                 (Q, _) = la.qr(Q, mode='economic')
    # print("Qshape",Q.shape)
    Y=np.dot(Q,A)
    # print("Yshape",Y.shape)
    # UB,sigmaB,V_TB=la.svd(Y,full_matrices=False)
    # U_new=np.dot(UB,Q)
    # A_new=np.dot((U_new.T*sigmaB),V_TB)
    

    # return A_new,U_new,sigmaB,V_TB
    B=np.dot(Y,Y.T)
    # print("Bshape",B.shape)
    B=(B+conjugate_transpose(B))/2
    w,v= la.eigh(B,eigvals_only=False, overwrite_a=True,
                       turbo=True, eigvals=None, type=1, check_finite=False)
    
    # print(w.shape,v[:,:l].shape,Q.shape)


    
    v[:, :A.shape[1]] = v[:,A.shape[1] - 1::-1]
    w = w[::-1]
#     D,T=w[:l],v[:,:l]
    D,T=w[:k],v[:,:k]
#     print("D",D)
    Stilde=csqrt.sqrt(D)
    # print("stilde",Stilde)
    Stilde = np.real(Stilde[np.isreal(Stilde)])
    # print("stilde",Stilde)
    k=len(Stilde)
#     print("newstilde",Stilde)
    T=T[:,:k]
    # print("Tshape",T.shape)
    sinv=la.pinv(np.diag(Stilde))
#     sinv=safeinv(np.diag(Stilde))
    # print("Sinvershape:",sinv.shape)
    Vtilde=np.dot(Y.T,np.dot(T,sinv))
    # print("Vtidleshape:",Vtilde.shape)
    
    Utilde=np.dot(A,Vtilde)#np.dot(np.dot(A,Vtilde),sinv)
    # print("Utilde",Utilde.shape)
    U,S,Q_T=la.svd(Utilde,full_matrices=False)
    V=np.dot(Vtilde,Q_T.T)
    # print(U.shape)
    # print(S.shape)
    # print(V.shape)
    # A_new=np.dot(np.dot(U,np.diag(S)),V.T)
    A_new=np.dot((U*S),V.T)
    # print(A_new.shape)
    
    # print(la.norm(A - A_new, 2))
    return A_new,U, S, V

def normal_SVD(A,k,p,n_iter):
    print("normalsvd")
    U,S,V_T=la.svd(A,full_matrices=False)
    A_new=np.dot((U*S),V_T)
    return A_new,U, S, V_T