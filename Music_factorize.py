# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 08:10:15 2016

@author: god
"""
#from math import sqrt
import numpy as np

def nmf_euc(Y,K,iter):

    H = np.random.rand(Y.shape[0],K)
    U = np.random.rand(K,Y.shape[1])
    eps = np.finfo(float).eps
#    onemat = np.ones(Y.shape)
    
    for i in range(0,iter):
        H = H * Y.dot(U.T) / (H.dot(U.dot(U.T)) + eps)
        U = U * H.T.dot(Y) / (H.T.dot(H.dot(U)) + eps)  
    return H,U
    

    