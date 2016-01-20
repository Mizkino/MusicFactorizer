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
    # U = U / U.max()
    # for ks in range(K):
    #     H[:,ks] = H[:,ks]/sum(H[:,ks])
    eps = np.finfo(float).eps
#    onemat = np.ones(Y.shape)

    for i in range(0,iter):
        H = H * Y.dot(U.T) / (H.dot(U.dot(U.T)) + eps)
        U = U * H.T.dot(Y) / (H.T.dot(H.dot(U)) + eps)
        U = U / U.max()
    return H,U

def m_fact_euc(Y,K,j,iter,H0,O0):
    tau = 44
    # U=(k,y2) G = (j,tau) O =(k,y2)
    G0 = np.zeros(j,tau)
