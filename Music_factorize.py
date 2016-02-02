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
    for ks in range(K):
        H[:,ks] = H[:,ks]/np.sum(H[:,ks])
    eps = np.finfo(float).eps
#    onemat = np.ones(Y.shape)

    for i in range(0,iter):
        H = H * Y.dot(U.T) / (H.dot(U.dot(U.T)) + eps)
        U = U * H.T.dot(Y) / (H.T.dot(H.dot(U)) + eps)
        # U = U / U.max()
        for ks in range(K):
            H[:,ks] = H[:,ks]/np.sum(H[:,ks])
    return H,U

def m_fact_euc(Y,K,envs,iter,H,U):
    tau = 40
    # U=(k,y2) G = (j,tau) O =(k,y2)
    G = np.zeros((envs,tau))
    O = np.zeros((U.shape[0],U.shape[1],envs))
    X = np.zeros(Y.shape)
    eps = np.finfo(float).eps

    for js in range(envs):
        ranj = random.rand()
        taus = range(G.shape[1])
        G[js,:] = np.exp((-1)*ranj*taus)
        G[js,:] = G[js,:]/np.sum(G[js,:])
        O[:,:,js] = U

    U = calc_U(U,G,O,K,U.shape[1],tau,envs)
    X = H.dot(U)
    for ks in range(K):
        H[:,ks] = H[:,ks]/np.sum(H[:,ks])

    for it in range(iter):
        H = H * Y.dot(U.T) / (H.dot(U.dot(U.T)) + eps)
        for ks in range(K):
            H[:,ks] = H[:,ks]/np.sum(H[:,ks])
        U = calc_U(U,G,O,K,U.shape[1],tau,envs)
        X = H.dot(U)


def calc_U(U,G,O,K,t2,tau,envs):
    for ks in range(K):
        for ts in range(t2):
            U[ks,ts] = np.sum(np.sum(G[:,0:min(ts,tau)+1].T*O[ks,slice(ts,max(0,ts-tau)-1,-1),:]))
    return U
