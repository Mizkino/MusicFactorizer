# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 04:46:40 2016

@author: god
"""
from scipy import fft# , ifft
from scipy import ifft
import numpy as np
import math

def split_frames(x,w_size,step):
    N = int((len(x) - w_size)/step)
    frames = np.zeros((w_size,N),dtype=int)
    for i in range(N):
        frames[:,i] = x[ i*step : i*step+w_size]
    return [frames,N]

def stft(x,w_size,step):
    window = np.hamming(w_size)
    frames,N = split_frames(x,w_size,step)
    
    spectrogram = np.zeros((w_size,N),dtype=np.complex)
    for i in range(N):
        spectrogram[:,i] = fft(frames[:,i] * window)
    return spectrogram
