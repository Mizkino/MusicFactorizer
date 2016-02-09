# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 04:46:40 2016

@author: god
"""
from scipy import fft
from scipy import ifft
import numpy as np
import math


def split_frames(x, w_size, step):
    N = int((len(x) - w_size) / step)
    frames = np.zeros((w_size, N), dtype=int)
    for i in range(N):
        frames[:, i] = x[i * step: i * step + w_size]
    return [frames, N]


def stft(x, w_size, step):
    window = np.hanning(w_size)
    frames, N = split_frames(x, w_size, step)

    spectrogram = np.zeros((w_size, N), dtype=np.complex)
    for i in range(N):
        spectrogram[:, i] = fft(frames[:, i] * window)
    return spectrogram


def istft(spectrogram, w_size, step):
    
    if spectrogram.shape[0] != w_size:
        print ("Mismatch w_size and spectrogram")

    eps = np.finfo(float).eps
    window = np.hanning(w_size)
    # spectrogram.shape = w_size , bins
    spectr_len = spectrogram.shape[1]
    reconst_len = w_size + (spectr_len - 1) * step

    reconst_x = np.zeros(reconst_len, dtype=float)
    windowsum = np.zeros(reconst_len, dtype=float)
    windowsq = window * window

    # Overlap add
    for i in range(0, spectr_len):
        s = i * step
        e = i * step + w_size
        r = ifft(spectrogram[:, i]).real
        # r = abs(ifft(spectrogram[:, i]))

        reconst_x[s:e] += r * window
        windowsum[s:e] += windowsq

    # Normalize by window

    # for i in range(0,reconst_len)
    #     if windowsum[i] > eps
    #         reconst_x[i] /= windowsum[i]
    pos = (windowsum != 0)
    reconst_x[pos] /= windowsum[pos]
    return reconst_x.astype("int16")
