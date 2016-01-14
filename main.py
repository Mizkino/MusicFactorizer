# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:12:11 2016

@author: god
"""
import wave
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import func_stft

WINSIZE=256

infile= 'song.wav'
outfile= 'out.wav'

def read_signal(filename, winsize):
    wf = wave.open(filename,'rb')
    fs = wf.getnframes()
    str = wf.readframes(fs)
    params = ((wf.getnchannels(), wf.getsampwidth(),
               wf.getframerate(), wf.getnframes(),
               wf.getcomptype(), wf.getcompname()))
    siglen=((int )(len(str)/2/winsize) + 1) * winsize
    signal=sp.zeros(siglen, sp.int16)
    signal[0:len(str)/2] = sp.fromstring(str,sp.int16)
    return [signal, params]

def get_frame(signal, winsize, no):
    shift=winsize/2
    start=no*shift
    end = start+winsize
    return signal[start:end]

def add_signal(signal, frame, winsize, no ):
    shift=winsize/2
    start=no*shift
    end=start+winsize
    signal[start:end] = signal[start:end] + frame

def write_signal(filename, params ,signal):
    wf=wave.open(filename,'wb')
    wf.setparams(params)
    s=sp.int16(signal).tostring()
    wf.writeframes(s)

def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print ("チャンネル数:", wf.getnchannels())
    print ("サンプル幅:", wf.getsampwidth())
    print ("サンプリング周波数:", wf.getframerate())
    print ("フレーム数:", wf.getnframes())
    print ("パラメータ:", wf.getparams())
    print ("長さ（秒）:", float(wf.getnframes()) / wf.getframerate())

if __name__ == '__main__':
    wf = wave.open("test/test.wav", "rb")
    printWaveInfo(wf)

    buffer = wf.readframes(wf.getnframes())
    print (len(buffer))  # バイト数 = 1フレーム2バイト x フレーム数

    # bufferはバイナリなので2バイトずつ整数（-32768から32767）にまとめる
    data = np.frombuffer(buffer, dtype="int16")

    # プロット
    # plt.plot(data)
    # plt.show()

    spectrogram = func_stft.stft(data, 2048, 1024)

    plt.imshow(abs(spectrogram), aspect = "auto", origin = "lower")
    plt.show()
    
#signal, params = read_signal(infile,WINSIZE)
#nf = len(signal)/(WINSIZE/2) - 1
#sig_out=sp.zeros(len(signal),sp.float32)
#window = sp.hanning(WINSIZE)
#
#for no in xrange(nf):
#    y = get_frame(signal, WINSIZE, no)
#    Y = sp.fft(y*window)
#    # Y = G * Y # 何らかのフィルタ処理
#    y_o = sp.real(sp.ifft(Y))
#    add_signal(sig_out, y_o, WINSIZE, no)
#
#write_signal(outfile, params, sig_out)
