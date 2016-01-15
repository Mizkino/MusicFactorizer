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

def printWaveInfo(wf):
    """WAVEファイルの情報を取得"""
    print ("チャンネル数:", wf.getnchannels())
    print ("サンプル幅:", wf.getsampwidth())
    print ("サンプリング周波数:", wf.getframerate())
    print ("フレーム数:", wf.getnframes())
    print ("パラメータ:", wf.getparams())
    print ("長さ（秒）:", float(wf.getnframes()) / wf.getframerate())

if __name__ == '__main__':
    # set params
    input_fname = "test/test.wav"
    output_name = "out.wav"
    stft_wsize = 2048
    stft_step = 1024

    wf = wave.open("test/test.wav", "rb")
    printWaveInfo(wf)
    ps = wf.getparams()
    buffer = wf.readframes(wf.getnframes())
    # bufferはバイナリなので2バイトずつ整数（-32768から32767）にまとめる
    data = np.frombuffer(buffer, dtype="int16")
    wf.close()

    spectrogram = func_stft.stft(data, stft_wsize, stft_step)
    # spectrogram = func_stft.stft_hitono(data, np.hanning(stft_wsize), stft_step)

    # プロット
    # plt.plot(data)
    # plt.show()
    #    pxx, freqs, bins, im = plt.specgram(data,NFFT=2048,Fs=wf.getframerate(),noverlap=0,window=np.hamming(2048))
    #    plt.axis([0, float(wf.getnframes()) / wf.getframerate(), 0, wf.getframerate() / 2])
    # plt.imshow(abs(spectrogram[:stft_wsize/2,:]), aspect = "auto", origin = "lower")
    # plt.show()
    #
    # Y = np.abs(spectrogram)
    # phase = np.angle(spectrogram)

    reconst = func_stft.istft(spectrogram,stft_wsize,stft_step)
    # reconst = func_stft.istft_hitono(spectrogram,np.hanning(stft_wsize),stft_step)
    # print ("recon.shape=",reconst.shape,"      reconst.dtype=",reconst.dtype)
    reconst = reconst.astype("int16")
    # plt.subplot(211)
    # plt.plot(data)
    # plt.subplot(212)
    # plt.plot(reconst)
    # plt.show()
    # wavfile.write(output_name,fs,reconst)
    w = wave.Wave_write(output_name)
    w.setparams((
        ps.nchannels,                        # channel
        ps.sampwidth,                        # byte width
        ps.framerate,                    # sampling rate
        len(reconst),                 # number of frames
        ps.comptype, ps.compname  # no compression
    ))
    w.writeframes(reconst.tostring())
    w.close()
