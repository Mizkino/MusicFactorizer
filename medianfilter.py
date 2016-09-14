import numpy as np
import wave
import sys
import os
import pyaudio
from scipy.signal import medfilt
from func_stft import stft, istft


def medianfilter(S, axis=0, length=15 ):
    # axis : 0 横方向, 1 縦方向
    w = ( (1,length)[axis] , (length,1)[axis] )
    S2 = medfilt(S,w)
    return S2

def open_file(self):
    input_fname = "test/test.wav"
    # input_fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
    wf = wave.open(input_fname, "rb")
    printWaveInfo(wf)
    SD.ps = wf.getparams()
    buffer = wf.readframes(wf.getnframes())
    SD.data = np.frombuffer(buffer, dtype="int16")

def make_spectrogram(self):

    wsize
    # global stft_step = self.step
    Org_spectrogram = stft(SD.data[self.offset:self.length + 1], self.wsize, self.step)
    global Ymean
    Ymean = np.mean(np.abs(self.spectrogram))
    self.spw.disp_spectrogram(self.spectrogram)
    self.spw.show()


def printWaveInfo(wf):
    print ("Channels:", wf.getnchannels())
    print ("Sampwidth:", wf.getsampwidth())
    print ("Framerate:", type(wf.getframerate()))
    print ("Frames:", type(wf.getnframes()))


def save_as_wave(filename, parameters, mdata):
    write_wave = wave.Wave_write(filename)
    write_wave.setparams(parameters)
    write_wave.writeframes(mdata.tostring())
    write_wave.close()

"""
main
"""
if len(sys.argv) > 2:
    l = int(sys.argv[2])
elif len(sys.argv) < 1:
    print("SrcFileName is Needed")
    sys.exit()
filens = (sys.argv[1]).split("/")
fname = filens[-1][:-4]
filepath = "test"+str(l)

if (not os.path.exists(filepath)) or (not os.path.isdir(filepath)):
    os.mkdir(filepath)



wsize = 2048
step = 1024
input_fname = sys.argv[1]
# input_fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
wf = wave.open(input_fname, "rb")
printWaveInfo(wf)
paras = wf.getparams()
buffer = wf.readframes(wf.getnframes())
mdata = np.frombuffer(buffer, dtype="int16")

Org_spectr = stft(mdata, wsize, step)
phase = np.angle(Org_spectr)

print("median1")
Ht = medianfilter(abs(Org_spectr),0,l)
recon1 = istft( Ht * np.exp(0 + 1j) * phase, wsize, step)

save_as_wave(filepath+"/"+fname+"median1.wav",paras,recon1)

print("median2")
Hf = medianfilter(abs(Org_spectr),1,l)
recon2 = istft( Hf * np.exp(0 + 1j) * phase, wsize, step)

save_as_wave(filepath+"/"+fname+"median2.wav",paras,recon2)

print("wener")

MH = (Ht*Ht)/((Ht*Ht)+(Hf*Hf))
MP = (Hf*Hf)/((Ht*Ht)+(Hf*Hf))

recon3 = istft( Org_spectr * MH, wsize, step)
save_as_wave(filepath+"/"+fname+"2wener1.wav",paras,recon3)

recon4 = istft( Org_spectr * MP, wsize, step)
save_as_wave(filepath+"/"+fname+"2wener2.wav",paras,recon4)


MH = Ht/(Ht+Hf)
MP = Hf/(Ht+Hf)

recon5 = istft( Org_spectr * MH, wsize, step)
save_as_wave(filepath+"/"+fname+"1wener1.wav",paras,recon5)

recon6 = istft( Org_spectr * MP, wsize, step)
save_as_wave(filepath+"/"+fname+"1wener2.wav",paras,recon6)
