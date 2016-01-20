# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:12:11 2016

@author: god
"""
import sys
import wave
import numpy as np
import scipy as sp
# import pyqtgraph as pqg
import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
# import PySide.QtCore as QtCore
# import PySide.QtGui as QtGui

import matplotlib
import matplotlib.pyplot as plt
# import the Qt4Agg FigureCanvas object, that binds Figure to

# Qt4Agg backend. It also inherits from QWidget
#from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
# import matplotlib.backends.backend_qt4agg
# import matplotlib.backends.backend_agg
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# local
from func_stft import stft, istft
import music_factorize

# set params
input_fname = "test/test.wav"
output_name = "out.wav"
stft_wsize = 2048
stft_step = 1024
fs = 0
K = 10
max_iter = 100


class SoundData():

    def __init__(self):
        ps = []
        data = []


class BarPlot():

    def __init__(self, parent=None, width=8, height=6):
        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        self.fig = Figure((width, height), dpi=100)
        self.canvas = FigureCanvas(self.fig)  # pass a figure to the canvas
        self.canvas.setParent(parent)
        self.axes = self.fig.add_subplot(111)

    def draw_wave(self, data, offs, leng):
        self.axes.clear()
        self.axes.axis('off')
        print("datashape = ",data.shape)
        self.axes.plot(data[offs:leng + 1])
        self.canvas.draw()

    def draw_spectrogram(self, spectrogram, fmax, fmin):
        self.axes.clear()
        maxbin = (int)(fmax / 44100 * 2048)
        minbin = (int)(fmin / 44100 * 2048)

        sp = self.axes.imshow(abs(spectrogram[minbin:maxbin, :]), aspect="auto", origin="lower")
        self.fig.subplots_adjust(right=0.8)
        cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        self.fig.colorbar(sp, cax=cbar_ax)
        self.canvas.draw()

    def draw_vert_h(self, H1, fmax, fmin):
        self.axes.clear()
        self.axes.axis('off')
        maxbin = (int)(fmax / 44100 * 2048)
        minbin = (int)(fmin / 44100 * 2048)
        print("H1shape = ",H1.shape)
        x = range(minbin,maxbin)
        self.axes.barh(H1[minbin:maxbin],x)
        self.canvas.draw()

class MusicFactorWindow(QtGui.QWidget):

    def __init__(self,K):
        # init
        super(MusicFactorWindow, self).__init__()
        # instant var
        self.fmax = 22050
        self.fmin = 0
        self.K = K
        self.Hbar = []
        self.Ubar = []
        self.H = []
        self.U = []
        self.phase = []
        self.design_mfw(K)

    def design_mfw(self,K):

        self.main_frame = QtGui.QWidget()
        self.spbar = BarPlot(self, width=6, height=6)
        for ks in range(self.K):
            Hbar = BarPlot(self, width=6, height=float(6/self.K))
            Ubar = BarPlot(self, width=6, height=float(6/self.K))
            self.Hbar.append(Hbar)
            self.Ubar.append(Ubar)

        # set option slider and button etc
        self.maxs = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.maxs.setMinimum(1)
        self.maxs.setMaximum(100)
        self.maxs.setValue(50)
        self.maxs.valueChanged.connect(self.get_fmax)
        self.mins = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.mins.setMinimum(0)
        self.mins.setMaximum(100)
        self.mins.setValue(0)
        self.mins.valueChanged.connect(self.get_fmin)

        # H and U Layout
        hsbox = QtGui.QHBoxLayout()
        usbox = QtGui.QVBoxLayout()
        for ks in range(self.K):
            hsbox.addWidget(self.Hbar[ks].canvas)
            usbox.addWidget(self.Ubar[ks].canvas)

        # H,U and spectrogram Layout
        grid = QtGui.QGridLayout()
        grid.addLayout(hsbox, 0, 1)
        grid.addLayout(usbox, 1, 0)
        grid.addWidget(self.spbar.canvas,1,1)

        # bars layout
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.mins)
        hbox.addWidget(self.maxs)

        # set layout all
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addLayout(grid)  # add canvas to the layout
        self.main_frame.setLayout(vbox)
        # set widget
        # self.addWidget(self.main_frame)
        self.setLayout(vbox)

    def disp_musicfactor(self,H,U,phase):
        self.H = H
        self.U = U
        self.phase = phase
        self.K = H.shape[1]
        self.design_mfw(self.K)
        # self.spbar.draw_spectrogram(H.dot(U),self.fmax, self.fmin)
        self.disp_hs(self.K)
        self.disp_us(self.K)

    def get_fmax(self, value):
        self.fmax = 200 * value
        self.spbar.draw_spectrogram(self.H.dot(self.U), self.fmax, self.fmin)

    def get_fmin(self, value):
        self.fmin = 5 * value
        self.spbar.draw_spectrogram(self.H.dot(self.U), self.fmax, self.fmin)

    def disp_hs(self,K):
        for ks in range(K):
            self.Hbar[ks].draw_vert_h(self.H[:,ks],self.fmax, self.fmin)

    def disp_us(self,K):
        length = len(self.U[0,:])
        print("length = ",length)
        for ks in range(K):
            self.Ubar[ks].draw_wave(self.U[ks,:],0,length)


class SpectrogramWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(SpectrogramWindow, self).__init__()
        # instant var
        self.fmax = 22050
        self.fmin = 0
        # self.main_frame = QtGui.QWidget()
        self.barplot = BarPlot(self, width=6, height=6)
        self.spectrogram = []

        # set option slider and button etc
        self.maxs = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.maxs.setMinimum(1)
        self.maxs.setMaximum(100)
        self.maxs.setValue(50)
        self.maxs.valueChanged.connect(self.get_fmax)
        self.mins = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.mins.setMinimum(0)
        self.mins.setMaximum(100)
        self.mins.setValue(0)
        self.mins.valueChanged.connect(self.get_fmin)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.mins)
        hbox.addWidget(self.maxs)
        # set layout
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.barplot.canvas)  # add canvas to the layout
        # self.main_frame.setLayout(vbox)
        # set widget
        # self.addWidget(self.main_frame)
        self.setLayout(vbox)

    def disp_spectrogram(self, spectrogram):
        self.spectrogram = spectrogram
        self.barplot.draw_spectrogram(spectrogram, self.fmax, self.fmin)

    def get_fmax(self, value):
        self.fmax = 200 * value
        self.barplot.draw_spectrogram(self.spectrogram, self.fmax, self.fmin)

    def get_fmin(self, value):
        self.fmin = 5 * value
        self.barplot.draw_spectrogram(self.spectrogram, self.fmax, self.fmin)


class WaveformWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(WaveformWindow, self).__init__()
        # self.main_frame = QtGui.QWidget()
        self.barplot = BarPlot(self, width=8, height=1.5)
        # set layout
        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.barplot.canvas)  # add canvas to the layout
        # self.main_frame.setLayout(vbox)
        # set widget
        # self.addWidget(self.main_frame)
        self.setLayout(vbox)

    def disp_wave(self, sd, offs, leng):
        self.barplot.draw_wave(sd.data, offs, leng)


class ApplicationWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(ApplicationWindow, self).__init__()

        # instant var
        self.length = 0
        self.offset = 0
        self.sd = SoundData()
        self.wsize = 2048
        self.step = 1024
        self.spectrogram = []
        self.K = 10
        self.envs = 5
        self.iter = 100

        # open_file and params row
        open_horizen = QtGui.QHBoxLayout()
        self.fileb = QtGui.QPushButton('Open File', self)
        self.fileb.clicked.connect(self.open_file)
        self.sampling_l = QtGui.QLabel(self)
        self.sampling_l.setText('SamplRate:')
        self.sampling_p = QtGui.QLabel(self)
        self.frames_l = QtGui.QLabel(self)
        self.frames_l.setText('Hz         Frames:')
        self.frames_p = QtGui.QLabel(self)

        open_horizen.addWidget(self.fileb)
        open_horizen.addWidget(self.sampling_l)
        open_horizen.addWidget(self.sampling_p)
        open_horizen.addWidget(self.frames_l)
        open_horizen.addWidget(self.frames_p)

        # waveform row
        wave_horizen = QtGui.QHBoxLayout()
        self.waveb = QtGui.QPushButton('Waveform', self)
        self.waveb.clicked.connect(self.make_waveform)
        self.offset_l = QtGui.QLabel(self)
        self.offset_l.setText('Offset:')
        self.offset_t = QtGui.QLineEdit()
        self.length_l = QtGui.QLabel(self)
        self.length_l.setText('Length:')
        self.length_t = QtGui.QLineEdit()

        wave_horizen.addWidget(self.waveb)
        wave_horizen.addWidget(self.offset_l)
        wave_horizen.addWidget(self.offset_t)
        wave_horizen.addWidget(self.length_l)
        wave_horizen.addWidget(self.length_t)

        # spectrogram row
        spectrogram_horizen = QtGui.QHBoxLayout()
        self.specb = QtGui.QPushButton('Spectrogram', self)
        self.specb.clicked.connect(self.make_spectrogram)
        self.stftw_l = QtGui.QLabel(self)
        self.stftw_l.setText('FFT_frame:')
        self.stftw_t = QtGui.QLineEdit()
        self.stftw_t.setText(str(self.wsize))
        self.stfts_l = QtGui.QLabel(self)
        self.stfts_l.setText('FFT_step:')
        self.stfts_t = QtGui.QLineEdit()
        self.stfts_t.setText(str(self.step))

        spectrogram_horizen.addWidget(self.specb)
        spectrogram_horizen.addWidget(self.stftw_l)
        spectrogram_horizen.addWidget(self.stftw_t)
        spectrogram_horizen.addWidget(self.stfts_l)
        spectrogram_horizen.addWidget(self.stfts_t)

        # spectrofactor row
        specfact_horizen = QtGui.QHBoxLayout()
        self.sfacb = QtGui.QPushButton('Spectrogram', self)
        self.sfacb.clicked.connect(self.calc_NMF)
        self.K_l = QtGui.QLabel(self)
        self.K_l.setText('NMF_K:')
        self.K_t = QtGui.QLineEdit()
        self.K_t.setText(str(self.K))
        self.envs_l = QtGui.QLabel(self)
        self.envs_l.setText('Envelopes:')
        self.envs_t = QtGui.QLineEdit()
        self.envs_t.setText(str(self.envs))
        self.iter_l = QtGui.QLabel(self)
        self.iter_l.setText('iter:')
        self.iter_t = QtGui.QLineEdit()
        self.iter_t.setText(str(self.iter))

        specfact_horizen.addWidget(self.sfacb)
        specfact_horizen.addWidget(self.K_l)
        specfact_horizen.addWidget(self.K_t)
        specfact_horizen.addWidget(self.envs_l)
        specfact_horizen.addWidget(self.envs_t)
        specfact_horizen.addWidget(self.iter_l)
        specfact_horizen.addWidget(self.iter_t)

        # general Layout
        grid = QtGui.QGridLayout()
        grid.setRowStretch(0, 1)
        grid.addLayout(open_horizen, 1, 0)
        grid.addLayout(wave_horizen, 2, 0)
        grid.addLayout(spectrogram_horizen, 3, 0)
        grid.addLayout(specfact_horizen, 4, 0)
        grid.setRowStretch(5, 1)

        self.setLayout(grid)

        self.setWindowTitle('File Dialog')
        self.setGeometry(50, 50, 500, 200)

        # instantiate Windows
        self.wfw = WaveformWindow()
        self.spw = SpectrogramWindow()
        self.mfw = MusicFactorWindow(self.K)

    def disp_refresh(self):
        self.sampling_p.setText(str(self.sd.ps.framerate))
        self.frames_p.setText(str(self.sd.ps.nframes))
        self.offset_t.setText(str(self.offset))
        self.length_t.setText(str(self.length))

    def open_file(self):
        global input_fname
        # input_fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
        wf = wave.open(input_fname, "rb")
        printWaveInfo(wf)
        self.sd.ps = wf.getparams()
        buffer = wf.readframes(wf.getnframes())
        self.sd.data = np.frombuffer(buffer, dtype="int16")
        wf.close()
        self.length = self.sd.ps.nframes
        self.disp_refresh()

    def make_waveform(self):
        self.length = int(self.length_t.text())
        self.offset = int(self.offset_t.text())
        self.wfw.disp_wave(self.sd, self.offset, self.length)
        self.wfw.show()

    def make_spectrogram(self):
        self.length = int(self.length_t.text())
        self.offset = int(self.offset_t.text())
        self.wsize = int(self.stftw_t.text())
        self.step = int(self.stfts_t.text())
        self.spectrogram = stft(self.sd.data[self.offset:self.length + 1], self.wsize, self.step)
        self.spw.disp_spectrogram(self.spectrogram)
        self.spw.show()

    def calc_NMF(self):
        # self.length = int(self.length_t.text())
        # self.offset = int(self.offset_t.text())
        # self.wsize = int(self.stftw_t.text())
        # self.step = int(self.stfts_t.text())
        self.spectrogram = stft(self.sd.data[self.offset:self.length + 1], self.wsize, self.step)

        self.K = int(self.K_t.text())
        self.envs = int(self.envs_t.text())
        self.iter = int(self.iter_t.text())
        Y = np.abs(self.spectrogram)
        phase = np.angle(self.spectrogram)
        H, U = music_factorize.nmf_euc(Y, self.K, self.iter)
        self.mfw.disp_musicfactor(H,U,phase)
        self.mfw.show()
#
# class FileWidget(QtGui.QWidget):
#
#     def __init__(self):
#         super(FileWidget, self).__init__()
#         self.button = QtGui.QPushButton("click me!")
#         self.button.clicked.connect(self.open_FileDialog)
#         self.ledit = QtGui.QLineEdit("")
#
#         hbox = QtGui.QHBoxLayout()
#         hbox.addWidget(self.button)
#         hbox.addWidget(self.ledit)
#         self.setLayout(hbox)
#
#         self.move(50, 50)
#
#     def open_FileDialog(self):
#         filename = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
#         self.ledit.setText(filename)


def printWaveInfo(wf):
    print ("Channels:", wf.getnchannels())
    print ("Sampwidth:", wf.getsampwidth())
    print ("Framerate:", type(wf.getframerate()))
    print ("Frames:", type(wf.getnframes()))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # file_w = FileWidget()
    ex = ApplicationWindow()
    # file_w.show()
    ex.show()
    sys.exit(app.exec_())

    # wf = wave.open(input_fname, "rb")
    # printWaveInfo(wf)
    # ps = wf.getparams()
    # buffer = wf.readframes(wf.getnframes())
    # # bufferはバイナリなので2バイトずつ整数（-32768から32767）にまとめる
    # data = np.frombuffer(buffer, dtype="int16")
    # wf.close()
    #
    # spectrogram = stft(data, stft_wsize, stft_step)
    #
    # # plt.plot(data)
    # # plt.show()
    # #    pxx, freqs, bins, im = plt.specgram(data,NFFT=2048,Fs=wf.getframerate(),noverlap=0,window=np.hamming(2048))
    # #    plt.axis([0, float(wf.getnframes()) / wf.getframerate(), 0, wf.getframerate() / 2])
    # # plt.imshow(abs(spectrogram[:stft_wsize/2,:]), aspect = "auto", origin = "lower")
    # # plt.show()
    # #
    # Y = np.abs(spectrogram)
    # phase = np.angle(spectrogram)
    #
    # H, U = music_factorize.nmf_euc(Y, K, max_iter)
    # reconst = istft(H.dot(U) * np.exp((0 + 1j) * phase), stft_wsize, stft_step)
    #
    # # reconst = istft(Y * np.exp((0 + 1j) * phase), stft_wsize, stft_step)
    #
    # # plt.subplot(211)
    # # plt.plot(data)
    # # plt.subplot(212)
    # # plt.plot(reconst)
    # # plt.show()
    # # wavfile.write(output_name,fs,reconst)
    # w = wave.Wave_write(output_name)
    # w.setparams((
    #     ps.nchannels,             # channel
    #     ps.sampwidth,             # byte width
    #     ps.framerate,             # sampling rate
    #     len(reconst),             # number of frames
    #     ps.comptype, ps.compname  # no compression
    # ))
    # w.writeframes(reconst.tostring())
    # w.close()
