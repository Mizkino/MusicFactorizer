# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:12:11 2016

@author: god
"""
import sys
import threading
import wave
import pyaudio
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
SD = []
Ymean = 0


class SoundData():

    def __init__(self):
        ps = []
        data = []

# class AudioPlayer():


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
        self.axes.autoscale(tight="True")
        self.axes.axis('off')
        # self.axes.set_xticks([])
        # self.axes.set_yticks([])
        print("datashape = ", data.shape)
        self.axes.plot(data[offs:leng + 1])
        self.canvas.draw()

    def draw_spectrogram(self, spectrogram, fmax, fmin, cscl):
        self.axes.clear()
        global stft_wsize
        maxbin = (int)(fmax / 44100 * stft_wsize)
        minbin = (int)(fmin / 44100 * stft_wsize)

        # vman = (np.max(abs(spectrogram[minbin:maxbin, :])) - np.min(abs(spectrogram[minbin:maxbin, :]))) * (100-cscl) /200
        # vmax = np.max(abs(spectrogram[minbin:maxbin, :])) - vman
        # vmin = np.min(abs(spectrogram[minbin:maxbin, :])) + vman
        # print("cscl, vman, vmax, vmin =", cscl,", ", vman,", ",vmax,", ",vmin)
        vmax = np.max(abs(spectrogram[minbin:maxbin, :])) * cscl / 100
        vmin = np.min(abs(spectrogram[minbin:maxbin, :])) * 100 / cscl

        self.axes.imshow(abs(spectrogram[minbin:maxbin, :]), aspect="auto", origin="lower", vmin=vmin,vmax=vmax)
        posxt = self.axes.get_xticks()
        posyt = self.axes.get_yticks()
        self.axes.set_xticks(posxt[1:len(posxt)-1])
        self.axes.set_xticklabels(np.ceil(posxt[1:len(posxt)]/44100*1024))
        self.axes.set_yticks(posyt[1:len(posyt)-1])
        self.axes.set_yticklabels(np.ceil(posyt[1:len(posyt)]/44100*2048*1000))

        # self.fig.subplots_adjust(right=0.8)
        # cbar_ax = self.fig.add_axes([0.85, 0.15, 0.05, 0.7])
        # # cbar_ax.majorticks_off()
        # self.fig.colorbar(sp, cax=cbar_ax,ticks=[])
        self.canvas.draw()

    def draw_vert_h(self, H, fmax, fmin):
        self.fig.subplots_adjust(wspace=0.0)
        for ax in self.fig.axes:  # delete axes first
            self.fig.delaxes(ax)
        self.axes.clear()
        global stft_wsize
        maxbin = (int)(fmax / 44100 * stft_wsize)
        minbin = (int)(fmin / 44100 * stft_wsize)
        x = range(minbin, maxbin)
        for K in range(H.shape[1]):
            self.axes = self.fig.add_subplot(1, H.shape[1], K + 1)
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            self.axes.set_ylim(minbin, maxbin)
            self.axes.barh(x, H[minbin:maxbin, K])
        self.canvas.draw()

    def draw_hors_u(self, U):
        self.fig.subplots_adjust(hspace=0.0)
        for ax in self.fig.axes:  # delete axes first
            self.fig.delaxes(ax)
        self.axes.clear()
        for K in range(U.shape[0]):
            self.axes = self.fig.add_subplot(U.shape[0], 1, K + 1)
            self.axes.set_xticks([])
            self.axes.set_yticks([])
            self.axes.plot(U[K, :],)
        self.canvas.draw()


class MusicFactorWindow(QtGui.QWidget):

    def __init__(self, K):
        # init
        super(MusicFactorWindow, self).__init__()

        # instant var
        global SD
        global stft_wsize
        global stft_step
        self.fmax = 5000
        self.fmin = 0
        self.cscl = 100
        self.H = []
        self.U = []
        self.phase = []
        self.K = K
        self.willstop = True
        self.reconst = []
        self.clist = [True] * self.K

        # self.main_frame = QtGui.QWidget()
        self.spbar = BarPlot(self, width=6, height=6)
        self.Hbar = BarPlot(self, width=6, height=6)
        self.Ubar = BarPlot(self, width=6, height=6)

        # set label for sliders
        self.minl = QtGui.QLabel(self)
        self.minl.setText('Freq. Min')
        self.maxl = QtGui.QLabel(self)
        self.maxl.setText('Freq. Max')
        self.scll = QtGui.QLabel(self)
        self.scll.setText('Colorbar Scale')

        hlbox = QtGui.QHBoxLayout()
        hlbox.addWidget(self.minl)
        hlbox.addWidget(self.maxl)
        hlbox.addWidget(self.scll)

        # set option slider and button etc
        self.maxs = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.maxs.setMinimum(10)
        self.maxs.setMaximum(100)
        self.maxs.setValue(50)
        self.maxs.valueChanged.connect(self.get_fmax)
        self.maxs.sliderReleased.connect(self.disp_hs)
        self.mins = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.mins.setMinimum(0)
        self.mins.setMaximum(100)
        self.mins.setValue(0)
        self.mins.valueChanged.connect(self.get_fmin)
        self.mins.sliderReleased.connect(self.disp_hs)
        self.scls = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.scls.setMinimum(1)
        self.scls.setMaximum(100)
        self.scls.setValue(100)
        self.scls.valueChanged.connect(self.get_cscl)

        # bars layout
        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.mins)
        hbox.addWidget(self.maxs)
        hbox.addWidget(self.scls)

        # Designing Check box
        chbox = QtGui.QHBoxLayout()
        self.checks = []

        for ks in range(self.K):
            cbox = QtGui.QCheckBox("K" + str(ks), self)
            cbox.setChecked(QtCore.Qt.Checked)
            cbox.stateChanged.connect(self.change_check)

            self.checks.append(cbox)
            chbox.addWidget(self.checks[ks])

        # design upper left
        self.playb = QtGui.QPushButton('Play', self)
        self.playb.clicked.connect(self.push_ps_button)
        luvbox = QtGui.QVBoxLayout()
        luvbox.addWidget(self.playb)
        luvbox.addLayout(chbox)

        # set layout general
        grid = QtGui.QGridLayout()
        grid.addWidget(self.Hbar.canvas, 3, 0)
        grid.addWidget(self.Ubar.canvas, 2, 1)
        grid.addWidget(self.spbar.canvas, 3, 1)
        grid.addLayout(hlbox, 0, 0, 1, 2)
        grid.addLayout(hbox, 1, 0, 1, 2)
        grid.addLayout(luvbox, 2, 0)

        self.setLayout(grid)
        # self.setGeometry(500, 500)

    def change_check(self):
        for ks in range(self.K):
            self.clist[ks] = self.checks[ks].isChecked()
        self.reconst_music()

    def lsee_mstftm(self,X):
        V = X
        for i in range(5):
            v_aud = istft(V * np.exp((0 + 1j) * self.phase), stft_wsize, stft_step)
            V = stft(v_aud, stft_wsize, stft_step)
            while(V.shape[1] < X.shape[1]):
                V = np.c_[V,[0 for i in range(V.shape[0])]]
            while(X.shape[1] < V.shape[1]):
                X = np.c_[V,[0 for i in range(X.shape[0])]]
            V = X * V / np.abs(V)
        return V


    def reconst_music(self):
        print("NOW!! RECONSTRUCTION!!")
        print("wsize, step =", stft_wsize, ", ", stft_step)
        print(self.clist)
        global Ymean
        print("Ym:",Ymean,"  , Xmean:",np.mean(self.H.dot(self.U)))
        # scale = Ymean / np.mean(self.H.dot(self.U))
        scale = 1
        Vphase = self.phase
        Ht = np.zeros(self.H.shape)
        Ut = np.zeros(self.U.shape)
        for ks in range(self.K):
            if self.clist[ks]:
                Ht[:, ks] = self.H[:, ks]
                Ut[ks, :] = self.U[ks, :]
        self.reconst = istft(scale * Ht.dot(Ut) * np.exp((0 + 1j) * Vphase), stft_wsize, stft_step)


    def push_ps_button(self):
        print("reconst type,len= ", type(self.reconst), ", ", len(self.reconst))
        if(not self.willstop):
            self.willstop = True
            return
        self.willstop = False
        self.playb.setText("Stop")
        t = threading.Thread(target=self.playing_m)
        t.setDaemon(True)
        t.start()

    def playing_m(self):
        mdata = self.reconst.tostring()
        print("len type mdata = ", len(mdata), ", ", type(mdata))

        # prepare stream
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(SD.ps.sampwidth),
                        channels=SD.ps.nchannels,
                        rate=SD.ps.framerate,
                        output=True)
        print ("Channels:", SD.ps.nchannels)
        print ("Sampwidth:", SD.ps.sampwidth)
        print ("Frames:", SD.ps.nframes)

        for t in range(0, len(mdata), 2048):
            if(self.willstop):
                break
            stream.write(mdata[t: min(t + 2048, len(mdata))])
#            print("mdataikuze::",mdata[t: min(t + 2048, len(mdata))])
        self.willstop = True
        self.playb.setText("Play")
        stream.close()
        p.terminate()

    def disp_musicfactor(self, H, U, phase):
        self.H = H
        self.U = U
        self.phase = phase
        self.K = H.shape[1]
        # self.design_mfw(self.K)
        print("drawSpectrogram")
        self.spbar.draw_spectrogram(H.dot(U), self.fmax, self.fmin, self.cscl)
        # self.disp_hs(self.K)
        # self.disp_us(self.K)
        print("drawH")
        self.disp_hs()
        print("drawU")
        self.disp_us(H, U)
        V = self.lsee_mstftm(self.H.dot(self.U))
        self.phase = np.angle(V)


    def get_fmax(self, value):
        # self.fmax = 200 * value
        self.fmax = 22050 * ((value / 100)**2)
        self.spbar.draw_spectrogram(self.H.dot(self.U), self.fmax, self.fmin, self.cscl)
        # self.Hbar.draw_vert_h(self.H, self.fmax, self.fmin)

    def get_fmin(self, value):
        self.fmin = 5 * value
        self.spbar.draw_spectrogram(self.H.dot(self.U), self.fmax, self.fmin, self.cscl)
        # self.Hbar.draw_vert_h(self.H, self.fmax, self.fmin)

    def get_cscl(self, value):
        self.cscl = value
        self.spbar.draw_spectrogram(self.H.dot(self.U), self.fmax, self.fmin, self.cscl)

    def disp_hs(self):
        self.Hbar.draw_vert_h(self.H, self.fmax, self.fmin)

    def disp_us(self, H, U):
        self.Ubar.draw_hors_u(U)


class SpectrogramWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(SpectrogramWindow, self).__init__()
        # instant var
        self.fmax = 5000
        self.fmin = 0
        self.cscl = 100
        # self.main_frame = QtGui.QWidget()
        self.barplot = BarPlot(self, width=6, height=6)
        self.spectrogram = []

        # set label for sliders
        self.minl = QtGui.QLabel(self)
        self.minl.setText('Freq. Min')
        self.maxl = QtGui.QLabel(self)
        self.maxl.setText('Freq. Max')
        self.scll = QtGui.QLabel(self)
        self.scll.setText('Colorbar Scale')

        hlbox = QtGui.QHBoxLayout()
        hlbox.addWidget(self.minl)
        hlbox.addWidget(self.maxl)
        hlbox.addWidget(self.scll)

        # set option slider and button etc
        self.maxs = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.maxs.setMinimum(10)
        self.maxs.setMaximum(100)
        self.maxs.setValue(50)
        self.maxs.valueChanged.connect(self.get_fmax)
        self.mins = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.mins.setMinimum(0)
        self.mins.setMaximum(100)
        self.mins.setValue(0)
        self.mins.valueChanged.connect(self.get_fmin)
        self.scls = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.scls.setMinimum(1)
        self.scls.setMaximum(100)
        self.scls.setValue(100)
        self.scls.valueChanged.connect(self.get_cscl)

        hbox = QtGui.QHBoxLayout()
        hbox.addWidget(self.mins)
        hbox.addWidget(self.maxs)
        hbox.addWidget(self.scls)

        # set layout
        vbox = QtGui.QVBoxLayout()
        vbox.addLayout(hlbox)
        vbox.addLayout(hbox)
        vbox.addWidget(self.barplot.canvas)  # add canvas to the layout
        # self.main_frame.setLayout(vbox)
        # set widget
        # self.addWidget(self.main_frame)
        self.setLayout(vbox)

    def disp_spectrogram(self, spectrogram):
        self.spectrogram = spectrogram
        self.barplot.draw_spectrogram(self.spectrogram, self.fmax, self.fmin, self.cscl)

    def get_fmax(self, value):
        # self.fmax = 200 * value
        self.fmax = 22050 * ((value / 100)**2)
        self.barplot.draw_spectrogram(self.spectrogram, self.fmax, self.fmin, self.cscl)

    def get_fmin(self, value):
        self.fmin = 5 * value
        self.barplot.draw_spectrogram(self.spectrogram, self.fmax, self.fmin, self.cscl)

    def get_cscl(self, value):
        self.cscl = value
        self.barplot.draw_spectrogram(self.spectrogram, self.fmax, self.fmin, self.cscl)


class WaveformWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(WaveformWindow, self).__init__()
        self.barplot = BarPlot(self, width=6, height=1.5)
        # set layout
        vbox = QtGui.QHBoxLayout()
        vbox.addWidget(self.barplot.canvas)  # add canvas to the layout
        self.setLayout(vbox)

    def disp_wave(self, SD, offs, leng):
        self.barplot.draw_wave(SD.data, offs, leng)


class ApplicationWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(ApplicationWindow, self).__init__()

        # instant var
        self.length = 0
        self.offset = 0
        global SD
        SD = SoundData()
        self.wsize = 2048
        self.step = 1024
        self.spectrogram = []
        self.K = 5
        self.envs = 5
        self.iter = 100
        self.willstop = True

        # open_file, play and params row
        open_horizen = QtGui.QHBoxLayout()
        self.fileb = QtGui.QPushButton('Open File', self)
        self.fileb.clicked.connect(self.open_file)
        self.playb = QtGui.QPushButton('Play', self)
        self.playb.clicked.connect(self.push_ps_button)
        self.sampling_l = QtGui.QLabel(self)
        self.sampling_l.setText('SamplRate:')
        self.sampling_p = QtGui.QLabel(self)
        self.frames_l = QtGui.QLabel(self)
        self.frames_l.setText('Hz         Frames:')
        self.frames_p = QtGui.QLabel(self)

        open_horizen.addWidget(self.fileb)
        open_horizen.addWidget(self.playb)
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
        self.offset_t.returnPressed.connect(self.make_waveform)
        self.length_l = QtGui.QLabel(self)
        self.length_l.setText('Length:')
        self.length_t = QtGui.QLineEdit()
        self.length_t.returnPressed.connect(self.make_waveform)

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
        self.sfacb = QtGui.QPushButton('MusicFactor', self)
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
        self.open_file()

    def disp_refresh(self):
        self.sampling_p.setText(str(SD.ps.framerate))
        self.frames_p.setText(str(SD.ps.nframes))
        self.offset_t.setText(str(self.offset))
        self.length_t.setText(str(self.length))

    def open_file(self):
        global input_fname
        # input_fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
        wf = wave.open(input_fname, "rb")
        printWaveInfo(wf)
        SD.ps = wf.getparams()
        buffer = wf.readframes(wf.getnframes())
        SD.data = np.frombuffer(buffer, dtype="int16")
        wf.close()
        self.length = SD.ps.nframes
        self.disp_refresh()

    def push_ps_button(self):
        if(not self.willstop):
            self.willstop = True
            return
        self.willstop = False
        self.playb.setText("Stop")
        t = threading.Thread(target=self.playing_m)
        t.setDaemon(True)
        t.start()

    def playing_m(self):

        # prepare stream
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(SD.ps.sampwidth),
                        channels=SD.ps.nchannels,
                        rate=SD.ps.framerate,
                        output=True)
        mdata = SD.data.tostring()
        for t in range(self.offset * 2, self.length * 2, 2048):
            if(self.willstop):
                break
            stream.write(mdata[t: min(t + 2048, self.length * 2)])
        self.willstop = True
        self.playb.setText("Play")
        stream.close()
        p.terminate()

    def make_waveform(self):
        if(not self.length_t.text()):
            self.length_t.setText("0")
        elif(int(self.length_t.text()) > SD.ps.nframes):
            self.length_t.setText(str(SD.ps.nframes))
        if(not self.offset_t.text()):
            self.offset_t.setText("0")
        elif(int(self.offset_t.text()) > SD.ps.nframes):
            self.offset_t.setText(str(SD.ps.nframes))

        self.length = int(self.length_t.text())
        self.offset = int(self.offset_t.text())
        self.wfw.disp_wave(SD, self.offset, self.length)
        self.wfw.show()

    def make_spectrogram(self):
        if (self.spectrogram == [] or self.length != int(self.length_t.text()) or self.offset != int(self.offset_t.text()) or self.wsize != int(self.stftw_t.text()) or self.step != int(self.stfts_t.text())):
            self.length = int(self.length_t.text())
            self.offset = int(self.offset_t.text())
            self.wsize = int(self.stftw_t.text())
            self.step = int(self.stfts_t.text())
            global stft_wsize
            stft_wsize = self.wsize
            # global stft_step = self.step
            self.spectrogram = stft(SD.data[self.offset:self.length + 1], self.wsize, self.step)
            global Ymean
            Ymean = np.mean(np.abs(self.spectrogram))
        self.spw.disp_spectrogram(self.spectrogram)
        self.spw.show()

    def calc_NMF(self):
        if (self.spectrogram == [] or self.length != int(self.length_t.text()) or self.offset != int(self.offset_t.text()) or self.wsize != int(self.stftw_t.text()) or self.step != int(self.stfts_t.text())):
            self.length = int(self.length_t.text())
            self.offset = int(self.offset_t.text())
            self.wsize = int(self.stftw_t.text())
            self.step = int(self.stfts_t.text())
            global stft_wsize
            stft_wsize = self.wsize
            # global stft_step = self.step
            self.spectrogram = stft(SD.data[self.offset:self.length + 1], self.wsize, self.step)
            global Ymean
            Ymean = np.mean(np.abs(self.spectrogram))

        self.K = int(self.K_t.text())
        self.envs = int(self.envs_t.text())
        self.iter = int(self.iter_t.text())
        self.mfw = MusicFactorWindow(self.K)

        Y = np.abs(self.spectrogram)
        phase = np.angle(self.spectrogram)
        H, U = music_factorize.nmf_euc(Y, self.K, self.iter)
        self.mfw.disp_musicfactor(H, U, phase)
        self.mfw.show()
        self.mfw.reconst_music()


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
