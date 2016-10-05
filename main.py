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
FACTORIZE = "SimpleNMF"

class GlobalData():

    def __init__(self):
        self.params = []
        self.orig_data = []
        self.recon_data = []
        self.orig_spectrogram = []
        self.recon_spectrogram = []
        self.stft_wsize = 2048
        self.stft_step = 1024
        self.length = 0
        self.offset = 0
        self.K = 5
        self.envs = 5
        self.max_iter = 100
        self.Ymean = 0
        self.Y = []
        self.phase = []
        self.H = []
        self.U = []
        self.clist = [True] * self.K

    def factorize(self):
        if FACTORIZE == "SimpleNMF" :
        # SimpleNMF
            self.Y = np.abs(self.orig_spectrogram)
            self.phase = np.angle(self.orig_spectrogram)
            self.H, self.U = music_factorize.nmf_euc(self.Y, self.K, self.max_iter)

    def reconst(self):
        print("NOW!! RECONSTRUCTION!!")
        print("wsize, step =", self.stft_wsize, ", ", self.stft_step)
        print(self.clist)
        print("Ym:",self.Ymean,"  , Xmean:",np.mean(self.H.dot(self.U)))
        scale = 1
        V = self.lsee_mstftm(self.H.dot(self.U))
        self.phase = np.angle(V)
        Vphase = self.phase

        if FACTORIZE == "SimpleNMF" :
        # SimpleNMF
            Ht = np.zeros(self.H.shape)
            Ut = np.zeros(self.U.shape)
            for ks in range(self.K):
                if self.clist[ks]:
                    Ht[:, ks] = self.H[:, ks]
                    Ut[ks, :] = self.U[ks, :]
            self.recon_spectrogram = Ht.dot(Ut)
            self.recon_data = istft(scale * Ht.dot(Ut) * np.exp((0 + 1j) * Vphase), self.stft_wsize, self.stft_step)

    def lsee_mstftm(self,X):
        V = X * self.phase
        eps = np.finfo(float).eps
        for i in range(10):
            v_aud = istft(V * np.exp((0 + 1j)), SD.stft_wsize, SD.stft_step)
            V = stft(v_aud, SD.stft_wsize, SD.stft_step)
            while(V.shape[1] < X.shape[1]):
                V = np.c_[V,[eps for i in range(V.shape[0])]]
            while(X.shape[1] < V.shape[1]):
                X = np.c_[V,[eps for i in range(X.shape[0])]]
            V = X * V / np.abs(V)
        return V



SD = GlobalData()


class BarPlot():

    def __init__(self, parent=None, width=8, height=6):
        # Create the mpl Figure and FigCanvas objects.
        # 5x4 inches, 100 dots-per-inch
        global SD
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
        self.axes.plot(data[offs:leng + 1])
        self.canvas.draw()

    def draw_spectrogram(self, spectrogram, fmax, fmin, cscl):
        self.axes.clear()
        maxbin = (int)(fmax / 44100 * SD.stft_wsize)
        minbin = (int)(fmin / 44100 * SD.stft_wsize)

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
        for tick in self.axes.xaxis.get_major_ticks():
            tick.label.set_fontsize('x-small')
        for tick in self.axes.yaxis.get_major_ticks():
            tick.label.set_fontsize('x-small')


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
        maxbin = (int)(fmax / 44100 * SD.stft_wsize)
        minbin = (int)(fmin / 44100 * SD.stft_wsize)
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

    def __init__(self):
        # init
        super(MusicFactorWindow, self).__init__()

        # instant var
        global SD
        self.fmax = 5000
        self.fmin = 0
        self.cscl = 100
        self.willstop = True
        SD.clist = [True] * SD.K

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
        chbox = QtGui.QGridLayout()
        self.checks = []

        for ks in range(SD.K):
            cbox = QtGui.QCheckBox("K" + str(ks), self)
            cbox.setChecked(QtCore.Qt.Checked)
            cbox.stateChanged.connect(self.change_check)

            self.checks.append(cbox)
            chbox.addWidget(self.checks[ks],int(ks/5),ks%5*2)

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
        for ks in range(SD.K):
            SD.clist[ks] = self.checks[ks].isChecked()
        SD.reconst()
        self.spbar.draw_spectrogram(SD.recon_spectrogram, self.fmax, self.fmin, self.cscl)


    def push_ps_button(self):
        print("reconst type,len= ", type(SD.recon_data), ", ", len(SD.recon_data))
        if(not self.willstop):
            self.willstop = True
            return
        self.willstop = False
        self.playb.setText("Stop")
        t = threading.Thread(target=self.playing_m)
        t.setDaemon(True)
        t.start()

    def playing_m(self):
        mdata = SD.recon_data.tostring()
        print("len type mdata = ", len(mdata), ", ", type(mdata))

        # prepare stream
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(SD.params.sampwidth),
                        channels=SD.params.nchannels,
                        rate=SD.params.framerate,
                        output=True)
        print ("Channels:", SD.params.nchannels)
        print ("Sampwidth:", SD.params.sampwidth)
        print ("Frames:", SD.params.nframes)

        for t in range(0, len(mdata), 2048):
            if(self.willstop):
                break
            stream.write(mdata[t: min(t + 2048, len(mdata))])
#            print("mdataikuze::",mdata[t: min(t + 2048, len(mdata))])
        self.willstop = True
        self.playb.setText("Play")
        stream.close()
        p.terminate()

    def disp_musicfactor(self):
        # self.design_mfw(SD.K)
        print("drawSpectrogram")
        self.disp_spectrogram()
        print("drawH")
        self.disp_hs()
        print("drawU")
        self.disp_us()

    def get_fmax(self, value):
        # self.fmax = 200 * value
        self.fmax = 22050 * ((value / 100)**2)
        self.spbar.draw_spectrogram(SD.recon_spectrogram, self.fmax, self.fmin, self.cscl)
        # self.Hbar.draw_vert_h(SD.H, self.fmax, self.fmin)

    def get_fmin(self, value):
        self.fmin = 5 * value
        self.spbar.draw_spectrogram(SD.recon_spectrogram, self.fmax, self.fmin, self.cscl)
        # self.Hbar.draw_vert_h(SD.H, self.fmax, self.fmin)

    def get_cscl(self, value):
        self.cscl = value
        self.spbar.draw_spectrogram(SD.recon_spectrogram, self.fmax, self.fmin, self.cscl)

    def disp_hs(self):
        self.Hbar.draw_vert_h(SD.H, self.fmax, self.fmin)

    def disp_us(self):
        self.Ubar.draw_hors_u(SD.U)

    def disp_spectrogram(self):
        self.spbar.draw_spectrogram(SD.recon_spectrogram, self.fmax, self.fmin, self.cscl)



class SpectrogramWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(SpectrogramWindow, self).__init__()
        # instant var
        global SD
        self.fmax = 5000
        self.fmin = 0
        self.cscl = 100
        # self.main_frame = QtGui.QWidget()
        self.barplot = BarPlot(self, width=6, height=6)

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

    def disp_spectrogram(self):
        self.barplot.draw_spectrogram(SD.orig_spectrogram, self.fmax, self.fmin, self.cscl)

    def get_fmax(self, value):
        self.fmax = 22050 * ((value / 100)**2)
        self.barplot.draw_spectrogram(SD.orig_spectrogram, self.fmax, self.fmin, self.cscl)

    def get_fmin(self, value):
        self.fmin = 5 * value
        self.barplot.draw_spectrogram(SD.orig_spectrogram, self.fmax, self.fmin, self.cscl)

    def get_cscl(self, value):
        self.cscl = value
        self.barplot.draw_spectrogram(SD.orig_spectrogram, self.fmax, self.fmin, self.cscl)


class WaveformWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(WaveformWindow, self).__init__()
        self.barplot = BarPlot(self, width=6, height=1.5)
        # set layout
        vbox = QtGui.QHBoxLayout()
        vbox.addWidget(self.barplot.canvas)  # add canvas to the layout
        self.setLayout(vbox)

    def disp_wave(self, SD):
        self.barplot.draw_wave(SD.orig_data, SD.offset, SD.length)


class MainApplicationWindow(QtGui.QWidget):

    def __init__(self):
        # init
        super(MainApplicationWindow, self).__init__()

        # instant var
        global SD
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
        self.specb.clicked.connect(self.show_spectrogram)
        self.stftw_l = QtGui.QLabel(self)
        self.stftw_l.setText('FFT_frame:')
        self.stftw_t = QtGui.QLineEdit()
        self.stftw_t.setText(str(SD.stft_wsize))
        self.stfts_l = QtGui.QLabel(self)
        self.stfts_l.setText('FFT_step:')
        self.stfts_t = QtGui.QLineEdit()
        self.stfts_t.setText(str(SD.stft_step))

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
        self.K_t.setText(str(SD.K))
        self.envs_l = QtGui.QLabel(self)
        self.envs_l.setText('Envelopes:')
        self.envs_t = QtGui.QLineEdit()
        self.envs_t.setText(str(SD.envs))
        self.iter_l = QtGui.QLabel(self)
        self.iter_l.setText('iter:')
        self.iter_t = QtGui.QLineEdit()
        self.iter_t.setText(str(SD.max_iter))

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
        self.sampling_p.setText(str(SD.params.framerate))
        self.frames_p.setText(str(SD.params.nframes))
        self.offset_t.setText(str(SD.offset))
        self.length_t.setText(str(SD.length))

    def open_file(self):
        global input_fname
        # input_fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file')
        wf = wave.open(input_fname, "rb")
        printWaveInfo(wf)
        SD.params = wf.getparams()
        buffer = wf.readframes(wf.getnframes())
        SD.orig_data = np.frombuffer(buffer, dtype="int16")
        wf.close()
        SD.length = SD.params.nframes
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
        mdata = SD.orig_data.tostring()
        print("len type mdata = ", len(mdata), ", ", type(mdata))
        # prepare stream
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(SD.params.sampwidth),
                        channels=SD.params.nchannels,
                        rate=SD.params.framerate,
                        output=True)
        print ("Channels:", SD.params.nchannels)
        print ("Sampwidth:", SD.params.sampwidth)
        print ("Frames:", SD.params.nframes)

        for t in range(SD.offset * 2, SD.length * 2, 2048):
            if(self.willstop):
                break
            stream.write(mdata[t: min(t + 2048, SD.length * 2)])
        self.willstop = True
        self.playb.setText("Play")
        stream.close()
        p.terminate()

    def make_waveform(self):
        if(not self.length_t.text()):
            self.length_t.setText("0")
        elif(int(self.length_t.text()) > SD.params.nframes):
            self.length_t.setText(str(SD.params.nframes))
        if(not self.offset_t.text()):
            self.offset_t.setText("0")
        elif(int(self.offset_t.text()) > SD.params.nframes):
            self.offset_t.setText(str(SD.params.nframes))

        SD.length = int(self.length_t.text())
        SD.offset = int(self.offset_t.text())
        self.wfw.disp_wave(SD)
        self.wfw.show()

    def make_spectrogram(self):
        if (SD.orig_spectrogram == [] or SD.length != int(self.length_t.text()) or SD.offset != int(self.offset_t.text()) or SD.stft_wsize != int(self.stftw_t.text()) or SD.stft_step != int(self.stfts_t.text())):
            SD.length = int(self.length_t.text())
            SD.offset = int(self.offset_t.text())
            SD.stft_wsize = int(self.stftw_t.text())
            SD.stft_step = int(self.stfts_t.text())
            SD.orig_spectrogram = stft(SD.orig_data[SD.offset:SD.length + 1], SD.stft_wsize, SD.stft_step)
            SD.Ymean = np.mean(np.abs(SD.orig_spectrogram))

    def show_spectrogram(self):
        self.make_spectrogram()
        self.spw.disp_spectrogram()
        self.spw.show()

    def calc_NMF(self):
        self.make_spectrogram()

        SD.K = int(self.K_t.text())
        SD.envs = int(self.envs_t.text())
        SD.max_iter = int(self.iter_t.text())
        self.mfw = MusicFactorWindow()

        SD.factorize()
        SD.reconst()
        self.mfw.disp_musicfactor()
        self.mfw.show()
        # SD.reconst()
        # self.mfw.reconst_music()


def printWaveInfo(wf):
    print ("Channels:", wf.getnchannels())
    print ("Sampwidth:", wf.getsampwidth())
    print ("Framerate:", type(wf.getframerate()))
    print ("Frames:", type(wf.getnframes()))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    # file_w = FileWidget()
    ex = MainApplicationWindow()
    # file_w.show()
    ex.show()
    sys.exit(app.exec_())
