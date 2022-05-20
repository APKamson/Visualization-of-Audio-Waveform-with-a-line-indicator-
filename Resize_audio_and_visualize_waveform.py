# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:17:08 2022

@author: aihig
"""


import matplotlib.pyplot as plt
from librosa.core import convert
from scipy.interpolate import interp1d
from scipy.io import wavfile
import librosa
# sphinx_gallery_thumbnail_number = 5
import numpy as np
import os
import glob
import wavio
import pylops


def apply_transfer(signal, transfer, interpolation='linear'):
    constant = np.linspace(-1, 1, len(transfer))
    interpolator = interp1d(constant, transfer, interpolation)
    return interpolator(signal)

def limiter(x, treshold=0.8):
    transfer_len = 1000
    transfer = np.concatenate([ np.repeat(-1, int(((1-treshold)/2)*transfer_len)),
                                np.linspace(-1, 1, int(treshold*transfer_len)),
                                np.repeat(1, int(((1-treshold)/2)*transfer_len)) ])
    return apply_transfer(x, transfer)

# smooth compression: if factor is small, its near linear, the bigger it is the
# stronger the compression
def arctan_compressor(x, factor=2):
    constant = np.linspace(-1, 1, 1000)
    transfer = np.arctan(factor * constant)
    transfer /= np.abs(transfer).max()
    return apply_transfer(x, transfer)


filenames = glob.glob("./Audio/*.wav")

for fname in filenames:
    fID = fname.split('/')[-1].split('.')[0] 
    
    # Convert to an audio file with given set of name
    pathname, extension = os.path.split(fname.split('/')[-1])
    Name, extension = os.path.splitext(extension)
    
    
    # load wav file
    x, fs = librosa.load(os.path.join('./', 
                                            fname.split('/')[-1]), sr = 4000)
    
    # Some Preprocessing
    chunk = 4              # 4 sec
    x = x[1:chunk*fs]
    x = x-np.mean(x)
    x = x/abs(x).max()
    
    # x2 = limiter(x)
    # x2 = np.int16(x2 * 32767 * A)
    
    
    # plt.subplot(211)
    # plt.plot(x)
    # plt.subplot(212)
    # plt.plot(x2)
    
    
    # timestamp
    t = (np.linspace(0,len(x)/fs,num=len(x)))
    
    Frame_rate = 25
    # Plot waveform
    # for i in range(0, int(Frame_rate*n)):
    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     ax.plot(t, x, "g", lw=.5)
    #     ax.grid()
    #     ax.axvline(i/Frame_rate, color='red')
    #     plt.savefig(os.path.join('assdasd'+ str(1000+i) + '.png'), dpi=150)
        
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(t, x, "k", lw=.5)
    ax.grid() 
    plt.xlabel('t (sec)')
    plt.ylabel('Amplitude')
    ax.set_xlim(left=0.0,right=chunk)
    plt.savefig(os.path.join('./images/' + Name + '.png'), dpi=150)
    
    # Save resized audio and figures
    # wavio.write('hallo_me.wav',x2, fs, sampwidth=3)
    
    x2 = limiter(x)
    x2 = np.int16(x2*32767)
    wavfile.write(os.path.join('./Audio/Quiz/' + Name + '_Q.wav'), fs, x2)
    
    # x3 = arctan_compressor(x)
    # x3 = np.int16(x3 * 32767 * 1.5)
    # wavfile.write("output_comp.wav", fs, x3)


