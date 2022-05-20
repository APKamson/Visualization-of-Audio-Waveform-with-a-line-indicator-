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
from PIL import Image
# sphinx_gallery_thumbnail_number = 5
import numpy as np
import cv2
import os
import glob
import wavio
import pylops
import io


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

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


def convert_frames_to_video(sig,pathOut,fps, chunk, fs):
    frame_array = []
    # timestamp
    t = (np.linspace(0,len(sig)/fs,num=len(sig)))
    # Plot waveform
    for i in range(0, int(fps*chunk)+1):
        fig, ax = plt.subplots(figsize=(15, 4))
        ax.plot(t, sig, "k", lw=.5)
        ax.grid()
        plt.xlabel('t (sec)')
        plt.ylabel('Amplitude')
        ax.set_xlim(left=0.0,right=chunk)
        ax.axvline(i/fps, color='red')
        
        # export
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png', dpi = 150)
        # buf.seek(0)
        # img = np.array(Image.open(buf))
        
        plt.savefig('plot.png', bbox_inches='tight',pad_inches = 0, dpi=100)
        plt.clf () 
        img = cv2.imread('plot.png')
        height, width, layers = img.shape
        size = (width, height)
        # print(img, size)
        #inserting the frames into an image array
        frame_array.append(img)
        # plt.savefig(os.path.join('assdasd'+ str(1000+i) + '.png'), dpi=150)
        
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(frame_array)):
        # writing to a image array
        print(len(frame_array)) 
        out.write(frame_array[i])
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    out.release()

def main():
    # load wav file
    filename = '0bb533b8-6bdf-4d37-a5dc-b129944f8c18'
    sig, fs = librosa.load(os.path.join('./Audio/' + filename + '.wav'), sr = 4000)
    chunk = 4               # 4 sec
    sig = sig[1:chunk*fs]
    sig = sig/abs(sig).max()
    
    # Save as corresponding audio file
    x2 = limiter(sig)
    x2 = np.int16(x2 * 32767 * 1)
    wavfile.write(os.path.join(filename + 'audio' + '.wav'), fs, x2)
    
    # Creat video for wave visualization
    pathOut = os.path.join(filename + '1.mp4')
    fps = 30
    convert_frames_to_video(sig, pathOut, fps, chunk, fs)
    
if __name__=="__main__":
    main()
    
# Save resized audio and figures
# wavio.write('hallo_me.wav',x2, fs, sampwidth=3)






