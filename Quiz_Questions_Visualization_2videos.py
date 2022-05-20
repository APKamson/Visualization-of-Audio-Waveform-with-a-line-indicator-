# -*- coding: utf-8 -*-
"""
Created on Tue May 17 17:20:25 2022

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


def convert_frames_to_video(filename,pathOut,fps, chunk, fs):
    frame_array = []
    # Plot waveform
    for i in range(0, int(fps*chunk)+1):
        # fig, ax = plt.subplots(figsize=(15, 4))
        # ax.plot( sig, "k", lw=.5)
        # ax.grid()
        # plt.xlabel('t (sec)')
        # plt.ylabel('Amplitude')
        # ax.set_xlim(left=0.0,right=chunk)
        # ax.axvline(i/fps, color='red')
        
        plt.rcParams["figure.figsize"] = [15, 4]
        plt.rcParams["figure.autolayout"] = True
        im = plt.imread(os.path.join('./images/Edited/' + filename + '.png'))
        fig, ax = plt.subplots()
        ax.imshow(im, extent = [0, 4, -1.1, 1], aspect='auto')
        plt.xlabel('t (sec)')
        plt.ylabel('Amplitude')
        ax.axvline(i/fps, color='red')
        
        # export
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png', dpi = 150)
        # buf.seek(0)
        # img = np.array(Image.open(buf))
        
        plt.savefig('./images/plot.png', bbox_inches='tight',pad_inches = 0, dpi=300)
        plt.clf () 
        img = cv2.imread('./images/plot.png')
        height, width, layers = img.shape
        size = (width, height)
        # print(img, size)
        #inserting the frames into an image array
        frame_array.append(img)
        # plt.savefig(os.path.join('assdasd'+ str(1000+i) + '.png'), dpi=150)
        
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    # Deallocating memories taken for window creation
    cv2.destroyAllWindows() 
    out.release()

def main():
    # load wav file
    filename = '0c5eecf3-7ee6-420b-b29c-459974ee881d_DN_BW_Opt_1'
    sig, fs = librosa.load(os.path.join('./Audio/' + filename + '.wav'), sr = 4000)
    chunk = 4               # 4 sec
    sig = sig[1:chunk*fs]
    
    # Save as corresponding audio file
    # x2 = limiter(sig)
    # x2 = np.int16(x2 * 32767 * 1.5)
    # wavfile.write(os.path.join(filename + 'audio' + '.wav'), fs, x2)
    
    # Creat video for wave visualization
    pathOut = os.path.join('./Video/' + filename + '_Q.avi')
    fps = 30
    convert_frames_to_video(filename, pathOut, fps, chunk, fs)
    
if __name__=="__main__":
    main()
    
# Save resized audio and figures
# wavio.write('hallo_me.wav',x2, fs, sampwidth=3)






