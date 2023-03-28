# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:44:08 2022

@author: artem
"""

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from IPython import display

import os


df = pd.read_csv('C:/Users/artem/Desktop/2022_12_15_22_01_36_record.csv')

# need to install ffmpeg from "conda install -c conda-forge ffmpeg"

# # Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)


for index, trial in df.groupby(df.trialNr):
   
    if index == 1:
        plt.figure()
        plt.xlim(0, df.resX[0])
        plt.ylim(0, df.resY[0])
        plt.scatter(trial.user_pred_px_x, trial.user_pred_px_y)
        plt.gca().invert_yaxis()
    
    
    
    
    
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.set(xlim=(0, df.resX[0]), ylim=(0, df.resY[0]))
        plt.gca().invert_yaxis()
        plt.axis('off')
        x = np.array(trial.user_pred_px_x)
        y = np.array(trial.user_pred_px_y)
        time = np.array(trial.sampTime)
        
        scat = ax.scatter(x, y, s=20, edgecolor='b')
        
        
        def animate(i):
            scat.set_offsets(np.c_[x[i], y[i]])
           
        
        anim = FuncAnimation(
            fig, animate, interval=600, frames=len(x)-1, repeat=True)
            
            
            
         
        plt.draw()
        plt.show() 

# writervideo = animation.FFMpegWriter(fps=30)
# anim.save('deepEye.mp4', writer=writervideo)
# plt.close()

