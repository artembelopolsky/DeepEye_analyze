# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:57:23 2019

@author: Jonathan
"""

# =============================================================================
# =============================================================================
# #  FIXATION DETECTION USING THE IDENTIFICATION BY 2-MEANS CLUSTERING (I2MC) ALGORITHM
#
# Translated to python from Matlab by Jonathan van Leeuwen, 2019 - www.github.com/jonathanvanleeuwen/I2MC
# Tested on Python 3.6

# =============================================================================
# Initialize
# =============================================================================
import os
import sys
import numpy as np
import import_funcs as imp
import I2MC_funcs
import plot_funcs as plot
import matplotlib.pyplot as plt
import time 
import pandas as pd

# =============================================================================
# NECESSARY VARIABLES
# =============================================================================
def runI2MC(fName, plotData = False):
    start = time.time()
    # Load file to extract meta info
    origData = pd.read_csv(fName)
    
    opt = {}
    # General variables for eye-tracking data
    opt['xres'] = origData.resX[0] # maximum value of horizontal resolution in pixels
    opt['yres'] = origData.resY[0] # maximum value of vertical resolution in pixels
    opt['missingx'] = 9999 # missing value for horizontal position in eye-tracking data (example data uses -xres). used throughout functions as signal for data loss
    opt['missingy'] = 9999 # missing value for vertical position in eye-tracking data (example data uses -yres). used throughout functions as signal for data loss
    # Get sample frequency, chooses the closests frequency from array of possible freqs
    freqs = np.arange(10,2002,2) # sets frequency to even number, to allow for easy downsampling
    calcFreq = int(1000/np.median(np.diff(origData.sampTime.values))) # sampling frequency of data 
    opt['freq'] = freqs[np.abs(freqs-calcFreq).argmin()] # sampling frequency of data  closest to allowed frequencys 
    
    # Variables for the calculation of visual angle
    # These values are used to calculate noise measures (RMS and BCEA) of
    # fixations. The may be left as is, but don't use the noise measures then.
    # If either or both are empty, the noise measures are provided in pixels
    # instead of degrees.
    opt['scrSz'] = [50.9174, 28.6411] # screen size in cm
    opt['disttoscreen'] = 65.0 # distance to screen in cm.
    
    # =============================================================================
    # OPTIONAL VARIABLES
    # =============================================================================
    # The settings below may be used to adopt the default settings of the
    # algorithm. Do this only if you know what you're doing.
    
    # # STEFFEN INTERPOLATION
    opt['windowtimeInterp'] = 0.1 # max duration (s) of missing values for interpolation to occur
    opt['edgeSampInterp'] = 2 # amount of data (number of samples) at edges needed for interpolation
    opt['maxdisp'] = opt['xres']*0.2*np.sqrt(2) # maximum displacement during missing for interpolation to be possible
    
    # # K-MEANS CLUSTERING
    opt['windowtime'] = 0.2 # time window (s) over which to calculate 2-means clustering (choose value so that max. 1 saccade can occur)
    opt['steptime'] = 0.02 # time window shift (s) for each iteration. Use zero for sample by sample processing
    opt['maxerrors'] = 100.0 # maximum number of errors allowed in k-means clustering procedure before proceeding to next file
    opt['downsamples'] = [2] # [2, 5, 10]) # downsample levels (can be empty)
    opt['downsampFilter'] = 0 # use chebychev filter when downsampling? 1: yes, 0: no. requires signal processing toolbox. is what matlab's downsampling functions do, but could cause trouble (ringing) with the hard edges in eye-movement data
    
    # # FIXATION DETERMINATION
    opt['cutoffstd'] = 2.0 # number of standard deviations above mean k-means weights will be used as fixation cutoff
    opt['onoffsetThresh'] = 3 # number of MAD away from median fixation duration. Will be used to walk forward at fixation starts and backward at fixation ends to refine their placement and stop algorithm from eating into saccades
    opt['maxMergeDist'] = 30.0 # maximum Euclidean distance in pixels between fixations for merging
    opt['maxMergeTime'] = 30.0 # maximum time in ms between fixations for merging
    opt['minFixDur'] = 40.0 # minimum fixation duration after merging, fixations with shorter duration are removed from output
    
    # =============================================================================
    # SETUP directory handeling
    # =============================================================================
    # Write the final fixation output file 
    fixFileHeader = ['FixStart', 'FixEnd', 'FixDur', 'XPos', 'YPos', 'FlankedByDataLoss',
                     'Fraction Interpolated', 'WeightCutoff', 'RMSxy', 'BCEA', 
                     'FixRangeX', 'FixRangeY']
    fixDF = pd.DataFrame(data = [], columns = fixFileHeader)
    # =============================================================================
    # START ALGORITHM 
    # =============================================================================
    ## IMPORT DATA
    print('\n\n\nImporting and processing: "{}"'.format(fName))
    data = {}
    data['time'], data['average_X'], data['average_Y'] = imp.importDeepEye(fName)
    # RUN FIXATION DETECTION
    fix,_,_ = I2MC_funcs.I2MC(data,opt)
    
    if fix != False:
        if plotData == True:
            # pre-allocate name for saving file
            f = I2MC_funcs.plotResults(data,fix,[opt['xres'], opt['yres']])
            
        # # Write data to dataframe
        fixDF['FixStart'] = fix['startT']
        fixDF['FixEnd'] = fix['endT']
        fixDF['FixDur'] = fix['dur']
        fixDF['XPos'] = fix['xpos']
        fixDF['YPos'] = fix['ypos']
        fixDF['FlankedByDataLoss'] = fix['flankdataloss']
        fixDF['Fraction Interpolated'] = fix['fracinterped']
        fixDF['WeightCutoff'] = fix['cutoff']
        fixDF['RMSxy'] = fix['RMSxy']
        fixDF['BCEA'] = fix['BCEA']
        fixDF['FixRangeX'] = fix['fixRangeX']
        fixDF['FixRangeY'] = fix['fixRangeY']
        
        # Add meta information
        # metaInfo = ['resX', 'resY', 'scrW_cm'] 
        
       
        # for k in metaInfo:
        #    fixDF[k] = origData[k][0]
    else:
        print('No fixations detected, quiting')
    print('\n\nI2MC took {}s to finish!'.format(time.time()-start))
    return fixDF
