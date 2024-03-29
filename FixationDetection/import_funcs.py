# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:57:23 2019

@author: Jonathan
"""

# =============================================================================
# Import modules
# =============================================================================
import numpy as np
import pandas as pd

# =============================================================================
# Import TobiiTX 300
# =============================================================================
def importTobiiTX300(fname, nskip=1, res=[1920,1080], missingx=-1920, missingy=-1080):
    '''
    Imports data from Tobii TX300 as returned by Tobii SDK. 
    
    
    Parameters
    ----------
    fname : string
        The file (filepath) 
    nskip : int
        Number of header lines
    res : tupple
        The (X,Y) resolution of the screen
    missingx : ??
        The value reflecting mising values for X coordinates in the dataset
    missingy : ??
        The value reflecting mising values for Y coordinates in the dataset    
    
    Returns
    -------
    t : np.array
        The sample times from the dataset
    L_X : np.array
        X positions from the left eye
    L_Y : np.array
        Y positions from the left eye
    R_X : np.array
        X positions from the right eye
    R_Y : np.array
        Y positions from the right eye
    
    Example
    --------
    >>> 
    >>> data = {}
    >>> fname = '..\\example data\\participant1\\1.txt'
    >>> nskip = 1
    >>> res = [1920,1080]
    >>> missingx = -1920
    >>> missingy = -1080
    >>> data['time'], data['L_X'], data['L_Y'], data['R_X'], data['R_Y'] = importTobiiTX300(fname, nskip, res, missingx, missingy)
    >>> print('Time: {}\nL_X: {}\nL_Y: {}\nR_X: {}\nR_Y: {}\n'.format(data['time'][:5], data['L_X'][:5], data['L_Y'][:5], data['R_X'][:5], data['R_Y'][:5]))
    >>>
    Time: [ 0.     3.333  6.685 10.032 13.392]
    L_X: [-1920.    -1920.      956.544   950.016   957.504]
    L_Y: [-1080.    -1080.      536.112   544.212   536.436]
    R_X: [-1920.    -1920.      939.648   931.776   938.688]
    R_Y: [-1080.    -1080.      480.924   492.804   479.196]
    '''  
    # Load all data
    dat = np.loadtxt(fname, skiprows=nskip)
    
    # Extract required data
    t = dat[:,27]
    L_X = dat[:,7] * res[0]
    L_Y = dat[:,8] * res[1]
    L_V = dat[:,13]
    R_X = dat[:,20] * res[0]
    R_Y = dat[:,21] * res[1]
    R_V = dat[:,26]
    
    ###
    # sometimes we have weird peaks where one sample is (very) far outside the
    # monitor. Here, count as missing any data that is more than one monitor
    # distance outside the monitor.
    
    # Left eye
    lMiss1 = np.logical_or(L_X<-res[0], L_X>2*res[0])
    lMiss2 = np.logical_or(L_Y<-res[1], L_Y>2*res[1])
    lMiss = np.logical_or(np.logical_or(lMiss1, lMiss2), L_V > 1)
    L_X[lMiss] = missingx
    L_Y[lMiss] = missingy
    
    # Right eye
    rMiss1 = np.logical_or(R_X<-res[0], R_X>2*res[0])
    rMiss2 = np.logical_or(R_Y<-res[1], R_Y>2*res[1])
    rMiss = np.logical_or(np.logical_or(rMiss1, rMiss2), R_V > 1)
    R_X[rMiss] = missingx
    R_Y[rMiss] = missingy
    
    return t, L_X, L_Y, R_X, R_Y


# =============================================================================
# Write your own func
# =============================================================================
def importDeepEye(fName, missingXY = 9999):
    data = pd.read_csv(fName)
    
    
    # Remove missing values and make sure data is not string
    data = data[data.fName.notna()]
    data.frameNr = data.frameNr.apply(pd.to_numeric, errors='coerce') # if framerNr is not a number, it is replaces with nan
    data = data[data.frameNr.notna()] # filter out rows where frameNr is a nan     
    
    data = data[data.sampTime.notna()]
    data = data[data.user_pred_px_x.notna()]
    data = data[data.user_pred_px_y.notna()]
    data = data.apply(pd.to_numeric, errors='ignore')
    
    # Sort data in time
    data =  data.sort_values('frameNr')
    data = data.reset_index(drop=True)
    data = data.drop_duplicates(subset=['user_pred_px_x', 'user_pred_px_y'], ignore_index=True)

    
    # Extract meta information
    resX = data.resX[0]
    resY = data.resY[0]
    
    # Extract data in pixel coords
    x = data.user_pred_px_x.values
    y = data.user_pred_px_y.values
    t = data.sampTime.values
    ###
    # sometimes we have weird peaks where one sample is (very) far outside the
    # monitor. Here, count as missing any data that is more than one monitor
    # distance outside the monitor.
    missX = np.logical_or(x<-resX, x>2*resX)
    missY = np.logical_or(y<-resY, y>2*resY)
    miss = np.logical_or(missX, missY)
    x[miss] = missingXY
    y[miss] = missingXY
    
    return t, x, y
