# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 11:56:00 2023

@author: artem
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from plot_utils import create_plot_to_base64, cam_to_pix, pix_to_cam,wrap_pix_to_cam, dot_error, makeHeat

# def plotValidationResults():

   
# debug
temp_csv_id = '2023_09_24_09_08_25'   
type_dataset = 'test' 

# load training or testing data for calibration
print('Loading the user validation data')
df = pd.read_csv(f'{temp_csv_id}_{type_dataset}.csv')
df = df[~df.frameNr.isin(['frameNr'])] # remove rows for which frameNr contains'frameNr' header
df = df.apply(pd.to_numeric, errors='ignore') # when header is written twice, some floats are str, fix this

# filter out the rows where important variables are nan
df = df[df.sampTime.notna()]
df = df[df.user_pred_px_x.notna()]
df = df[df.user_pred_px_y.notna()]
df = df[df.x.notna()]
df = df[df.y.notna()]
df = df[df.scrW_cm.notna()]
df = df[df.resX.notna()]
df = df[df.resY.notna()]

#df = df.dropna() # remove rows containing NaN values

#df.drop_duplicates(subset=['user_pred_px_x', 'user_pred_px_y'], inplace=True) # removes duplicated rows in place
# df = df.dropna() # remove rows containing NaN values
df = df.sort_values('frameNr').reset_index(drop=True)

user_predictions_px = np.array(df[['user_pred_px_x', 'user_pred_px_y']])
ground_truths_px = np.array(df[['x','y']])        
scale_cm_in_px = df.scrW_cm.astype(float)[0]/df.resX.astype(float)[0]  

heatmap = makeHeat([df.resX[0], df.resY[0]], user_predictions_px[:,0], user_predictions_px[:,1])

f, ax = plt.subplots()
f.set_size_inches(df.resX[0]/100., df.resY[0]/100.)            
            
ax.imshow(heatmap, cmap=cm.hot, extent=[0, df.resX[0], df.resY[0], 0], alpha = 0.5, aspect='equal')                   

plt.scatter(user_predictions_px[:, 0], user_predictions_px[:, 1], c='r', s=10, alpha=0.5)
plt.scatter(ground_truths_px[:, 0], ground_truths_px[:, 1], c='g', s=10, alpha=0.5)
            
plt.axis('off')            