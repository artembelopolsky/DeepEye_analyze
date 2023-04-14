# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 09:45:38 2023

@author: artem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import astropy.convolution as krn

def makeHeat(screenRes, xPos, yPos):
        xMax = screenRes[0]
        yMax = screenRes[1]
        xMin = 0
        yMin = 0
        kernelPar = 50

        # Input handeling
        xlim = np.logical_and(xPos < xMax, xPos > xMin)
        ylim = np.logical_and(yPos < yMax, yPos > yMin)
        xyLim = np.logical_and(xlim, ylim)
        dataX = xPos[xyLim]
        dataX = np.floor(dataX)
        dataY = yPos[xyLim]
        dataY = np.floor(dataY)

        # initiate map and gauskernel
        gazeMap = np.zeros([int((xMax-xMin)),int((yMax-yMin))])+0.0001
        gausKernel = krn.Gaussian2DKernel(kernelPar)

        # Rescale the position vectors (if xmin or ymin != 0)
        dataX -= xMin
        dataY -= yMin

        # Now extract all the unique positions and number of samples
        xy = np.vstack((dataX, dataY)).T
        uniqueXY, idx, counts = uniqueRows(xy)
        uniqueXY = uniqueXY.astype(int)
        # populate the gazeMap
        gazeMap[uniqueXY[:,0], uniqueXY[:,1]] = counts

        # Convolve the gaze with the gauskernel
        heatMap = np.transpose(krn.convolve_fft(gazeMap,gausKernel))
        heatMap = heatMap/np.max(heatMap)

        return heatMap

def uniqueRows(x):
    y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx, counts = np.unique(y, return_index=True, return_counts = True)
    uniques = x[idx]
    return uniques, idx, counts


def np_euclidean_distance(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1))

def dot_error(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    eucl_dist = np_euclidean_distance(y_true, y_pred)
    # Get indices of unique dot positions
    u, indices = np.unique(y_true, axis=0, return_inverse=True)
    # Make dataframe for each sample of unique dot label and error distance
    df_dict = {'unique_dot': indices, 'eucl_distance': eucl_dist, 'true_x': y_true[:,0],
                'true_y': y_true[:,1], 'pred_x': y_pred[:,0], 'pred_y': y_pred[:,1]}
    df = pd.DataFrame(df_dict)
    # Group by unique dot position, compute median error per dot, average across dots
    mean_dot_error = df.groupby('unique_dot').eucl_distance.median().mean()
    std_dot_error = df.groupby('unique_dot').eucl_distance.median().std()

    return float(mean_dot_error), df, float(std_dot_error)



# path_to_folders = 'C:/Users/artem/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1'
# path_to_folders = 'D:/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1'

path_to_folders = 'C:/Users/artem/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/lab_pilot'
# path_to_folders = 'D:/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1'


folder_names = os.listdir(path_to_folders)

pp_list = []

for fn in folder_names:
    path = os.path.join(path_to_folders, fn, fn+'_test_all.csv')
    
    
        
    df = pd.read_csv(path)
    
    mask_dup = df.duplicated(keep=False)
    idx_dup = df.index[mask_dup == True].tolist()
    idx_dup[:0] = [-1]
    idx_dup.extend([df.shape[0]])
    
    df_list = []
    count_datasets = 0
    for i in range(len(idx_dup)):
        if i < len(idx_dup) - 1:
            a = df.iloc[idx_dup[i]+1:idx_dup[i+1]]
            a = a.apply(pd.to_numeric, errors='ignore') # when header is written twice, some floats are str, fix this 
            a['num_calibration_dots'] = a.unique_dot.max() + 1
            a['dataset_num'] = count_datasets
                   
            df_list.append(a)
            count_datasets += 1
    
    
    # if there are more than 4 datasets, I should remove the recalibrated ones
    b = pd.concat(df_list)
    b['subj_nr'] = fn
    
    pp_list.append(b)

c = pd.concat(pp_list)

y_true = np.array(a[['true_x','true_y']])
y_pred = np.zeros(y_true.shape)
dist = np_euclidean_distance(y_true, y_pred)



# for name, i in b.groupby('dataset_num'):
#     plt.figure()
#     plt.scatter(np.array(i.pred_x), np.array(i.pred_y))
#     plt.scatter(i.true_x, i.true_y)
#     plt.gca().invert_yaxis()
    
    
d = c[c.num_calibration_dots == 13]
d = d[d.subj_nr != '2023_03_13_13_27_40']
d = d[d.subj_nr != '2023_03_27_11_43_30']

# for name, i in d.groupby(['subj_nr','unique_dot']):
#     median_pred_x = i.pred_x.median()
#     median_pred_y = i.pred_y.median()
#     true_x = i.true_x.mean()
#     true_y = i.true_y.mean()    
    
    # plt.figure()
    # plt.scatter(np.array(median_pred_x), np.array(median_pred_y))
    # plt.scatter(true_x, true_y)
    # plt.gca().invert_yaxis()
    
d2 = d.groupby(['subj_nr', 'unique_dot'], as_index=False).median()


# plt.scatter(d.true_x, d.true_y)
# plt.scatter(d.pred_x, d.pred_y)

df = d.reset_index()

user_predictions_px = np.array(df[['pred_x', 'pred_y']])
ground_truths_px = np.array(df[['true_x','true_y']])        
scale_cm_in_px = df.scrW_cm.astype(float)[0]/df.resX.astype(float)[0]  
 

heatmap = makeHeat([df.resX[0], df.resY[0]], user_predictions_px[:,0], user_predictions_px[:,1])

f, ax = plt.subplots()
f.set_size_inches(df.resX[0]/100., df.resY[0]/100.)            
            
ax.imshow(heatmap, cmap=cm.hot, extent=[0, df.resX[0], df.resY[0], 0], alpha = 0.5, aspect='equal')                   

# plt.scatter(user_predictions_px[:, 0], user_predictions_px[:, 1], c='r', s=10, alpha=0.5)
plt.scatter(ground_truths_px[:, 0], ground_truths_px[:, 1], c='g', s=40, alpha=0.5)
            
# plt.axis('off')            

median_pred_x = df.groupby('unique_dot').pred_x.median()
median_pred_y = df.groupby('unique_dot').pred_y.median()
std_pred_x = df.groupby('unique_dot').pred_x.std()
std_pred_y = df.groupby('unique_dot').pred_y.std()
true_x = df.groupby('unique_dot').true_x.mean()
true_y = df.groupby('unique_dot').true_y.mean() 


 # Plot median errors, lines
plt.scatter(median_pred_x, median_pred_y, c='b', s=40)
plt.plot([median_pred_x, true_x], [median_pred_y, true_y], c='black')


# Add text with error per dot
err =  np.array(df.groupby('unique_dot').eucl_distance.median())
err_cm = err * scale_cm_in_px
# for x,y,e in zip(np.array(true_x), np.array(true_y), np.round(err_cm, 1)):
#     plt.text(x, y, e, fontsize=18)

# Standard deviation per each dot
stdev_err =  np.array(df.groupby('unique_dot').eucl_distance.std())
stdev_err_cm = stdev_err * scale_cm_in_px

# convert to cm
mean_dot_error_cm = np.round(err_cm.mean(), 1)
std_dot_error_cm = np.round(stdev_err_cm.mean(), 1)
# plt.title(f'Mean error: {mean_dot_error_cm}cm, Std: {np.round(std_dot_error_cm,1)}cm', fontsize=26)

# calculate the distance between median of all samples (as plotted)
dist = np_euclidean_distance(np.array([median_pred_x, median_pred_y]).T, np.array([true_x, true_y]).T)
dist_cm = dist *  scale_cm_in_px
std_pred_x_cm = std_pred_x * scale_cm_in_px
std_pred_y_cm = std_pred_y * scale_cm_in_px
plt.title(f'Mean error: {np.round(dist_cm.mean(),1)}cm, Std: ({np.round(std_pred_x_cm.mean(),1)}cm, {np.round(std_pred_y_cm.mean(),1)}cm)', fontsize=26)


for x,y,e in zip(np.array(true_x), np.array(true_y), np.round(dist_cm, 1)):
    plt.text(x, y, e, fontsize=18)


# Save plot
# plt.savefig('calibration13.jpg', dpi=100, pad_inches=0)




###### Read log file

# path = 'C:/Users/artem/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/2023_03_20_09_25_13/2023_03_20_09_25_13_log.csv'
        
# df = pd.read_csv(path)

# idx_dup = df.index[df['Unnamed: 0'] == 0].tolist()
# idx_dup.extend([df.shape[0]])

# df_list = []
# for i in range(len(idx_dup)):
#     if i < len(idx_dup) - 1:
#         a = df.iloc[idx_dup[i]:idx_dup[i+1]]
#         a['num_calibration_dots'] = a.shape[0]
#         a = a.apply(pd.to_numeric, errors='ignore') # when header is written twice, some floats are str, fix this        
#         df_list.append(a)        


# # if there are more than 4 datasets, I should remove the recalibrated ones
# b = pd.concat(df_list)
