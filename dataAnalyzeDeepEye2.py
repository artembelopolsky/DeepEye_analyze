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
import scipy.stats as stats

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


path_to_folders = 'C:/Users/artem/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/online'
# path_to_folders = 'D:/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/online'

# get all folder names
folder_names = os.listdir(path_to_folders)

pp_list = []
for fn in folder_names:
    path = os.path.join(path_to_folders, fn, fn+'_test_all.csv')       
        
    df = pd.read_csv(path)
        
    
    # Find the headers via duplicates and use it to split into datasets
    # Make indices of datasets
    mask_dup = df.duplicated(keep=False)
    idx_dup = df.index[mask_dup == True].tolist()
    idx_dup[:0] = [-1] # add lower index
    idx_dup.extend([df.shape[0]]) # add upper index
    
    # Use indices to parse datasets
    df_list = []
    count_datasets = 0
    last_numCalibDots = []
    for i in range(len(idx_dup)):
        if i < len(idx_dup) - 1:
            a = df.iloc[idx_dup[i]+1:idx_dup[i+1]]
            a = a.apply(pd.to_numeric, errors='ignore') # when header is written twice, some floats are str, fix this 
            a['dataset_num'] = count_datasets
            a['eucl_dist_px_orig'] = np_euclidean_distance(np.array(a[['x','y']]), np.array(a[['user_pred_px_x','user_pred_px_y']]))
            scale_cm_in_px = a.scrW_cm/a.resX
            a['eucl_dist_cm_orig'] = a.eucl_dist_px_orig * scale_cm_in_px 
            
            if pd.api.types.is_string_dtype(a.sona_pp_id) == True:
                a['platform'] = 'PROLIFIC'
            else:
                a['platform'] = 'SONA'
           
            # Label 25-dot conditions based on preceeding dataset
            if a.numCalibDots.iloc[0] == 25:
                print(f'last: {last_numCalibDots[-1]}')
                if last_numCalibDots[-1] == 9:
                    a['condition'] = '25_9'
                elif last_numCalibDots[-1] == 13:
                    a['condition'] = '25_13'
                
            else:
                a['condition'] = a.numCalibDots.astype(str)
                
            last_numCalibDots.append(a.numCalibDots.iloc[-1]) # log last value                
            
            # Accumulate all dataset per subject
            df_list.append(a)
            count_datasets += 1
    
    
    # if there are more than 4 datasets, remove the recalibrated ones, pick the last one
    last_numCalibDots = pd.Series(last_numCalibDots)
    idx_good_datasets = last_numCalibDots.loc[last_numCalibDots.shift(-1) != last_numCalibDots] # shift dataset by one row and get indices
    df_list = [df_list[i] for i in list(idx_good_datasets.index)] # pick only the 4 datasets
    assert(len(df_list) == 4)
    
    # Concatenate all datasets per subject
    b = pd.concat(df_list)
    
    # Add a subj_nr column
    b['subj_nr'] = fn    
    
    # Accumulate datasets across subjects
    pp_list.append(b)

# Concatenate all subjects in one df
df_all = pd.concat(pp_list)



# Every display resolution is scaled to this one since all dots are drawn in % display size in px
target_resX = 1280.0
target_resY = 800.0    

# To do:
# How many attempts
# Filter out failed last attemtps
# How to deal with missing data for some dots (e.g. for '2023_04_13_13_53_28')

df_all = df_all.reset_index()


# Select subset
"""

'2023_04_15_11_42_39' - amazing performance, but did not do 25_9
"""
# df_all = df_all[df_all.numCalibDots == 9]
# df_all = df_all[(df_all.subj_nr == '2023_04_15_12_22_19')]
# Exclude subjects
df_all = df_all[df_all.subj_nr != '2023_04_07_13_59_57'] # my pilot data
df_all = df_all[df_all.subj_nr != '2023_04_07_13_45_47'] # my pilot data


# user_predictions_px = np.array(df_all[['user_pred_px_x', 'user_pred_px_y']])
df_all['user_pred_px_x_scaled'] = df_all.user_pred_px_x/df_all.resX * target_resX
df_all['user_pred_px_y_scaled'] = df_all.user_pred_px_y/df_all.resY * target_resY
# ground_truths_px = np.array(df_all[['x','y']])
df_all['x_scaled'] = np.round(df_all.x/df_all.resX * target_resX)
df_all['y_scaled'] = np.round(df_all.y/df_all.resY * target_resY)

df_all['scale_cm_in_px'] = df_all.scrW_cm.astype(float)/df_all.resX.astype(float)
scale_cm_in_px = df_all.scale_cm_in_px.mean() # average scaling factor
# scale_cm_in_px = df_all.scrW_cm.astype(float)[0]/df_all.resX.astype(float)[0]  

# Get indices of unique dot positions (unique rows)
u, indices = np.unique(np.array([df_all.x_scaled, df_all.y_scaled]).T, axis=0, return_inverse=True)
df_all['unique_dot'] = indices



""" 
Plotting

"""
fig2, ax2 = plt.subplots(nrows=2, ncols=2)
fig2.set_size_inches((8.5, 7.0), forward=False)
fig2.tight_layout()
cell = [[0,0],[0,1],[1,0], [1,1]] 

count_plots2 = 0

# Iterate per condition
for name, df in df_all.groupby('condition'):


    heatmap = makeHeat([target_resX, target_resY], np.array(df.user_pred_px_x_scaled), np.array(df.user_pred_px_y_scaled))
    
    # f, ax = plt.subplots()
    # f.set_size_inches(target_resX/100., target_resY/100.)            
                
    # ax.imshow(heatmap, cmap=cm.hot, extent=[0, target_resX, target_resY, 0], alpha = 0.5, aspect='equal')                   
    
    # plt.scatter(df.user_pred_px_x_scaled, df.user_pred_px_y_scaled, c='r', s=10, alpha=0.5)
    # plt.scatter(df.x_scaled, df.y_scaled, c='g', s=40, alpha=0.5)
                
    # plt.axis('off')  
    
    median_pred_x = df.groupby('unique_dot').user_pred_px_x_scaled.median()
    median_pred_y = df.groupby('unique_dot').user_pred_px_y_scaled.median()
    std_pred_x = df.groupby('unique_dot').user_pred_px_x_scaled.std()
    std_pred_y = df.groupby('unique_dot').user_pred_px_y_scaled.std()
    true_x = df.groupby('unique_dot').x_scaled.mean()
    true_y = df.groupby('unique_dot').y_scaled.mean() 
    
    
    # Plot median errors, lines
    # plt.scatter(median_pred_x, median_pred_y, c='b', s=40)
    # plt.plot([median_pred_x, true_x], [median_pred_y, true_y], c='black')
    
    
    # calculate the distance between median of all samples (as plotted)
    dist = np_euclidean_distance(np.array([median_pred_x, median_pred_y]).T, np.array([true_x, true_y]).T)
    dist_cm = dist *  scale_cm_in_px
    std_pred_x_cm = std_pred_x * scale_cm_in_px
    std_pred_y_cm = std_pred_y * scale_cm_in_px
    # plt.title(f'Condition:{df.condition.iloc[0]}\n Mean error: {np.round(dist_cm.mean(),1)}cm, Std (x,y): ({np.round(std_pred_x_cm.mean(),1)}cm, {np.round(std_pred_y_cm.mean(),1)}cm)', fontsize=26)
    
    
    # for x,y,e in zip(np.array(true_x), np.array(true_y), np.round(dist_cm, 1)):
    #     plt.text(x, y, e, fontsize=18)


    # Save plot
    # plt.savefig(f'calibration{df.condition.iloc[0]}.jpg', dpi=100, pad_inches=0)
    
            
    row = cell[count_plots2][0]
    column = cell[count_plots2][1]
    # Plot heatmap
    ax2[row, column].imshow(heatmap, cmap=cm.hot, extent=[0, target_resX, target_resY, 0], alpha = 0.5, aspect='equal')  
    # Plot true pos and predicted median errors, lines
    ax2[row, column].scatter(df.x_scaled, df.y_scaled, c='g', s=40, alpha=0.5)
    ax2[row, column].scatter(median_pred_x, median_pred_y, c='b', s=40)
    ax2[row, column].plot([median_pred_x, true_x], [median_pred_y, true_y], c='black')
    # Title
    ax2[row, column].set_title(f'Condition:{df.condition.iloc[0]}\n Mean error: {np.round(dist_cm.mean(),1)}cm, Std (x,y): ({np.round(std_pred_x_cm.mean(),1)}cm, {np.round(std_pred_y_cm.mean(),1)}cm)')
    # Error numbers
    for x,y,e in zip(np.array(true_x), np.array(true_y), np.round(dist_cm, 1)):
        ax2[row, column].text(x, y, e, fontsize=18)
    
    count_plots2 += 1        
  
    
 # Save plot
fig2.suptitle(f'N={df.subj_nr.unique().size}', fontsize=16)
fig2.subplots_adjust(top=0.9)
fig2.savefig('calibration.jpg', dpi=1000, pad_inches=0)
"""
Plotting mean E.d. and SD per condition
"""

fig, ax = plt.subplots(nrows=2, ncols=4)
fig.set_size_inches((8.5, 11), forward=False)

count_plots = 0
for name, i in df_all.groupby('condition'):
    
    # Get median per each unique dot, separately per subject and condition
    summary_df_ed = i.groupby(['subj_nr', 'condition', 'unique_dot'])[['eucl_dist_px_orig', 'eucl_dist_cm_orig']].median().reset_index()
    summary_df_std = i.groupby(['subj_nr', 'condition', 'unique_dot'])[['eucl_dist_px_orig', 'eucl_dist_cm_orig']].std().reset_index()
    
    # Aggregate over dots (mean E.d. and SD E.d.)
    agg_Ed = summary_df_ed.groupby(['subj_nr', 'condition'])[['eucl_dist_px_orig', 'eucl_dist_cm_orig']].mean().reset_index()
    agg_SD = summary_df_std.groupby(['subj_nr', 'condition'])[['eucl_dist_px_orig', 'eucl_dist_cm_orig']].mean().reset_index()
    
    # Get subj nr for the largest error
    larg_err_subj = agg_Ed.where(agg_Ed.eucl_dist_cm_orig==agg_Ed.eucl_dist_cm_orig.max()).dropna().subj_nr
    print(f'Maximum E.d. error is: {agg_Ed.eucl_dist_cm_orig.max(), larg_err_subj}')
    print('\nMean Euclidean distance:')
    print(agg_Ed)
    print('\nStandard deviation of Euclidean distances:')
    print(agg_SD)
    
    # Plot euclidean distances per subject
    ax[0, count_plots].title.set_text(f'Condition:{i.condition.iloc[0]}\nEuclidean distances')
    ax[0, count_plots].set_ylim(0,4.5)
    ax[0, count_plots].scatter(np.ones(agg_Ed.eucl_dist_cm_orig.size),agg_Ed.eucl_dist_cm_orig)
    ax[0, count_plots].scatter(1,agg_Ed.eucl_dist_cm_orig.mean())
    
    # Plot SD per subject
    ax[1, count_plots].title.set_text(f'Condition:{i.condition.iloc[0]}\nSDs')
    ax[1, count_plots].set_ylim(0,4.5)
    ax[1, count_plots].scatter(np.ones(agg_SD.eucl_dist_cm_orig.size),agg_SD.eucl_dist_cm_orig)
    ax[1, count_plots].scatter(1,agg_SD.eucl_dist_cm_orig.mean())
    
    
    
    
    count_plots += 1



    
# Save plot
fig.tight_layout()
fig.suptitle(f'N={df.subj_nr.unique().size}', fontsize=16)
fig.subplots_adjust(top=0.9)
fig.savefig('summary.jpg', dpi=1000)

# T-tests
x = df_all.groupby(['subj_nr','condition']).eucl_dist_cm_orig.mean()
x = x.unstack()
print('T-Test: 13 vs. 9')
print(stats.ttest_rel(np.array(x['13']), np.array(x['9'])))
print('T-Test: 25_13 vs. 25_9:')
print(stats.ttest_rel(np.array(x['25_13']), np.array(x['25_9'])))
print(x.mean())

# Subject descriptive statistics
descr_stats = df_all.groupby(['platform', 'subj_nr', 'sona_pp_id'])['unique_dot'].count().reset_index()
print(f'Descriptive Stats:\n {descr_stats}')


# Read the training file
path_to_folders = 'C:/Users/artem/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/online'
# path_to_folders = 'D:/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/online'

# get all folder names
folder_names = os.listdir(path_to_folders)

path = os.path.join(path_to_folders, fn, fn+'_train_all.csv')       
    
df = pd.read_csv(path)
