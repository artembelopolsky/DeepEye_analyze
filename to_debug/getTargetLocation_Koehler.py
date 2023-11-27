# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 14:06:40 2023

@author: artem
"""

import pandas as pd
import numpy as np
import os

path = 'C:/Users/artem/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/Test_SceneViewing'
filename = 'random_cues_final.csv'

# read design file
df1 = pd.read_csv(os.path.join(path, filename))

# drop column
df1 = df1.drop('Unnamed: 0', axis=1)

# get list of filenames with Target Locations
filenames_cues = os.listdir(os.path.join(path, 'Koehler_etal_dataset/Cues'))

# extract target locations from a filename
target_locations = [i.split('_')[-1][:-4] for i in filenames_cues]

# extract image number from a filename
image_numbers = [int(i.split('_')[1]) for i in filenames_cues]

df2 = pd.DataFrame({'filenames_cues': filenames_cues, 'Image No.': image_numbers, 'target_locations': target_locations})

# Merge DataFrames based on the 'Image No.' column
merged_df = pd.merge(df1, df2, on='Image No.', how='inner')

# Create new column with all variables
def outputVars(row1, row2, row3):
    output = f'[{row1}, "{row2}", "{row3}"],'
    
    return output
    
    
merged_df['ImageNumCueWordTargetloc'] = merged_df.apply(lambda row: outputVars(row['Image No.'], row['Cue'], row['target_locations']), axis=1)


# Save new design file for SceneViewing Experiment
merged_df.to_csv(os.path.join(path, 'sceneViewing_design.csv'), index=False)

