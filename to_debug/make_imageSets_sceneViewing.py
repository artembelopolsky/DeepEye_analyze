# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 11:35:49 2023

@author: Artem
"""
import pandas as pd
import numpy as np
import os

# Specify the file path and delimiter (in this case, tab '\t')
file_path = 'D:/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/Test_SceneViewing'
file_name = 'selected_cues.txt'
delimiter = '\t'

# Read the text file using pd.read_table() and specify the delimiter
df = pd.read_table(os.path.join(file_path, file_name), sep=delimiter)

# Add a column with image filename
df['ImageFileName'] = df.iloc[:,0].apply(lambda x: f'"imagesExp/image_r_{x}.jpg",')


# Add a column with Image Number and cue Word ready for copy/paste into js
df['ImageNumCueWord'] = df.apply(lambda row: f"[{row['Image No.']}, '{row['Cue']}'],", axis=1)


# Now, df contains your data from the text file
print(df)



def has_adjacent_duplicates(df):
    
    # Find identical consecutive rows in Cue
    identical_consecutive_rows = df.Cue == df.Cue.shift()

    # Select rows with identical consecutive rows
    rows_with_identical_consecutive = df.Cue[identical_consecutive_rows]
    
    return rows_with_identical_consecutive.any()

# Randomize all the rows
df_randomized = df.sample(frac=1).reset_index(drop=True)

# The df_randomized DataFrame now contains all rows from df randomized
print(df_randomized)


# Loop until there are no adjacent duplicates
# not really necessary, since each set will be randomized during experiment
while has_adjacent_duplicates(df_randomized):
    # Randomize all the rows
    df_randomized = df.sample(frac=1).reset_index(drop=True)
    


# Label first half as imageSetA and the second half as imageSetB
# Calculate the halfway point of the DataFrame
halfway_point = len(df_randomized) // 2

# Label the first half of the rows as 'imageSetA'
df_randomized.loc[:halfway_point - 1, 'Label'] = 'imageSetA'

# Label the second half of the rows as 'imageSetB'
df_randomized.loc[halfway_point:, 'Label'] = 'imageSetB'

# Save dataframe
df_randomized.to_csv(os.path.join(file_path, 'random_cues.csv'))


