# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:22:51 2023

@author: Artem
"""


import numpy as np
import pandas as pd

def trainUserModel():

    # experiment_id = request.headers['experiment-id'] # experiment folder name
    # temp_csv_id = request.headers['participant-id']
    # type_dataset = request.headers['type-dataset'] # train or test calibration dataset
    # final_batch = request.headers['final-batch']
    
    # debug
    temp_csv_id = '2023_10_09_12_12_19'   
    type_dataset = 'train' 

    # load training or testing data for calibration
    print('Loading the user calibration data')
    df = pd.read_csv(f'{temp_csv_id}_{type_dataset}.csv')
        
    
    # # header columns are: embeddings vector (0-127), ground_truth_x, ground_truth_y, frameNr
    # predictions = np.array(df.iloc[:, :-3])
    # ground_truths = np.array(df.iloc[:, -3:-1])

    # # train and save the model  
    # print('gridsearch for best params :')
    # params = grid_searchSVR(predictions, ground_truths)
    # #print(params)
    # print('Training SVR')
    # SVR_model = trainSVR(
    #     predictions,
    #     ground_truths,
    #     params['C'],
    #     params['gamma'],
    #     params['epsilon']
    #     )

    # # save svr model in user folder
    # with open(f'/mnt/{experiment_id}/{temp_csv_id}/{temp_csv_id}_model.pkl','wb') as file:
    #     pickle.dump(SVR_model,file)

    # # delete the current training dataset
    # os.remove(f'/mnt/{experiment_id}/{temp_csv_id}/{temp_csv_id}_{type_dataset}.csv')
    
    # if final_batch == 'keep_traindataset':
    #     # save file for all training attempts
    #     output_file = f'/mnt/{experiment_id}/{temp_csv_id}/{temp_csv_id}_{type_dataset}_all.csv'
    #     df.to_csv(output_file, mode='a', index=True) # write header every time to keep track of training attemtps
        
    # return json.dumps('Done model training')


temp_csv_id = '2023_10_09_12_12_19'   
type_dataset = 'train' 

# load training or testing data for calibration
print('Loading the user calibration data')
df = pd.read_csv(f'{temp_csv_id}_{type_dataset}.csv')

# filter out the rows where nan is present
df = df.dropna()

df.isnull().values.any()