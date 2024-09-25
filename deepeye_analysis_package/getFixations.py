import os
import numpy as np
import pandas as pd
from FixationDetection.I2MC import runI2MC


def extract_fixations(df, path):
    """
    Extracts fixation information from eye-tracking data using the I2MC algorithm, merges the results
    with the original data, and computes additional metrics such as fixation duration and distance
    from the previous fixation.

    The function:
    1. Preprocesses the dataframe by filtering and sorting the data.
    2. Runs the I2MC fixation detection algorithm to obtain fixations.
    3. Adds extracted fixation information (e.g., fixation start/end times, fixation coordinates)
       to the original dataframe.
    4. Computes additional metrics, such as the distance from the previous fixation and 
       the previous fixation coordinates.
    5. Saves the processed dataframe as a CSV file and returns it.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing eye-tracking data. It should include columns such as 'sampTime', 
        'user_pred_px_x', 'user_pred_px_y', and 'event'.
    
    path : str
        The path to the data file for running the I2MC fixation detection algorithm.

    Returns
    -------
    pandas.DataFrame
        The preprocessed dataframe, filtered to include only rows where the target was presented.
    """
    
    # Preprocess the input dataframe: Remove invalid or missing data
    df = df[df.fName.notna()]  # Filter rows where fName is not NaN
    df['frameNr'] = pd.to_numeric(df['frameNr'], errors='coerce')  # Convert frameNr to numeric, set non-numeric to NaN
    df = df[df['frameNr'].notna()]  # Filter out rows where frameNr is NaN
    
    df['sampTime'] = pd.to_numeric(df['sampTime'], errors='coerce')  # Convert sampTime to numeric
    df = df[df['sampTime'].notna()]  # Filter rows where sampTime is NaN
    df = df[df['user_pred_px_x'].notna() & df['user_pred_px_y'].notna()]  # Filter rows with valid x and y coordinates
    df = df.apply(pd.to_numeric, errors='ignore')  # Convert any possible strings to numbers
    df = df.sort_values('frameNr').reset_index(drop=True)  # Sort by frame number
    df = df.drop_duplicates(subset=['sampTime'], ignore_index=True)  # Drop duplicate sampTime entries
    
    # Run the I2MC fixation detection algorithm to extract fixation data
    fixDF = runI2MC(path, plotData=False)

    # Initialize arrays for storing extracted fixation data
    FixXPos = np.zeros(df.shape[0])
    FixYPos = np.zeros(df.shape[0])
    FixStartEnd = np.empty(df.shape[0], dtype='U10')
    FixStartEnd.fill('')  # Fill array with empty strings
    FixDur = np.zeros(df.shape[0])
    DistFromPrevFix = np.zeros(df.shape[0])
    PrevFixXPos = np.zeros(df.shape[0])
    PrevFixYPos = np.zeros(df.shape[0])
    PrevFixSampTime = np.zeros(df.shape[0])

    prev_fix_x = None  # Keep track of the previous fixation's x-coordinate
    prev_fix_y = None  # Keep track of the previous fixation's y-coordinate
    prev_fix_sampTime = 0  # Previous fixation's sample time
    idx = 0  # Index for iterating through fixDF

    # Iterate through the dataframe to accumulate fixation information
    for index, row in df.iterrows():
        if idx >= fixDF.shape[0]:
            break  # Stop if we run out of fixations to process
        
        # Move to the next fixation if the sample time exceeds the current fixation's end time
        if row['sampTime'] > fixDF['FixEnd'].iloc[idx]:
            idx += 1
            if idx >= fixDF.shape[0]:
                break
        
        # Check if the current sample time falls within a fixation
        if fixDF['FixStart'].iloc[idx] <= row['sampTime'] <= fixDF['FixEnd'].iloc[idx]:
            FixXPos[index] = fixDF['XPos'].iloc[idx]
            FixYPos[index] = fixDF['YPos'].iloc[idx]
        
        # Label fixation start and end
        if row['sampTime'] == fixDF['FixStart'].iloc[idx]:
            FixStartEnd[index] = 'fix_start'
            if prev_fix_x is not None:
                # Calculate the distance from the previous fixation
                DistFromPrevFix[index] = np.sqrt((fixDF['XPos'].iloc[idx] - prev_fix_x) ** 2 + 
                                                 (fixDF['YPos'].iloc[idx] - prev_fix_y) ** 2)
                PrevFixXPos[index] = prev_fix_x
                PrevFixYPos[index] = prev_fix_y
                PrevFixSampTime[index] = prev_fix_sampTime
        
        elif row['sampTime'] == fixDF['FixEnd'].iloc[idx]:
            FixStartEnd[index] = 'fix_end'
            FixDur[index] = fixDF['FixDur'].iloc[idx]
            prev_fix_x = fixDF['XPos'].iloc[idx]
            prev_fix_y = fixDF['YPos'].iloc[idx]
            prev_fix_sampTime = row['sampTime']

    # Add extracted fixation data to the original dataframe
    df['FixXPos'] = FixXPos
    df['FixYPos'] = FixYPos
    df['FixStartEnd'] = FixStartEnd
    df['FixDur'] = FixDur
    df['DistFromPrevFix'] = DistFromPrevFix
    df['PrevFixSampTime'] = PrevFixSampTime
    df['PrevFixXPos'] = PrevFixXPos
    df['PrevFixYPos'] = PrevFixYPos

    # Filter out invalid fixations and coordinates
    df = df[(df['FixXPos'] > 0) & (df['FixYPos'] > 0) & 
            (df['user_pred_px_x'] > 0) & (df['user_pred_px_y'] > 0)]

    # Save the pre-processed dataframe to a CSV file
    output_file = os.path.splitext(path)[0] + '_extra.csv'
    df.to_csv(output_file, index=False)

    return df


def batch_extract_fixations(path_to_data):
    """
    Process all participant data files in the specified folder using the extract_fixations function.

    This function iterates through all subdirectories (or files) in the given folder, applies the 
    extract_fixations function to each participant's data file, and skips files that do not exist 
    or cause errors during reading.

    Parameters
    ----------
    path_to_data : str
        Path to the folder containing participant data files.

    Returns
    -------
    None
        The function processes the data and saves the results to new CSV files for each participant.
    """

    # Get all folder names (or participant directories) in the data folder
    folder_names = os.listdir(path_to_data)

    # Process each participant's data file
    for fn in folder_names:
        path_to_file = os.path.join(path_to_data, fn, fn + '_record.csv')

        print(f'Processing participant {fn}...')

        # Read the file and handle any potential errors
        try:
            # Use 'on_bad_lines' to skip rows with issues, such as improperly formatted lines
            df = pd.read_csv(path_to_file, on_bad_lines='skip')
        except FileNotFoundError:
            print(f'File does not exist: {path_to_file}')
            continue
        except Exception as e:
            print(f'Error reading file {path_to_file}: {e}')
            continue

        # Apply extract_fixations to the dataframe
        try:
            extract_fixations(df, path_to_file)
        except Exception as e:
            print(f'Error during fixation extraction for {fn}: {e}')
            continue

    print('Processing complete.')
   

# Example usage:
# path_to_data = 'D:/Dropbox/Appliedwork/CognitiveSolutions/Projects/DeepEye/TechnicalReports/TechnicalReport1/Test_Spaak/data/approved/tmp'
# batch_extract_fixations(path_to_data)

