"""
Collection of functions for processing the data from the DeepEye eye tracker

"""
import pandas as pd
import ast  # For literal_eval


def getFixationLatency(df):
    """
    Calculate fixation latency and fixation order for each trial in the provided dataframe.

    The function performs the following steps:
    1. Retrieves the first timestamp of when the target was presented for each trial.
    2. Merges this timestamp into the main dataframe.
    3. Extracts the rows where a fixation starts and computes the fixation latency (the time between when the target is presented and the fixation starts).
    4. Clips negative latencies to zero (to handle fixations that carry over from before the target was presented).
    5. Ranks the fixations in order of their occurrence within each trial.
    6. Returns a modified dataframe that includes the fixation latency and order for each fixation event (start and end).

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing eye-tracking data time-locked to the event of interest (e.g.'target_on') with the following columns:
        - 'trialNr': int, trial number.
        - 'sampTime': float, sample time when a data point was recorded.
        - 'FixStartEnd': str, either 'fix_start' or 'fix_end' indicating the fixation state.
    
    Returns
    -------
    pandas.DataFrame
        A modified dataframe with additional columns:
        - 'FixLatency': time between when the target was presented and when fixation started.
        - 'FixationOrder': the rank order of each fixation within a trial.
        Only the rows corresponding to 'fix_start' and 'fix_end' are included.
    """
    
    # Step 1: Get the first sample time when the target was presented for each trial
    sampTime_df = df.drop_duplicates(subset=['trialNr'], keep='first', ignore_index=True)[['trialNr', 'sampTime']]
    sampTime_df.columns = ['trialNr', 'targSampTime']  # Rename columns for clarity
    
    # Step 2: Merge target presentation time back into the main dataframe
    df = pd.merge(df, sampTime_df, on="trialNr", how="left")
    
    # Step 3: Select rows where fixation started
    fl_df = df[df['FixStartEnd'] == 'fix_start'].copy()
    
    # Step 4: Compute fixation latency (sampTime - targSampTime)
    fl_df['FixLatency'] = fl_df['sampTime'] - fl_df['targSampTime']
    
    # Step 5: Clip negative fixation latencies to zero
    fl_df['FixLatency'] = fl_df['FixLatency'].clip(lower=0)
    
    # Step 6: Rank the fixation latencies within each trial
    fl_df['FixationOrder'] = fl_df.groupby('trialNr')['FixLatency'].rank(method='first')
    
    # Step 7: Extract relevant columns for merging back into the main dataframe
    fl_df = fl_df[['sampTime', 'FixLatency', 'FixationOrder']]
    
    # Step 8: Filter the original dataframe to include only 'fix_start' and 'fix_end' rows
    df_start_end = df[df['FixStartEnd'].isin(['fix_start', 'fix_end'])]
    
    # Step 9: Merge the fixation latency and order back into the filtered dataframe
    df_modified = pd.merge(df_start_end, fl_df, on="sampTime", how="left")
    
    return df_modified


def handle_carryover_fixations_and_merge(df, max_event_duration):
    """
    Identify, label, and correct carryover fixations, and merge fixation start and end events.

    This function handles two types of carryover fixations:
    1. Fixation that starts in the previous trial and ends at the beginning of the current trial.
    2. Fixation that starts near the end of the current trial and finishes after the trial has ended.

    The function:
    - Identifies carryover fixations and inserts missing events.
    - Merges fixation start and end events within trials.
    - Handles fixations that cross trial boundaries (carryover fixations).
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing eye-tracking fixation data with columns:
        - 'trialNr': trial number
        - 'FixStartEnd': indicating 'fix_start' or 'fix_end'
        - 'sampTime': timestamp of the sample
        - 'FixDur': fixation duration
        - 'FixXPos': X position of the fixation
        - 'FixYPos': Y position of the fixation
        - 'targSampTime': timestamp when the target appeared
        - 'frameNr': frame number for sorting
    
    max_event_duration : int
        Maximum duration of a trial or event in milliseconds.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with merged fixation start and end events, including inserted events to account for carryover fixations.
    """
    
    fixcarryover_groups = []
    
    # Step 1: Handle carryover fixations by iterating over each trial
    for trial_nr, group in df.groupby('trialNr'):
        
        # fix_start = group[group.FixStartEnd == 'fix_start']
        # fix_end = group[group.FixStartEnd == 'fix_end']

        # Handle case where trial starts with 'fix_end' (carryover from previous trial)
        if group.FixStartEnd.iloc[0] == 'fix_end':
            group = group.copy()
            group.FixStartEnd.iloc[0] = 'fix_end_carryover_inserted_start'
            group.FixDur.iloc[0] = group.sampTime.iloc[0] - group.targSampTime.iloc[0]
            
            # Insert a new 'fix_start' event at the beginning of the trial
            first_row = group.iloc[0:1].copy() # Make a copy of the first row 
            first_row.index = [-1]  # Assign a negative index for the new row
            first_row.FixStartEnd = 'fix_start_carryover_inserted_start'
            first_row.FixDur = 0
            first_row.FixLatency = 0
            group = pd.concat([first_row, group]).sort_index().reset_index(drop=True) # Prepend the copied first row to the original DataFrame
            group['FixationOrder'] = group['FixLatency'].rank() # Re-rank the order of fixations in the trial

        # Handle case where trial ends with 'fix_start' (carryover to next trial)
        if group.FixStartEnd.iloc[-1] == 'fix_start':
            group = group.copy()
            group.FixStartEnd.iloc[-1] = 'fix_start_carryover_inserted_end'
            group.FixDur.iloc[-1] = 0
            
            # Insert a new 'fix_end' event at the end of the trial
            last_row = group.iloc[[-1]].copy()
            last_row.index += 1  # Shift index by 1 for the new row
            last_row.FixStartEnd = 'fix_end_carryover_inserted_end'
            last_row.FixDur = (last_row.targSampTime + max_event_duration) - last_row.sampTime
            last_row.FixLatency = 0
            last_row.FixationOrder = 0
            group = pd.concat([group, last_row]).sort_index().reset_index(drop=True)

        # Accumulate groups into a list
        fixcarryover_groups.append(group)

    # Step 2: Concatenate all modified groups back into a single DataFrame
    fc_df = pd.concat(fixcarryover_groups)

    # Step 3: Merge fixation events within trials

    # Merge fix_start and fix_end for normal fixations
    df_fix_start = fc_df[fc_df.FixStartEnd == 'fix_start'].drop('FixDur', axis=1)
    df_fix_end = fc_df[fc_df.FixStartEnd == 'fix_end'][["FixXPos", "FixYPos", "FixDur"]]
    df_merged = pd.merge(df_fix_start, df_fix_end, on=["FixXPos", "FixYPos"])

    # Handle carryover fixations that missed the fix_start event
    df_fix_start_insert_start = fc_df[fc_df.FixStartEnd == 'fix_start_carryover_inserted_start'].drop('FixDur', axis=1)
    df_fix_end_insert_start = fc_df[fc_df.FixStartEnd == 'fix_end_carryover_inserted_start'][["FixXPos", "FixYPos", "FixDur"]]
    df_merged_insert_start = pd.merge(df_fix_start_insert_start, df_fix_end_insert_start, on=["FixXPos", "FixYPos"])

    # Handle carryover fixations that missed the fix_end event
    df_fix_start_insert_end = fc_df[fc_df.FixStartEnd == 'fix_start_carryover_inserted_end'].drop('FixDur', axis=1)
    df_fix_end_insert_end = fc_df[fc_df.FixStartEnd == 'fix_end_carryover_inserted_end'][["FixXPos", "FixYPos", "FixDur"]]
    df_merged_insert_end = pd.merge(df_fix_start_insert_end, df_fix_end_insert_end, on=["FixXPos", "FixYPos"])

    # Step 4: Concatenate all carryover fixations
    df_carryover = pd.concat([df_merged_insert_start, df_merged_insert_end], ignore_index=True).sort_values(by='frameNr')

    # Step 5: Concatenate carryover fixations with within-trial fixations
    df_final = pd.concat([df_merged, df_carryover], ignore_index=True).sort_values(by='frameNr')

    return df_final



def addAOI(df):
    """
    Assign Areas of Interest (AOI) to each fixation based on predefined bounding boxes for stimuli.

    This function processes a DataFrame containing fixation data and assigns each fixation to a bounding box 
    (i.e., an Area of Interest or AOI). Each bounding box is defined by its coordinates, and the function 
    checks whether the fixation coordinates fall within one of these bounding boxes. If a fixation does not 
    fall within any bounding box, it is assigned to 'None'.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing fixation data with at least the following columns:
        - 'FixXPos': float, X-coordinate of the fixation point.
        - 'FixYPos': float, Y-coordinate of the fixation point.
        - 'bboxes': list of bounding boxes, where each box is defined as [x, y, width, height].
        - 'bboxesNames': list of names corresponding to each bounding box.
        - 'padding': float, padding to apply around each bounding box.
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame with additional columns:
        - 'AOI_bbox': The index of the bounding box the fixation falls within, or 'None'.
        - 'AOI_stim': The name of the stimulus (bounding box) the fixation is assigned to, or 'None'.
    """
    
    def is_point_in_box(point, box):
        """
        Determine if a point (FixXPos, FixYPos) is within a bounding box.

        Parameters
        ----------
        point : tuple
            A tuple (x, y) representing the fixation coordinates.
        box : tuple
            A tuple ((x1, y1), (x2, y2)) representing the bounding box, where (x1, y1) is the top-left corner
            and (x2, y2) is the bottom-right corner.

        Returns
        -------
        bool
            True if the point is within the bounding box, otherwise False.
        """
        px, py = point
        (x1, y1), (x2, y2) = box
        return x1 <= px <= x2 and y1 <= py <= y2

    def get_bounding_box_assignment(boxes, point):
        """
        Determine the bounding box index that a point belongs to.

        Parameters
        ----------
        boxes : list of tuples
            List of bounding boxes, each defined as ((x1, y1), (x2, y2)).
        point : tuple
            A tuple (x, y) representing the fixation coordinates.

        Returns
        -------
        int or 'None'
            The index of the bounding box if the point belongs to any box, otherwise 'None'.
        """
        for i, box in enumerate(boxes):
            if is_point_in_box(point, box):
                return i
        return 'None'

    # Initialize lists to hold the assignments
    bbox_assignments = []
    stim_assignments = []

    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        padding = row.padding  # Get padding for the bounding boxes
        bboxesNames = row.bboxesNames  # List of bounding box names
        bboxes_coords = row.bboxes  # Bounding box coordinates

        if isinstance(bboxesNames, str):
            bboxesNames = ast.literal_eval(bboxesNames)

        if isinstance(bboxes_coords, str):
            bboxes_coords = ast.literal_eval(bboxes_coords)

        # Ensure bboxes_coords is a list of lists (e.g., [[x, y, width, height], ...])
        if all(isinstance(coord, (int, float)) for coord in bboxes_coords):
            bboxes_coords = [bboxes_coords]

        # Initialize bounding boxes with padding
        bounding_boxes = []
        for coord in bboxes_coords:
            x1 = coord[0] - padding
            y1 = coord[1] - padding
            x2 = coord[0] + coord[2] + padding * 2
            y2 = coord[1] + coord[3] + padding * 2
            bounding_boxes.append(((x1, y1), (x2, y2)))

        # Get fixation coordinates
        point = (row.FixXPos, row.FixYPos)

        # Determine which bounding box (if any) contains the fixation point
        assignment = get_bounding_box_assignment(bounding_boxes, point)

        # Assign the corresponding bounding box name, or 'None' if no match
        if assignment != 'None':
            bboxName = bboxesNames[assignment]
        else:
            bboxName = 'None'

        # Append results to the lists
        bbox_assignments.append(assignment)
        stim_assignments.append(bboxName)

    # Add the AOI assignments to the DataFrame
    df['AOI_bbox'] = bbox_assignments
    df['AOI_stim'] = stim_assignments

    # Reset index and return the updated DataFrame
    df = df.reset_index(drop=True)

    return df
