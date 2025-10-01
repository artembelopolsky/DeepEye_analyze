import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import astropy.convolution as krn

def _get_padding_for_box(padding, box_idx, bboxes_names=None):
    """
    Helper function to get horizontal and vertical padding for a specific bounding box.
    
    Parameters
    ----------
    padding : float, list, or dict
        Padding specification
    box_idx : int
        Index of the current bounding box
    bboxes_names : list, optional
        Names of bounding boxes (used when padding is a dict)
    
    Returns
    -------
    tuple
        (horizontal_padding, vertical_padding)
    """
    if isinstance(padding, (int, float)):
        # Same padding on all sides
        return float(padding), float(padding)
    elif isinstance(padding, list) and len(padding) == 2:
        # [horizontal, vertical] padding for all boxes
        return float(padding[0]), float(padding[1])
    elif isinstance(padding, dict) and bboxes_names:
        # Specific padding per box name
        box_name = bboxes_names[box_idx] if box_idx < len(bboxes_names) else f'box_{box_idx}'
        if box_name in padding:
            pad_spec = padding[box_name]
            if isinstance(pad_spec, (int, float)):
                return float(pad_spec), float(pad_spec)
            elif isinstance(pad_spec, list) and len(pad_spec) == 2:
                return float(pad_spec[0]), float(pad_spec[1])
        # Default to 0 if box name not found
        return 0.0, 0.0
    else:
        # Default to 0 padding
        return 0.0, 0.0

def draw_bbox(ax, bboxes_coords, colormap='viridis', offset_left=0, offset_top=0, padding=0, bboxes_names=None):
    """
    Draws bounding boxes on the given axes based on provided coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the bounding boxes.
    bboxes_coords : list or str
        A list of bounding boxes, each defined as [x, y, width, height]. 
        If provided as a string, it will be converted to a list using ast.literal_eval().
    colormap : str, optional
        The colormap to use for the bounding box colors (default is 'viridis').
    offset_left : float, optional
        Horizontal offset for bounding boxes (default is 0).
    offset_top : float, optional
        Vertical offset for bounding boxes (default is 0).
    padding : float, list, or dict, optional
        Padding to add around each bounding box. Can be:
        - float: Same padding on all sides for all boxes
        - list: [h_pad, v_pad] for horizontal and vertical padding for all boxes
        - dict: {'box_name': [h_pad, v_pad]} for specific padding per box (default is 0).
    bboxes_names : list, optional
        List of names for each bounding box, used when padding is a dict (default is None).

    Returns
    -------
    str
        'done' when all bounding boxes are drawn.
    """
    # Convert string to list if necessary
    if isinstance(bboxes_coords, str):
        bboxes_coords = ast.literal_eval(bboxes_coords)
    
    # Ensure bboxes_coords is a list of lists
    if all(isinstance(coord, (int, float)) for coord in bboxes_coords):
        bboxes_coords = [bboxes_coords]

    # Generate a colormap
    cmap = plt.get_cmap(colormap)
    colors = cmap(np.linspace(0, 1, len(bboxes_coords)))

    # Iterate through bounding boxes and draw them
    for idx, bbox_coords in enumerate(bboxes_coords):
        bbox_left = offset_left + float(bbox_coords[0])
        bbox_top = offset_top + float(bbox_coords[1])
        bbox_width = float(bbox_coords[2])
        bbox_height = float(bbox_coords[3])

        # Calculate padding for this specific box
        h_pad, v_pad = _get_padding_for_box(padding, idx, bboxes_names)
        
        # Draw a rectangle with padding
        rect = patches.Rectangle((bbox_left - h_pad, bbox_top - v_pad), 
                                 bbox_width + h_pad * 2, bbox_height + v_pad * 2, 
                                 fill=False, edgecolor=colors[idx], linewidth=2)
        ax.add_patch(rect)

    return 'done'


def draw_stimuli(ax, img_paths, img_coords, path_to_analysis):
    """
    Draws stimuli images on the given axes based on provided image paths and coordinates.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to draw the stimuli.
    img_paths : list or str
        A list of image file paths or a string representing the list.
    img_coords : list or str
        A list of image coordinates, where each entry is [x, y, width, height]. 
        If provided as a string, it will be converted to a list using ast.literal_eval().
    path_to_analysis : str
        The base path to the directory where the images are stored.

    Returns
    -------
    str
        'done' when all stimuli images are drawn.
    """
    # Convert strings to lists if necessary
    if isinstance(img_paths, str):
        img_paths = ast.literal_eval(img_paths)

    if isinstance(img_coords, str):
        img_coords = ast.literal_eval(img_coords)

    # Adjust the path to point to the stimuli
    new_path = os.path.dirname(path_to_analysis)

    # Iterate through image paths and coordinates to plot images
    for img_path, img_coord in zip(img_paths, img_coords):
        img_path_full = os.path.join(new_path, img_path)
        image = mpimg.imread(img_path_full)

        # Define the extent of the image in the plot (left, right, bottom, top)
        left, top, width, height = img_coord
        extent = [left, left + width, top + height, top]

        # Plot the image on the axes
        ax.imshow(image, extent=extent)

    return 'done'


def plot2d(df, fn, path_to_analysis=False, condition=None, bboxes=True, stimuli=True, save=True, padding=None):
    """
    Plots 2D eye-tracking data for each trial, with optional bounding boxes and stimuli images.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the eye-tracking data for multiple trials.
    fn : str
        The filename prefix for saving the plot.
    path_to_analysis : str
        The path to the directory where results will be saved.
    condition : str or list, optional
        The column name(s) in df used for generating a title for each plot. 
        Can be a single column name (str) or a list of column names (default is None).
    bboxes : bool, optional
        Whether to draw bounding boxes on the plot (default is True).
    stimuli : bool, optional
        Whether to draw stimuli images on the plot (default is True).
    save : bool, optional
        Whether to save the plot to a file (default is True).
    padding : float, list, or dict, optional
        Padding for bounding boxes. If None, uses padding from DataFrame (default is None).

    Returns
    -------
    None
    """
    # Iterate over each trial and generate plots
    for trialNr, group in df.groupby('trialNr'):
        fig, ax = plt.subplots()
        plt.style.use('ggplot')
        plt.grid(False)

        # Set the plot title based on the condition
        if condition:
            # Handle both single column and multiple columns
            if isinstance(condition, list):
                # Join multiple column values with underscores
                title_parts = [str(group[col].iloc[0]) for col in condition]
                title = '_'.join(title_parts)
            else:
                # Single column
                title = str(group[condition].iloc[0])
            plt.title(f'{fn}_{title}_trial{int(trialNr)}')
        else:
            title = f'trial_{int(trialNr)}'

        # Draw stimuli if enabled
        if stimuli:
            draw_stimuli(ax, group.image_paths.iloc[0], group.image_coords.iloc[0], path_to_analysis)

        # Draw bounding boxes if enabled
        if bboxes:
            # Use padding parameter if provided, otherwise get from DataFrame
            if padding is not None:
                plot_padding = padding
            else:
                plot_padding = group.padding.iloc[0]  # Get padding from DataFrame
                # Handle case where padding is NaN (when dictionary was assigned to DataFrame)
                if pd.isna(plot_padding):
                    plot_padding = 0  # Default to no padding
            
            bboxes_names = group.bboxesNames.iloc[0] if 'bboxesNames' in group.columns else None
            draw_bbox(ax, group.bboxes.iloc[0], colormap='viridis', padding=plot_padding, bboxes_names=bboxes_names)

        # Plot raw eye samples
        raw_h = plt.scatter(group.user_pred_px_x, group.user_pred_px_y, c='orange', alpha=0.5, edgecolors='black')

        # Plot fixations (filter out zeros)
        fix_h = plt.scatter(group.FixXPos[group.FixXPos > 0], group.FixYPos[group.FixYPos > 0], 
                            c='blue', alpha=0.5, edgecolors='black')

        # Set plot limits
        plt.xlim((0, df.resX.iloc[0]))
        plt.ylim((df.resY.iloc[0], 0))

        # Set axis labels
        plt.xlabel('Horizontal eye position (pixels)')
        plt.ylabel('Vertical eye position (pixels)')

        # Add a legend
        plt.legend((raw_h, fix_h), ('raw samples', 'fixations'), scatterpoints=1)

        # Save the plot if enabled
        if save:
            plt.savefig(os.path.join(path_to_analysis, f'{fn}_{title}_{int(trialNr)}'), dpi=300, pad_inches=0)
            plt.close()


def makeHeat(screenRes, xPos, yPos):
    """
    Generate a heatmap based on gaze position data, using Gaussian convolution.

    Parameters:
    -----------
    screenRes : tuple
        A tuple (xMax, yMax) representing the resolution of the screen.
    xPos : array-like
        A list or numpy array of x-coordinates of gaze positions.
    yPos : array-like
        A list or numpy array of y-coordinates of gaze positions.

    Returns:
    --------
    heatMap : numpy.ndarray
        A 2D array representing the heatmap of gaze data with values scaled between 0 and 1.

    Raises:
    -------
    ValueError:
        If the input screen resolution is not a tuple of two integers.
        If the length of xPos and yPos do not match.
        If the inputs are not numeric arrays or lists.
    """

    try:
        if not isinstance(screenRes, (tuple, list)) or len(screenRes) != 2:
            raise ValueError("screenRes must be a tuple or list of two elements representing screen dimensions.")
        
        xMax, yMax = int(screenRes[0]), int(screenRes[1])
        xMin, yMin = 0, 0
        kernelPar = 50

        # Input validation for xPos and yPos
        if len(xPos) != len(yPos):
            raise ValueError("xPos and yPos must have the same length.")
        
        if not isinstance(xPos, (list, np.ndarray)) or not isinstance(yPos, (list, np.ndarray)):
            raise ValueError("xPos and yPos must be array-like structures (lists or numpy arrays).")

        xPos = np.array(xPos)
        yPos = np.array(yPos)

        # Ensure gaze points are within the screen resolution
        xlim = np.logical_and(xPos < xMax, xPos > xMin)
        ylim = np.logical_and(yPos < yMax, yPos > yMin)
        xyLim = np.logical_and(xlim, ylim)

        dataX = np.floor(xPos[xyLim])  # Filter and round x coordinates
        dataY = np.floor(yPos[xyLim])  # Filter and round y coordinates

        # Initialize gaze map and Gaussian kernel
        gazeMap = np.zeros([int((xMax - xMin)), int((yMax - yMin))]) + 0.0001
        gausKernel = krn.Gaussian2DKernel(kernelPar)

        # Rescale the position vectors (if xmin or ymin != 0)
        dataX -= xMin
        dataY -= yMin

        # Extract unique positions and count occurrences
        xy = np.vstack((dataX, dataY)).T
        uniqueXY, idx, counts = uniqueRows(xy)
        uniqueXY = uniqueXY.astype(int)

        # Populate the gazeMap with counts
        gazeMap[uniqueXY[:, 0], uniqueXY[:, 1]] = counts

        # Convolve the gaze map with the Gaussian kernel
        heatMap = np.transpose(krn.convolve_fft(gazeMap, gausKernel))
        heatMap = heatMap / np.max(heatMap)  # Normalize heatmap to [0, 1]

        return heatMap
    
    except Exception as e:
        raise RuntimeError(f"An error occurred during heatmap generation: {str(e)}")


def uniqueRows(x):
    """
    Identify unique rows in a 2D array and return them along with their indices and counts.

    Parameters:
    -----------
    x : numpy.ndarray
        A 2D array of coordinates.

    Returns:
    --------
    uniques : numpy.ndarray
        A 2D array of unique rows.
    idx : numpy.ndarray
        The indices of the unique rows in the original array.
    counts : numpy.ndarray
        The number of occurrences of each unique row.
    """
    y = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
    _, idx, counts = np.unique(y, return_index=True, return_counts=True)
    uniques = x[idx]
    return uniques, idx, counts

