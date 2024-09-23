import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def draw_bbox(ax, bboxes_coords, colormap='viridis', offset_left=0, offset_top=0, padding=0):
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
    padding : float, optional
        Padding to add around each bounding box (default is 0).

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

        # Draw a rectangle with padding
        rect = patches.Rectangle((bbox_left - padding, bbox_top - padding), 
                                 bbox_width + padding * 2, bbox_height + padding * 2, 
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


def plot2d(df, fn, path_to_analysis, condition=None, bboxes=True, stimuli=True, save=True):
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
    condition : str, optional
        The column name in df used for generating a title for each plot (default is None).
    bboxes : bool, optional
        Whether to draw bounding boxes on the plot (default is True).
    stimuli : bool, optional
        Whether to draw stimuli images on the plot (default is True).
    save : bool, optional
        Whether to save the plot to a file (default is True).

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
            title = group[condition].iloc[0]
            plt.title(f'{fn}_{title}_trial{int(trialNr)}')
        else:
            title = f'trial_{int(trialNr)}'

        # Draw stimuli if enabled
        if stimuli:
            draw_stimuli(ax, group.image_paths.iloc[0], group.image_coords.iloc[0], path_to_analysis)

        # Draw bounding boxes if enabled
        if bboxes:
            padding = group.padding.iloc[0]  # Get padding from DataFrame
            draw_bbox(ax, group.bboxes.iloc[0], colormap='viridis', padding=padding)

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
