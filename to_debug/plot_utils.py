import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ast
import astropy.convolution as krn
from PIL import Image
import io
import base64
import sys
import pandas as pd

sys.path.append('./utils/FixationDetection')
# from I2MC import runI2MC

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

def cam_to_pix(cmX, cmY, screen_resolution, top_left_tocam_cm, scrW_cm):

    screenResX=screen_resolution[0]
    screenResY=screen_resolution[1]
    cm = np.array((cmX, cmY))
    screenRes = np.array([screenResX, screenResY])
    scaleW_cm_px = 1. * screenRes[0]/scrW_cm

    pix_xy = cm - top_left_tocam_cm
    pix_xy = scaleW_cm_px * pix_xy

    #pix_xy[1] = -1*(pix_xy[1] - screenResY) # this is for laptop with camera at the bottom left
    pix_xy[1] = -1*(pix_xy[1])
#   pix_xy[0] = -1*(pix_xy[0])

    return list(pix_xy)

def pix_to_cam(pix_x, pix_y, screen_resolution, top_left_tocam_cm, scrW_cm):

    screenResX=screen_resolution[0]
    screenResY=screen_resolution[1]
    screenRes = np.array([screenResX, screenResY])

    pix_xy = np.array((pix_x, pix_y))

    pix_xy = (1,-1.) * pix_xy # flip the y-axis

    # convert pixels to cm
    scaleW_px_cm = 1. * scrW_cm/screenRes[0] # 1 pixel size in cm
    cam_xy = pix_xy * scaleW_px_cm

    # normalize to camera as origin
    cam_xy = cam_xy - top_left_tocam_cm # top_left_tocam_cm x,y should both be positive when the camera is in the top middle of the monitor

    return cam_xy

def wrap_pix_to_cam(pix_x, pix_y, screen_resolution, top_left_tocam_cm, scrW_cm, frameNr):
    a = pix_to_cam(pix_x, pix_y, screen_resolution, top_left_tocam_cm, scrW_cm)
    return a[0], a[1], frameNr

def runFixationDetection(screen_resolution, temp_csv_id):
        fixDF = runI2MC(f'{temp_csv_id}', screen_resolution, plotData = False)
        # fixDF.to_csv(resultsFile[:-4]+'_fixations.csv', index=False)
        return fixDF

def create_plot_to_base64(predictions_px, display_img, screen_resolution, dpiX, temp_csv_id):
    xPredictions = []
    yPredictions = []

    for pred in predictions_px:
        pred = ast.literal_eval(pred)
        xPredictions.append(pred[0])
        yPredictions.append(pred[1])

    heatmap = makeHeat((screen_resolution[0],screen_resolution[1]), np.array(xPredictions), np.array(yPredictions))
    f, ax = plt.subplots()

    #Background image
    bgIm = Image.open(io.BytesIO(base64.b64decode(display_img)))
    ax.imshow(bgIm, aspect='equal', extent = [0, screen_resolution[0], screen_resolution[1], 0])

    #Plot heatmap
    #plt.switch_backend('Agg')
    plt.axis('off')
    ax.imshow(heatmap, cmap=cm.hot, extent=[0, screen_resolution[0],screen_resolution[1], 0], alpha = 0.5, aspect='equal')

    #Plot raw predictions
    plt.scatter(np.array(xPredictions), np.array(yPredictions), s=10, alpha=0.5)

    try:
        fixDF = runFixationDetection(screen_resolution, temp_csv_id)
        #Plot fixations
        ax.scatter(fixDF['XPos'].values, fixDF['YPos'].values, c='r', s=20, zorder=2)
    except:
        print('Fixation detection failed')

    stringIObytes = io.BytesIO()
    plt.savefig(stringIObytes, dpi=int(dpiX), format='jpg', bbox_inches='tight', pad_inches = 0)

    stringIObytes.seek(0)
    base64_jpgData = base64.b64encode(stringIObytes.read()).decode()
    stringIObytes.close()
    plt.close()

    return base64_jpgData

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
