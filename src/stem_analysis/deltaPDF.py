import numpy as np
from scipy.interpolate import RBFInterpolator
from skimage.feature import match_template, peak_local_max
from skimage.filters import window
from scipy.ndimage import median_filter, rotate

def _median_absolute_deviation(data):
    """
    Calculates the Mean Absolute Deviation (MAD) of a dataset using numpy.
    
    Args:
        data (list or np.array): The input dataset.
        
    Returns:
        float: The mean absolute deviation.
    """
    median = np.median(data)
    absolute_deviations = np.abs(data - median)
    mad = np.median(absolute_deviations)
    return mad

def Karen(dp, window_size):
    """
    Applies the Karen algorithm to an image to enhance features and reduce noise.
    
    Args:
        dp (np.array): The input image data as a 2D numpy array.
        window_size (int): The size of the window for local processing.
        pad_mode (str): The mode for padding the image borders (default is "edge").
        
    Returns:
        np.array: The processed image after applying the Karen algorithm.
    """
    dp_out = dp.copy()

    med = median_filter(dp_out, size=window_size, mode="nearest")
    mad = median_filter(np.abs(dp_out - med), size=window_size, mode="nearest")

    asigma = np.abs(mad*3*1.4826)
    mask = np.logical_or(dp_out < (med - asigma), dp_out > (med + asigma))
    dp_out[mask] = (med+2.2*mad)[mask]
            
    return dp_out

def PunchAndFill(dp, peaks, radius):
    """
    Punches circular regions around specified peaks in an image and fills them using RBF interpolation.
    
    Args:
        dp (np.array): The input image data as a 2D numpy array.
        peaks (list of tuples): List of (row, column) coordinates for the peaks to be punched.
        radius (int): The radius of the circular regions to be punched.
        
    Returns:
        np.array: The image with punched regions filled using RBF interpolation.
    """
    rr, cc = np.mgrid[0:dp.shape[0], 0:dp.shape[1]]
    punched = dp.copy().astype(np.float64)
    for peak in peaks:
        mask = (rr - peak[0])**2 + (cc - peak[1])**2 < radius**2
        punched[mask] = np.nan

    valid_mask = ~np.isnan(punched)
    coords = np.column_stack((rr[valid_mask], cc[valid_mask]))
    values = punched[valid_mask]

    missing_mask = ~valid_mask
    missing_coords = np.column_stack((rr[missing_mask], cc[missing_mask]))

    rbf = RBFInterpolator(
        coords,
        values,
        kernel='multiquadric',
        epsilon=2.5,
        smoothing=2.5,
        neighbors=50
    )

    punched_rbf = punched.copy()
    punched_rbf[missing_mask] = rbf(missing_coords)
    
    return punched_rbf
    

    