import numpy as np
# import xarray as xr

def get_metadata_from_hdr(filepath):
    """Get metadata stored in .hdr file from PNCCD

    Parameters
    ----------
    filepath : string
        Path to .hdr file

    Returns
    -------
    metadata : dict
        A dictionary storing all metadata
    """

    with open(filepath) as f:
        content_list = (f.read()).splitlines()
    f.close()

    metadata = dict()
    for c in content_list:
        try:
            key, value = c.split("=")
            metadata[key] = value
        except:
            pass

    return metadata


def get_metadata_from_json(filepath):
    """Get metadata stored in .json file exported from Nion Swift

    Parameters
    ----------
    filepath : string
        Path to .hdr file

    Returns
    -------
    metadata : dict
        A dictionary storing all metadata
    """

    ### To be filled

    return metadata

        
def numpy_to_xarray(data_array, dims, calibration = None, attrs = None):
    """Convert numpy array to xarray array

    Parameters
    ----------
    data_array : N-dimensional numpy array
        data array in numpy
    
        dims : tuple of N elements
            Name of each dimension. Number of elements should match the dimension of the data_array.
        
        Calibration : list of tuples
            Calibration for each dimension, with the form of [("unit", offset, scale), ...]. Number of elements should match the dimension of the data_array

        attrs: dict
            Metadata needs to be included, in the form of dictionary
    """
    data_xarray = xr.DataArray(data_array)

    if calibration == None:
        calibration = [("px", 0, 1)]*data_array.ndim
    
    coords = {}
    units = []

    for i, (dim, cal) in enumerate(zip(dims, calibration)):
        coord = (np.arange(data_array.shape[i]) + cal[1])*cal[2]
        coords[dim] = coord
        units.append(cal[0])
    
    if attrs == None:
        attrs = {}
    attrs['units'] = units

    data_xarray = xr.DataArray(data_array, coords = coords, attrs = attrs)

    return data_xarray
