"""
GPU/CPU dispatch helpers for stem_tools functions that run outside LiberTEM's
UDF executor (dpc.py, utils.py). UDFs.py does not need this module: it uses
LiberTEM's own self.xp / get_backends() device dispatch instead.
"""
import numpy as np

try:
    import cupy as _cupy
    HAS_CUPY = bool(_cupy.cuda.is_available())
except Exception:
    _cupy = None
    HAS_CUPY = False


def get_array_module(*arrays):
    """Return the cupy module if any argument is a cupy array and cupy is usable, else numpy."""
    if HAS_CUPY:
        for a in arrays:
            if isinstance(a, _cupy.ndarray):
                return _cupy
    return np


def get_ndimage_module(xp):
    """Return the scipy.ndimage-compatible module matching the given array module."""
    if HAS_CUPY and xp is _cupy:
        import cupyx.scipy.ndimage as ndi
        return ndi
    import scipy.ndimage as ndi
    return ndi


def to_device(arr, xp):
    """Move arr onto the given array module (no-op if already there, e.g. numpy)."""
    return xp.asarray(arr)


def to_numpy(arr):
    """Bring a possibly-cupy array back to a plain numpy array."""
    if HAS_CUPY and isinstance(arr, _cupy.ndarray):
        return _cupy.asnumpy(arr)
    return np.asarray(arr)
