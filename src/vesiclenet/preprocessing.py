"""Image input, validation, normalization, and background correction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
import tifffile


def validate_image(image: ArrayLike, *, ndim: int | tuple[int, ...] = (2, 3), allow_nan: bool = False) -> NDArray[np.float64]:
    """Return a floating image after shape and finiteness validation."""
    arr = np.asarray(image, dtype=np.float64)
    accepted = (ndim,) if isinstance(ndim, int) else ndim
    if arr.ndim not in accepted: raise ValueError(f"expected {accepted} dimensions, received {arr.ndim}")
    if any(size == 0 for size in arr.shape): raise ValueError("image dimensions must be nonempty")
    if np.isinf(arr).any() or (not allow_nan and np.isnan(arr).any()): raise ValueError("image contains non-finite values")
    return arr


def load_image(path: str | Path, *, channel_axis: int | None = -1, channel: int | None = None) -> NDArray[np.float64]:
    """Load TIFF data. Select a channel from a 3D image when ``channel`` is given."""
    arr = validate_image(tifffile.imread(Path(path)), ndim=(2, 3))
    if arr.ndim == 3 and channel is not None:
        axis = channel_axis % arr.ndim
        if not 0 <= channel < arr.shape[axis]: raise IndexError("channel index out of range")
        arr = np.take(arr, channel, axis=axis)
    elif arr.ndim == 3 and channel is None:
        raise ValueError("3D TIFF requires an explicit channel index or downstream channel-axis handling")
    return arr


def normalize_image(image: ArrayLike, *, lower_percentile: float = 1.0, upper_percentile: float = 99.8) -> NDArray[np.float64]:
    """Percentile-scale to [0,1]; constant input maps to zeros."""
    arr = validate_image(image, ndim=(2,3))
    if not 0 <= lower_percentile < upper_percentile <= 100: raise ValueError("invalid percentile range")
    lo, hi = np.percentile(arr, [lower_percentile, upper_percentile])
    if hi <= lo: return np.zeros_like(arr)
    return np.clip((arr-lo)/(hi-lo), 0, 1)


def subtract_background(image: ArrayLike, *, radius: float = 20.0, method: str = "gaussian") -> NDArray[np.float64]:
    """Subtract a smooth estimated background and clip negative residuals.

    ``gaussian`` uses a broad Gaussian low-pass estimate; ``median`` uses a
    disk-like median footprint. Radius is an approximate pixel scale.
    """
    arr = validate_image(image, ndim=(2,3))
    if radius <= 0: raise ValueError("radius must be positive")
    if method == "gaussian": bg = ndi.gaussian_filter(arr, sigma=radius)
    elif method == "median": bg = ndi.median_filter(arr, size=max(3, int(2*radius)//2*2+1))
    else: raise ValueError("method must be 'gaussian' or 'median'")
    return np.maximum(arr-bg, 0)
