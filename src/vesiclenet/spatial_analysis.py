"""Object and spatial distribution summaries for 2D label images."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.distance import cdist


def spatial_metrics(labels: ArrayLike, *, pixel_size: float = 1.0) -> dict[str, object]:
    """Summarize object count/area, nearest-neighbor distance, and spatial spread.

    Nearest-neighbor distances use centroid Euclidean distance in 2D. Spatial
    distribution reports centroid dispersion and Clark-Evans R only as descriptive
    summaries; edge correction and an appropriate null model are not applied.
    """
    lab = np.asarray(labels)
    if lab.ndim != 2 or not np.isfinite(lab).all(): raise ValueError("labels must be a finite 2D array")
    if pixel_size <= 0: raise ValueError("pixel_size must be positive")
    ids = np.unique(lab); ids = ids[ids != 0]
    areas = np.array([(lab == i).sum() for i in ids], dtype=float) * pixel_size**2
    centers = np.array([np.argwhere(lab == i).mean(axis=0) * pixel_size for i in ids]) if ids.size else np.empty((0,2))
    if len(centers) > 1:
        d = cdist(centers, centers); np.fill_diagonal(d, np.inf); nn = d.min(axis=1)
        area = lab.shape[0]*lab.shape[1]*pixel_size**2
        density = len(ids)/area
        expected = 0.5/np.sqrt(density) if density else np.nan
        ce = float(nn.mean()/expected) if expected else np.nan
    else: nn = np.array([], dtype=float); ce = None
    spread = float(np.sqrt(np.mean(np.sum((centers-centers.mean(axis=0))**2, axis=1)))) if len(centers) else None
    return {"object_count": int(len(ids)), "areas": areas.tolist(), "mean_area": float(areas.mean()) if len(areas) else None,
            "median_area": float(np.median(areas)) if len(areas) else None, "centroids": centers.tolist(),
            "nearest_neighbor_distances": nn.tolist(), "mean_nearest_neighbor_distance": float(nn.mean()) if len(nn) else None,
            "centroid_rms_spread": spread, "clark_evans_r": ce}
