"""Threshold-based segmentation and mask comparison metrics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, threshold_yen, threshold_li

@dataclass(frozen=True)
class SegmentationResult:
    """Binary foreground mask, connected-component labels, and threshold."""
    mask: NDArray[np.bool_]
    labels: NDArray[np.int32]
    threshold: float


def segment_channel(image: ArrayLike, *, method: str = "otsu", threshold: float | None = None,
                    min_size: int = 1, connectivity: int = 1) -> SegmentationResult:
    """Segment 2D image using a named global threshold or explicit threshold."""
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2 or not arr.size: raise ValueError("image must be a nonempty 2D array")
    if not np.isfinite(arr).all(): raise ValueError("image must contain only finite values")
    if min_size < 1 or connectivity not in (1,2): raise ValueError("invalid min_size or connectivity")
    if threshold is None:
        if method == "otsu": t = float(threshold_otsu(arr))
        elif method == "yen": t = float(threshold_yen(arr))
        elif method == "li": t = float(threshold_li(arr))
        else: raise ValueError("method must be otsu, yen, or li when threshold is omitted")
    else: t = float(threshold)
    if not np.isfinite(t): raise ValueError("threshold must be finite")
    mask = arr > t
    structure = ndi.generate_binary_structure(2, connectivity)
    labels, _ = ndi.label(mask, structure=structure)
    if min_size > 1:
        counts = np.bincount(labels.ravel()); remove = counts < min_size; remove[0] = False
        mask = ~remove[labels]; labels, _ = ndi.label(mask, structure=structure)
    return SegmentationResult(mask.astype(bool), labels.astype(np.int32), t)


def segmentation_metrics(prediction: ArrayLike, truth: ArrayLike) -> dict[str, float]:
    """Compute pixel Dice, IoU, precision, and recall; empty-empty scores are 1."""
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if p.shape != t.shape: raise ValueError("prediction and truth shapes must match")
    tp = int(np.count_nonzero(p&t)); fp = int(np.count_nonzero(p&~t)); fn = int(np.count_nonzero(~p&t))
    dice = 1.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn)
    iou = 1.0 if tp+fp+fn == 0 else tp/(tp+fp+fn)
    precision = 1.0 if tp+fp == 0 else tp/(tp+fp)
    recall = 1.0 if tp+fn == 0 else tp/(tp+fn)
    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}
