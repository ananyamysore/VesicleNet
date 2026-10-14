"""Pixel-level colocalization metrics and threshold sensitivity."""
from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike


def colocalization_metrics(channel_a: ArrayLike, channel_b: ArrayLike, *, threshold_a: float | None = None,
                           threshold_b: float | None = None) -> dict[str, float | None]:
    """Return Pearson r and Manders M1/M2 using explicit thresholds.

    Pearson is computed over all finite paired pixels, independent of thresholds.
    M1 is the fraction of above-threshold A intensity in pixels where B passes its
    threshold; M2 is symmetric. NaNs are omitted pairwise. Constant inputs return
    Pearson None. Zero thresholded intensity totals return None for the corresponding M.
    """
    a, b = np.asarray(channel_a, float), np.asarray(channel_b, float)
    if a.shape != b.shape or a.ndim < 1: raise ValueError("channels must have matching non-scalar shapes")
    valid = np.isfinite(a)&np.isfinite(b); a, b = a[valid], b[valid]
    if not a.size: return {"pearson": None, "manders_m1": None, "manders_m2": None}
    ta = float(np.mean(a) if threshold_a is None else threshold_a)
    tb = float(np.mean(b) if threshold_b is None else threshold_b)
    if not np.isfinite([ta,tb]).all(): raise ValueError("thresholds must be finite")
    pearson = None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
    aa = np.where(a > ta, a, 0); bb = np.where(b > tb, b, 0)
    sum_a, sum_b = aa.sum(), bb.sum()
    m1 = None if sum_a == 0 else float(aa[b > tb].sum()/sum_a)
    m2 = None if sum_b == 0 else float(bb[a > ta].sum()/sum_b)
    return {"pearson": pearson, "manders_m1": m1, "manders_m2": m2}


def threshold_sensitivity(channel_a: ArrayLike, channel_b: ArrayLike, thresholds_a: ArrayLike,
                          thresholds_b: ArrayLike | None = None) -> list[dict[str, float | None]]:
    """Evaluate Manders metrics on a Cartesian grid of explicit channel thresholds."""
    ta, tb = np.asarray(thresholds_a, float), np.asarray(thresholds_a if thresholds_b is None else thresholds_b, float)
    if ta.ndim != 1 or tb.ndim != 1 or not np.isfinite(ta).all() or not np.isfinite(tb).all(): raise ValueError("thresholds must be finite 1D arrays")
    return [{"threshold_a": float(x), "threshold_b": float(y), **colocalization_metrics(channel_a, channel_b, threshold_a=float(x), threshold_b=float(y))} for x in ta for y in tb]
