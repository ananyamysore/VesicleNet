"""End-to-end two-channel analysis and machine-readable export."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .preprocessing import load_image, normalize_image, subtract_background, validate_image
from .segmentation import segment_channel
from .colocalization import colocalization_metrics
from .spatial_analysis import spatial_metrics


def analyze_image(image: Any, *, output_dir: str | Path | None = None, channel_axis: int = -1,
                  thresholds: tuple[float, float] | None = None, threshold_method: str = "otsu",
                  background_radius: float | None = None, pixel_size: float = 1.0) -> dict[str, Any]:
    """Analyze a two-channel image and optionally export JSON, CSV, and PNG.

    Input may be a NumPy-like array shaped YXC (or specified ``channel_axis``) or
    a two-dimensional TIFF path is deliberately rejected: two channels are required.
    Segmentation uses global thresholding. Pixel colocalization and object overlap
    are returned separately.
    """
    raw = load_image(image) if isinstance(image, (str, Path)) else validate_image(image, ndim=(3,))
    axis = channel_axis % raw.ndim
    if raw.shape[axis] != 2: raise ValueError("pipeline requires exactly two channels")
    ch = [np.take(raw, i, axis=axis) for i in range(2)]
    processed = [normalize_image(subtract_background(c, radius=background_radius) if background_radius else c) for c in ch]
    segments = [segment_channel(processed[i], method=threshold_method,
                                threshold=None if thresholds is None else thresholds[i]) for i in range(2)]
    pixel = colocalization_metrics(processed[0], processed[1],
                                   threshold_a=None if thresholds is None else thresholds[0],
                                   threshold_b=None if thresholds is None else thresholds[1])
    obj = []
    for i in range(2):
        own, other = segments[i].labels, segments[1-i].mask
        ids = np.unique(own); ids = ids[ids != 0]
        touched = [int(np.any((own == ident) & other)) for ident in ids]
        obj.append({"object_count": int(len(ids)), "overlapping_object_count": int(sum(touched)),
                    "fraction_overlapping_objects": float(np.mean(touched)) if touched else None})
    result: dict[str, Any] = {"schema_version": "1.0", "analysis": {"pixel_size": pixel_size,
              "threshold_method": threshold_method, "thresholds": [s.threshold for s in segments],
              "background_radius": background_radius},
              "pixel_colocalization": pixel,
              "object_overlap": {"channel_0": obj[0], "channel_1": obj[1]},
              "spatial": {"channel_0": spatial_metrics(segments[0].labels, pixel_size=pixel_size),
                          "channel_1": spatial_metrics(segments[1].labels, pixel_size=pixel_size)}}
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out/"results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        rows = [{"channel": i, **{k:v for k,v in spatial_metrics(segments[i].labels, pixel_size=pixel_size).items() if not isinstance(v, list)}} for i in range(2)]
        pd.DataFrame(rows).to_csv(out/"spatial_summary.csv", index=False)
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(2): axes[i].imshow(processed[i], cmap="gray"); axes[i].set_title(f"Channel {i}"); axes[i+2].imshow(segments[i].mask, cmap="gray"); axes[i+2].set_title(f"Mask {i}")
        for ax in axes: ax.axis("off")
        fig.tight_layout(); fig.savefig(out/"analysis.png", dpi=150); plt.close(fig)
    return result
