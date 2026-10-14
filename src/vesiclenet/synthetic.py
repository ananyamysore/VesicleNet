"""Deterministic synthetic microscopy data for validation and examples.

Synthetic data test code behavior under stated assumptions; they do not establish
performance on biological images.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

FloatImage = NDArray[np.float64]
BoolMask = NDArray[np.bool_]

@dataclass(frozen=True)
class SyntheticDataset:
    """Two-channel image with known per-channel object masks and instance labels."""
    image: FloatImage
    masks: tuple[BoolMask, BoolMask]
    labels: tuple[NDArray[np.int32], NDArray[np.int32]]
    metadata: dict[str, float | int]


def generate_synthetic_image(shape: tuple[int, int] = (128, 128), *, n_objects: int = 24,
                             overlap: float = 0.5, noise: float = 0.05, blur: float = 0.8,
                             radius: int = 4, seed: int = 0) -> SyntheticDataset:
    """Create reproducible paired images with disk-shaped fluorescent objects.

    ``overlap`` is the fraction of channel-1 object centers placed at channel-0
    centers. Remaining objects are independently placed. ``noise`` is Gaussian
    standard deviation relative to unit peak intensity; ``blur`` is Gaussian sigma.
    """
    if len(shape) != 2 or min(shape) < 8: raise ValueError("shape must be a 2D tuple with dimensions >= 8")
    if n_objects < 0 or radius < 1: raise ValueError("n_objects must be nonnegative and radius positive")
    if not 0 <= overlap <= 1 or noise < 0 or blur < 0: raise ValueError("overlap must be in [0,1]; noise and blur must be nonnegative")
    rng = np.random.default_rng(seed); h, w = shape
    yy, xx = np.ogrid[:h, :w]
    centers: list[list[tuple[int,int]]] = [[], []]
    margin = radius + 1
    for channel in range(2):
        for i in range(n_objects):
            if channel == 1 and i < round(n_objects * overlap) and centers[0]:
                cy, cx = centers[0][i % len(centers[0])]
            else:
                cy = int(rng.integers(margin, max(margin + 1, h-margin)))
                cx = int(rng.integers(margin, max(margin + 1, w-margin)))
            centers[channel].append((cy, cx))
    images=[]; masks=[]; labels=[]
    for channel in range(2):
        lab = np.zeros(shape, dtype=np.int32); intensity = np.zeros(shape, dtype=float)
        for idx, (cy,cx) in enumerate(centers[channel], 1):
            disk = (yy-cy)**2 + (xx-cx)**2 <= radius**2
            lab[disk] = idx; intensity[disk] += 1.0
        if blur: intensity = ndi.gaussian_filter(intensity, blur)
        if noise: intensity += rng.normal(0, noise, shape)
        images.append(np.clip(intensity, 0, None)); masks.append(lab > 0); labels.append(lab)
    image = np.stack(images, axis=-1).astype(np.float64)
    scale = float(image.max())
    if scale: image /= scale
    return SyntheticDataset(image, (masks[0], masks[1]), (labels[0], labels[1]),
                            {"seed": seed, "n_objects_per_channel": n_objects, "overlap_fraction": overlap, "noise": noise, "blur_sigma": blur, "radius": radius})
