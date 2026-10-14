"""VesicleNet: reproducible fluorescence microscopy analysis."""
from .synthetic import SyntheticDataset, generate_synthetic_image
from .preprocessing import load_image, validate_image, normalize_image, subtract_background
from .segmentation import SegmentationResult, segment_channel, segmentation_metrics
from .colocalization import colocalization_metrics, threshold_sensitivity
from .spatial_analysis import spatial_metrics
from .pipeline import analyze_image

__all__ = ["SyntheticDataset", "generate_synthetic_image", "load_image", "validate_image", "normalize_image", "subtract_background", "SegmentationResult", "segment_channel", "segmentation_metrics", "colocalization_metrics", "threshold_sensitivity", "spatial_metrics", "analyze_image"]
__version__ = "0.1.0"
