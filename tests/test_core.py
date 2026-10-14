import numpy as np
import pytest
from vesiclenet.synthetic import generate_synthetic_image
from vesiclenet.preprocessing import normalize_image, validate_image
from vesiclenet.segmentation import segment_channel, segmentation_metrics
from vesiclenet.colocalization import colocalization_metrics, threshold_sensitivity
from vesiclenet.spatial_analysis import spatial_metrics
from vesiclenet.pipeline import analyze_image


def test_synthetic_is_deterministic_and_masks_known():
    a = generate_synthetic_image(shape=(64,64), n_objects=8, seed=21)
    b = generate_synthetic_image(shape=(64,64), n_objects=8, seed=21)
    np.testing.assert_array_equal(a.image, b.image)
    assert a.image.shape == (64,64,2)
    assert all(mask.any() for mask in a.masks)


def test_validation_normalization_and_nan():
    with pytest.raises(ValueError): validate_image(np.array([[np.nan]]))
    np.testing.assert_array_equal(normalize_image(np.ones((3,3))), np.zeros((3,3)))


def test_segmentation_and_empty_metrics():
    truth = np.zeros((12,12), bool); truth[3:8,4:9] = True
    result = segment_channel(truth.astype(float), threshold=.5)
    assert segmentation_metrics(result.mask, truth)["dice"] == 1
    assert segmentation_metrics(np.zeros_like(truth), np.zeros_like(truth))["iou"] == 1


def test_colocalization_known_and_constants():
    a = np.arange(16, dtype=float).reshape(4,4); b = a.copy()
    result = colocalization_metrics(a,b,threshold_a=2,threshold_b=2)
    assert result["pearson"] == pytest.approx(1)
    assert result["manders_m1"] == pytest.approx(1)
    assert colocalization_metrics(np.ones((2,2)), b)["pearson"] is None
    assert len(threshold_sensitivity(a,b,[2,5],[2])) == 2


def test_spatial_and_pipeline_exports(tmp_path):
    labels = np.zeros((20,20), int); labels[2:4,2:4] = 1; labels[10:12,10:12] = 2
    metrics = spatial_metrics(labels)
    assert metrics["object_count"] == 2 and len(metrics["nearest_neighbor_distances"]) == 2
    data = generate_synthetic_image(shape=(48,48),n_objects=5,seed=3)
    res = analyze_image(data.image, output_dir=tmp_path)
    assert "pixel_colocalization" in res and "object_overlap" in res
    assert {"results.json","spatial_summary.csv","analysis.png"}.issubset({p.name for p in tmp_path.iterdir()})


def test_mismatched_inputs_raise():
    with pytest.raises(ValueError): colocalization_metrics(np.zeros((2,2)), np.zeros((3,3)))
    with pytest.raises(ValueError): segmentation_metrics(np.zeros((2,2)), np.zeros((3,3)))
