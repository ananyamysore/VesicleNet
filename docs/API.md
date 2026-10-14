# API reference

## Synthetic data

`generate_synthetic_image(shape=(128, 128), *, n_objects=24, overlap=0.5, noise=0.05, blur=0.8, radius=4, seed=0) -> SyntheticDataset`

Returns channels-last float image, binary masks, instance labels, and metadata. `overlap` controls the fraction of channel-1 centers placed at channel-0 centers. This simplified generator does not model photophysics, camera noise, PSF variation, cell geometry, or realistic vesicle biology.

## Preprocessing

- `load_image(path, *, channel_axis=-1, channel=None)`: read TIFF; optionally extract one plane from a 3D TIFF.
- `validate_image(image, *, ndim=(2, 3), allow_nan=False)`: validate shape and finite values, then cast to float64.
- `normalize_image(image, *, lower_percentile=1.0, upper_percentile=99.8)`: percentile-scale to [0,1].
- `subtract_background(image, *, radius=20.0, method="gaussian")`: smooth-background subtraction with nonnegative output.

## Segmentation

`segment_channel(image, *, method="otsu", threshold=None, min_size=1, connectivity=1) -> SegmentationResult`

Returns boolean mask, integer connected-component labels, and applied threshold. `segmentation_metrics(prediction, truth)` returns pixel Dice, IoU, precision, and recall.

## Colocalization

`colocalization_metrics(channel_a, channel_b, *, threshold_a=None, threshold_b=None)` returns Pearson and Manders M1/M2. Pearson is threshold independent; M1/M2 use strict greater-than thresholds. `threshold_sensitivity(channel_a, channel_b, thresholds_a, thresholds_b=None)` evaluates the Cartesian threshold grid.

## Spatial analysis

`spatial_metrics(labels, *, pixel_size=1.0)` returns object count, area distribution, centroids, nearest-neighbor distances, centroid RMS spread, and descriptive Clark–Evans R.

## Pipeline

`analyze_image(image, *, output_dir=None, channel_axis=-1, thresholds=None, threshold_method="otsu", background_radius=None, pixel_size=1.0)` accepts a two-channel array or TIFF path. It returns a JSON-serializable nested dictionary and optionally writes JSON, CSV, and PNG files.
