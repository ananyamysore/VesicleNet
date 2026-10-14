# VesicleNet

VesicleNet is a lightweight, reproducible Python toolkit for quantitative analysis of two-channel fluorescence microscopy images, with a focus on vesicle segmentation, colocalization, and spatial summaries. It is a research tool, not a validated clinical or biological assay. Synthetic examples validate implementation behavior under their stated assumptions; they do not establish performance on real specimens.

## Installation

Requires Python 3.11 or newer.

```bash
python -m pip install .
# For development and tests:
python -m pip install '.[test]'
```

## Quick start

```python
from vesiclenet import generate_synthetic_image, analyze_image

sample = generate_synthetic_image(shape=(128, 128), n_objects=24,
                                  overlap=0.5, noise=0.05, blur=0.8, seed=12)
report = analyze_image(sample.image, output_dir="results")
print(report["pixel_colocalization"])
print(report["object_overlap"])
```

Run a TIFF through the CLI:

```bash
vesiclenet image.tif --channel-axis -1 --threshold-method otsu --output results
```

The CLI expects exactly two channels along the selected axis. Input arrays must be 3D; use `load_image(path, channel=...)` to extract a single channel from a 3D TIFF. `analyze_image` writes `results.json`, `spatial_summary.csv`, and `analysis.png` when an output directory is provided.

Generate the bundled synthetic example:

```bash
python examples/generate_example.py
```

## Analysis and interpretation

- **Preprocessing:** TIFF input is read with tifffile. Normalization uses the 1st and 99.8th percentiles by default and clips to [0,1]. Optional background subtraction estimates a smooth Gaussian or median background and clips negative residuals. It may remove broad biological structures; inspect parameter choices and raw images.
- **Segmentation:** The current implementation uses global Otsu, Yen, or Li thresholds and connected components. An explicit threshold overrides the automatic method. This is a simple baseline, not an adaptive or biologically validated segmentation model. Thresholding does not resolve touching vesicles.
- **Pixel-level colocalization:** Pearson correlation summarizes paired pixel intensity association over all finite pixels; it is not a measure of object overlap and can be affected by background, intensity range, and spatial autocorrelation. Manders M1/M2 are intensity fractions above explicitly specified channel thresholds that coincide with above-threshold pixels in the other channel. Omitting thresholds uses each channel's mean for these coefficients. Threshold sensitivity can be evaluated with `threshold_sensitivity`.
- **Object-level overlap:** The pipeline separately reports the fraction and count of segmented objects with any foreground pixel intersecting the other channel's mask. This permissive definition is not an object correspondence model and does not compensate for registration error.
- **Spatial summaries:** Areas and centroid distances are reported in pixels or scaled by `pixel_size`. The Clark–Evans R shown is a descriptive ratio against a homogeneous Poisson expectation; it has no edge correction or inferential p-value. Use a domain-appropriate spatial null model before drawing biological conclusions.

Images with NaNs are supported by pixel colocalization (finite pixel pairs only); other processing functions reject NaNs. Constant image normalization returns zeros and constant-channel Pearson correlation is `None`. Empty masks yield perfect overlap scores only when both masks are empty; undefined intensity fractions are `None`. Dimensions, channel count, thresholds, and pixel scale are validated.

## Reproducibility

Synthetic images use NumPy's `default_rng` with a caller-supplied seed and return object masks, instance labels, and generation metadata. The benchmark script reports runtime and environment versions for the current machine; it intentionally contains no precomputed or fabricated performance claims:

```bash
python benchmarks/benchmark_pipeline.py
```

## Tests

```bash
python -m pip install '.[test]'
pytest
```

Tests cover deterministic synthetic generation, edge handling, known mask metrics, colocalization on controlled arrays, spatial summaries, and pipeline exports. No real microscopy dataset is bundled or downloaded. Compare the chosen metrics and preprocessing with established packages (for example, scikit-image's thresholding/labeling routines and established colocalization implementations such as ImageJ/Fiji coloc2) using representative data and matching threshold/ROI conventions before interpreting study results. Exact numerical equivalence is not claimed because conventions differ.

## API documentation

See [docs/API.md](docs/API.md) for function signatures and [docs/assumptions.md](docs/assumptions.md) for method assumptions and limitations.
