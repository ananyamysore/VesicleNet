"""Reproducible lightweight runtime benchmark; reports measurements, not claims."""
import platform, time
import numpy as np
import scipy, skimage
from vesiclenet.synthetic import generate_synthetic_image
from vesiclenet.pipeline import analyze_image

if __name__ == "__main__":
    sample = generate_synthetic_image(shape=(512,512), n_objects=120, seed=42)
    start = time.perf_counter(); analyze_image(sample.image); elapsed = time.perf_counter()-start
    print({"seconds": elapsed, "shape": sample.image.shape, "seed": 42, "python": platform.python_version(),
           "numpy": np.__version__, "scipy": scipy.__version__, "scikit_image": skimage.__version__})
