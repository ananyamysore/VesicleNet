"""Generate a freely usable synthetic example TIFF and analysis outputs."""
from pathlib import Path
import tifffile
from vesiclenet.synthetic import generate_synthetic_image
from vesiclenet.pipeline import analyze_image

out = Path("example-output"); out.mkdir(exist_ok=True)
sample = generate_synthetic_image(shape=(128,128), n_objects=20, overlap=.6, seed=2025)
tifffile.imwrite(out/"synthetic_two_channel.tif", sample.image.astype("float32"))
analyze_image(sample.image, output_dir=out/"analysis")
print(f"Wrote synthetic example to {out}")
