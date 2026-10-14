# Scientific assumptions and limitations

VesicleNet currently provides conventional image processing rather than a learned or novel research algorithm. It assumes two spatially aligned channels, a 2D plane per channel, and meaningful foreground/background separation by global intensity threshold. The software does not register channels, estimate chromatic aberration, model point-spread functions, correct photobleaching, or identify cells/ROIs.

Pearson r is an image-wide linear association measure. Manders coefficients are threshold-dependent intensity fractions and can be inflated by diffuse background or by broad thresholds. Pixel colocalization should not be interpreted as molecular interaction. Pipeline object overlap is a binary intersection event; it says nothing about chance overlap, object pairing, or physical interaction.

Nearest-neighbor and centroid summaries use Euclidean distances in the image plane. Clark–Evans R here has no edge correction; image boundaries and nonuniform cell geometry can bias it. Statistical inference requires a study-specific null model and independent biological replicates, not pixels as independent observations.

Synthetic output provides exact masks for software checks only. Its disks, center placement, additive Gaussian noise, and Gaussian blur are intentionally simple. It is not a substitute for validation on appropriately licensed, representative microscopy images with expert annotations. Compare outputs with established tools under matched preprocessing and threshold conventions before use in a study.
