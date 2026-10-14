from vesiclenet.synthetic import generate_synthetic_image
from vesiclenet.pipeline import analyze_image

def test_pipeline_reports_reproducible_analysis():
    sample = generate_synthetic_image(shape=(64,64), n_objects=10, overlap=1, noise=0, blur=.3, seed=9)
    report = analyze_image(sample.image)
    assert report["analysis"]["threshold_method"] == "otsu"
    assert report["object_overlap"]["channel_0"]["object_count"] > 0
