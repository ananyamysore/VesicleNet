"""Command line entry point."""
import argparse
from .pipeline import analyze_image

def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a two-channel TIFF with VesicleNet")
    parser.add_argument("image", help="two-channel TIFF (channel axis last by default)")
    parser.add_argument("--output", default="vesiclenet-results", help="output directory")
    parser.add_argument("--channel-axis", type=int, default=-1)
    parser.add_argument("--threshold-method", choices=["otsu", "yen", "li"], default="otsu")
    parser.add_argument("--pixel-size", type=float, default=1.0)
    args = parser.parse_args()
    analyze_image(args.image, output_dir=args.output, channel_axis=args.channel_axis,
                  threshold_method=args.threshold_method, pixel_size=args.pixel_size)
    print(f"Results written to {args.output}")

if __name__ == "__main__": main()
