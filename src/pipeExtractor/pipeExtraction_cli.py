import argparse
import os
from pipeline import extract_features_for_pointcloud


def main():
    ap = argparse.ArgumentParser(description="Pipe extraction from LAS to GeoJSON/OBJ")
    ap.add_argument("--input", required=True, help="Path to LAS/LAZ file")
    ap.add_argument(
        "--config_path",
        default="./config.json",
        help="Path to configuration file (JSON) with parameters",
    )
    ap.add_argument("--gt_path", default=None, help="Path to Ground Truth GeoJSON file")
    ap.add_argument(
        "--output_dir",
        default="./output",
        help="Path to output directory for extracted files",
    )
    ap.add_argument(
        "--eval", default=False, help="Whether to run evaluation after extraction"
    )
    args = ap.parse_args()

    args.output_dir = os.path.abspath(args.output_dir)
    print(f"Output directory set to: {args.output_dir}")

    if args.gt_path is None and args.eval:
        print("No ground truth path provided, skipping evaluation.")
        args.eval = False

    extract_features_for_pointcloud(
        pc_path=args.input,
        gt_path=args.gt_path,
        config_path=args.config_path,
        output_dir=args.output_dir,
        eval=args.eval,
    )
