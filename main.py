import argparse
import os
from util import load_las
from pipe_extraction.pipe_extraction_hough_parallel import extract_pipes


def main():
    ap = argparse.ArgumentParser(
        description="Z-Slice Hough-Liniendetektion aus LAS → OBJ (CloudCompare)"
    )
    ap.add_argument("--input", required=True, help="Pfad zur LAS/LAZ-Datei")
    ap.add_argument(
        "--config_path",
        default="./config.json",
        help="Pfad zur Konfigurationsdatei (JSON) mit Parametern",
    )
    args = ap.parse_args()

    xyz_pipes = load_las(args.input, ignoreZ=False, filterClass=1)
    pipes = extract_pipes(xyz_pipes, args.config_path, os.path.basename(args.input))

    # xyz_pipeComponents = load_las(args.input, ignoreZ=False, filterClass=2)


if __name__ == "__main__":
    main()
