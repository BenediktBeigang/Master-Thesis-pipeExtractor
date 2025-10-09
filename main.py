import argparse
import os
from custom_types import Segment3DArray
from export_geojson import export_geojson
from pipeComponent_extraction.armatureExtraction import extract_pipeComponents
from util import load_las, prepare_output_directory
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

    prepare_output_directory("./output/")
    pointcloudName = os.path.basename(args.input).split(".")[0]

    print("#######################")
    print("### Pipe Extraction ###")
    print("#######################")
    xyz_pipes = load_las(args.input, ignoreZ=False, filterClass=1)
    pipes: Segment3DArray = extract_pipes(
        xyz=xyz_pipes,
        config_path=args.config_path,
        pointcloudName=pointcloudName,
    )

    print("\n\n")

    print("#################################")
    print("### Pipe Component Extraction ###")
    print("#################################")
    xyz_pipeComponents = load_las(args.input, ignoreZ=False, filterClass=2)
    pipeComponents = extract_pipeComponents(
        xyz=xyz_pipeComponents,
        pipes=pipes,
        pointcloudName=pointcloudName,
        near_pipe_filter=True,
    )

    export_geojson(
        pipes,
        pipeComponents,
        f"./output/geojson/{pointcloudName}.geojson",
    )


if __name__ == "__main__":
    main()
