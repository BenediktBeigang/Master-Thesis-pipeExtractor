import argparse
import os
from custom_types import PipeComponentArray
from util import load_las, prepare_output_directory
from export_geojson import export_geojson
from eval.pipeEval import pipeEval
from eval.componentEval import componentEval
from eval.load_geojson import load_geojson
from pipe_extraction.pipe_extraction_hough_parallel import extract_pipes
from pipeComponent_extraction.pipeComponentExtraction import extract_pipeComponents


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

    prepare_output_directory("./output/", clean=False)
    pointcloudName = os.path.basename(args.input).split(".")[0]

    if True:
        print("#######################")
        print("### Pipe Extraction ###")
        print("#######################")
        xyz_pipes = load_las(args.input, ignoreZ=False, filterClass=1)
        pipes, snapped_chains = extract_pipes(
            xyz=xyz_pipes,
            config_path=args.config_path,
            pointcloudName=pointcloudName,
        )

        print("\n\n")
        print("#################################")
        print("### Pipe Component Extraction ###")
        print("#################################")
        xyz_pipeComponents = load_las(args.input, ignoreZ=False, filterClass=2)
        pipeComponents: PipeComponentArray | None = extract_pipeComponents(
            xyz=xyz_pipeComponents,
            config_path=args.config_path,
            pipes=pipes,
            pointcloudName=pointcloudName,
            apply_poisson=True,
            poisson_radius=0.02,
            near_pipe_filter=True,
        )

        export_geojson(
            pipes,
            snapped_chains,
            pipeComponents,
            f"./output/geojson/{pointcloudName}.geojson",
        )

    if True:
        print("\n\n")
        print("########################")
        print("### Pipe Evaluation  ###")
        print("########################")
        ground_truth = [
            "../Master-Thesis/poiExtraction/ontras_0_ground_truth.json",
            "../Master-Thesis/poiExtraction/ontras_1_ground_truth.json",
            "../Master-Thesis/poiExtraction/ontras_2_ground_truth.json",
            "../Master-Thesis/poiExtraction/ontras_3_ground_truth.json",
        ]
        ground_truth_pipes, ground_truth_components, ground_truth_pipes_asChain = (
            load_geojson(ground_truth[3])
        )
        detected_pipes, detected_components, detected_pipes_asChain = load_geojson(
            f"./output/geojson/{pointcloudName}.geojson"
        )

        pipeEval(ground_truth_pipes, detected_pipes, pointcloudName)
        componentEval(ground_truth_components, detected_components, pointcloudName)


if __name__ == "__main__":
    main()
