import argparse
import gc
import glob
import os
from custom_types import PipeComponentArray
from util import load_las, prepare_output_directory
from export_geojson import export_geojson
from eval.pipeEval import pipeEval
from eval.componentEval import componentEval
from eval.load_geojson import load_geojson
from pipe_extraction.pipe_extraction_hough_parallel import extract_pipes
from pipeComponent_extraction.pipeComponentExtraction import extract_pipeComponents


def main(
    pc_path: str,
    gt_path: str = None,
    config_path: str = "./config.json",
    eval: bool = False,
):
    """
    Main function to extract pipes from a point cloud and optionally evaluate against ground truth.

    Parameters
    ----------
    pc_path : str
        Path to the input point cloud file.
    gt_path : str, optional
        Path to the ground truth GeoJSON file.
    config_path : str, optional
        Path to the configuration file (JSON) with parameters.
    eval : bool, optional
        Whether to run evaluation after extraction.
    """
    prepare_output_directory("./output/", clean=False)
    path_to_pc = args.input if pc_path is None else pc_path
    pointcloudName = os.path.basename(path_to_pc).split(".")[0]
    pointcloudName = pointcloudName + "_" + os.path.basename(config_path).split(".")[0]

    if True:
        print("#######################")
        print("### Pipe Extraction ###")
        print("#######################")

        xyz_pipes = load_las(
            path_to_pc,
            ignoreZ=False,
            filterClass=1,
        )
        pipes, snapped_chains = extract_pipes(
            xyz=xyz_pipes,
            config_path=config_path,
            pointcloudName=pointcloudName,
        )
        del xyz_pipes
        gc.collect()

        print("\n\n")
        print("#################################")
        print("### Pipe Component Extraction ###")
        print("#################################")
        xyz_pipeComponents = load_las(
            path_to_pc,
            ignoreZ=False,
            filterClass=2,
        )
        pipeComponents: PipeComponentArray | None = extract_pipeComponents(
            xyz=xyz_pipeComponents,
            config_path=config_path,
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
            pointscloudName=f"./output/geojson/{pointcloudName}.geojson",
        )
        del pipes, snapped_chains, pipeComponents
        gc.collect()

    if gt_path is None:
        print("No ground truth path provided, skipping evaluation.")
        return

    if eval:
        print("\n\n")
        print("########################")
        print("### Pipe Evaluation  ###")
        print("########################")
        ground_truth_pipes, ground_truth_components, ground_truth_pipes_asChain = (
            load_geojson(gt_path)
        )
        detected_pipes, detected_components, detected_pipes_asChain = load_geojson(
            f"./output/geojson/{pointcloudName}.geojson"
        )

        pipeEval(ground_truth_pipes, detected_pipes, pointcloudName)
        componentEval(ground_truth_components, detected_components, pointcloudName)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipe extraction from LAS to GeoJSON/OBJ")
    ap.add_argument("--input", required=True, help="Path to LAS/LAZ file")
    ap.add_argument("--gt_path", default=None, help="Path to Ground Truth GeoJSON file")
    ap.add_argument(
        "--config_path",
        default="./config.json",
        help="Path to configuration file (JSON) with parameters",
    )
    ap.add_argument(
        "--eval", default=False, help="Whether to run evaluation after extraction"
    )
    args = ap.parse_args()

    main(
        pc_path=args.input,
        gt_path=args.gt_path,
        config_path=args.config_path,
        eval=args.eval,
    )
