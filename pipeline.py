import gc
import os
from custom_types import PipeComponentArray
from util import load_las_and_split, prepare_output_directory
from export_geojson import export_geojson
from eval.pipeEval import pipeEval
from eval.componentEval import componentEval
from eval.load_geojson import load_geojson
from pipe_extraction.pipe_extraction_hough_parallel import extract_pipes
from pipeComponent_extraction.pipeComponentExtraction import extract_pipeComponents


def extract_features_for_pointcloud(
    pc_path: str,
    gt_path: str = None,
    config_path: str = "./config.json",
    output_dir: str = "./output",
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
    prepare_output_directory(output_dir)
    pointcloudName = os.path.basename(pc_path).split(".")[0]

    print("##########################")
    print("### Loading Pointcloud ###")
    print("##########################")
    try:
        xyz_pipes, xyz_pipeComponents = load_las_and_split(pc_path, ignoreZ=False)
        if len(xyz_pipes) == 0:
            raise RuntimeError("No pipe points (Class 1) found in the LAS file.")
    except Exception as e:
        print(f"Failed to load and split LAS file: {e}")
        print("Skipping extraction.")
        return

    print("#######################")
    print("### Pipe Extraction ###")
    print("#######################")

    pipes, snapped_chains = extract_pipes(
        xyz=xyz_pipes,
        config_path=config_path,
        pointcloudName=pointcloudName,
        output_dir=output_dir,
    )
    del xyz_pipes
    gc.collect()

    print("\n\n")
    print("#################################")
    print("### Pipe Component Extraction ###")
    print("#################################")

    if len(xyz_pipeComponents) > 0:
        pipeComponents: PipeComponentArray | None = extract_pipeComponents(
            xyz=xyz_pipeComponents,
            config_path=config_path,
            pipes=pipes,
            pointcloudName=pointcloudName,
            output_dir=output_dir,
            apply_poisson=True,
            poisson_radius=0.02,
            near_pipe_filter=True,
        )
    else:
        print("No pipe component points (Class 2) found in the LAS file.")
        pipeComponents = None

    export_geojson(
        pipes,
        snapped_chains,
        pipeComponents,
        pointscloudName=os.path.join(
            output_dir, "geojson", f"{pointcloudName}.geojson"
        ),
    )
    del pipes, snapped_chains, pipeComponents
    gc.collect()

    if not eval:
        return

    if gt_path is None:
        print("No ground truth path provided, skipping evaluation.")
        return

    print("\n\n")
    print("########################")
    print("### Pipe Evaluation  ###")
    print("########################")
    ground_truth_pipes, ground_truth_components, ground_truth_pipes_asChain = (
        load_geojson(gt_path)
    )
    detected_pipes, detected_components, detected_pipes_asChain = load_geojson(
        os.path.join(output_dir, "geojson", f"{pointcloudName}.geojson")
    )
    if len(detected_pipes) > 0 and len(ground_truth_pipes) > 0:
        pipeEval(ground_truth_pipes, detected_pipes, pointcloudName)
    if len(detected_components) > 0 and len(ground_truth_components) > 0:
        componentEval(ground_truth_components, detected_components, pointcloudName)
