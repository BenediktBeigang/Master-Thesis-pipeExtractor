import argparse
import gc
import glob
import os
from custom_types import PipeComponentArray, Segment3DArray
from util import load_las, prepare_output_directory
from export_geojson import export_geojson
from eval.pipeEval import pipeEval
from eval.componentEval import componentEval
from eval.load_geojson import load_geojson
from pipe_extraction.pipe_extraction_hough_parallel import extract_pipes
from pipeComponent_extraction.pipeComponentExtraction import extract_pipeComponents


def main(pc_path: str = None, gt_path: str = None, config_path: str = "./config.json", raw_segments_input: Segment3DArray = None):
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
        if raw_segments_input is None:
            pipes, snapped_chains = extract_pipes(
                xyz=xyz_pipes,
                config_path=config_path,
                pointcloudName=pointcloudName,
            )
        else:
            pipes, snapped_chains = extract_pipes(
                xyz=xyz_pipes,
                config_path=config_path,
                pointcloudName=pointcloudName,
                raw_segments_input=raw_segments_input,
            )
        del xyz_pipes
        gc.collect()

        # print("\n\n")
        # print("#################################")
        # print("### Pipe Component Extraction ###")
        # print("#################################")
        # xyz_pipeComponents = load_las(
        #     path_to_pc,
        #     ignoreZ=False,
        #     filterClass=2,
        # )
        # pipeComponents: PipeComponentArray | None = extract_pipeComponents(
        #     xyz=xyz_pipeComponents,
        #     config_path=config_path,
        #     pipes=pipes,
        #     pointcloudName=pointcloudName,
        #     apply_poisson=True,
        #     poisson_radius=0.02,
        #     near_pipe_filter=True,
        # )

        export_geojson(
            pipes,
            snapped_chains=None,
            pipeComponents=None,
            pointscloudName=f"./output/geojson/{pointcloudName}.geojson",
        )
        # del pipes, snapped_chains, pipeComponents
        del pipes
        gc.collect()

    if gt_path is None:
        print("No ground truth path provided, skipping evaluation.")
        return

    if True:
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
        # componentEval(ground_truth_components, detected_components, pointcloudName)


if __name__ == "__main__":
    # main(
    #     "/mnt/c/Users/bened/Downloads/0904/ontras_3_predicted_0904_t1_sampled.las",
    #     "../Master-Thesis/ground_truth/ontras_3_ground_truth.json",
    #     "./config.json",
    # )
    
    # Alle Konfigurationsdateien aus dem config_variations Ordner laden
    config_files = glob.glob("config_variations/*_config.json")
    config_files.sort()  # Sortieren für konsistente Reihenfolge
    
    print(f"Gefundene Konfigurationsdateien: {len(config_files)}")
    
    i = 0  # Beispiel-Punktwolke Nummer
    
    # xyz_pipes = load_las(
    #     f"/mnt/c/Users/bened/Downloads/0904/ontras_{i}_predicted_0904_t1_sampled.las",
    #     ignoreZ=False,
    #     filterClass=1,
    # )
    # pipes_raw, snapped_chains = extract_pipes(
    #     xyz=xyz_pipes,
    #     config_path="./config.json",
    #     pointcloudName=f"ontras_{i}_raw_segments",
    #     get_raw_segments=True
    # )
    # del xyz_pipes
    # gc.collect()
    
    # Äußere Schleife über alle Konfigurationen
    for config_idx, config_path in enumerate(config_files, 1):
        config_name = os.path.basename(config_path).replace("_config.json", "")
        print(f"\n{'='*60}")
        print(f"KONFIGURATION {config_idx}/{len(config_files)}: {config_name}")
        print(f"Config-Datei: {config_path}")
        print(f"{'='*60}")
        
        # Innere Schleife über alle Punktwolken
        pc_path = f"/mnt/c/Users/bened/Downloads/0904/ontras_{i}_predicted_0904_t1_sampled.las"
        gt_path = f"../Master-Thesis/ground_truth/ontras_{i}_ground_truth.json"
        
        print(f"\n--- Verarbeite ontras_{i} mit Config {config_name} ---")
        
        try:
            main(pc_path, gt_path, config_path=config_path, raw_segments_input=None)
            print(f"✓ ontras_{i} erfolgreich verarbeitet")
        except Exception as e:
            print(f"✗ Fehler bei ontras_{i}: {str(e)}")
            continue
        
        print(f"\n--- Konfiguration {config_name} abgeschlossen ---")
    
    print(f"\n{'='*60}")
    print("ALLE KONFIGURATIONEN ABGESCHLOSSEN")
    print(f"{'='*60}")
