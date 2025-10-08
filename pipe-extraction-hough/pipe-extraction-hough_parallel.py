#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D-Liniendetektion in Z-Slices einer LAS-Punktwolke per Rasterisierung + Hough,
Export aller Linien als OBJ (Polylines) für CloudCompare.

Abhängigkeiten:
    numpy
    laspy        (>=2.0 empfohlen; Fallback für 1.7.0 integriert)
    scikit-image (skimage)
    matplotlib   (für Bildexport)

Beispiel:
    python slice_hough_obj.py \
        --input in.las \
        --thickness 0.10 \
        --cell-size 0.03 \
        --min-count 3 \
        --use-canny \
        --canny-sigma 1.2 \
        --min-line-length-m 0.40 \
        --max-line-gap-m 0.10 \
        --top-k 300 \
        --output lines.obj \
        --save-images \
        --image-output-dir slice_images
"""

from itertools import repeat
import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import math
from multiprocessing import get_context
import os
import time
import sys

import numpy as np

from calcSlice import get_z_slices
from clustering_hough import cluster_segments
from custom_types import Segment3DArray
from export import (
    write_clusters_as_obj,
    write_obj_lines,
    write_segments_as_geojson,
)
from merge_segments import merge_segments_in_clusters
from parallel_slices import share_xyz_array, _init_shm, worker_process_slice
from parallel_snapping import snap_segments_to_point_cloud_data_parallel
from util import load_config, load_las, prepare_output_directory


def main():
    startTime = time.time()
    checkpointTime = startTime

    ap = argparse.ArgumentParser(
        description="Z-Slice Hough-Liniendetektion aus LAS → OBJ (CloudCompare)"
    )
    ap.add_argument("--input", required=True, help="Pfad zur LAS/LAZ-Datei")
    ap.add_argument(
        "--config_path",
        default="./config.json",
        help="Pfad zur Konfigurationsdatei (JSON) mit Parametern",
    )
    ap.add_argument(
        "--output",
        default="houghOutput.obj",
        help="Ausgabe-OBJ",
    )
    args = ap.parse_args()

    print(f"Lade config.json...")
    config = load_config(args.config_path)

    print(f"Lade Punktwolke: {args.input}")
    xyz = load_las(args.input)
    if xyz.size == 0:
        print("Leere Punktwolke.", file=sys.stderr)
        sys.exit(1)

    print(f"Punktwolke geladen: {xyz.shape[0]} Punkte")

    pointcloud_name = os.path.basename(args.input)

    # Berechne alle Z-Slices
    slices = get_z_slices(xyz, config["slice_thickness"])

    # Shared Memory vorbereiten
    shm, shape, dtype_str = share_xyz_array(xyz)

    # Tasks in der Reihenfolge der Slices bauen (bewahrt Sortierung)
    tasks = [(i, zc, zmin, zmax) for i, (zc, zmin, zmax) in enumerate(slices)]

    # Verarbeite alle Slices
    all_segments: Segment3DArray = np.empty((0, 2, 3), dtype=np.float64)
    total_processed = 0

    prepare_output_directory("./output/")

    print(f"Phase 1: Approximating lines...")
    print(f"Phase 1a): Process {len(slices)} slices in parallel...")

    # robustes Startverfahren wählen:
    # - Linux: 'fork' nutzt COW, spart anfänglich RAM, ist ok wenn du SharedMemory sowieso nutzt
    # - Windows/macOS: 'spawn' ist Standard, SHM funktioniert dort genau für diesen Use-Case
    ctx = get_context()  # Standard-Startmethode des OS
    max_workers = os.cpu_count() or 4
    chunksize = 8  # feinabstimmen bei vielen Slices

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_shm,
            initargs=(shm.name, shape, dtype_str),
        ) as ex:
            for slice_idx, segments in ex.map(
                worker_process_slice,
                tasks,
                repeat(args.config_path),
                chunksize=chunksize,
            ):
                if len(segments) > 0:
                    all_segments = np.vstack([all_segments, segments])
                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"Finished: {total_processed}/{len(slices)} Slices")
    finally:
        # SHM nur im Hauptprozess schließen/unlinken
        shm.close()
        shm.unlink()
        print()

    print(
        f"Phase 1a): Finished in {time.time() - checkpointTime:.2f}s - {time.time() - startTime:.2f}s"
    )
    checkpointTime = time.time()

    if len(all_segments) == 0:
        print("No lines found in any slice.", file=sys.stderr)
        sys.exit(0)

    # distance_tolerance_m = 1
    # angle_tolerance_deg = 5
    # angle_tolerance_radians = math.radians(angle_tolerance_deg)
    # rho_scale = (distance_tolerance_m) / (2 * math.sin(angle_tolerance_radians))
    # epsilon = math.sqrt(2) * 2 * math.sin(angle_tolerance_radians)
    # print(f"rho_scale: {rho_scale:.2f}, eps_euclid: {epsilon:.2f}")

    # Phase 2: Cluster über alle Slices
    print("Phase 1b): Cluster and merge over all segments and slices...")
    clusterAndMerge_args = config["global_cluster_and_merge"]
    result_phase1b_clustering = cluster_segments(
        all_segments,
        eps_euclid=clusterAndMerge_args["epsilon"],
        min_samples=clusterAndMerge_args["min_samples"],
        rho_scale=clusterAndMerge_args["rho_scale"],
        preserve_noise=True,
    )

    result_phase1b_clustering = result_phase1b_clustering["clusters"]
    write_clusters_as_obj(
        slice_idx=-1,
        segments=all_segments,
        clusters=result_phase1b_clustering,
        output_dir="./output/obj",
    )

    all_segments = merge_segments_in_clusters(
        all_segments,
        result_phase1b_clustering,
        gap_threshold=clusterAndMerge_args["max_line_gap"],
        min_length=clusterAndMerge_args["min_line_length"],
    )

    print(
        f"Phase 1b): Finished in {time.time() - checkpointTime:.2f}s - {time.time() - startTime:.2f}s"
    )
    checkpointTime = time.time()

    write_obj_lines(
        all_segments,
        f"{pointcloud_name}_approx.obj",
    )

    phase_2_enabled = True
    if phase_2_enabled:
        print("Phase 2: Snap segments to original point cloud data...")
        all_segments = snap_segments_to_point_cloud_data_parallel(
            xyz,
            all_segments,
            args.config_path,
        )

        write_obj_lines(
            all_segments,
            f"{pointcloud_name}_snapped.obj",
        )
        print(
            f"Phase 2: Finished in {time.time() - checkpointTime:.2f}s - {time.time() - startTime:.2f}s"
        )
        checkpointTime = time.time()

    write_segments_as_geojson(
        all_segments,
        f"{pointcloud_name}_pipes.geojson",
    )

    print(f"\nFinished!")
    print(f"Processed slices: {len(slices)}")
    print(f"Total lines found: {len(all_segments)}")
    print(f"Output file: {args.output}")
    endTime = time.time()
    print(f"Total time taken: {endTime - startTime:.2f} seconds")


if __name__ == "__main__":
    main()
