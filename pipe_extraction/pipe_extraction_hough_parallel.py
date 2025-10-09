#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import os
import time
import sys
import numpy as np
from pipe_extraction.calcSlice import get_z_slices
from pipe_extraction.clustering_hough import cluster_segments
from custom_types import Segment3DArray
from pipe_extraction.export import (
    write_clusters_as_obj,
    write_obj_lines,
)
from pipe_extraction.merge_segments import merge_segments_in_clusters
from pipe_extraction.parallel_slices import (
    share_xyz_array,
    _init_shm,
    worker_process_slice,
)
from pipe_extraction.parallel_snapping import snap_segments_to_point_cloud_data_parallel
from util import load_config


def extract_pipes(
    xyz: np.ndarray, config_path: str, pointcloudName: str
) -> Segment3DArray:
    """
    Extracts pipes from a point cloud.

    Parameters:
    -----------
    xyz : np.ndarray
        The input point cloud as a Nx3 numpy array.
    config_path : str
        Path to the configuration JSON file.
    pointcloudName : str
        Base name for output files (without extension).

    Returns:
    --------
    Segment3DArray
        An array of extracted pipe segments.
    """
    startTime = time.time()
    checkpointTime = startTime

    print(f"Load config.json...")
    config = load_config(config_path)

    if xyz.size == 0:
        print("Empty Pointcloud.", file=sys.stderr)
        sys.exit(1)

    print(f"Pointcloud loaded: {xyz.shape[0]} points")

    # Compute all Z-slices
    slices = get_z_slices(xyz, config["slice_thickness"])

    # Prepare Shared Memory
    shm, shape, dtype_str = share_xyz_array(xyz)

    # Build tasks in the order of slices (preserves sorting)
    tasks = [(i, zc, zmin, zmax) for i, (zc, zmin, zmax) in enumerate(slices)]

    # Process all slices
    all_segments: Segment3DArray = np.empty((0, 2, 3), dtype=np.float64)
    total_processed = 0

    print(f"Phase 1: Approximating lines...")
    print(f"Phase 1a): Process {len(slices)} slices in parallel...")

    # Choose robust start method for multiprocessing:
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
                repeat(config_path),
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

    # Phase 2: Cluster over all slices
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
        segments=all_segments,
        clusters=result_phase1b_clustering,
        output_path=f"./output/obj/{pointcloudName}_cluster.obj",
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
        f"./output/obj/{pointcloudName}_approx.obj",
    )

    phase_2_enabled = True
    if phase_2_enabled:
        print("Phase 2: Snap segments to original point cloud data...")
        all_segments = snap_segments_to_point_cloud_data_parallel(
            xyz,
            all_segments,
            config_path,
        )

        write_obj_lines(
            all_segments,
            f"./output/obj/{pointcloudName}_snapped.obj",
        )
        print(
            f"Phase 2: Finished in {time.time() - checkpointTime:.2f}s - {time.time() - startTime:.2f}s"
        )
        checkpointTime = time.time()

    print(f"\nFinished!")
    print(f"Processed slices: {len(slices)}")
    print(f"Total lines found: {len(all_segments)}")
    endTime = time.time()
    print(f"Total time taken: {endTime - startTime:.2f} seconds")

    return all_segments
