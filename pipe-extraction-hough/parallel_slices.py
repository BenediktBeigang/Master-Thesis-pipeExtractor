# parallel_slices.py
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, get_context
import numpy as np

from util import load_config

# Globals, die im Worker gefüllt werden
_SHM = None
_XYZ = None
_SHAPE = None
_DTYPE = None


def _init_shm(shm_name, shape, dtype_str):
    """Wird einmal pro Worker-Prozess aufgerufen."""
    global _SHM, _XYZ, _SHAPE, _DTYPE
    _SHM = shared_memory.SharedMemory(name=shm_name)
    _SHAPE = tuple(shape)
    _DTYPE = np.dtype(dtype_str)
    _XYZ = np.ndarray(_SHAPE, dtype=_DTYPE, buffer=_SHM.buf)


def worker_process_slice(task, config_path):
    """
    task: (slice_idx, z_center, zmin, zmax)
    args_dict: nur primitive Werte (float/bool/int), damit Pickling klein bleibt
    """
    from calcSlice import find_lines_in_slice  # Import im Worker (sauber bei spawn)
    from clustering_hough import cluster_segments
    from merge_segments import merge_segments_in_clusters

    slice_idx, z_center, zmin, zmax = task

    config = load_config(config_path)
    args = config["slice_cluster_and_merge"]

    # Hough-Detektion im Slice
    segments_world = find_lines_in_slice(
        _XYZ, z_center, zmin, zmax, config["hough"], slice_idx
    )
    if not segments_world:
        return (slice_idx, [])

    # Lokales Clustering
    result = cluster_segments(
        segments_world,
        eps_euclid=args["epsilon"],
        min_samples=args["min_samples"],
        rho_scale=args["rho_scale"],
        preserve_noise=args["preserve_noise"],
    )
    if "clusters" not in result or not result["clusters"]:
        return (slice_idx, [])

    # Merge innerhalb des Slices (kurze Fragmente raus, Segmente verschmelzen)
    merged = merge_segments_in_clusters(
        segments_world,
        result["clusters"],
        gap_threshold=args["max_line_gap"],
        min_length=args["min_line_length"],
    )
    return (slice_idx, merged)


def share_xyz_array(xyz: np.ndarray):
    """Legt xyz in Shared Memory und gibt (shm, shape, dtype_str) zurück."""
    shm = shared_memory.SharedMemory(create=True, size=xyz.nbytes)
    shm_arr = np.ndarray(xyz.shape, dtype=xyz.dtype, buffer=shm.buf)
    shm_arr[:] = xyz  # einmalige Kopie in SHM
    return shm, xyz.shape, str(xyz.dtype)
