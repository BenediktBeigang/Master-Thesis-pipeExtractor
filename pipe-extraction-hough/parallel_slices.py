# parallel_slices.py
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, get_context
import numpy as np

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


def worker_process_slice(task, args_dict):
    """
    task: (slice_idx, z_center, zmin, zmax)
    args_dict: nur primitive Werte (float/bool/int), damit Pickling klein bleibt
    """
    from calcSlice import process_single_slice  # Import im Worker (sauber bei spawn)
    from clustering_hough import cluster_segments
    from merge_segments import merge_segments_in_clusters

    slice_idx, z_center, zmin, zmax = task

    class A:  # leichter Namespace-Ersatz
        pass

    a = A()
    a.cell_size = args_dict["cell_size"]
    a.canny_sigma = args_dict["canny_sigma"]
    a.min_line_length_m = args_dict["min_line_length_m"]
    a.max_line_gap_m = args_dict["max_line_gap_m"]

    # Lokale Cluster-/Merge-Parameter (wie synchron)
    a.local_eps_euclid = args_dict["local_eps_euclid"]
    a.local_min_samples = args_dict["local_min_samples"]
    a.local_rho_scale = args_dict["local_rho_scale"]
    a.local_preserve_noise = args_dict["local_preserve_noise"]
    a.local_gap_threshold = args_dict["local_gap_threshold"]
    a.local_min_length = args_dict["local_min_length"]
    a.local_z_max = args_dict["local_z_max"]
    merge_segments = args_dict["merge_segments"]

    # Hough-Detektion im Slice
    segments_world = process_single_slice(_XYZ, z_center, zmin, zmax, a, slice_idx)
    if not segments_world:
        return (slice_idx, [])

    if merge_segments:
        # Lokales Clustering (streng) wie im synchronen Code
        result = cluster_segments(
            segments_world,
            eps_euclid=a.local_eps_euclid,
            min_samples=a.local_min_samples,
            rho_scale=a.local_rho_scale,
            preserve_noise=a.local_preserve_noise,
        )
        if "clusters" not in result or not result["clusters"]:
            return (slice_idx, [])

        # Merge innerhalb des Slices (kurze Fragmente raus, Segmente verschmelzen)
        merged = merge_segments_in_clusters(
            segments_world,
            result["clusters"],
            gap_threshold=a.local_gap_threshold,
            min_length=a.local_min_length,
            z_max=a.local_z_max,
        )
    else:
        merged = segments_world  # ohne Merge, nur Clustering
    return (slice_idx, merged)


def share_xyz_array(xyz: np.ndarray):
    """Legt xyz in Shared Memory und gibt (shm, shape, dtype_str) zurück."""
    shm = shared_memory.SharedMemory(create=True, size=xyz.nbytes)
    shm_arr = np.ndarray(xyz.shape, dtype=xyz.dtype, buffer=shm.buf)
    shm_arr[:] = xyz  # einmalige Kopie in SHM
    return shm, xyz.shape, str(xyz.dtype)
