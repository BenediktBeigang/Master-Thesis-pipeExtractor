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

    slice_idx, z_center, zmin, zmax = task

    class A:  # leichter Namespace-Ersatz
        pass

    a = A()
    a.cell_size = args_dict["cell_size"]
    a.canny_sigma = args_dict["canny_sigma"]
    a.min_line_length_m = args_dict["min_line_length_m"]
    a.max_line_gap_m = args_dict["max_line_gap_m"]

    segments_world = process_single_slice(_XYZ, z_center, zmin, zmax, a, slice_idx)
    return (slice_idx, segments_world)


def share_xyz_array(xyz: np.ndarray):
    """Legt xyz in Shared Memory und gibt (shm, shape, dtype_str) zurück."""
    shm = shared_memory.SharedMemory(create=True, size=xyz.nbytes)
    shm_arr = np.ndarray(xyz.shape, dtype=xyz.dtype, buffer=shm.buf)
    shm_arr[:] = xyz  # einmalige Kopie in SHM
    return shm, xyz.shape, str(xyz.dtype)
