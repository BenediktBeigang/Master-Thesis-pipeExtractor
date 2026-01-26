import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, get_context
import numpy as np
from scipy.spatial import KDTree
from pipeExtractor.custom_types import (
    Point3DArray,
    Segment3DArray,
    Segment3DArray_Empty,
)
from pipeExtractor.util import load_config

# Globals, which are filled in the worker
_SHM = None
_XYZ = None
_SHAPE = None
_DTYPE = None
_KDTREE = None


def _init_shm(shm_name, shape, dtype_str):
    """Wird einmal pro Worker-Prozess aufgerufen."""
    global _SHM, _XYZ, _SHAPE, _DTYPE, _KDTREE
    _SHM = shared_memory.SharedMemory(name=shm_name)
    _SHAPE = tuple(shape)
    _DTYPE = np.dtype(dtype_str)
    _XYZ = np.ndarray(_SHAPE, dtype=_DTYPE, buffer=_SHM.buf)
    XY = _XYZ[:, :2].astype(float, copy=False)
    _KDTREE = KDTree(XY)


def worker_process_segment(task, config_path):
    """
    task: (segment_idx, segment)
    args_dict: Parameter für das Segment-Processing
    """
    from pipeExtractor.pipe_extraction.snapToPipe import process_single_segment

    segment_idx, segment = task

    if _XYZ is None or _KDTREE is None:
        print("Fehler: _XYZ oder _KDTREE ist None im Worker!")
        return (segment_idx, Segment3DArray_Empty(), [])

    xy = _XYZ[:, :2].astype(float, copy=False)
    args = load_config(config_path)["snap_to_pipe"]

    # Process the segment
    snapped_segments, seg_sample_data = process_single_segment(
        segment,
        _KDTREE,
        _XYZ,
        xy,
        args,
    )

    return (segment_idx, snapped_segments, seg_sample_data)


def share_xyz_array(xyz: np.ndarray):
    """Legt xyz in Shared Memory und gibt (shm, shape, dtype_str) zurück."""
    shm = shared_memory.SharedMemory(create=True, size=xyz.nbytes)
    shm_arr = np.ndarray(xyz.shape, dtype=xyz.dtype, buffer=shm.buf)
    shm_arr[:] = xyz  # einmalige Kopie in SHM
    return shm, xyz.shape, str(xyz.dtype)


def snap_segments_to_point_cloud_data_parallel(
    xyz: np.ndarray,
    segments: Segment3DArray,
    pointcloudName: str,
    config_path: str,
    output_dir: str,
    max_workers: int = -1,
) -> tuple[Segment3DArray, list[Point3DArray]]:
    """
    Uses the approximated segments that are close to the real pipes in the pointcloud and snaps to them.
    This function is parallelized and uses shared memory to avoid copying the point cloud data to each worker.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Point cloud data (x, y, z) in world space
    segments : Segment3DArray, shape (M, 2, 3)
        List of segments to be snapped, each defined by two endpoints (x1, y1, z1) and (x2, y2, z2)
    config_path : str
        Path to the configuration file (JSON) containing parameters for snapping
    max_workers : int, optional
        Maximum number of parallel worker processes, by default -1 (all available cores)

    Returns
    -------
    Segment3DArray, shape (N, 2, 3)
    """
    from pipeExtractor.custom_types import Segment3DArray_Empty
    from pipeExtractor.pipe_extraction.export import export_sample_vectors_to_obj

    if not isinstance(xyz, np.ndarray) or xyz.shape[1] < 2:
        raise ValueError("xyz has to be [N,3].")

    # Prepare Shared Memory
    shm, shape, dtype_str = share_xyz_array(xyz)

    # Create tasks: (segment_idx, segment)
    tasks = [(i, seg) for i, seg in enumerate(segments)]

    # Initialize result arrays
    out_segments: Segment3DArray = Segment3DArray_Empty()
    sample_data = []
    segment_samples: list[list[dict]] = [[] for _ in range(len(segments))]
    total_processed = 0

    # Parallelization
    ctx = get_context()
    if max_workers == -1:
        max_workers = os.cpu_count() or 4
    chunksize = max(1, len(tasks) // (max_workers * 4))  # adaptive chunksize

    print(
        f"Processing {len(segments)} segments in parallel with {max_workers} workers..."
    )

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_shm,
            initargs=(shm.name, shape, dtype_str),
        ) as executor:
            from itertools import repeat

            # Process all segments in parallel
            for segment_idx, snapped_segments, seg_sample_data in executor.map(
                worker_process_segment,
                tasks,
                repeat(config_path),
                chunksize=chunksize,
            ):
                if len(snapped_segments) > 0:
                    out_segments = np.vstack([out_segments, snapped_segments])
                sample_data.extend(seg_sample_data)
                segment_samples[segment_idx] = seg_sample_data

                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"Finished: {total_processed}/{len(segments)} segments")

    finally:
        # SHM only in the main process close/unlink
        shm.close()
        shm.unlink()

    print(f"Processing complete: {total_processed} segments processed")

    # Export of the Sample Data as a .obj for Visualization
    args = load_config(config_path)["snap_to_pipe"]
    export_sample_vectors_to_obj(
        sample_data,
        args["tangential_length"],
        args["normal_length"],
        pointcloudName,
        output_dir,
    )

    snapped_chains: list[Point3DArray] = []
    for seg_sample_data in segment_samples:
        if not seg_sample_data:
            snapped_chains.append(np.empty((0, 3), dtype=float))
            continue
        seg_points = []
        for sample in seg_sample_data:
            c_xy = np.asarray(sample["c_xy"]).reshape(-1)
            if c_xy.shape[0] != 2:
                raise ValueError("c_xy muss 2 Elemente enthalten.")
            seg_points.append([c_xy[0], c_xy[1], sample["z"]])
        snapped_chains.append(np.asarray(seg_points, dtype=float))

    return out_segments, snapped_chains
