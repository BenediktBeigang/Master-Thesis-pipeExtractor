# parallel_grabPipe.py
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory, get_context
import numpy as np
from scipy.spatial import KDTree
from custom_types import Segment3DArray, Segment3DArray_Empty

# Globals, die im Worker gefüllt werden
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


def worker_process_segment(task, args_dict):
    """
    task: (segment_idx, segment)
    args_dict: Parameter für das Segment-Processing
    """
    from snapToPipe import process_single_segment

    segment_idx, segment = task

    if _XYZ is None or _KDTREE is None:
        print("Fehler: _XYZ oder _KDTREE ist None im Worker!")
        return (segment_idx, Segment3DArray_Empty(), [])

    # Extract parameters from args_dict
    min_samples = args_dict["min_samples"]
    samples_per_meter = args_dict["samples_per_meter"]
    tangential_half_width = args_dict["tangential_half_width"]
    normal_length = args_dict["normal_length"]
    min_pts = args_dict["min_pts"]
    poisson_radius = args_dict["poisson_radius"]
    quantization_precision = args_dict["quantization_precision"]

    xy = _XYZ[:, :2].astype(float, copy=False)

    # Process the segment
    snapped_segments, seg_sample_data = process_single_segment(
        segment,
        min_samples,
        samples_per_meter,
        tangential_half_width,
        normal_length,
        _KDTREE,
        _XYZ,
        xy,
        min_pts,
        poisson_radius,
        quantization_precision,
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
    segments,
    normal_length: float = 1.0,
    tangential_half_width: float = 0.25,
    min_pts: int = 4,
    samples_per_meter: float = 1.0,
    min_samples: int = 3,
    poisson_radius: float = 0.02,
    quantization_precision: float = 0.01,
    max_workers: int = -1,
) -> Segment3DArray:
    """
    Uses the approximated segments that are close to the real pipes in the pointcloud and snaps to them.
    This function is parallelized and uses shared memory to avoid copying the point cloud data to each worker.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Point cloud data (x, y, z) in world space
    segments : Segment3DArray, shape (M, 2, 3)
        List of segments to be snapped, each defined by two endpoints (x1, y1, z1) and (x2, y2, z2)
    normal_length : float, optional
        Length of the bounding box in normal direction (meters), by default 1.0
    tangential_half_width : float, optional
        Half-width of the bounding box in tangential direction (meters), by default 0.25
    min_pts : int, optional
        Minimum number of points required in the bounding box, by default 4
    samples_per_meter : float, optional
        Number of sample points per meter of segment length, by default 1.0
    min_samples : int, optional
        Minimum number of sample points along the segment, by default 3
    quantization_precision : float, optional
        Precision for quantizing z-values (meters), by default 0.01
    max_workers : int, optional
        Maximum number of parallel worker processes, by default -1 (all available cores)

    Returns
    -------
    Segment3DArray, shape (N, 2, 3)
    """
    from custom_types import Segment3DArray_Empty
    from export import export_sample_vectors_to_obj

    if not isinstance(xyz, np.ndarray) or xyz.shape[1] < 2:
        raise ValueError("xyz has to be [N,3].")

    # Shared Memory vorbereiten
    shm, shape, dtype_str = share_xyz_array(xyz)

    # Parameter-Dictionary für Worker
    args_dict = dict(
        min_samples=min_samples,
        samples_per_meter=samples_per_meter,
        tangential_half_width=tangential_half_width,
        normal_length=normal_length,
        min_pts=min_pts,
        poisson_radius=poisson_radius,
        quantization_precision=quantization_precision,
    )

    # Tasks erstellen: (segment_idx, segment)
    tasks = [(i, seg) for i, seg in enumerate(segments)]

    # Ergebnis-Arrays initialisieren
    out_segments: Segment3DArray = Segment3DArray_Empty()
    sample_data = []
    total_processed = 0

    # Parallelisierung
    ctx = get_context()
    if max_workers == -1:
        max_workers = os.cpu_count() or 4
    chunksize = max(1, len(tasks) // (max_workers * 4))  # adaptive chunksize

    print(f"Verarbeite {len(segments)} Segmente parallel mit {max_workers} Workern...")

    try:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_shm,
            initargs=(shm.name, shape, dtype_str),
        ) as executor:
            from itertools import repeat

            # Alle Segmente parallel verarbeiten
            for segment_idx, snapped_segments, seg_sample_data in executor.map(
                worker_process_segment,
                tasks,
                repeat(args_dict),
                chunksize=chunksize,
            ):
                if len(snapped_segments) > 0:
                    out_segments = np.vstack([out_segments, snapped_segments])
                sample_data.extend(seg_sample_data)

                total_processed += 1
                if total_processed % 10 == 0:
                    print(f"Verarbeitet: {total_processed}/{len(segments)} Segmente")

    finally:
        # SHM nur im Hauptprozess schließen/unlinken
        shm.close()
        shm.unlink()

    print(f"Parallelisierung abgeschlossen: {total_processed} Segmente verarbeitet")

    # Export der Sample-Daten
    export_sample_vectors_to_obj(sample_data, tangential_half_width, normal_length)

    return out_segments
