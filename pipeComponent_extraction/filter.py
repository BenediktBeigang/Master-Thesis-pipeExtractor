import numpy as np
from typing import Tuple

from custom_types import Segment3DArray


def _points_to_segment_min_distances_chunked(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    chunk_size: int = 200_000,
) -> np.ndarray:
    # ...existing code...
    N, D = points.shape
    M = a.shape[0]
    v = b - a  # (M,D)
    len_sq = np.einsum("ij,ij->i", v, v)  # (M,)

    dists = np.empty(N, dtype=float)
    thr = 1e-30

    for start in range(0, N, chunk_size):
        chunk = points[start : start + chunk_size]  # (K, D)
        K = chunk.shape[0]

        w = chunk[:, None, :] - a[None, :, :]  # heavy but chunked
        inv_len = np.where(len_sq > thr, 1.0 / len_sq, 0.0)  # (M,)

        t = np.einsum("kmd,md->km", w, v) * inv_len  # (K,M)
        t = np.clip(t, 0.0, 1.0)
        proj_diff = w - t[:, :, None] * v[None, :, :]  # (K,M,D)
        d2 = np.einsum("kmd,kmd->km", proj_diff, proj_diff)  # (K,M)
        min_d2 = np.min(d2, axis=1)
        dists[start : start + K] = np.sqrt(min_d2)

    return dists


def _normalize_pipes_to_arrays(
    pipes: Segment3DArray, ignore_z: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unterstützt ausschliesslich Segment3DArray:
      ndarray mit Shape (M, 2, 3) wobei jede Zeile [p1, p2] ist.
    Gibt arrays a,b zurück (Start- und Endpunkte). Wenn ignore_z True,
    werden nur x,y-Spalten zurückgegeben.
    """
    arr = np.asarray(pipes, dtype=float)

    # empty input
    if arr.size == 0:
        return np.empty((0, 2 if ignore_z else 3)), np.empty((0, 2 if ignore_z else 3))

    # Expect shape (M,2,3)
    if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] >= (2 if ignore_z else 3):
        if ignore_z:
            a = arr[:, 0, :2].astype(float)
            b = arr[:, 1, :2].astype(float)
        else:
            a = arr[:, 0, :3].astype(float)
            b = arr[:, 1, :3].astype(float)
        return a, b

    raise ValueError("pipes must be a Segment3DArray with shape (M,2,3)")


def filter_points_by_pipe_distance_vectorized(
    points: np.ndarray,
    pipes: Segment3DArray,
    distance_threshold: float = 1.0,
    ignore_z: bool = False,
    chunk_size: int = 200_000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized filter using chunking. Returns (kept_points, kept_indices).
    Accepts pipes as:
      - ndarray (M,6) columns [p1x,p1y,p1z,p2x,p2y,p2z]
      - structured ndarray with fields 'p1_x' etc.
      - iterable of dicts with those keys
    """
    pts_all = np.asarray(points)
    if pts_all.size == 0:
        return np.empty((0, pts_all.shape[1])), np.empty((0,), dtype=int)

    if len(pipes) == 0:
        return np.empty((0, pts_all.shape[1])), np.empty((0,), dtype=int)

    # Prepare arrays
    if ignore_z:
        pts = pts_all[:, :2]
    else:
        pts = pts_all[:, :3]

    a, b = _normalize_pipes_to_arrays(pipes, ignore_z)
    if a.shape[0] == 0:
        return np.empty((0, pts_all.shape[1])), np.empty((0,), dtype=int)

    dists = _points_to_segment_min_distances_chunked(pts, a, b, chunk_size=chunk_size)
    mask = dists < float(distance_threshold)
    kept_indices = np.nonzero(mask)[0]
    return kept_indices
