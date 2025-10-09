import datetime
from typing import Optional, Union
import numpy as np
from sklearn.cluster import HDBSCAN
import time
from custom_types import PipeComponentArray, Segment3DArray
from pipeComponent_extraction.export import write_obj_boxes
from pipeComponent_extraction.filter import filter_points_by_pipe_distance_vectorized


def ensure_bool_mask(
    mask_like: Optional[Union[np.ndarray, list]], length: int
) -> np.ndarray:
    if mask_like is None:
        return np.ones(length, dtype=bool)
    arr = np.asarray(mask_like)
    if arr.dtype == bool and arr.shape[0] == length:
        return arr
    mask = np.zeros(length, dtype=bool)
    mask[arr.astype(int)] = True
    return mask


def extract_pipeComponents(
    xyz: np.ndarray,
    pipes: Segment3DArray,
    pointcloudName: str,
    near_pipe_filter: bool = False,
) -> PipeComponentArray:
    start_time = time.time()
    n_points = len(xyz)

    # 1) pipe-based filter
    if near_pipe_filter:
        print("Filtering points near pipes...")
        indicies = filter_points_by_pipe_distance_vectorized(
            xyz, pipes, distance_threshold=0.5, ignore_z=True
        )
        mask = ensure_bool_mask(indicies, n_points)
        kept = mask.sum()
        print(f"Pipe filter: kept {kept}/{n_points} points.")
    else:
        mask = np.ones(n_points, dtype=bool)

    if mask.sum() == 0:
        print("Warning: No points after filter")
        return

    # 2) Clustern nur auf XY
    print("Clustering points with HDBSCAN...")
    xyz_filtered = xyz[mask]
    xy_filtered = xyz_filtered[:, :2]

    print(f"Clustering {len(xy_filtered)} XY-points with HDBSCAN")
    clusterer = HDBSCAN(
        min_cluster_size=20,
        metric="euclidean",
        cluster_selection_method="eom",
        n_jobs=-1,
    )
    clusterer.fit(xy_filtered)
    labels_filtered = clusterer.labels_

    # 3) Labels zurück auf Original-Indexraum mappen
    full_labels = -1 * np.ones(n_points, dtype=int)
    original_indices_of_filtered = np.nonzero(mask)[0]
    full_labels[original_indices_of_filtered] = labels_filtered

    # 4) Für jedes Cluster einfache Bounding-Box und Centroid berechnen (auf 3D-Punkte)
    unique = set(labels_filtered)
    unique.discard(-1)
    cluster_boxes = {}
    means = {}
    components: PipeComponentArray = []
    for c in sorted(unique):
        orig_inds = np.where(full_labels == c)[0]
        pts3d = xyz[orig_inds]
        mins = pts3d.min(axis=0)
        maxs = pts3d.max(axis=0)
        centroid = pts3d.mean(axis=0)
        cluster_boxes[c] = (mins.tolist(), maxs.tolist())
        means[c] = centroid.tolist()
        components.append((mins, maxs, centroid))

    # 5) Export
    write_obj_boxes(cluster_boxes, f"./output/obj/pipeComponent_bbox.obj")

    print("")
    print("Finished!")
    print("Total components found:", len(unique))
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return components
