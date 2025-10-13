from typing import Optional, Tuple, Union
import numpy as np
from sklearn.cluster import HDBSCAN
import time
from custom_types import PipeComponentArray, Point3D, Segment3DArray
from pipeComponent_extraction.export import write_obj_boxes
from pipeComponent_extraction.filter import filter_points_by_pipe_distance_vectorized
from util import load_config


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


def _nearest_pipe_info(point_xy: Point3D, pipes: Segment3DArray) -> Tuple[float, float]:
    best_dist = np.inf
    best_z = 0.0
    for pipe in pipes:
        start_xy = pipe[0, :2]
        end_xy = pipe[1, :2]
        seg = end_xy - start_xy
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq == 0:
            t = 0.0
        else:
            t = np.dot(point_xy - start_xy, seg) / seg_len_sq
            t = float(np.clip(t, 0.0, 1.0))
        closest_xy = start_xy + t * seg
        dist = float(np.linalg.norm(point_xy - closest_xy))
        if dist < best_dist:
            best_dist = dist
            best_z = float(pipe[0, 2] + t * (pipe[1, 2] - pipe[0, 2]))
    return best_dist, best_z


def extract_pipeComponents(
    xyz: np.ndarray,
    config_path: str,
    pipes: Segment3DArray,
    pointcloudName: str,
    near_pipe_filter: bool = False,
) -> PipeComponentArray:
    start_time = time.time()
    n_points = len(xyz)

    print(f"Load config.json...")
    config = load_config(config_path)["pipeComponent_clustering"]

    # 1) pipe-based filter
    if near_pipe_filter:
        print("Filtering points near pipes...")
        indicies = filter_points_by_pipe_distance_vectorized(
            xyz,
            pipes,
            distance_threshold=config["pipe_distance_threshold"],
            ignore_z=True,
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
        min_cluster_size=config["min_points"],
        metric=config["metric"],
        cluster_selection_method=config["cluster_selection_method"],
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
        centroid = pts3d.mean(axis=0)
        if pipes.size == 0:
            continue
        dist_xy, pipe_z = _nearest_pipe_info(centroid[:2], pipes)
        if dist_xy > config["pipe_distance_threshold"]:
            continue
        centroid[2] = pipe_z
        mins = pts3d.min(axis=0)
        maxs = pts3d.max(axis=0)
        cluster_boxes[c] = (mins.tolist(), maxs.tolist())
        means[c] = centroid.tolist()
        components.append((mins, maxs, centroid))

    # 5) Export
    write_obj_boxes(
        cluster_boxes, f"./output/obj/{pointcloudName}_pipeComponent_bbox.obj"
    )

    print("")
    print("Finished!")
    print("Total clusters found:", len(unique), "-> components:", len(components))
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")

    return components
