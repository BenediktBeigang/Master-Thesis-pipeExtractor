from typing import List, Tuple
from custom_types import Point3D
from eval.export import export_points_to_obj
from eval.metrics.pipeTruePositive import is_component_in_tolerance


def component_detection_metric(
    ground_truth_components: List[Point3D],
    detected_components: List[Point3D],
    pointcloudName: str,
    tolerance: float = 0.5,
) -> Tuple[int, int, int, int, List[float], List[float]]:
    """
    Calculates detection metrics for pipe segments based on ground truth and detections.

    ## How it works
    Each ground-truth component is compared against all detected components. If a detected
    component lies within the tolerance, it is counted as found and its XY/Z distances are
    recorded. Remaining ground-truth components are marked as missed. False positives are
    derived from detections without a ground-truth match.

    Parameters
    ------------
    ground_truth_components : list
        Ground-truth components, each represented as a tuple of two 3D points.
    detected_components : list
        Detected components, each represented as a tuple of two 3D points.
    pointcloudName : str
        Identifier for the point cloud (currently unused).
    tolerance : float
        Maximum allowed distance between corresponding points to count as found.

    Returns
    --------
    found : int
        Number of ground-truth components with a matching detection.
    missed : int
        Number of ground-truth components without a matching detection.
    false_positives : int
        Number of detections without a corresponding ground-truth match.
    xy_distances : List[float]
        XY-plane distances for matched components.
    z_distances : List[float]
        Z-axis distances for matched components.
    """

    found = 0
    missed = 0
    xy_distances: List[float] = []
    z_distances: List[float] = []
    true_positives: set[int] = set()

    # --- Hauptlogik ---
    for gt in ground_truth_components:
        label = "not_found"

        for det_idx, det in enumerate(detected_components):
            validComponent, xy_dis, z_dis = is_component_in_tolerance(
                det, gt, tolerance
            )

            if validComponent:
                true_positives.add(det_idx)
                label = "found" if label == "not_found" else label

                # XY/Z-Distanzen sammeln
                xy_distances.append(xy_dis)
                z_distances.append(z_dis)

        match label:
            case "found":
                found += 1
            case _:
                missed += 1

    false_positives = len(detected_components) - len(true_positives)

    matched_components = [detected_components[idx] for idx in sorted(true_positives)]
    export_points_to_obj(
        matched_components,
        f"./output/obj/{pointcloudName}_detected_components.obj",
    )

    return (
        found,
        missed,
        false_positives,
        xy_distances,
        z_distances,
    )
