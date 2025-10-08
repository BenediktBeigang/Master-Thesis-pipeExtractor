import math
import numpy as np
from custom_types import (
    ListOfPoint3DArrays,
    Point2D,
    Point2DArray,
    Point3D,
    Point3DArray,
    Segment3D,
    Segment3D_Create,
    Segment3DArray,
    Segment3DArray_Empty,
)
from skimage.measure import LineModelND, ransac
from util import project_point_to_line


def _buckets_by_delta_z(
    points_3D_chain: Point3DArray,
    samples_per_meter: float,
    max_angle_change_deg: float,
    min_points_per_bucket: int = 2,
    merge_jump_threshold: float = 3.0,
) -> ListOfPoint3DArrays:
    """
    Groups a chain of 3D points into buckets based on z-coordinate differences.

    ## How it works
    This function merges a sequence of 3D points into buckets by analyzing the delta-z between consecutive accepted points.
    Diagonal movements are filtered out by checking the mean angle of z-differences, because the pipes are expected to be mostly horizontal.
    Gaps caused by overlapping pipes, for example, are taken into account by combining buckets.
    This occurs when the endpoints of two buckets are below merge_jump_threshold and the average z-values are similar.

    Parameters
    -----------
    points_3D_chain : Point3DArray, shape (N, 3)
        Array of shape (N, 3) containing ordered 3D points (x, y, z).
    samples_per_meter : float
        Number of samples per meter along the chain, used to compute angles.
    max_angle_change_deg : float
        Maximum allowed angle change (in degrees) between consecutive points
        to be considered part of the same bucket.
    min_points_per_bucket : int, optional
        Minimum number of points required in a bucket to be kept, by default 2
    merge_jump_threshold : float, optional
        Maximum distance between the end of one bucket and the start of the next
        to consider merging them, by default 3.0

    Returns
    --------
    ListOfPoint3DArrays
        List of numpy arrays, each containing a sequence of 3D points that
        form a coherent bucket based on z-coordinate progression. Each bucket
        contains at least 2 points.
    """
    deltaZ_threshold = math.sin(math.radians(max_angle_change_deg)) / samples_per_meter

    if points_3D_chain is None or len(points_3D_chain) < 2:
        return []

    z = points_3D_chain[:, 2]
    n = len(points_3D_chain)

    buckets = []
    current = [points_3D_chain[0]]

    # Vereinfacht: nur Mean der z-Differenzen
    dz_values = []

    for i in range(1, n):
        current_dz = z[i - 1] - z[i]

        # Ersten Punkt immer akzeptieren
        if len(dz_values) == 0:
            current.append(points_3D_chain[i])
            dz_values.append(current_dz)
            continue

        mean_dz = np.mean(dz_values)

        # Bei Richtungswechsel neuen Bucket starten
        if abs(current_dz - mean_dz) > deltaZ_threshold:
            if len(current) >= min_points_per_bucket:
                buckets.append(np.array(current))

            # Neuer Bucket mit Überlappung
            current = [points_3D_chain[i - 1], points_3D_chain[i]]
            dz_values = [current_dz]
        else:
            current.append(points_3D_chain[i])
            dz_values.append(current_dz)

    # Add last bucket
    if len(current) >= min_points_per_bucket:
        buckets.append(np.array(current))

    # Ignore all buckets that:
    # - have less than min_points_per_bucket
    # - with a mean angle change of more than 20 degrees
    buckets = [
        b
        for b in buckets
        if (
            len(b) >= min_points_per_bucket
            and np.mean(
                np.abs(np.arctan(np.diff(b[:, 2]) * samples_per_meter) * 180 / np.pi)
            )
            < 15
        )
    ]

    # Merge buckets that are closer than merge_jump_threshold
    merged_buckets = []
    if len(buckets) > 1:
        current_bucket = buckets[0]
        for i in range(1, len(buckets)):
            dist = np.linalg.norm(current_bucket[-1] - buckets[i][0])
            if (
                dist < merge_jump_threshold
                and abs(np.mean(current_bucket[:, 2]) - np.mean(buckets[i][:, 2])) < 0.1
            ):
                current_bucket = np.vstack((current_bucket, buckets[i]))
            else:
                merged_buckets.append(current_bucket)
                current_bucket = buckets[i]
        merged_buckets.append(current_bucket)
        buckets = merged_buckets
    return buckets


def fit_ransac_line_and_project_endpoints(
    points_3D: Point3DArray,
    residual_threshold: float,
    min_samples: float = 2,
    max_trials: int = 1000,
) -> Segment3D:
    """
    Fits a 2D line to the XY-projection of 3D points using RANSAC and projects the chain endpoints onto it.

    ## How it works
    Performs robust line fitting on the XY-coordinates of the input 3D points using RANSAC to handle outliers.
    The fitted line is then used to create a 3D segment by projecting the first and last points of the chain onto the line while preserving their original Z-coordinates.

    Parameters
    -----------
    points_3D : Point3DArray, shape (N, 3)
        Array of 3D points (x, y, z) forming an ordered chain. Must contain at least 2 points.
    residual_threshold : float, default=0.2
        Maximum distance from a point to the fitted line for it to be considered an inlier
        during RANSAC fitting.
    min_samples : float, default=2
        Minimum number of samples required to fit the line model. Should be 2 for a 2D line.
    max_trials : int, default=1000
        Maximum number of RANSAC iterations to attempt before giving up.

    Returns
    --------
    Segment3D
        A 3D line segment represented by two 3D points: the start and end points of the
        original chain projected onto the fitted line in XY, with their original Z-coordinates.

    Raises:
    -------
    ValueError
        If less than 2 points are provided or if RANSAC line fitting fails.
    """
    if len(points_3D) < 2:
        raise ValueError("Zu wenige Punkte für Linien-Fit.")

    points_2D: Point2DArray = points_3D[:, :2]

    # Fit mit skimage LineModelND + ransac (robust, N-D geeignet)
    model = LineModelND()
    model_est, inliers = ransac(
        points_2D,
        LineModelND,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials,
    )

    if model_est is None:
        raise ValueError("RANSAC line fitting failed.")

    # model_est.params: (origin, direction_unit_vector)
    origin = model_est.params[0]
    direction = model_est.params[1]

    # Endpunkte der Kette (unabhängig von Inlier-Status) auf die gefittete Linie projizieren
    start_pt: Point2D = points_2D[0]
    end_pt: Point2D = points_2D[-1]
    proj_start: Point2D = project_point_to_line(start_pt, origin, direction)
    proj_end: Point2D = project_point_to_line(end_pt, origin, direction)

    segment_start: Point3D = np.array([proj_start[0], proj_start[1], points_3D[0, 2]])
    segment_end: Point3D = np.array([proj_end[0], proj_end[1], points_3D[-1, 2]])
    return Segment3D_Create(segment_start, segment_end)


def extract_segments(
    points_3D: Point3DArray,
    samples_per_meter: float,
    max_angle_change_deg: float = 15.0,
    ransac_residual_threshold: float = 0.05,
) -> Segment3DArray:
    """
    Extracts 3D line segments from a chain of 3D points by clustering points based on their Z-coordinates and fitting lines using RANSAC.

    ## How it works
    This function segments a sequence of 3D points into coherent buckets based on Z-coordinate differences.
    Each bucket is then processed to fit a 3D line segment using RANSAC, ensuring robustness against outliers.
    The resulting segments preserve the original Z-coordinates of the endpoints.

    Parameters
    -----------
    points_3D : Point3DArray, shape (N, 3)
        Array of 3D points (x, y, z) forming an ordered chain. Must contain at least 2 points.
    dz_threshold : float, default=0.3
        Maximum allowed deviation from the running mean of z-differences
        for a point to be considered an inlier when forming buckets.
    ransac_residual_threshold : float, default=0.05
        Maximum distance from a point to the fitted line for it to be considered an inlier
        during RANSAC fitting of each bucket.

    Returns
    --------
    Segment3DArray
        An array of 3D line segments, each represented by two 3D points: the start and end points of the segment.
        The segments are derived from fitting lines to the clustered buckets of points.
    """
    if (
        not isinstance(points_3D, np.ndarray)
        or points_3D.ndim != 2
        or points_3D.shape[1] != 3
    ):
        raise ValueError("points_3D muss ein (N,3)-Array sein.")

    if len(points_3D) < 2:
        return Segment3DArray_Empty()

    buckets = _buckets_by_delta_z(
        points_3D,
        samples_per_meter,
        max_angle_change_deg=max_angle_change_deg,
    )

    # buckets = _buckets_by_dbscan_z(
    #     points_3D,
    #     eps=0.2,
    #     min_samples=2,
    # )

    segments_out: Segment3DArray = Segment3DArray_Empty()
    for sub_chain in buckets:
        a, b = fit_ransac_line_and_project_endpoints(
            sub_chain, residual_threshold=ransac_residual_threshold
        )
        segment_2x3 = np.vstack([a, b]).astype(float).reshape(1, 2, 3)
        segments_out = np.vstack([segments_out, segment_2x3])
        continue
    return segments_out
