import numpy as np
from sklearn.cluster import DBSCAN
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


# def _buckets_by_dbscan_z(
#     points_3D_chain: Point3DArray,
#     eps: float = 0.2,
#     min_samples: int = 2,
# ) -> ListOfPoint3DArrays:
#     """
#     Groups a chain of 3D points into buckets using DBSCAN clustering on z-coordinates.

#     ## How it works
#     This function segments a sequence of 3D points by clustering their z-coordinates using DBSCAN.
#     Each cluster becomes one bucket, regardless of sequence continuity. Clusters with too few points are discarded.

#     Parameters
#     -----------
#     points_3D_chain : Point3DArray, shape (N, 3)
#         Array of shape (N, 3) containing ordered 3D points (x, y, z).
#     eps : float, default=0.2
#         Maximum distance between two samples for one to be considered
#         in the neighborhood of the other for DBSCAN clustering.
#     min_samples : int, default=2
#         Minimum number of samples in a neighborhood for a point to be
#         considered as a core point in DBSCAN.

#     Returns
#     --------
#     ListOfPoint3DArrays
#         List of numpy arrays, each containing points from one DBSCAN cluster.
#         Points maintain their original order within each bucket.
#     """
#     if points_3D_chain is None or len(points_3D_chain) < 2:
#         return []

#     # Extrahiere z-Koordinaten für Clustering
#     z_coords = points_3D_chain[:, 2].reshape(-1, 1)

#     # DBSCAN Clustering auf z-Koordinaten
#     clustering = DBSCAN(eps=eps, min_samples=min_samples)
#     cluster_labels = clustering.fit_predict(z_coords)

#     # Sammle Indizes für jeden Cluster
#     label_to_indices = {}
#     for i, label in enumerate(cluster_labels):
#         if label == -1:  # Noise points ignorieren
#             continue
#         if label not in label_to_indices:
#             label_to_indices[label] = []
#         label_to_indices[label].append(i)

#     # Erstelle Buckets - ein Cluster = ein Bucket
#     buckets = []
#     for label, indices in label_to_indices.items():
#         if len(indices) > 2:  # Cluster mit <= 2 Punkten rausfliegen
#             # Behalte ursprüngliche Reihenfolge bei
#             indices.sort()
#             bucket_points = points_3D_chain[indices]
#             buckets.append(bucket_points)

#     # Debug-Output
#     if len(buckets) > 1:
#         print(f"{len(buckets)} DBSCAN Buckets (eps={eps}, min_samples={min_samples})")
#         for bi, b in enumerate(buckets):
#             z_range = b[:, 2].max() - b[:, 2].min()
#             print(f"  Bucket {bi}: {len(b)} Punkte, z-Range: {z_range:.3f}")

#     return buckets


def _buckets_by_delta_z(
    points_3D_chain: Point3DArray,
    deltaZ_threshold: float,
    outlier_z_threshold: float,
    min_points_per_bucket: int = 2,
) -> ListOfPoint3DArrays:
    """
    Groups a chain of 3D points into buckets based on z-coordinate differences.

    ## How it works
    This function segments a sequence of 3D points by analyzing the delta-z between consecutive accepted points.
    It uses an adaptive approach with outlier detection to handle noisy data while preserving meaningful transitions in the z-direction.
    Outliers are single points that deviate significantly from the running mean of delta-z values (imagine three points forming a triangle).
    Otherwise a new bucket is started when a significant change in delta-z is detected.

    Parameters
    -----------
    points_3D_chain : Point3DArray, shape (N, 3)
        Array of shape (N, 3) containing ordered 3D points (x, y, z).
    deltaZ_threshold : float
        Maximum allowed deviation from the running mean of z-differences
        for a point to be considered an inlier.
    outlier_z_threshold : float
        Maximum z-difference threshold used for look-ahead outlier detection.
        If skipping a potential outlier results in a z-difference below this
        threshold, the point is considered an outlier and ignored.

    Returns
    --------
    ListOfPoint3DArrays
        List of numpy arrays, each containing a sequence of 3D points that
        form a coherent bucket based on z-coordinate progression. Each bucket
        contains at least 2 points.
    """
    if points_3D_chain is None or len(points_3D_chain) < 2:
        return []

    z = points_3D_chain[:, 2]
    n = len(points_3D_chain)

    buckets = []
    current = [points_3D_chain[0]]
    # Statistik über akzeptierte Punkte
    count = 0
    mean_dz = 0.0

    # Index des zuletzt AKZEPTIERTEN Punktes (nicht nur letzter iterierter)
    last_accepted = 0

    def accept(idx, dz):
        nonlocal count, mean_dz, last_accepted
        current.append(points_3D_chain[idx])
        last_accepted = idx

        # Online-Mean (Welford light)
        count += 1
        mean_dz += (dz - mean_dz) / count

    # Starte mit erstem akzeptierten Übergang
    # (der zweite Punkt wird gleich verarbeitet)
    for i in range(1, n):
        dz = z[last_accepted] - z[i]

        if count == 0:
            # Warm-up: die ersten 1–2 Punkte immer akzeptieren
            accept(i, dz)
            continue

        # Inlier?
        if abs(dz - mean_dz) <= deltaZ_threshold:
            accept(i, dz)
            continue

        # Kandidat Outlier? Mit Look-Ahead relativ zu last_acc prüfen:
        if i + 1 < n:
            dz_without_potential_outlier = z[last_accepted] - z[i + 1]
            if abs(dz_without_potential_outlier) <= outlier_z_threshold:
                # i ist Outlier -> ignoriere ihn komplett
                # (kein Update von mean/count/last_acc)
                continue

        # Mindestens ein echter Richtungswechsel -> Bucket schließen
        if len(current) >= 2:
            buckets.append(np.array(current))
        else:
            # Sicherheitsnetz, sollte praktisch nicht auftreten
            buckets.append(
                np.array([points_3D_chain[last_accepted], points_3D_chain[i]])
            )

        # Neuen Bucket starten: Naht erhalten, daher mit last_acc und i
        current = [points_3D_chain[last_accepted], points_3D_chain[i]]
        # Statistik neu initialisieren ab diesem Sprung
        mean_dz = dz
        count = 1
        last_accepted = i

    if len(current) >= 2:
        buckets.append(np.array(current))

    multibuckets = len(buckets) > 1
    if multibuckets:
        print(f"{len(buckets)} Buckets")
        for bi, b in enumerate(buckets):
            print(f"{len(b)}")

    # Mindestens 2 Punkte pro Bucket
    buckets = [b for b in buckets if len(b) >= min_points_per_bucket]

    if multibuckets:
        print(
            f"{len(buckets)} Buckets nach min_points_per_bucket={min_points_per_bucket}"
        )

    return buckets


def fit_ransac_line_and_project_endpoints(
    points_3D: Point3DArray,
    residual_threshold: float = 0.2,
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
    dz_threshold: float = 0.2,
    outlier_z_threshold: float = 0.2,
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
    outlier_z_threshold : float, default=0.1
        Maximum z-difference threshold used for look-ahead outlier detection
        when forming buckets.
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
        deltaZ_threshold=dz_threshold,
        outlier_z_threshold=outlier_z_threshold,
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
