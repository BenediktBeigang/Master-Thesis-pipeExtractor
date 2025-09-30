#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import math
import numpy as np
from scipy.stats import qmc
from scipy.spatial import KDTree
from custom_types import Point2D, Point3D, Segment3DArray
from samplePointMerge import extract_segments


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _compute_quantized_z(z_values: np.ndarray, bucket_size: float) -> float:
    """
    Extracts the the highest z-value where points can be found and ignoring outliers.
    With it the top edge of the pipe can be found.

    ## How it works
    This function creates a histogram of Z-values using the specified bucket size,
    calculates a threshold as half the mean of all bucket counts, and returns
    the center of the last (highest) bucket that exceeds this threshold.

    Parameters
    ----------
    z_values : np.ndarray
        Array of Z-coordinate values to be quantized
    bucket_size : float
        Size of each bucket for the histogram binning

    Returns
    -------
    float
        The center point of the selected bucket. Returns 0.0 if no Z-values
        are provided, or the mean Z-value if only one bucket is needed.
    """
    if len(z_values) == 0:
        return 0.0

    # Buckets erstellen
    z_min, z_max = z_values.min(), z_values.max()
    num_buckets = max(1, int(np.ceil((z_max - z_min) / bucket_size)))

    if num_buckets == 1:
        return z_values.mean()

    # Histogramm erstellen
    counts, bin_edges = np.histogram(z_values, bins=num_buckets, range=(z_min, z_max))

    # Schwellwert als Mittelwert der Häufigkeiten
    threshold = counts.mean() / 2

    # Ersten Bucket über Schwellwert finden
    over_threshold = np.where(counts >= threshold)[0]
    if len(over_threshold) == 0:
        # Fallback: höchsten Peak nehmen
        bucket_idx = np.argmax(counts)
    else:
        bucket_idx = over_threshold[-1]  # Letzten (höchsten) Index nehmen

    # Bucket-Mittelpunkt zurückgeben
    bucket_center = (bin_edges[bucket_idx] + bin_edges[bucket_idx + 1]) / 2

    # print(
    #     f"Pos: {pos}, Z-Buckets: {counts}, Threshold: {threshold:.2f}, Center: {bucket_center:.3f}"
    # )

    # if int(pos[0]) == 875320 and int(pos[1]) == 5715369:
    #     print(f"Variant: {variant}")
    #     print(f"Z-Werte: {len(z_values)}")
    #     print(f"counts: {counts}, mean: {counts.mean():.2f}")
    #     print(f"threshold: {threshold:.2f}, center: {bucket_center:.3f}")
    # import matplotlib.pyplot as plt

    # bucket_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # plt.bar(bucket_centers, counts, width=bucket_size, align="center", alpha=0.7)
    # plt.axhline(y=threshold, color="r", linestyle="--", label="Threshold")
    # plt.axvline(
    #     x=bucket_center, color="g", linestyle="--", label="Selected Bucket Center"
    # )
    # plt.xlabel("Z Value")
    # plt.ylabel("Count")
    # plt.title("Z Value Buckets")
    # plt.legend()
    # plt.show()

    return bucket_center


def _bbox_centroid_for_endpoint(
    cloud_2D: np.ndarray,
    cloud_3D: np.ndarray,
    sample_point_xy: Point2D,
    t_hat: np.ndarray,
    n_hat: np.ndarray,
    tangential_half_width: float,
    normal_length: float,
    min_pts: int,
    poisson_radius: float,
    quantization_precision: float,
) -> tuple[Point2D, float, int]:
    """
    Calculate the centroid of points within an oriented bounding box around a segment endpoint.

    ## How it works
    This function creates a local coordinate system at the given endpoint and selects points that fall within a rectangular bounding box.
    The selected points are then subsampled using Poisson disk sampling and without the z-value to ensure uniform distribution.
    The centroid of the selected points is calculated in 2D (x, y).
    Then the z-value is determined by quantizing the z-values of the selected points.

    Local Coordinate System:
        - x-axis: Segment direction (t_hat)
        - y-axis: Normal direction (n_hat)
        - Bounding box: |x| <= tangential_half_width AND |y| <= normal_length

    Parameters
    ----------
    cloud_2D : np.ndarray, shape (N, 2)
        2D point cloud coordinates (x, y) in world space
    cloud_3D : np.ndarray, shape (N, 3)
        3D point cloud coordinates (x, y, z) in world space
    endpoint_xy : np.ndarray, shape (2,)
        2D coordinates of the segment endpoint around which to create the bounding box
    t_hat : np.ndarray, shape (2,)
        Unit vector in segment direction (tangent direction)
    n_hat : np.ndarray, shape (2,)
        Unit vector perpendicular to segment (normal direction)
    tangential_half_width : float
        Half-width of the bounding box in tangential direction (meters)
    normal_length : float
        Full length of the bounding box in normal direction (meters)
    min_pts : int
        Minimum number of points required in the bounding box
    poisson_radius : float
        Radius for Poisson disk sampling (must be > 0)
    quantization_precision : float
        Precision for quantizing z-values (meters)

    Returns
    -------
    tuple[np.ndarray, float, int]
        - selected_xy : Point2D, shape (2,)
            2D centroid coordinates of the selected points
        - selected_z : float
            Quantized z-coordinate of the selected points
        - point_count : int
            Number of points used for centroid calculation

    Raises
    ------
    ValueError
        If poisson_radius <= 0
    """
    D = cloud_2D - sample_point_xy[None, :]  # (N,2)
    x_local = D @ t_hat
    y_local = D @ n_hat

    # Sammle Punkte in beide Richtungen gleichzeitig (lokale Orientierungs-BBox)
    mask = (np.abs(x_local) <= tangential_half_width) & (
        np.abs(y_local) <= normal_length
    )
    idx = np.nonzero(mask)[0]
    if idx.size < min_pts:
        return sample_point_xy, 0.0, 0

    if poisson_radius <= 0:
        raise ValueError("poisson_radius must be > 0")

    # --- Poisson-Disk-Subsampling im lokalen Rechteck [-w, w] x [-L, L] ---
    # 1) Generiere Poisson-Disk-Samples im lokalen Domain-Rechteck
    w = tangential_half_width
    L = normal_length
    eng = qmc.PoissonDisk(
        d=2,
        radius=float(poisson_radius),
        l_bounds=np.array([-w, -L]),
        u_bounds=np.array([+w, +L]),
        rng=int(42),
    )
    # fill_space füllt bis keine Kandidaten mehr passen (für d=2 meist flott)
    S = eng.fill_space()  # (M,2) lokale Samplepunkte

    # Fallback: Wenn aus irgendeinem Grund keine Samples entstehen, nimm alle Punkte
    if S.size == 0:
        selected_xy = cloud_2D[idx].mean(axis=0)
        z_values = cloud_3D[idx, 2]
        selected_z = _compute_quantized_z(z_values, quantization_precision)
        return selected_xy, selected_z, idx.size

    # 2) Mappe Samples auf die nächstgelegenen Originalpunkte in der Box (lokal!)
    P_local = np.column_stack((x_local[idx], y_local[idx]))  # (K,2)
    kdt = KDTree(P_local)
    nn_idx = kdt.query(S, k=1)[1]  # (M,) Indizes innerhalb von 'idx'
    nn_idx = np.unique(nn_idx)  # eindeutige Auswahl

    if nn_idx.size < min_pts:
        # Wenn zu wenige Punkte übrig bleiben, fallback auf alle Box-Punkte
        selected_xy = cloud_2D[idx].mean(axis=0)
        z_values = cloud_3D[idx, 2]
        selected_z = _compute_quantized_z(z_values, quantization_precision)
        return selected_xy, selected_z, idx.size

    # 3) Schwerpunkt in Weltkoordinaten über der subsampleten Teilmenge
    idx_sub = idx[nn_idx]
    selected_xy = cloud_2D[idx_sub].mean(axis=0)
    z_values = cloud_3D[idx_sub, 2]
    selected_z = _compute_quantized_z(z_values, quantization_precision)
    return selected_xy, selected_z, idx_sub.size


def snap_segments_to_point_cloud_data(
    xyz: np.ndarray,
    segments: Segment3DArray,
    normal_length: float = 1.0,
    tangential_half_width: float = 0.25,
    min_pts: int = 4,
    samples_per_meter: float = 1.0,
    min_samples: int = 3,
    poisson_radius: float = 0.02,
    quantization_precision: float = 0.01,
) -> Segment3DArray:
    """
    Uses the approximated segments that are close to the real pipes in the pointcloud and snaps to them.

    ## How it works
    The function samples points along the segment, creating an oriented bounding box around each sample point
    and calculating the centroid of the points within the box.
    The new snapped segment is then created from these centroids and RANSAC.
    The length of the original segment is preserved, by projecting the endpoints of the sampling chain onto the result line by RANSAC.

    Parameters
    ----------
    xyz : np.ndarray, shape (N, 3)
        Point cloud data (x, y, z) in world space
    segments_2x3 : Segment3DArray, shape (M, 2, 3)
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

    Returns
    -------
    Segment3DArray, shape (N, 2, 3)
    """
    if not isinstance(xyz, np.ndarray) or xyz.shape[1] < 2:
        raise ValueError("xyz has to be [N,3].")
    XY = xyz[:, :2].astype(float, copy=False)

    kdtree = KDTree(XY)

    sample_data = []
    out_segments: Segment3DArray = np.empty((0, 2, 3), dtype=np.float64)
    for index, seg in enumerate(segments):
        if index % 5 == 0:
            print(f"Segment {index}/{len(segments)}")

        seg = np.asarray(seg, dtype=float)
        if seg.shape != (2, 3):
            raise ValueError(f"Segment hat Form {seg.shape}, erwartet (2,3).")

        p1 = seg[0].copy()
        p2 = seg[1].copy()
        p1_xy, p2_xy = p1[:2], p2[:2]

        # Segmentlänge und Richtung
        segment_vec = p2 - p1
        segment_length = np.linalg.norm(segment_vec)

        if segment_length < 1e-6:
            out_segments = np.vstack([out_segments, seg])  # degeneriert
            continue

        # Anzahl Sample-Punkte basierend auf Segmentlänge
        num_samples = max(min_samples, int(np.ceil(segment_length * samples_per_meter)))

        # Tangente/Normale aus XY
        t_hat = _unit(p2_xy - p1_xy)
        n_hat = np.array([-t_hat[1], t_hat[0]])

        # Sample-Punkte entlang des Segments generieren (inkl. Endpunkte)
        t_values = np.linspace(0, 1, num_samples)
        collected_points = []
        collected_weights = []

        # Maximale Suchradius für alle Sample-Punkte dieses Segments
        search_radius = math.sqrt(tangential_half_width**2 + normal_length**2)

        for t in t_values:
            # Interpolierter Punkt auf dem Segment
            sample_point_xyz: Point3D = p1 + t * segment_vec
            sample_point_xy: Point2D = sample_point_xyz[:2]

            # Nur relevante Punkte per KDTree
            candidate_indices = kdtree.query_ball_point(sample_point_xy, search_radius)
            if len(candidate_indices) < min_pts:
                continue

            XY_candidates = XY[candidate_indices]
            xyz_candidates = xyz[candidate_indices]

            # BBox-Centroid für diesen Sample-Punkt
            grabed_xy, grabed_z, n_pts = _bbox_centroid_for_endpoint(
                XY_candidates,
                xyz_candidates,
                sample_point_xy,
                t_hat,
                n_hat,
                tangential_half_width,
                normal_length,
                min_pts,
                poisson_radius,
                quantization_precision=quantization_precision,
            )

            # Nur verwenden wenn genügend Punkte gefunden
            if n_pts > 0:
                # 3D-Punkt: XY aus BBox, Z interpoliert
                c_3d = np.array([grabed_xy[0], grabed_xy[1], grabed_z])
                collected_points.append(c_3d)
                collected_weights.append(n_pts)  # Gewichtung nach Anzahl Punkte
                sample_data.append(
                    {
                        "t_hat": t_hat.copy(),
                        "n_hat": n_hat.copy(),
                        "c_xy": grabed_xy.copy(),
                        "sample_point_xy": sample_point_xy.copy(),
                        "z": grabed_z,
                    }
                )

        # Fallback: Wenn keine gültigen Punkte gefunden, ursprüngliches Segment behalten
        if len(collected_points) < 2:
            seg_reshaped = seg.astype(float).reshape(1, 2, 3)
            out_segments = np.vstack([out_segments, seg_reshaped])
            continue

        collected_points = np.array(collected_points)

        extracted = extract_segments(collected_points)
        out_segments = np.vstack([out_segments, extracted])

        _export_sample_vectors_to_obj(sample_data, tangential_half_width, normal_length)

    return out_segments


def _export_sample_vectors_to_obj(
    sample_data: list, tangential_half_width: float, normal_length: float
):
    """
    Exportiert Sample-Point-Vektoren in eine OBJ-Datei.

    Für jeden Sample-Point werden exportiert:
    - Tangentialvektor (t_hat) skaliert mit tangential_half_width
    - Normalenvektor (n_hat) skaliert mit normal_length
    - Resultierender Punkt (c_xy) als Punkt
    """
    with open("./sample_vectors.obj", "w") as f:
        f.write("# Sample Point Vectors Export\n")
        f.write(f"# Tangential half width: {tangential_half_width}\n")
        f.write(f"# Normal length: {normal_length}\n")
        f.write(f"# Total sample points: {len(sample_data)}\n\n")

        vertex_count = 0

        for i, data in enumerate(sample_data):
            t_hat = data["t_hat"]
            n_hat = data["n_hat"]
            c_xy = data["c_xy"]
            sample_xy = data["sample_point_xy"]
            z = data["z"]

            f.write(f"# Sample point {i}\n")

            # Startpunkt (Sample-Point-Position)
            start_3d = [sample_xy[0], sample_xy[1], z]
            f.write(f"v {start_3d[0]:.6f} {start_3d[1]:.6f} {start_3d[2]:.6f}\n")

            # Endpunkt des Tangentialvektors (beide Richtungen)
            t_end1_3d = [
                start_3d[0] + t_hat[0] * tangential_half_width,
                start_3d[1] + t_hat[1] * tangential_half_width,
                z,
            ]
            t_end2_3d = [
                start_3d[0] - t_hat[0] * tangential_half_width,
                start_3d[1] - t_hat[1] * tangential_half_width,
                z,
            ]
            f.write(f"v {t_end1_3d[0]:.6f} {t_end1_3d[1]:.6f} {t_end1_3d[2]:.6f}\n")
            f.write(f"v {t_end2_3d[0]:.6f} {t_end2_3d[1]:.6f} {t_end2_3d[2]:.6f}\n")

            # Endpunkt des Normalenvektors (beide Richtungen)
            n_end1_3d = [
                start_3d[0] + n_hat[0] * normal_length,
                start_3d[1] + n_hat[1] * normal_length,
                z,
            ]
            n_end2_3d = [
                start_3d[0] - n_hat[0] * normal_length,
                start_3d[1] - n_hat[1] * normal_length,
                z,
            ]
            f.write(f"v {n_end1_3d[0]:.6f} {n_end1_3d[1]:.6f} {n_end1_3d[2]:.6f}\n")
            f.write(f"v {n_end2_3d[0]:.6f} {n_end2_3d[1]:.6f} {n_end2_3d[2]:.6f}\n")

            # Resultierender Punkt (c_xy)
            result_3d = [c_xy[0], c_xy[1], z]
            f.write(f"v {result_3d[0]:.6f} {result_3d[1]:.6f} {result_3d[2]:.6f}\n")

            # Linien definieren (OBJ verwendet 1-basierte Indizes)
            base_idx = vertex_count + 1

            # Tangentialvektor-Linien (Kreuz)
            f.write(f"l {base_idx} {base_idx + 1}\n")  # Start -> t_end1
            f.write(f"l {base_idx} {base_idx + 2}\n")  # Start -> t_end2

            # Normalenvektor-Linien (Kreuz)
            f.write(f"l {base_idx} {base_idx + 3}\n")  # Start -> n_end1
            f.write(f"l {base_idx} {base_idx + 4}\n")  # Start -> n_end2

            # Linie zum resultierenden Punkt
            f.write(f"l {base_idx} {base_idx + 5}\n")  # Start -> result

            vertex_count += 6  # 6 Vertices pro Sample-Point
            f.write("\n")
