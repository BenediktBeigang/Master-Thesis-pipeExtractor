#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing: Endpunkte von Segmenten (2x3) per orientierter BBox snappen.
Eingabe/Output sind IDENTISCH zum merge_segments_in_clusters-Format:
    List[np.ndarray] mit je shape (2,3): [[x1,y1,z1],[x2,y2,z2]]
"""

from __future__ import annotations
import math
import numpy as np
from scipy.stats import qmc
from scipy.spatial import KDTree
import open3d as o3d


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


# Änderungen:
# 3d Punktwolke übergeben und an betreffenden stellen die 2d XY Koordinaten extrahieren
# Am Ende mit den indizes der bbox punkte, die z-Werte der gefundenen Punkte extrahieren
# quantisieren mit buckets der größe 0.05, mittelwert bestimmen und als schwellwert nehmen
# z-Wert für resultierender Punkt ist bucket bei dem das erste mal der schwellwert überschritten wird
def _bbox_centroid_for_endpoint(
    XY: np.ndarray,
    endpoint_xy: np.ndarray,
    t_hat: np.ndarray,
    n_hat: np.ndarray,
    tangential_half_width: float,
    normal_length: float,
    min_pts: int,
    poisson_radius: float | None = None,
    poisson_seed: int = 42,
):
    """
    Lokales Koordsystem am Endpunkt:
      x: Segmentrichtung (t_hat)
      y: Normalenrichtung (n_hat)
    Auswahl-Box: |x| <= w  UND  |y| <= L (beide Richtungen gleichzeitig)
    Mit optionalem Poisson-Disk-Subsampling in der lokalen Box.
    """
    D = XY - endpoint_xy[None, :]  # (N,2)
    x_local = D @ t_hat
    y_local = D @ n_hat

    # Sammle Punkte in beide Richtungen gleichzeitig (lokale Orientierungs-BBox)
    mask = (np.abs(x_local) <= tangential_half_width) & (
        np.abs(y_local) <= normal_length
    )
    idx = np.nonzero(mask)[0]
    if idx.size < min_pts:
        return endpoint_xy, 0

    # Ohne Subsampling: wie gehabt Mittelwert über alle Box-Punkte
    if poisson_radius is None or poisson_radius <= 0:
        return XY[idx].mean(axis=0), idx.size

    # --- Poisson-Disk-Subsampling im lokalen Rechteck [-w, w] x [-L, L] ---
    # 1) Generiere Poisson-Disk-Samples im lokalen Domain-Rechteck
    w = tangential_half_width
    L = normal_length
    eng = qmc.PoissonDisk(
        d=2,
        radius=float(poisson_radius),
        l_bounds=np.array([-w, -L]),
        u_bounds=np.array([+w, +L]),
        rng=int(poisson_seed),
    )
    # fill_space füllt bis keine Kandidaten mehr passen (für d=2 meist flott)
    S = eng.fill_space()  # (M,2) lokale Samplepunkte

    # Fallback: Wenn aus irgendeinem Grund keine Samples entstehen, nimm alle Punkte
    if S.size == 0:
        return XY[idx].mean(axis=0), idx.size

    # 2) Mappe Samples auf die nächstgelegenen Originalpunkte in der Box (lokal!)
    P_local = np.column_stack((x_local[idx], y_local[idx]))  # (K,2)
    kdt = KDTree(P_local)
    nn_idx = kdt.query(S, k=1)[1]  # (M,) Indizes innerhalb von 'idx'
    nn_idx = np.unique(nn_idx)  # eindeutige Auswahl
    if nn_idx.size < min_pts:
        # Wenn zu wenige Punkte übrig bleiben, fallback auf alle Box-Punkte
        return XY[idx].mean(axis=0), idx.size

    # 3) Schwerpunkt in Weltkoordinaten über der subsampleten Teilmenge
    idx_sub = idx[nn_idx]
    return XY[idx_sub].mean(axis=0), idx_sub.size


def adjust_segments_by_bbox_regression(
    xyz: np.ndarray,
    segments_2x3: list,
    *,
    normal_length: float = 1.0,  # Länge entlang Normalenrichtung (m)
    tangential_half_width: float = 0.25,  # halbe Breite entlang Segment (m)
    min_pts: int = 4,
    max_shift: float | None = None,  # Begrenzung des Shifts (z.B. = normal_length)
    samples_per_meter: float = 1.0,  # Anzahl Zwischenpunkte pro Meter Segmentlänge
    min_samples: int = 3,  # Mindestanzahl Samples (inkl. Endpunkte)
) -> list:
    """
    Parameters
    ----------
    xyz : np.ndarray (N,3)
    segments_2x3 : list of np.ndarray, jedes (2,3)
    samples_per_meter : float
        Anzahl der Sampling-Punkte pro Meter Segmentlänge
    min_samples : int
        Mindestanzahl von Sampling-Punkten entlang des Segments
    Returns
    -------
    list of np.ndarray, jedes (2,3)  # exakt wie merge_segments_in_clusters
    """
    if not isinstance(xyz, np.ndarray) or xyz.shape[1] < 2:
        raise ValueError("xyz muss [N,3] sein.")
    XY = xyz[:, :2].astype(float, copy=False)

    # print("Erstelle 2D KDTree...")
    # XY3D = np.c_[XY, np.zeros(len(XY))]

    # print("Erstelle Open3D KDTree mit Voxel-Downsampling...")
    # # Open3D Punktwolke
    # pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(XY3D))
    # print(f"Originalpunkte: {len(pcd.points)}")
    # # Voxel-Downsampling in 2D (Voxelgröße z.B. 0.02)
    # pcd_down = pcd.voxel_down_sample(voxel_size=0.02)
    # print(f"Downsampled Punkte: {len(pcd_down.points)}")
    # # Downsampled Koordinaten zurückholen
    # XY_down = np.asarray(pcd_down.points)[:, :2]
    # print(f"XY Downshape: {XY_down.shape}")
    # kdtree = KDTree(XY_down)

    kdtree = KDTree(XY)

    print("Justiere Segmente...")
    sample_data = []
    out_segments = []
    for index, seg in enumerate(segments_2x3):
        if index % 5 == 0:
            print(f"Segment {index}/{len(segments_2x3)}")

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
            out_segments.append(seg)  # degeneriert
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
            sample_point_3d = p1 + t * segment_vec
            sample_point_xy = sample_point_3d[:2]

            # Nur relevante Punkte per KDTree
            candidate_indices = kdtree.query_ball_point(sample_point_xy, search_radius)
            if len(candidate_indices) < min_pts:
                continue

            XY_candidates = XY[candidate_indices]

            # BBox-Centroid für diesen Sample-Punkt
            c_xy, n_pts = _bbox_centroid_for_endpoint(
                XY_candidates,
                sample_point_xy,
                t_hat,
                n_hat,
                tangential_half_width,
                normal_length,
                min_pts,
                # poisson_radius=0.02,
            )

            # Nur verwenden wenn genügend Punkte gefunden
            if n_pts > 0:
                # Optional: Shift begrenzen
                if max_shift is not None:
                    d = np.linalg.norm(c_xy - sample_point_xy)
                    if d > max_shift:
                        c_xy = sample_point_xy + (c_xy - sample_point_xy) * (
                            max_shift / d
                        )

                # 3D-Punkt: XY aus BBox, Z interpoliert
                c_3d = np.array([c_xy[0], c_xy[1], sample_point_3d[2]])
                collected_points.append(c_3d)
                collected_weights.append(n_pts)  # Gewichtung nach Anzahl Punkte

                sample_data.append(
                    {
                        "t_hat": t_hat.copy(),
                        "n_hat": n_hat.copy(),
                        "c_xy": c_xy.copy(),
                        "sample_point_xy": sample_point_xy.copy(),
                        "z": sample_point_3d[2],
                    }
                )

        # Fallback: Wenn keine gültigen Punkte gefunden, ursprüngliches Segment behalten
        if len(collected_points) < 2:
            out_segments.append(seg)
            continue

        collected_points = np.array(collected_points)
        collected_weights = np.array(collected_weights)

        # Gewichtete lineare Regression in 3D
        # Schwerpunkt der Punkte
        weights_sum = collected_weights.sum()
        centroid = np.average(collected_points, axis=0, weights=collected_weights)

        # Gewichtete Kovarianzmatrix
        X_centered = collected_points - centroid
        W = np.diag(collected_weights / weights_sum)
        cov_matrix = X_centered.T @ W @ X_centered

        # Hauptrichtung durch Eigenvector zum größten Eigenwert
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        main_direction = eigenvecs[:, -1]  # Eigenvektor zum größten Eigenwert

        # Projiziere alle Punkte auf die Hauptachse durch den Schwerpunkt
        proj_lengths = (
            X_centered @ main_direction
        )  # (n_points,) - Längen der Projektionen
        projections = (
            proj_lengths[:, np.newaxis] * main_direction[np.newaxis, :] + centroid
        )

        # Finde Endpunkte der projizierten Linie
        min_idx = np.argmin(proj_lengths)
        max_idx = np.argmax(proj_lengths)

        new_p1 = projections[min_idx]
        new_p2 = projections[max_idx]

        out_segments.append(np.vstack((new_p1, new_p2)))

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

    print(f"Sample-Vektoren exportiert nach: ./sample_vectors.obj")
