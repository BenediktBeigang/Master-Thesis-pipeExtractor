#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing: Endpunkte von Segmenten (2x3) per orientierter BBox snappen.
Eingabe/Output sind IDENTISCH zum merge_segments_in_clusters-Format:
    List[np.ndarray] mit je shape (2,3): [[x1,y1,z1],[x2,y2,z2]]
"""

from __future__ import annotations
import numpy as np
from scipy.stats import qmc
from scipy.spatial import KDTree


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


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


def adjust_segments_by_bbox(
    xyz: np.ndarray,
    segments_2x3: list,
    *,
    normal_length: float = 0.8,  # Länge entlang Normalenrichtung (m)
    tangential_half_width: float = 0.25,  # halbe Breite entlang Segment (m)
    min_pts: int = 8,
    max_shift: float | None = None,  # Begrenzung des Shifts (z.B. = normal_length)
) -> list:
    """
    Parameters
    ----------
    xyz : np.ndarray (N,3)
    segments_2x3 : list of np.ndarray, jedes (2,3)
    Returns
    -------
    list of np.ndarray, jedes (2,3)  # exakt wie merge_segments_in_clusters
    """
    if not isinstance(xyz, np.ndarray) or xyz.shape[1] < 2:
        raise ValueError("xyz muss [N,3] sein.")
    XY = xyz[:, :2].astype(float, copy=False)

    out_segments = []
    for seg in segments_2x3:
        seg = np.asarray(seg, dtype=float)
        if seg.shape != (2, 3):
            raise ValueError(f"Segment hat Form {seg.shape}, erwartet (2,3).")

        p1 = seg[0].copy()
        p2 = seg[1].copy()
        p1_xy, p2_xy = p1[:2], p2[:2]

        # Tangente/Normale aus XY
        t_hat = _unit(p2_xy - p1_xy)
        if np.allclose(t_hat, 0):
            out_segments.append(seg)  # degeneriert
            continue
        n_hat = np.array([-t_hat[1], t_hat[0]])

        # Ende 1
        c1_xy, n1 = _bbox_centroid_for_endpoint(
            XY,
            p1_xy,
            t_hat,
            n_hat,
            tangential_half_width,
            normal_length,
            min_pts,
            poisson_radius=0.02,
        )
        # Ende 2
        c2_xy, n2 = _bbox_centroid_for_endpoint(
            XY,
            p2_xy,
            t_hat,
            n_hat,
            tangential_half_width,
            normal_length,
            min_pts,
            poisson_radius=0.02,
        )

        # Optional Shift begrenzen
        if max_shift is not None and n1 > 0:
            d = np.linalg.norm(c1_xy - p1_xy)
            if d > max_shift:
                c1_xy = p1_xy + (c1_xy - p1_xy) * (max_shift / d)
        if max_shift is not None and n2 > 0:
            d = np.linalg.norm(c2_xy - p2_xy)
            if d > max_shift:
                c2_xy = p2_xy + (c2_xy - p2_xy) * (max_shift / d)

        # Nur XY snappen, Z des jeweiligen Endes unverändert lassen
        if n1 > 0:
            p1[:2] = c1_xy
        if n2 > 0:
            p2[:2] = c2_xy

        out_segments.append(np.vstack((p1, p2)))

    return out_segments


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

        for t in t_values:
            # Interpolierter Punkt auf dem Segment
            sample_point_3d = p1 + t * segment_vec
            sample_point_xy = sample_point_3d[:2]

            # BBox-Centroid für diesen Sample-Punkt
            c_xy, n_pts = _bbox_centroid_for_endpoint(
                XY,
                sample_point_xy,
                t_hat,
                n_hat,
                tangential_half_width,
                normal_length,
                min_pts,
                poisson_radius=0.02,
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

    return out_segments
