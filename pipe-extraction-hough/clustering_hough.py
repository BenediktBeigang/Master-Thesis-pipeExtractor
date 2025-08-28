import os
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import numpy as np


def features_and_clusters(
    segments,
    eps_euclid=0.35,  # Epsilon im (cos2θ, sin2θ, ρ/scale)-Raum
    min_samples=3,
    rho_scale=1.0,  # skaliert ρ relativ zum Winkelteil
):
    if not segments:
        return dict(X=None, labels=np.array([]), theta=None, rho=None, origin=None)

    segs = np.asarray(segments, dtype=float)  # (N, 2, 2) oder (N, 2, 3)

    # Bestimme Dimensionalität und extrahiere XY-Koordinaten für Clustering
    if segs.ndim == 3 and segs.shape[1] == 2:
        if segs.shape[2] == 2:  # 2D Input
            segs_xy = segs
        elif segs.shape[2] == 3:  # 3D Input - nur XY für Clustering verwenden
            segs_xy = segs[:, :, :2]  # (N, 2, 2) - nur X,Y
        else:
            raise ValueError(f"Unerwartete segments-Form: {segs.shape}")
    else:
        raise ValueError(f"Unerwartete segments-Form: {segs.shape}")

    # Clustering basiert nur auf XY-Koordinaten
    v = segs_xy[:, 1, :] - segs_xy[:, 0, :]  # Richtungsvektoren in XY
    # Segment-Richtungswinkel φ in [0, π): Richtung egal
    phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi)

    # *** WICHTIG: Hough-Normalwinkel θ = φ + 90° (mod π) ***
    theta = np.mod(phi + np.pi / 2.0, np.pi)  # Normalrichtung der Linie

    # globaler Schwerpunkt aller XY-Endpunkte -> als Ursprung verwenden
    all_pts_xy = segs_xy.reshape(-1, 2)
    origin_xy = all_pts_xy.mean(axis=0)

    # Für 3D: auch Z-Koordinate mitteln
    if segs.shape[2] == 3:
        all_pts_xyz = segs.reshape(-1, 3)
        origin_z = all_pts_xyz[:, 2].mean()
        origin = np.array([origin_xy[0], origin_xy[1], origin_z])
    else:
        origin = origin_xy

    segs_xy_centered = segs_xy - origin_xy

    # Ein Punkt auf der Linie reicht für ρ (hier: Segmentmittelpunkt in XY)
    mids = (segs_xy_centered[:, 0, :] + segs_xy_centered[:, 1, :]) * 0.5

    # Hough-Normalform: ρ = x*cosθ + y*sinθ  (θ ist *Normal*-Winkel!)
    rho = mids[:, 0] * np.cos(theta) + mids[:, 1] * np.sin(theta)

    # Winkel axial zyklisch einbetten => euklidisch messbar
    X = np.column_stack(
        [np.cos(2 * theta), np.sin(2 * theta), rho / max(rho_scale, 1e-12)]
    )

    # DBSCAN auf dem Feature-Vektor (basiert nur auf XY-Clustering)
    labels = DBSCAN(
        eps=eps_euclid, min_samples=min_samples, metric="euclidean"
    ).fit_predict(X)

    # Praktisch: Indexliste pro Cluster (ohne Rauschen = -1)
    clusters = {
        cid: np.where(labels == cid)[0] for cid in np.unique(labels) if cid != -1
    }

    return {
        "X": X,  # (cos2θ, sin2θ, ρ/scale) - basiert auf XY-Projektion
        "labels": labels,  # Cluster-Labels je Segment
        "clusters": clusters,  # dict: cid -> Indizes der Segmente
        "theta": theta,  # Normalwinkel je Segment (XY-Ebene)
        "rho": rho,  # ρ je Segment (zum XY-Ursprung)
        "origin": origin,  # benutzter Ursprung (2D oder 3D je nach Input)
    }


# def merge_clusters_to_segments(segments, clusters, gap_threshold=0.25, min_length=0.0):
#     """
#     Fügt pro Cluster kollineare Teilstücke zu größeren Geraden zusammen.
#     Annahme: 2D-Segmente (x,y). 3D wird unterstützt, Z wird als Cluster-Mittel übernommen.

#     Parameters
#     ----------
#     segments : list[ ((x1,y1),(x2,y2)) ] oder list[ ((x1,y1,z1),(x2,y2,z2)) ]
#     clusters : dict[int, np.ndarray]
#         Mapping: cluster_id -> Indizes in `segments`
#     gap_threshold : float
#         Maximale Lücke entlang der Projektionsgeraden, um noch als durchgehend zu gelten.
#     min_length : float
#         Mindestlänge eines erzeugten Segments. <= 0 deaktiviert die Prüfung.

#     Returns
#     -------
#     merged_segments : list
#         Liste zusammengesetzter Segmente als ((x1,y1),(x2,y2)) oder ((x1,y1,z1),(x2,y2,z2)).
#         Dimensionalität entspricht der der Eingabesegmente.
#     """
#     if not segments:
#         return []

#     segs = np.asarray(segments, dtype=float)  # (N,2,2) oder (N,2,3)
#     if segs.ndim != 3 or segs.shape[1] != 2 or segs.shape[2] not in (2, 3):
#         raise ValueError(
#             f"Unerwartete segments-Form: {segs.shape}. Erwartet (N,2,2) oder (N,2,3)."
#         )

#     D = segs.shape[2]  # 2 oder 3
#     merged_segments = []

#     for cid, idx in clusters.items():
#         idx = np.asarray(idx, dtype=int)
#         if idx.size == 0:
#             continue

#         seg_cluster = segs[idx]  # (K,2,D)
#         pts = seg_cluster.reshape(-1, D)  # (2K,D)

#         # 1) Mittel-Punkt aller Punkte
#         mean_pt = pts.mean(axis=0)  # (D,)

#         # 2) Mittelrichtung (axial) aus Segment-Richtungen (XY-Ebene)
#         v = seg_cluster[:, 1, :2] - seg_cluster[:, 0, :2]  # (K,2)
#         phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi)  # [0,π)
#         c = np.cos(2.0 * phi).mean()
#         s = np.sin(2.0 * phi).mean()
#         mean_phi = 0.5 * np.arctan2(s, c)  # axialer zirkularer Mittelwinkel

#         # 3) Projektionsgerade: mean_pt + t * u  (nur XY-Richtung)
#         u = np.array([np.cos(mean_phi), np.sin(mean_phi)])  # Einheitsrichtung in XY

#         # 4) Projektion aller Punkte auf die Gerade: t = (p_xy - mean_xy)·u
#         t_vals = (pts[:, :2] - mean_pt[:2]) @ u  # (2K,)

#         # 5) Entlang der Geraden sortieren
#         if t_vals.size == 0:
#             continue
#         order = np.argsort(t_vals)
#         t_sorted = t_vals[order]

#         # 6) Lücken schneiden -> Teilsegmente
#         start = t_sorted[0]
#         last = t_sorted[0]

#         def flush_segment(t0, t1):
#             length = float(t1 - t0)
#             if length <= 0:
#                 return
#             if min_length > 0 and length < min_length:
#                 return
#             p0_xy = mean_pt[:2] + t0 * u
#             p1_xy = mean_pt[:2] + t1 * u
#             if D == 2:
#                 merged_segments.append((tuple(p0_xy.tolist()), tuple(p1_xy.tolist())))
#             else:
#                 z_mean = float(pts[:, 2].mean())
#                 merged_segments.append(
#                     (
#                         (float(p0_xy[0]), float(p0_xy[1]), z_mean),
#                         (float(p1_xy[0]), float(p1_xy[1]), z_mean),
#                     )
#                 )

#         for t in t_sorted[1:]:
#             if (t - last) > gap_threshold:
#                 flush_segment(start, last)
#                 start = t
#             last = t
#         flush_segment(start, last)

#     return merged_segments


def merge_clusters_to_segments(
    segments, clusters, gap_threshold=0.25, min_length=0.0, useMaxZ=False
):
    """
    Fügt pro Cluster kollineare Teilstücke zu größeren Geraden zusammen.
    Annahme: 2D-Segmente (x,y). 3D wird unterstützt, Z wird als Cluster-Maximum übernommen.

    Parameters
    ----------
    segments : list[ ((x1,y1),(x2,y2)) ] oder list[ ((x1,y1,z1),(x2,y2,z2)) ]
    clusters : dict[int, np.ndarray]
        Mapping: cluster_id -> Indizes in `segments`
    gap_threshold : float
        Maximale Lücke entlang der Projektionsgeraden, um noch als durchgehend zu gelten.
    min_length : float
        Mindestlänge eines erzeugten Segments. <= 0 deaktiviert die Prüfung.

    Returns
    -------
    merged_segments : list
        Liste zusammengesetzter Segmente als ((x1,y1),(x2,y2)) oder ((x1,y1,z1),(x2,y2,z2)).
        Dimensionalität entspricht der der Eingabesegmente.
    """
    if not segments:
        return []

    segs = np.asarray(segments, dtype=float)  # (N,2,2) oder (N,2,3)
    if segs.ndim != 3 or segs.shape[1] != 2 or segs.shape[2] not in (2, 3):
        raise ValueError(
            f"Unerwartete segments-Form: {segs.shape}. Erwartet (N,2,2) oder (N,2,3)."
        )

    D = segs.shape[2]  # 2 oder 3
    merged_segments = []

    for cid, idx in clusters.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue

        seg_cluster = segs[idx]  # (K,2,D)
        pts = seg_cluster.reshape(-1, D)  # (2K,D)

        # 1) Mittel-Punkt aller Punkte (nur XY)
        mean_pt = pts.mean(axis=0)  # (D,)

        # 2) Mittelrichtung (axial) aus Segment-Richtungen (XY-Ebene)
        v = seg_cluster[:, 1, :2] - seg_cluster[:, 0, :2]  # (K,2)
        phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi)  # [0,π)
        c = np.cos(2.0 * phi).mean()
        s = np.sin(2.0 * phi).mean()
        mean_phi = 0.5 * np.arctan2(s, c)  # axialer zirkularer Mittelwinkel

        # 3) Projektionsgerade: mean_pt + t * u  (nur XY-Richtung)
        u = np.array([np.cos(mean_phi), np.sin(mean_phi)])  # Einheitsrichtung in XY

        # 4) Projektion aller Punkte auf die Gerade: t = (p_xy - mean_xy)·u
        t_vals = (pts[:, :2] - mean_pt[:2]) @ u  # (2K,)

        # 5) Entlang der Geraden sortieren
        if t_vals.size == 0:
            continue
        order = np.argsort(t_vals)
        t_sorted = t_vals[order]

        # 6) Lücken schneiden -> Teilsegmente
        start = t_sorted[0]
        last = t_sorted[0]

        def flush_segment(t0, t1):
            length = float(t1 - t0)
            if length <= 0:
                return
            if min_length > 0 and length < min_length:
                return
            p0_xy = mean_pt[:2] + t0 * u
            p1_xy = mean_pt[:2] + t1 * u
            if D == 2:
                merged_segments.append((tuple(p0_xy.tolist()), tuple(p1_xy.tolist())))
            else:
                z_max = float(pts[:, 2].max()) if useMaxZ else float(pts[:, 2].mean())
                merged_segments.append(
                    (
                        (float(p0_xy[0]), float(p0_xy[1]), z_max),
                        (float(p1_xy[0]), float(p1_xy[1]), z_max),
                    )
                )

        for t in t_sorted[1:]:
            if (t - last) > gap_threshold:
                flush_segment(start, last)
                start = t
            last = t
        flush_segment(start, last)

    return merged_segments
