import os
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
import numpy as np


def cluster_segments(
    segments,
    eps_euclid,  # Epsilon im (cos2θ, sin2θ, ρ/scale)-Raum
    min_samples,
    rho_scale,  # skaliert ρ relativ zum Winkelteil
    preserve_noise=False,
):
    """
    Cluster segments using DBSCAN, with Hough features.

    Parameter:
    - segments: Die zu clusternden Segmentdaten.
    - eps_euclid: Epsilon für den DBSCAN-Algorithmus.
    - min_samples: Minimale Anzahl von Proben für einen Cluster.
    - rho_scale: Skalierungsfaktor für den rho-Wert.
    - preserve_noise: Ob Rauschsegmente beibehalten werden sollen.
    """
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

    # DBSCAN auf dem Feature-Vektor
    labels = DBSCAN(
        eps=eps_euclid, min_samples=min_samples, metric="euclidean"
    ).fit_predict(X)

    # Cluster erstellen
    clusters = {
        cid: np.where(labels == cid)[0] for cid in np.unique(labels) if cid != -1
    }

    # Rauschen als Einzelsegment-Cluster behandeln
    if preserve_noise:
        noise_indices = np.where(labels == -1)[0]
        next_cluster_id = max(clusters.keys()) + 1 if clusters else 0
        for noise_idx in noise_indices:
            clusters[next_cluster_id] = np.array([noise_idx])
            labels[noise_idx] = next_cluster_id
            next_cluster_id += 1

    return {
        "X": X,  # (cos2θ, sin2θ, ρ/scale) - basiert auf XY-Projektion
        "labels": labels,  # Cluster-Labels je Segment
        "clusters": clusters,  # dict: cid -> Indizes der Segmente
        "theta": theta,  # Normalwinkel je Segment (XY-Ebene)
        "rho": rho,  # ρ je Segment (zum XY-Ursprung)
        "origin": origin,  # benutzter Ursprung (2D oder 3D je nach Input)
    }


def _z_value_per_segment(segments, idxs):
    """
    Liefert je Segment (ausgewählt durch idxs) einen einzigen Z-Wert:
    => 'mean':  (z0+z1)/2
    Fällt auf 0 zurück, wenn nur 2D-Daten vorliegen (keine Z-Spalte).
    """
    segs = segments[idxs]
    D = segs.shape[2]
    if D == 2:
        return np.zeros((segs.shape[0],), dtype=float)
    z0 = segs[:, 0, 2]
    z1 = segs[:, 1, 2]
    return (0.5 * (z0 + z1)).astype(float)


def subcluster_with_segement_z(
    segments,
    clusters,
    gap,
):
    """
    Teilt vorhandene Cluster (dict[id -> indices]) **ausschließlich über Z**:
    - Je Eltern-Cluster: Z-Werte der Segmente holen (Mittelwert der Endpunkte)
    - Aufsteigend sortieren und bei Sprung > gap einen neuen Block starten
    - Ergebnis: neues dict[new_id -> np.ndarray[int]] mit Original-Indizes

    Parameters
    ----------
    segments : np.ndarray, shape (N, 2, 2|3)
    clusters : dict[int, np.ndarray]
        Eltern-Cluster -> Segmentindizes (global auf 'segments')
    gap : float
        Gap-Schwelle in absoluten Z-Einheiten (z.B. 0.003 für 0.3cm).

    Returns
    -------
    new_clusters : dict[int, np.ndarray]
        Verfeinerte Cluster für direkte Nutzung in
        merge_clusters_to_segments_2d_zmax(segments, new_clusters, ...)
    """
    segs = np.asarray(segments, dtype=float)
    if segs.ndim != 3 or segs.shape[1] != 2 or segs.shape[2] not in (2, 3):
        raise ValueError(f"Unerwartete segments-Form: {segs.shape}")

    new_clusters = {}
    next_cid = 0

    # Für deterministisches Verhalten Eltern-IDs sortiert durchlaufen
    for parent_cid in sorted(clusters.keys()):
        idxs = np.asarray(clusters[parent_cid], dtype=int)
        if idxs.size == 0:
            continue

        # Z-Werte als absolute Mittelwerte der Segmentendpunkte
        z = _z_value_per_segment(segs, idxs)

        # Nach Z-Werten sortieren
        order = np.argsort(z)
        idxs_sorted = idxs[order]
        z_sorted = z[order]

        # Split nach Gap (ohne Normierung - absolute Z-Werte)
        start = 0
        for i in range(1, len(z_sorted)):
            if (z_sorted[i] - z_sorted[i - 1]) > gap:
                # Neuen Block starten
                block = idxs_sorted[start:i]
                if block.size > 0:
                    new_clusters[next_cid] = block
                    next_cid += 1
                start = i

        # Letzten Block hinzufügen
        block = idxs_sorted[start:]
        if block.size > 0:
            new_clusters[next_cid] = block
            next_cid += 1

    return new_clusters


def merge_segments_in_slice(segments, clusters, gap_threshold, min_length):
    """
    Fügt pro Cluster kollineare Teilstücke zu größeren Geraden zusammen.
    Annahme: 2D-Segmente (x,y). 3D wird unterstützt, Z wird als Cluster-Mittel übernommen.

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

        # 1) Mittel-Punkt aller Punkte
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
                z_mean = float(pts[:, 2].mean())
                merged_segments.append(
                    (
                        (float(p0_xy[0]), float(p0_xy[1]), z_mean),
                        (float(p1_xy[0]), float(p1_xy[1]), z_mean),
                    )
                )

        for t in t_sorted[1:]:
            if (t - last) > gap_threshold:
                flush_segment(start, last)
                start = t
            last = t
        flush_segment(start, last)

    return merged_segments


def merge_segments_global(
    segments,
    clusters,
    gap_threshold=2.0,
    min_length=0.0,
    dynamic_gap=False,  # optional: Schwelle aus Daten schätzen
    require_support=0,  # mind. so viele Segmente sollen ein Intervall tragen
):
    if not segments:
        return []

    segs = np.asarray(segments, dtype=float)
    if segs.ndim != 3 or segs.shape[1] != 2 or segs.shape[2] not in (2, 3):
        raise ValueError(f"Unerwartete segments-Form: {segs.shape}")

    D = segs.shape[2]
    merged = []

    for cid, idx in clusters.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue

        seg_cluster = segs[idx]  # (K,2,D)
        pts = seg_cluster.reshape(-1, D)  # (2K,D)

        # 1) Schwerpunkt (nur XY für Richtung)
        mean_xy = pts[:, :2].mean(axis=0)

        # 2) Axiale Mittelrichtung aus XY
        v = seg_cluster[:, 1, :2] - seg_cluster[:, 0, :2]
        phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi)
        c, s = np.cos(2 * phi).mean(), np.sin(2 * phi).mean()
        mean_phi = 0.5 * np.arctan2(s, c)
        u = np.array([np.cos(mean_phi), np.sin(mean_phi)])  # Einheitsrichtung in XY

        # 3) Intervalle pro Segment: [t_lo, t_hi]
        a = seg_cluster[:, 0, :2] - mean_xy  # (K,2)
        b = seg_cluster[:, 1, :2] - mean_xy
        t0 = a @ u  # (K,)
        t1 = b @ u  # (K,)
        t_lo = np.minimum(t0, t1)
        t_hi = np.maximum(t0, t1)

        # Unterstützungszähler pro Intervall (wie oft von Segmenten getragen)
        intervals = np.stack([t_lo, t_hi], axis=1)  # (K,2)
        order = np.argsort(intervals[:, 0])
        intervals = intervals[order]

        # Optional: dynamische Lückenschwelle
        if dynamic_gap and len(intervals) >= 3:
            dt = np.diff(np.sort(np.concatenate([t_lo, t_hi])))
            med = np.median(dt) if dt.size else 0.0
            gap_thr = max(gap_threshold, 3.0 * med)
        else:
            gap_thr = gap_threshold

        # 4) Intervall-Vereinigung mit Lücken-Toleranz
        if len(intervals) == 0:
            continue

        cur_start, cur_end = intervals[0]
        support = 1  # Startintervall trägt sich selbst
        union_intervals = []  # [(start, end, support_count_approx)]

        def flush_interval(s, e, supp):
            length = float(e - s)
            if length <= 0 or (min_length > 0 and length < min_length):
                return
            # Z-Höhe: höchstes Z aller Segmente im Cluster (einfachste Variante)
            if D == 3:
                z_top = float(seg_cluster[:, :, 2].max())
                p0_xy = mean_xy + s * u
                p1_xy = mean_xy + e * u
                merged.append(
                    (
                        (float(p0_xy[0]), float(p0_xy[1]), z_top),
                        (float(p1_xy[0]), float(p1_xy[1]), z_top),
                    )
                )
            else:
                p0_xy = mean_xy + s * u
                p1_xy = mean_xy + e * u
                merged.append((tuple(p0_xy.tolist()), tuple(p1_xy.tolist())))

        for st, en in intervals[1:]:
            # überlappend oder durch kleine Lücke getrennt?
            if st - cur_end <= gap_thr:
                cur_end = max(cur_end, en)
                support += 1
            else:
                if support >= require_support:
                    flush_interval(cur_start, cur_end, support)
                cur_start, cur_end = st, en
                support = 1

        if support >= require_support:
            flush_interval(cur_start, cur_end, support)

    return merged
