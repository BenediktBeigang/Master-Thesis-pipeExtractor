import numpy as np
from sklearn.cluster import DBSCAN
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
