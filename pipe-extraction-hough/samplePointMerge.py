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
    points_3D_chain: np.ndarray,
    deltaZ_threshold: float,
    outlier_z_threshold: float,
) -> ListOfPoint3DArrays:
    """
    Teilt eine geordnete 3D-Punktkette in Buckets, basierend auf delta-z.
    WICHTIG: Deltas werden stets relativ zum *zuletzt akzeptierten* Punkt
    berechnet. Outlier werden übersprungen und beeinflussen den gleitenden
    Mittelwert nicht.

    Kontext:
    Die Funktion dient dem Trennen von Rohren mit unterschiedlichen Höhen bzw. Neigungen.
    Es berücksichtigt einzelne Ausreißer, wenn z.B. ein angedocktes T-Stück nach oben wegführt.

    Parameters
    ----------
    points_3D : (N,3) array
    deltaZ_threshold : float
        Toleranz für Abweichung vom gleitenden Mittel (|dz - mean_dz|).
    outlier_threshold : float
        Look-Ahead-Regel: wenn dz_i + dz_{i+1} ~ 0 (innerhalb eps),
        gilt Punkt i als Outlier und wird ignoriert.

    Returns
    -------
    list[np.ndarray]
        Liste von Buckets (je ein (M_i,3)-Array, wobei M_i die Anzahl Punkte im i-ten Bucket ist).
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

    # Mindestens 2 Punkte pro Bucket
    buckets = [b for b in buckets if len(b) >= 2]
    return buckets


def fit_ransac_line_and_project_endpoints(
    points_3D: Point3DArray,
    residual_threshold: float = 0.2,
    min_samples: float = 2,
    max_trials: int = 1000,
) -> Segment3D:
    """
    Nutzt die Kette an Punkten um in der xy-Ebene eine Linie zu fitten.
    Das Ergebnis sind die Endpunkte projeziert auf die durch RANSAC gefittete Linie.
    Als Z-Werte werden die Original-Z-Werte der Endpunkte verwendet.

    Parameters:
    points_2d: (N,2) numpy array, geordnete Kette (aber Reihenfolge wird hier nur für Endpunkte verwendet)
    residual_threshold: maximaler orthogonaler Abstand für einen Inlier (in units of points)
    min_samples: wieviele Punkte pro RANSAC-Sample
    max_trials: maximale Iterationen
    return_inliers: wenn True, gibt zusätzlich die boolean mask der inliers zurück
    ---
    Rückgabe: (line_origin, line_dir, projected_start, projected_end, inlier_mask?)
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
    points_3D: np.ndarray,
    dz_threshold: float = 0.3,
    outlier_z_threshold: float = 0.1,
    ransac_residual_threshold: float = 0.05,
) -> Segment3DArray:
    """
    points_3D : (N,3) geordnete Punktkette (x,y,z).
    dz_threshold : Schwelle für Bucket-Schnitt nach delta-z-Abweichung.
    angle_tol_deg: Winkel-Toleranz für „Richtungs-Cluster“ in einem Bucket.
    approx_tol : Toleranz für skimage.measure.approximate_polygon (auf XY).
    max_nn_xy_dist : optionaler Grenzwert; wenn der nächste Originalpunkt zu weit
                     weg ist, wird trotzdem gemappt – setze das nur, wenn du hart
                     validieren willst.

    Rückgabe:
      Liste von Segmenten, je Segment als np.ndarray shape (2,3).
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

    segments_out: Segment3DArray = Segment3DArray_Empty()
    for B in buckets:
        a, b = fit_ransac_line_and_project_endpoints(
            B, residual_threshold=ransac_residual_threshold
        )
        segment_2x3 = np.vstack([a, b]).astype(float).reshape(1, 2, 3)
        segments_out = np.vstack([segments_out, segment_2x3])
        continue
    return segments_out
