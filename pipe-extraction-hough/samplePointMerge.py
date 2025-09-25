import numpy as np
from scipy.spatial import KDTree
from custom_types import ListOfPoint3DArrays, Segment3DArray


def _bucket_by_delta_z(
    points_3D: np.ndarray, deltaZ_threshold: float = 0.3, outlier_threshold: float = 0.2
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
    if points_3D is None or len(points_3D) < 2:
        return []

    z = points_3D[:, 2]
    n = len(points_3D)

    buckets = []
    current = [points_3D[0]]
    # Statistik über akzeptierte Deltas
    count = 0
    mean_dz = 0.0

    # Index des zuletzt AKZEPTIERTEN Punktes (nicht nur letzter iterierter)
    last_acc = 0

    def accept(idx, dz=None):
        nonlocal count, mean_dz, last_acc
        current.append(points_3D[idx])
        last_acc = idx
        if dz is not None:
            # Online-Mean (Welford light)
            count += 1
            mean_dz += (dz - mean_dz) / count

    # Starte mit erstem akzeptierten Übergang
    # (der zweite Punkt wird gleich verarbeitet)
    for i in range(1, n):
        dz = z[last_acc] - z[i]

        if count < 2:
            # Warm-up: die ersten 1–2 Deltas immer akzeptieren
            accept(i, dz)
            continue

        # Inlier?
        if abs(dz - mean_dz) <= deltaZ_threshold:
            accept(i, dz)
            continue

        # Kandidat Outlier? Mit Look-Ahead relativ zu last_acc prüfen:
        if i + 1 < n:
            dz_next = z[last_acc] - z[i + 1]
            if abs(dz + dz_next) <= outlier_threshold:
                # i ist Outlier -> ignoriere ihn komplett
                # (kein Update von mean/count/last_acc)
                continue

        # Kein Outlier -> echter Richtungs-/Steigungswechsel: Bucket schließen
        if len(current) >= 2:
            buckets.append(np.array(current))
        else:
            # Sicherheitsnetz, sollte praktisch nicht auftreten
            buckets.append(np.array([points_3D[last_acc], points_3D[i]]))

        # Neuen Bucket starten: Naht erhalten, daher mit last_acc und i
        current = [points_3D[last_acc], points_3D[i]]
        # Statistik neu initialisieren ab diesem Sprung
        mean_dz = dz
        count = 1
        last_acc = i

    if len(current) >= 2:
        buckets.append(np.array(current))

    # Mindestens 2 Punkte pro Bucket
    buckets = [b for b in buckets if len(b) >= 2]
    return buckets


def extract_segments(
    points_3D: np.ndarray,
    dz_threshold: float = 0.3,
    angle_tol_deg: float = 10.0,
    approx_tol: float = 1.0,
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
        return np.empty((0, 2, 3), dtype=np.float64)  # Empty Segment-Array

    # KD-Tree zum Rückholen der Z-Werte (nächstgelegener Originalpunkt im XY)
    XY_all = points_3D[:, :2].astype(float, copy=False)
    tree = KDTree(XY_all)

    segments_out: Segment3DArray = np.empty((0, 2, 3), dtype=np.float64)

    # 1) Buckets nach delta-z
    buckets = _bucket_by_delta_z(points_3D, deltaZ_threshold=dz_threshold)

    print(f"SamplePoints: {len(points_3D)}, Z-Buckets: {len(buckets)}")

    for B in buckets:
        # nutze alle punkte hintereinander als liniensegemente und füge sie segments_out hinzu

        a = B[0]
        b = B[-1]
        segment_2x3 = np.vstack([a, b]).astype(float).reshape(1, 2, 3)
        segments_out = np.vstack([segments_out, segment_2x3])

        continue
    return segments_out
