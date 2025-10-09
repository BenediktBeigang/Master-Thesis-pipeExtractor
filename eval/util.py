import math
import numpy as np
import os
from custom_types import Point3D


def map_classes(y_true, mapping):
    """
    Mappt Klassenlabels basierend auf einer Mapping-Liste.

    Args:
        y_true: Array/Liste mit ursprünglichen Klassenlabels
        mapping: Liste wo Index = ursprüngliche Klasse, Wert = neue Klasse

    Returns:
        Gemapptes Array mit neuen Klassenlabels
    """
    y_mapped = np.array([mapping[label] for label in y_true])
    return y_mapped


def get_las_files(input_path: str):
    """Sammelt alle LAS/LAZ-Dateien aus dem angegebenen Pfad."""
    las_files = []

    if os.path.isfile(input_path):
        if input_path.lower().endswith((".las", ".laz")):
            las_files.append(input_path)
    elif os.path.isdir(input_path):
        for filename in os.listdir(input_path):
            if filename.lower().endswith((".las", ".laz")):
                las_files.append(os.path.join(input_path, filename))

    return las_files


def _sub(a: Point3D, b: Point3D):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v):
    return math.sqrt(_dot(v, v))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def point_to_segment_distance_xy(p: Point3D, a: Point3D, b: Point3D) -> float:
    """
    Abstand Punkt -> Segment in XY-Ebene (ignoriert Z-Koordinate).
    Wenn die Projektion außerhalb des Segments liegt, wird der Abstand zum nächsten Endpunkt zurückgegeben.
    """
    # Projiziere alle Punkte auf XY-Ebene
    p_xy = (p[0], p[1], 0.0)
    a_xy = (a[0], a[1], 0.0)
    b_xy = (b[0], b[1], 0.0)

    ab = _sub(b_xy, a_xy)
    ab_norm_sq = _dot(ab, ab)

    if ab_norm_sq == 0.0:
        return _norm(_sub(p_xy, a_xy))

    ap = _sub(p_xy, a_xy)

    # Punkt liegt innerhalb -> lotrechter Abstand zur Linie
    ab_norm = math.sqrt(ab_norm_sq)
    return _norm(_cross(ab, ap)) / ab_norm


def point_to_infinite_line_distance_z(p: Point3D, a: Point3D, b: Point3D) -> float:
    """
    Z-Distanz zwischen Punkt und seiner Projektion auf die Linie.
    """
    # Berechne die Projektion des Punktes auf die Linie
    ab = _sub(b, a)
    ab_norm_sq = _dot(ab, ab)
    if ab_norm_sq == 0.0:
        return abs(p[2] - a[2])

    ap = _sub(p, a)
    t = _dot(ap, ab) / ab_norm_sq
    # Projektion auf die Linie
    proj_point = (a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2])

    return abs(p[2] - proj_point[2])
