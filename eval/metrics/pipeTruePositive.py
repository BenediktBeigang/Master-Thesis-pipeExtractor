import numpy as np
from custom_types import Point3D, Segment3D
from eval.util import _dot, _norm, point_to_segment_distance_xy, _sub


def calc_overlap_on_segment(seg_base, seg_proj):
    """Berechne Überlappung von seg_proj auf seg_base"""
    a_base, b_base = seg_base
    a_proj, b_proj = seg_proj

    ab_base = _sub(b_base, a_base)
    ab_base_norm_sq = _dot(ab_base, ab_base)

    # t-Parameter für beide Endpunkte von seg_proj auf seg_base
    aa_proj = _sub(a_proj, a_base)
    t_a_proj = _dot(aa_proj, ab_base) / ab_base_norm_sq

    ab_proj = _sub(b_proj, a_base)
    t_b_proj = _dot(ab_proj, ab_base) / ab_base_norm_sq

    # Überlappungsbereich auf seg_base
    t_start = max(0.0, min(t_a_proj, t_b_proj))
    t_end = min(1.0, max(t_a_proj, t_b_proj))

    # Überlappungslänge auf seg_base
    return max(0.0, t_end - t_start) * _norm(_sub(b_base, a_base))


def is_segment_in_tolerance(seg1: Segment3D, seg2: Segment3D, tolerance: float) -> bool:
    """
    Abstand zwischen zwei Segmenten in XY-Ebene.
    Gibt None zurück, wenn die Überlappung weniger als 50% beträgt.
    50% bedeutet, dass das kürzere Segment zu 50% vom längeren verdeckt wird.

    Args:
        seg1: Erstes Segment (a1, b1)
        seg2: Zweites Segment (a2, b2)

    Returns:
        float: Durchschnittlicher Abstand der Endpunkte, oder None bei <50% Überlappung
    """
    a1, b1 = seg1
    a2, b2 = seg2

    # Projiziere alle Punkte auf XY-Ebene
    a1_xy = (a1[0], a1[1], 0.0)
    b1_xy = (b1[0], b1[1], 0.0)
    a2_xy = (a2[0], a2[1], 0.0)
    b2_xy = (b2[0], b2[1], 0.0)

    # Berechne Segmentlängen in XY
    len1 = _norm(_sub(b1_xy, a1_xy))
    len2 = _norm(_sub(b2_xy, a2_xy))

    if len1 == 0.0 or len2 == 0.0:
        return False

    # Berechne Überlappung von seg2 auf seg1 und umgekehrt
    overlap_on_seg1 = calc_overlap_on_segment((a1_xy, b1_xy), (a2_xy, b2_xy))
    overlap_on_seg2 = calc_overlap_on_segment((a2_xy, b2_xy), (a1_xy, b1_xy))

    # Bestimme das kürzere Segment und dessen Überlappungsgrad
    if len1 <= len2:
        # seg1 ist kürzer oder gleich -> prüfe Überlappung von seg1
        overlap_ratio = overlap_on_seg1 / len1
    else:
        # seg2 ist kürzer -> prüfe Überlappung von seg2
        overlap_ratio = overlap_on_seg2 / len2

    # Prüfe 50% Überlappung
    if overlap_ratio < 0.5:
        return False

    # Berechne durchschnittlichen Abstand der Endpunkte
    dist_a2_to_seg1 = point_to_segment_distance_xy(a2, a1, b1)
    dist_b2_to_seg1 = point_to_segment_distance_xy(b2, a1, b1)

    return dist_a2_to_seg1 <= tolerance and dist_b2_to_seg1 <= tolerance


def is_component_in_tolerance(
    comp1: Point3D, comp2: Point3D, tolerance: float
) -> tuple[bool, float | None, float | None]:
    """
    Prüft, ob zwei Komponenten (Punkte) innerhalb der XY-Toleranz liegen.

    Args:
        comp1: Erstes Komponenten-Punkt (x, y, z)
        comp2: Zweites Komponenten-Punkt (x, y, z)
        tolerance: Maximal zulässiger Abstand in der XY-Ebene

    Returns:
        Tuple[bool, float | None, float | None]:
            - bool: True, wenn XY-Abstand <= Toleranz
            - float | None: XY-Abstand (None, falls außerhalb der Toleranz)
            - float | None: Absoluter Z-Abstand (None, falls außerhalb der Toleranz)
    """
    comp1 = np.asarray(comp1, dtype=float)
    comp2 = np.asarray(comp2, dtype=float)
    xy_vec = (
        comp2[0] - comp1[0],
        comp2[1] - comp1[1],
        0.0,
    )
    xy_distance = _norm(xy_vec)
    if xy_distance > tolerance:
        return False, None, None

    z_distance = abs(comp2[2] - comp1[2])
    return True, xy_distance, z_distance
