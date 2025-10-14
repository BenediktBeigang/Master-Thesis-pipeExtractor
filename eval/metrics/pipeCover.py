from custom_types import Point3D, Segment3D
from eval.util import _dot, _norm, _sub


def _project_point_on_segment_as_t(p: Point3D, a: Point3D, b: Point3D) -> float:
    """
    Projiziert einen Punkt P auf die Linie AB und gibt den Parameter t zurück.
    t = 0 bedeutet Punkt liegt bei A
    t = 1 bedeutet Punkt liegt bei B
    t kann auch außerhalb [0,1] liegen, wenn P außerhalb des Segments liegt.

    Returns:
        t (float): Parameter der Projektion auf der Linie
    """
    ab = _sub(b, a)
    ab_norm_sq = _dot(ab, ab)

    if ab_norm_sq == 0.0:
        return 0.0

    ap = _sub(p, a)
    t = _dot(ap, ab) / ab_norm_sq

    return max(0.0, min(1.0, t))


def calc_gt_cover(gt: Segment3D, det: Segment3D) -> tuple[float, float]:
    """
    Berechnet die Abdeckung eines Detection-Segments auf einem Ground Truth Segment.

    Args:
        gt: Ground Truth Segment (A, B)
        det: Detection Segment (P, Q)

    Returns:
        (t_start, t_end): Bereich der Abdeckung als t-Werte [0,1]
                         wobei t=0 am Anfang von gt und t=1 am Ende liegt
    """
    # Projiziere beide Endpunkte des Detection-Segments auf GT
    t_p = _project_point_on_segment_as_t(det[0], gt[0], gt[1])
    t_q = _project_point_on_segment_as_t(det[1], gt[0], gt[1])

    # Sortiere die t-Werte, damit t_start <= t_end
    t_start = min(t_p, t_q)
    t_end = max(t_p, t_q)

    return t_start, t_end


def calculate_coverage_absolute(gt_cover: dict) -> tuple[float, float]:
    """
    Berechnet für jedes GT-Segment die absolute Abdeckungslänge.

    Args:
        gt_cover: Dictionary mit GT-Segmenten als Keys und Listen von (t_start, t_end) Tupeln als Values

    Returns:
        Tuple mit zwei Listen:
        - covered_lengths: Liste der absoluten Abdeckungslängen für jedes GT-Segment
        - total_lengths: Liste der Gesamtlängen für jedes GT-Segment
    """
    covered_lengths = []
    total_lengths = []

    for gt, intervals in gt_cover.items():
        # Berechne Gesamtlänge des GT-Segments
        gt_length = _norm(_sub(gt[1], gt[0]))
        total_lengths.append(gt_length)

        if not intervals:
            covered_lengths.append(0.0)
            continue

        # Sortiere Intervalle nach t_start
        sorted_intervals = sorted(intervals, key=lambda x: x[0])

        # Merge überlappende oder angrenzende Intervalle
        merged = []
        current_start, current_end = sorted_intervals[0]

        for t_start, t_end in sorted_intervals[1:]:
            if t_start <= current_end:
                # Überlappung oder angrenzend -> merge
                current_end = max(current_end, t_end)
            else:
                # Kein Overlap -> speichere aktuelles Interval und starte neues
                merged.append((current_start, current_end))
                current_start, current_end = t_start, t_end

        # Füge letztes Interval hinzu
        merged.append((current_start, current_end))

        # Berechne totale Abdeckung in absoluten Einheiten (nur im Bereich [0, 1])
        total_coverage_t = 0.0
        for t_start, t_end in merged:
            # Clippe auf [0, 1]
            clipped_start = max(0.0, min(1.0, t_start))
            clipped_end = max(0.0, min(1.0, t_end))
            total_coverage_t += clipped_end - clipped_start

        # Konvertiere t-Wert in absolute Länge
        covered_length = total_coverage_t * gt_length
        covered_lengths.append(covered_length)

    return sum(covered_lengths), sum(total_lengths)
