import math
from typing import List
from custom_types import Segment3D
from eval.metrics.pipeCover import calc_gt_cover, calculate_coverage_absolute
from eval.metrics.pipeTruePositive import (
    is_segment_in_tolerance,
)
from eval.util import (
    _norm,
    point_to_infinite_line_distance_z,
    point_to_segment_distance_xy,
    _sub,
)
from eval.export import write_segments_to_obj


def _best_pair_maxdist_xy(gt: Segment3D, det: Segment3D) -> float:
    """
    Beste Endpunkt-Paarung basierend nur auf XY-Distanz.
    """
    A, B = gt
    P, Q = det

    # XY-Distanzen berechnen
    dAP_xy = math.sqrt((A[0] - P[0]) ** 2 + (A[1] - P[1]) ** 2)
    dBQ_xy = math.sqrt((B[0] - Q[0]) ** 2 + (B[1] - Q[1]) ** 2)
    dAQ_xy = math.sqrt((A[0] - Q[0]) ** 2 + (A[1] - Q[1]) ** 2)
    dBP_xy = math.sqrt((B[0] - P[0]) ** 2 + (B[1] - P[1]) ** 2)

    return min(max(dAP_xy, dBQ_xy), max(dAQ_xy, dBP_xy))


def _segment_key(
    segment: Segment3D,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    def _to_tuple(point):
        if hasattr(point, "tolist"):
            return tuple(point.tolist())
        return tuple(point)

    return _to_tuple(segment[0]), _to_tuple(segment[1])


def pipe_detection_metric(
    ground_truth_segments: List[Segment3D],
    detected_segments: List[Segment3D],
    pointcloudName: str,
    tolerance: float = 0.5,
) -> tuple[int, int, int, int, List[float], List[float], float, float, float]:
    """
    Berechnet Metriken zur Bewertung der Erkennung von Liniensegmenten.

    Es gibt drei Hauptwerte pro Ground Truth Segment:
    1. Vollständig erkannt
        - Die Segment Punkte am Ende des Segments sind innerhalb eines Toleranzbereichs
    2. Teilweise erkannt
        - Die Distanz der Punkte sind von der Ground Truth Linie kleiner als ein Toleranzwert
    3. Nicht erkannt
        - Weder vollständig noch teilweise erkannt

    Als Nebenmetrik wird die durchschnittliche Distanz der erkannten Segmente zu den Ground Truth Segmenten berechnet.
    Dabei werden nicht erkannte Segmente ignoriert.

    Args:
        ground_truth_segments (list): Liste von Ground Truth Segmenten, wobei jedes Segment ein Tupel aus zwei Punkten ist.
        detected_segments (list): Liste von erkannten Segmenten, wobei jedes Segment ein Tupel aus zwei Punkten ist.

    Returns:
        correct_detections (int): Anzahl der vollständig erkannten Segmente
        partial_detections (int): Anzahl der teilweise erkannten Segmente
        missed_detections (int): Anzahl der nicht erkannten Segmente
        avg_distance (float): Durchschnittliche Distanz der erkannten Segmente zu den Ground Truth Segmenten
    """

    correct = 0
    partial = 0
    missed = 0
    line_dist_xy_samples: List[float] = []
    line_dist_z_samples: List[float] = []
    true_positives = set()
    gt_cover = {_segment_key(gt): [] for gt in ground_truth_segments}

    missed_list = []

    # --- Hauptlogik ---
    for gt in ground_truth_segments:
        label = "not_found"
        gt_key = _segment_key(gt)

        for det in detected_segments:
            det_key = _segment_key(det)
            # Pipe partially correct?
            if is_segment_in_tolerance(det, gt, tolerance):
                true_positives.add(det_key)

                t_start, t_end = calc_gt_cover(gt, det)
                gt_cover[gt_key].append((t_start, t_end))

                # Upgrade zu partial, falls noch nicht correct
                label = "partial" if label == "not_found" else label

                # XY-Distanzen sammeln
                line_dist_xy_samples.append(
                    point_to_segment_distance_xy(det[0], gt[0], gt[1])
                )
                line_dist_xy_samples.append(
                    point_to_segment_distance_xy(det[1], gt[0], gt[1])
                )

                # Z-Distanzen als separate Metrik sammeln
                line_dist_z_samples.append(
                    point_to_infinite_line_distance_z(det[0], gt[0], gt[1])
                )
                line_dist_z_samples.append(
                    point_to_infinite_line_distance_z(det[1], gt[0], gt[1])
                )

                # Pipe correct?
                max_pair = _best_pair_maxdist_xy(gt, det)
                if max_pair <= tolerance:
                    label = "correct"

        # b) nach innerer Schleife: Zählen
        match label:
            case "correct":
                correct += 1
            case "partial":
                partial += 1
            case _:
                missed += 1
                missed_list.append(gt)

    false_positives = len(detected_segments) - len(true_positives)
    coverage, total_length = calculate_coverage_absolute(gt_cover)

    false_positive_list = [
        det for det in detected_segments if _segment_key(det) not in true_positives
    ]

    false_positive_length = sum(
        _norm(_sub(seg[1], seg[0])) for seg in false_positive_list
    )

    # write_segments_to_obj(missed_list, false_positive_list, pointcloudName)

    return (
        correct,
        partial,
        missed,
        false_positives,
        line_dist_xy_samples,
        line_dist_z_samples,
        coverage,
        total_length,
        false_positive_length,
    )
