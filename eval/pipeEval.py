import json
import numpy as np
from eval.pipeMetrics import pipe_detection_metric
from eval.plots.distancePlot import plot_boxplots_lineDistances
from eval.plots.pipeClassPlot import plot_segmentClasses


def pipeEval(ground_truth, detected_pipes, pointcloudName):
    # export_segments_to_obj(
    #     ground_truth_segments,
    #     "./ontras_3_ground_truth.obj",
    # )

    (
        correct,
        partial,
        missed,
        false_positives,
        line_dist_xy_samples,
        line_dist_z_samples,
        coverage,
        total_length,
        false_positive_length,
    ) = pipe_detection_metric(
        ground_truth, detected_pipes, pointcloudName, tolerance=0.5
    )

    result = {
        "pipe_count_ground_truth": len(ground_truth),
        "pipe_count_detected": len(detected_pipes),
        "correct": correct,
        "partial": partial,
        "missed": missed,
        "false_positives": false_positives,
        "coverage": f"{coverage:.2f} / {total_length:.2f} m  |  {(coverage/total_length*100):.2f} %",
        "coverage_iou": f"{coverage:.2f} / {total_length + false_positive_length:.2f} m  |  {(coverage / (total_length + false_positive_length) * 100):.2f} %",
        "distance_3D_xy_avg": (
            0
            if len(line_dist_xy_samples) == 0
            else sum(line_dist_xy_samples) / len(line_dist_xy_samples)
        ),
        "distance_3D_xy_median": np.median(line_dist_xy_samples),
        "distance_3D_z_avg": (
            0
            if len(line_dist_z_samples) == 0
            else sum(line_dist_z_samples) / len(line_dist_z_samples)
        ),
        "distance_3D_z_median": np.median(line_dist_z_samples),
        "distance_3D_xy_samples": line_dist_xy_samples,
        "distance_3D_z_samples": line_dist_z_samples,
    }

    with open(f"./output/metrics/{pointcloudName}_pipes.json", "w") as f:
        json.dump(result, f, indent=2)

    if len(line_dist_xy_samples) > 0 or len(line_dist_z_samples) > 0:
        plot_boxplots_lineDistances(
            line_dist_xy_samples,
            line_dist_z_samples,
            out_png=f"./output/plots/{pointcloudName}_boxplot_pipes.png",
            part="Endpunkte",
            title="Abstände der erkannten Rohre zu den Ground Truth Rohren",
            show=False,
        )

    plot_segmentClasses(
        correct,
        partial,
        missed,
        false_positives,
        coverage,
        missed_length=total_length - coverage,
        false_positive_length=false_positive_length,
        out_png=f"./output/plots/{pointcloudName}_segmentClasses.png",
        show=False,
    )
