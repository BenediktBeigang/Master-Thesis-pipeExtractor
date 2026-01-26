import json
import numpy as np
from pipeExtractor.eval.componentMetrics import component_detection_metric
from pipeExtractor.eval.plots.componentClassPlot import plot_componentClasses
from pipeExtractor.eval.plots.distancePlot import plot_boxplots_lineDistances


def componentEval(ground_truth, detected_components, pointcloudName):
    # export_segments_to_obj(
    #     ground_truth_segments,
    #     "./ontras_3_ground_truth.obj",
    # )

    (
        found,
        missed,
        false_positives,
        xy_distances,
        z_distances,
    ) = component_detection_metric(
        ground_truth, detected_components, pointcloudName, tolerance=0.5
    )

    result = {
        "pipe_count_ground_truth": len(ground_truth),
        "pipe_count_detected": len(detected_components),
        "found": found,
        "missed": missed,
        "false_positives": false_positives,
        "coverage": f"{found} / {len(ground_truth)}  |  {(found/len(ground_truth)*100):.2f} %",
        "coverage_iou": f"{found} / {len(ground_truth) + false_positives}  |  {(found / (len(ground_truth) + false_positives) * 100):.2f} %",
        "distance_xy_avg": (
            0 if len(xy_distances) == 0 else sum(xy_distances) / len(xy_distances)
        ),
        "distance_xy_median": np.median(xy_distances),
        "distance_z_avg": (
            0 if len(z_distances) == 0 else sum(z_distances) / len(z_distances)
        ),
        "distance_z_median": np.median(z_distances),
        "distance_xy_samples": xy_distances,
        "distance_z_samples": z_distances,
    }

    with open(f"./output/metrics/{pointcloudName}_components.json", "w") as f:
        json.dump(result, f, indent=2)

    if len(xy_distances) > 0 or len(z_distances) > 0:
        plot_boxplots_lineDistances(
            xy_distances,
            z_distances,
            out_png=f"./output/plots/{pointcloudName}_boxplot_components.png",
            part="Rohrbauteile",
            title="Abstände der erkannten Rohrbauteile zu den Ground Truth Rohrbauteilen",
            show=False,
        )

    plot_componentClasses(
        found,
        missed,
        false_positives,
        out_png=f"./output/plots/{pointcloudName}_componentClasses.png",
        show=False,
    )
