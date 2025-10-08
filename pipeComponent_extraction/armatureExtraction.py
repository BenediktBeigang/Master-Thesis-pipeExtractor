import datetime
import numpy as np
import laspy
from sklearn.cluster import HDBSCAN
import os
import json
import time

from lasUtil import load_las_xyz


def load_pipes(filepath):
    try:
        with open(filepath, "r") as f:
            pipes_data = json.load(f)

        # Konvertiere das JSON-Format zu dem erwarteten Format für filter_points_by_pipe_distance
        pipes = []
        for pipe in pipes_data:
            pipe_dict = {
                "p1_x": pipe["point1"]["x"],
                "p1_y": pipe["point1"]["y"],
                "p1_z": pipe["point1"]["z"],
                "p2_x": pipe["point2"]["x"],
                "p2_y": pipe["point2"]["y"],
                "p2_z": pipe["point2"]["z"],
                "id": pipe["id"],
                "length": pipe["length"],
                "fitness": pipe["fitness"],
                "original_pipes": pipe["original_pipes"],
            }
            pipes.append(pipe_dict)

        print(f"Loaded {len(pipes)} pipes from {filepath}")
        return pipes

    except FileNotFoundError:
        print(f"Fehler: JSON-Datei nicht gefunden: {filepath}")
        return []
    except json.JSONDecodeError:
        print(f"Fehler: Ungültiges JSON-Format in Datei: {filepath}")
        return []
    except KeyError as e:
        print(f"Fehler: Fehlender Schlüssel in JSON-Daten: {e}")
        return []
    except Exception as e:
        print(f"Unerwarteter Fehler beim Laden der Rohrdaten: {e}")
        return []


def compute_bounding_box(points):
    return points.min(axis=0), points.max(axis=0)


def write_obj_boxes(cluster_boxes, output_path="clusters_boxes.obj", timestamp=""):
    try:
        lines = []
        vert_count = 0
        for label, (mins, maxs) in cluster_boxes.items():
            xmin, ymin, zmin = mins
            xmax, ymax, zmax = maxs
            verts = [
                (xmin, ymin, zmin),  # 0
                (xmax, ymin, zmin),  # 1
                (xmax, ymax, zmin),  # 2
                (xmin, ymax, zmin),  # 3
                (xmin, ymin, zmax),  # 4
                (xmax, ymin, zmax),  # 5
                (xmax, ymax, zmax),  # 6
                (xmin, ymax, zmax),  # 7
            ]
            for v in verts:
                lines.append(f"v {v[0]} {v[1]} {v[2]}")

            # Kanten der Bounding Box als Linien
            edges = [
                # Untere Fläche (z = zmin)
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                # Obere Fläche (z = zmax)
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                # Vertikale Kanten
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for a, b in edges:
                lines.append(f"l {vert_count + a + 1} {vert_count + b + 1}")

            vert_count += 8
        with open(f"{timestamp}_{output_path}", "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"Fehler beim Schreiben der OBJ-Datei: {e}")


def write_json(
    means,
    X_original,
    labels,
    cluster_boxes,
    output_path="cluster.json",
    timestamp="",
):
    try:
        means_json = {}
        for c, m in means.items():
            pts = X_original[labels == c]
            means_json[str(c)] = {
                "centroid": {"x": float(m[0]), "y": float(m[1]), "z": float(m[2])},
                "point_count": int(len(pts)),
                "bounding_box": {
                    "min": {
                        "x": float(cluster_boxes[c][0][0]),
                        "y": float(cluster_boxes[c][0][1]),
                        "z": float(cluster_boxes[c][0][2]),
                    },
                    "max": {
                        "x": float(cluster_boxes[c][1][0]),
                        "y": float(cluster_boxes[c][1][1]),
                        "z": float(cluster_boxes[c][1][2]),
                    },
                },
            }

        with open(f"{timestamp}_{output_path}", "w") as f:
            json.dump(means_json, f, indent=2)
    except Exception as e:
        print(f"Fehler beim Schreiben der JSON-Datei: {e}")


def point_to_line_distance(point, line_start, line_end):
    """
    Calculate the shortest distance from a point to a line segment.
    """
    # Vektoren
    line_vec = line_end - line_start
    point_vec = point - line_start

    # Länge der Linie
    line_len_sq = np.dot(line_vec, line_vec)

    # Wenn die Linie ein Punkt ist
    if line_len_sq == 0:
        return np.linalg.norm(point_vec)

    # Parameter t für die Projektion des Punktes auf die Linie
    t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))

    # Nächster Punkt auf der Linie
    projection = line_start + t * line_vec

    # Distanz vom Punkt zur Projektion
    return np.linalg.norm(point - projection)


def filter_points_by_pipe_distance(
    points,
    pipes,
    distance_threshold: float = 1,
    ignore_z: bool = False,
):
    """
    Filter points that are within a certain distance from any of the pipe lines.
    Points close to pipes are kept (not filtered out).
    Returns filtered points and their original indices.
    """

    filtered_points = []
    filtered_indices = []
    for index, point in enumerate(points):
        if index % 50000 == 0:
            print(f"Processing point {index + 1}/{len(points)}: {point}")
        # Finde die minimale Distanz zu allen Rohrleitungen
        min_distance = float("inf")
        for pipe in pipes:
            if ignore_z:
                # Verwende nur X und Y Koordinaten für 2D-Vergleich
                line_start = np.array([pipe["p1_x"], pipe["p1_y"]])
                line_end = np.array([pipe["p2_x"], pipe["p2_y"]])
            else:
                # Verwende alle drei Koordinaten für 3D-Vergleich
                line_start = np.array([pipe["p1_x"], pipe["p1_y"], pipe["p1_z"]])
                line_end = np.array([pipe["p2_x"], pipe["p2_y"], pipe["p2_z"]])

            distance = point_to_line_distance(point, line_start, line_end)
            min_distance = min(min_distance, float(distance))

        # Punkt behalten, wenn er NICHT zu nah an einer Rohrleitung ist
        if min_distance < distance_threshold:
            filtered_points.append(point)
            filtered_indices.append(index)

    return np.array(filtered_points), np.array(filtered_indices)


def main(
    las_path,
    pipeJson_path,
    output_folder="./",
    ignore_z=False,
    near_pipe_filter=False,
):
    print(f"Loading LAS file: {las_path}")
    os.makedirs(output_folder, exist_ok=True)
    X_clustering = load_las_xyz(las_path, ignoreZ=ignore_z)
    X_original = load_las_xyz(las_path, ignoreZ=False)  # Für Bounding Boxes
    print(f"Loaded {len(X_clustering)} points from {las_path}")

    if near_pipe_filter:
        # Lade Rohre und filtere Punkte die weit weg von Rohren sind
        pipes = load_pipes(pipeJson_path)
        print(f"Filtering points based on pipe distances...")
        X_clustering, filtered_indices = filter_points_by_pipe_distance(
            X_clustering,
            pipes,
            distance_threshold=0.5,
            ignore_z=ignore_z,
        )
        # Filtere auch X_original mit denselben Indizes
        X_original = X_original[filtered_indices]
        print(f"Filtered non pipe points")

    MIN_POINTS = 20  # Minimalgröße eines Clusters

    # HDBSCAN mit sklearn-Implementierung
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    print(f"Starting Clustering for {len(X_clustering)} points.")
    clusterer = HDBSCAN(
        min_cluster_size=MIN_POINTS,
        min_samples=None,
        cluster_selection_method="eom",
        metric="euclidean",
        algorithm="auto",
        leaf_size=40,
        n_jobs=-1,
        store_centers="centroid",
    )
    clusterer.fit(X_clustering)

    print(f"Deleting clusters with less than {MIN_POINTS} points.")
    labels = clusterer.labels_
    unique = set(labels)
    if -1 in unique:
        unique.remove(-1)  # Rauschen

    cluster_boxes = {}
    means = {}
    for c in unique:
        pts = X_original[labels == c]  # Jetzt haben beide Arrays dieselbe Länge
        mins, maxs = compute_bounding_box(pts)
        cluster_boxes[c] = (mins, maxs)
        means[c] = pts.mean(axis=0)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Schreiben der Bounding Boxes in OBJ-Format
    write_obj_boxes(cluster_boxes, timestamp=timestamp)

    # JSON-Export der Cluster-Mittelwerte
    write_json(
        means,
        X_original,
        labels,
        cluster_boxes,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    start_time = time.time()

    main(
        las_path="./ontras_3_predicted_0916_t1_c2.las",
        pipeJson_path="./output/20250821_102717_all_pipes.json",
        ignore_z=True,
        near_pipe_filter=False,
    )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution took: {execution_time:.2f} seconds")
