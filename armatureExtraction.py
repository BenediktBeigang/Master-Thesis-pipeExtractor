import numpy as np
import laspy
from sklearn.cluster import HDBSCAN
import os


def load_las_points(filepath, ignore_z=False):
    las = laspy.read(filepath)
    if ignore_z:
        return np.vstack((las.x, las.y)).T
    return np.vstack((las.x, las.y, las.z)).T


def compute_bounding_box(points):
    return points.min(axis=0), points.max(axis=0)


def write_obj_boxes(cluster_boxes, output_path):
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
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def save_xyz_boxes(cluster_boxes, output_path):
    with open(output_path, "w") as f:
        for label, (mins, maxs) in cluster_boxes.items():
            corners = np.array(
                [
                    [mins[0], mins[1], mins[2]],
                    [mins[0], mins[1], maxs[2]],
                    [mins[0], maxs[1], mins[2]],
                    [mins[0], maxs[1], maxs[2]],
                    [maxs[0], mins[1], mins[2]],
                    [maxs[0], mins[1], maxs[2]],
                    [maxs[0], maxs[1], mins[2]],
                    [maxs[0], maxs[1], maxs[2]],
                ]
            )
            for c in corners:
                f.write(f"{c[0]} {c[1]} {c[2]} # cluster {label}\n")


def main(las_path, output_folder="./", ignore_z=False):
    os.makedirs(output_folder, exist_ok=True)
    X_clustering = load_las_points(las_path, ignore_z=ignore_z)
    X_original = load_las_points(las_path, ignore_z=False)  # Für Bounding Boxes

    # HDBSCAN mit sklearn-Implementierung
    print(f"Starting Clustering for {len(X_clustering)} points.")

    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html
    clusterer = HDBSCAN(
        min_cluster_size=20,
        min_samples=None,
        cluster_selection_method="eom",
        metric="euclidean",
        algorithm="auto",
        leaf_size=40,
        n_jobs=-1,
        store_centers="centroid",
    )
    clusterer.fit(X_clustering)

    labels = clusterer.labels_
    unique = set(labels)
    if -1 in unique:
        unique.remove(-1)  # Rauschen

    cluster_boxes = {}
    means = {}
    for c in unique:
        pts = X_original[labels == c]  # Verwende ursprüngliche 3D-Punkte
        mins, maxs = compute_bounding_box(pts)
        cluster_boxes[c] = (mins, maxs)
        means[c] = pts.mean(axis=0)

    obj_path = os.path.join(output_folder, "clusters_boxes.obj")
    try:
        write_obj_boxes(cluster_boxes, obj_path)
        print("Bounding boxes saved as OBJ:", obj_path)
    except Exception as e:
        print("OBJ-Export fehlgeschlagen:", e)
        xyz_path = os.path.join(output_folder, "clusters_boxes.xyz")
        save_xyz_boxes(cluster_boxes, xyz_path)
        print("Fallback: XYZ gespeichert:", xyz_path)

    means_path = os.path.join(output_folder, "cluster_means.txt")
    with open(means_path, "w") as f:
        for c, m in means.items():
            f.write(f"{c}: {m[0]} {m[1]} {m[2]}\n")
    print("Cluster-Mittelwerte gespeichert:", means_path)


if __name__ == "__main__":
    main(las_path="./ontras_3_predicted_0729_t3_class_2.las", ignore_z=True)


# birch oder DBSCAN ausprobieren
