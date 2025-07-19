#!/usr/bin/env python3
"""
Python-Skript zur Erkennung eines zylindrischen Rohrs in einer LAS-Punktwolke
und Export der Zylinderachse als OBJ-Datei für die Visualisierung in CloudCompare.

Abhängigkeiten:
  - pylas
  - numpy
  - pclpy

Usage:
  python detect_cylinder_and_export_obj.py input.las output.obj
"""
import argparse

import numpy as np
import pylas
from pclpy import pcl


def main():
    parser = argparse.ArgumentParser(
        description="Erkennt einen Zylinder in einer LAS-Datei und exportiert die Zylinderachse als OBJ."
    )
    parser.add_argument("input_las", help="Pfad zur Eingabe-LAS-Datei (z.B. input.las)")
    parser.add_argument(
        "output_obj", help="Pfad zur Ausgabe-OBJ-Datei (z.B. cylinder.obj)"
    )
    args = parser.parse_args()

    # 1. LAS-Datei einlesen
    las = pylas.read(args.input_las)
    points = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)

    # 2. Punktwolke in PCL-PointCloud umwandeln
    cloud = pcl.PointCloud.PointXYZ()
    cloud.from_array(points)

    # 3. Normalenschätzung
    ne = pcl.features.NormalEstimationOMP.PointXYZ_Normal()
    tree = pcl.search.KdTree.PointXYZ()
    ne.setSearchMethod(tree)
    ne.setInputCloud(cloud)
    ne.setKSearch(50)
    normals = pcl.PointCloud.Normal()
    ne.compute(normals)

    # 4. Zylinder-Segmentierung konfigurieren
    seg = pcl.segmentation.SACSegmentationFromNormals.PointXYZ_Normal()
    seg.setOptimizeCoefficients(True)
    seg.setModelType(pcl.sample_consensus.SACMODEL_CYLINDER)
    seg.setMethodType(pcl.sample_consensus.SAC_RANSAC)
    seg.setNormalDistanceWeight(0.1)
    seg.setMaxIterations(10000)
    seg.setDistanceThreshold(0.05)
    seg.setRadiusLimits(0.0, 1.0)
    seg.setInputCloud(cloud)
    seg.setInputNormals(normals)

    # 5. Zylinder ermitteln
    inliers = pcl.PointIndices()
    coefficients = pcl.ModelCoefficients()
    seg.segment(inliers, coefficients)
    if len(inliers.indices) == 0:
        print(
            "Kein Zylinder gefunden. Überprüfen Sie die Parameter oder die Punktwolke."
        )
        return

    # 6. Zylinder-Koeffizienten auslesen
    px, py, pz, dx, dy, dz, radius = coefficients.values
    axis_point = np.array([px, py, pz], dtype=np.float32)
    axis_dir = np.array([dx, dy, dz], dtype=np.float32)
    axis_dir /= np.linalg.norm(axis_dir)

    # 7. Parameter t für jedes Inlier entlang der Achse berechnen
    pts = points[inliers.indices]
    t_vals = np.dot(pts - axis_point, axis_dir)
    t_min, t_max = t_vals.min(), t_vals.max()

    # 8. Endpunkte der Zylinderachse
    start = axis_point + t_min * axis_dir
    end = axis_point + t_max * axis_dir

    # 9. OBJ-Datei schreiben (Liniensegment)
    with open(args.output_obj, "w") as f:
        f.write("# Cylinder axis from LAS detection\n")
        f.write(f"v {start[0]} {start[1]} {start[2]}\n")
        f.write(f"v {end[0]} {end[1]} {end[2]}\n")
        f.write("l 1 2\n")

    print(f"Erfolgreich exportiert: {args.output_obj}")


if __name__ == "__main__":
    main()
