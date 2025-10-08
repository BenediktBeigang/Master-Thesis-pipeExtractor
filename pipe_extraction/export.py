import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line_peaks
import laspy
import json
from custom_types import Segment3DArray


def save_slice_las(
    sliced_xyz: np.ndarray,
    slice_idx: int,
    output_dir: str,
) -> None:
    """
    Speichert die Punkte eines Slices als separate LAS-Datei.
    """
    os.makedirs(output_dir, exist_ok=True)
    las_path = os.path.join(output_dir, f"slice_{slice_idx:03d}_points.las")

    try:
        # Einfachste Lösung: Verwende die gleichen Header-Einstellungen wie bei großen LAS-Dateien
        header = laspy.LasHeader(point_format=0, version="1.2")

        # Keine Transformation - nur direkte Übernahme
        las = laspy.LasData(header)
        las.x = sliced_xyz[:, 0]
        las.y = sliced_xyz[:, 1]
        las.z = sliced_xyz[:, 2]

        las.write(las_path)
        print(f"  → Slice LAS gespeichert: {las_path} ({len(sliced_xyz)} Punkte)")

    except Exception as e:
        print(f"  → Fehler beim Speichern der LAS-Datei {las_path}: {e}")


def save_slice_obj(
    segments_world: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    slice_idx: int,
    z_center: float,
    output_dir: str,
) -> None:
    """
    Speichert die Linien eines Slices als separate OBJ-Datei.
    """
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, f"slice_{slice_idx:03d}_lines.obj")

    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"# OBJ generated for slice {slice_idx}\n")
        f.write(f"# Z-center: {z_center:.6f}m\n")
        f.write(f"# Total lines: {len(segments_world)}\n")

        vert_idx = 1
        for p0, p1 in segments_world:
            f.write(f"v {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}\n")
            f.write(f"v {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
            f.write(f"l {vert_idx} {vert_idx+1}\n")
            vert_idx += 2

    print(f"  → Slice OBJ gespeichert: {obj_path}")


def write_clusters_as_obj(
    slice_idx,
    segments,
    clusters,
    output_dir,
    z_value=0.0,
):
    """
    id : int                             # Slice-ID
    segments : list[ ((x1,y1),(x2,y2)), ... ] oder list[ ((x1,y1,z1),(x2,y2,z2)), ... ]
    clusters : dict[int, np.ndarray]  # {cluster_id: indices in segments}
    output_dir : str                   # Ausgabeverzeichnis
    z_value  : float                  # Z-Koordinate (2D -> setze 0.0)
    """
    segments = np.asarray(segments, dtype=float)  # (N, 2, 2) oder (N, 2, 3)
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, f"slice_{slice_idx:03d}_cluster.obj")

    # Eine einzige OBJ-Datei für alle Cluster
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# Wavefront OBJ (Polylines)\n")
        f.write(f"# Alle Cluster, {len(clusters)} Cluster insgesamt\n")

        v_counter = 0

        for cid, idx in clusters.items():
            idx = np.asarray(idx, dtype=int)
            if idx.size == 0:
                continue

            # Erstelle separates Objekt für jeden Cluster
            f.write(f"\n# Cluster {cid}, {len(idx)} Segmente\n")
            f.write(f"o cluster_{int(cid):03d}\n")
            f.write(f"g cluster_{int(cid):03d}_lines\n")

            # Schreibe jedes Segment als separate Linie
            for si in idx:
                segment = segments[si]

                # Prüfe ob 2D oder 3D Segmente
                if segment.shape[1] == 2:  # 2D: ((x1,y1),(x2,y2))
                    (x1, y1), (x2, y2) = segment
                    z1 = z2 = z_value
                elif segment.shape[1] == 3:  # 3D: ((x1,y1,z1),(x2,y2,z2))
                    (x1, y1, z1), (x2, y2, z2) = segment
                else:
                    raise ValueError(f"Unerwartetes Segment-Format: {segment.shape}")

                # Schreibe die beiden Vertices für dieses Segment
                f.write(f"v {x1:.9f} {y1:.9f} {z1:.9f}\n")
                f.write(f"v {x2:.9f} {y2:.9f} {z2:.9f}\n")

                # Erstelle eine Linie zwischen den beiden Vertices
                f.write(f"l {v_counter + 1} {v_counter + 2}\n")
                v_counter += 2


def write_clusters_as_json(
    slice_idx,
    segments,
    clusters,
    output_dir,
    z_value=0.0,
):
    """
    Exportiert Cluster als JSON-Datei.

    slice_idx : int                      # Slice-ID
    segments : list[ ((x1,y1),(x2,y2)), ... ] oder list[ ((x1,y1,z1),(x2,y2,z2)), ... ]
    clusters : dict[int, np.ndarray]     # {cluster_id: indices in segments}
    output_dir : str                     # Ausgabeverzeichnis
    z_value : float                      # Z-Koordinate (2D -> setze 0.0)
    """
    segments = np.asarray(segments, dtype=float)
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, f"slice_{slice_idx:03d}_clusters.json")

    # Erstelle JSON-Struktur
    cluster_data = {
        "slice_id": slice_idx,
        "total_segments": len(segments),
        "total_clusters": len(clusters),
        "z_value": z_value,
        "clusters": [],
    }

    for cid, idx in clusters.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue

        cluster_segments = []

        for si in idx:
            segment = segments[si]

            # Prüfe ob 2D oder 3D Segmente
            if segment.shape[1] == 2:  # 2D: ((x1,y1),(x2,y2))
                (x1, y1), (x2, y2) = segment
                z1 = z2 = z_value
            elif segment.shape[1] == 3:  # 3D: ((x1,y1,z1),(x2,y2,z2))
                (x1, y1, z1), (x2, y2, z2) = segment
            else:
                raise ValueError(f"Unerwartetes Segment-Format: {segment.shape}")

            # Segment als Dictionary mit Start- und Endpunkt
            segment_data = {
                "segment_index": int(si),
                "start_point": {"x": float(x1), "y": float(y1), "z": float(z1)},
                "end_point": {"x": float(x2), "y": float(y2), "z": float(z2)},
            }
            cluster_segments.append(segment_data)

        # Cluster-Information
        cluster_info = {
            "cluster_id": int(cid),
            "segment_count": len(cluster_segments),
            "segments": cluster_segments,
        }
        cluster_data["clusters"].append(cluster_info)

    # JSON speichern
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=2, ensure_ascii=False)

    print(f"  → Cluster JSON gespeichert: {json_path} ({len(clusters)} Cluster)")


def save_slice_images(
    binary: np.ndarray,
    segments_px: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    hough_space: np.ndarray,
    theta: np.ndarray,
    d: np.ndarray,
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    slice_idx: int,
    z_center: float,
    output_dir: str,
) -> None:
    """
    Speichert zwei Bilder für einen Slice:
    1. Das Binärbild mit den gefundenen Liniensegmenten
    2. Den Hough-Raum (Accumulator)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Binärbild mit Liniensegmenten
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Zeige das Binärbild
    ax.imshow(
        binary,
        cmap="gray",
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    )

    # Zeichne die gefundenen Liniensegmente
    for line in segments_px:
        p0, p1 = line
        # Konvertiere Pixel zu Weltkoordinaten für die Darstellung
        x0 = x_edges[int(p0[0])] + 0.5 * (
            x_edges[min(int(p0[0]) + 1, len(x_edges) - 1)] - x_edges[int(p0[0])]
        )
        y0 = y_edges[int(p0[1])] + 0.5 * (
            y_edges[min(int(p0[1]) + 1, len(y_edges) - 1)] - y_edges[int(p0[1])]
        )
        x1 = x_edges[int(p1[0])] + 0.5 * (
            x_edges[min(int(p1[0]) + 1, len(x_edges) - 1)] - x_edges[int(p1[0])]
        )
        y1 = y_edges[int(p1[1])] + 0.5 * (
            y_edges[min(int(p1[1]) + 1, len(y_edges) - 1)] - y_edges[int(p1[1])]
        )

        ax.plot([x0, x1], [y0, y1], "r-", linewidth=2, alpha=0.8)

    ax.set_title(
        f"Slice {slice_idx}: Binärbild mit Liniensegmenten (Z={z_center:.3f}m)"
    )
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.grid(True, alpha=0.3)

    # Speichere das erste Bild
    img1_path = os.path.join(output_dir, f"lines/slice_{slice_idx:03d}_lines.png")
    plt.savefig(img1_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 2: Hough-Raum
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Zeige den Hough-Raum
    im = ax.imshow(
        hough_space,
        cmap="hot",
        origin="lower",
        extent=[np.rad2deg(theta[0]), np.rad2deg(theta[-1]), d[0], d[-1]],
        aspect="auto",
    )

    # Finde und markiere die Peaks im Hough-Raum
    peaks = hough_line_peaks(
        hough_space,
        theta,
        d,
        min_distance=20,
        min_angle=10,
        threshold=0.3 * np.max(hough_space),
    )

    # Korrigierte Peak-Darstellung
    if len(peaks) >= 3 and len(peaks[0]) > 0:  # Prüfe ob Peaks gefunden wurden
        accums, angles, dists = peaks
        for i in range(len(accums)):
            angle_idx = int(angles[i])
            dist_idx = int(dists[i])
            if 0 <= angle_idx < len(theta) and 0 <= dist_idx < len(d):
                ax.plot(
                    np.rad2deg(theta[angle_idx]),
                    d[dist_idx],
                    "wo",
                    markersize=8,
                    markeredgecolor="black",
                )

    ax.set_title(f"Slice {slice_idx}: Hough-Raum (Z={z_center:.3f}m)")
    ax.set_xlabel("Winkel [Grad]")
    ax.set_ylabel("Distanz [Pixel]")

    # Colorbar
    plt.colorbar(im, ax=ax, label="Akkumulator-Werte")

    # Speichere das zweite Bild
    img2_path = os.path.join(output_dir, f"hough/slice_{slice_idx:03d}_hough.png")
    plt.savefig(img2_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  → Bilder gespeichert: {img1_path} und {img2_path}")


def write_obj_lines(
    segments_world: Segment3DArray,
    out_path: str,
) -> None:
    """
    Schreibt Polylinien als OBJ:
        v x y z
        v x y z
        l i j
    CloudCompare kann diese Linien laden.
    """
    print("Schreibe finale segmente in obj", out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# OBJ generated by slice_hough_obj.py\n")
        f.write(f"# Total lines: {len(segments_world)}\n")
        vert_idx = 1
        for p0, p1 in segments_world:
            f.write(f"v {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}\n")
            f.write(f"v {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
            f.write(f"l {vert_idx} {vert_idx+1}\n")
            vert_idx += 2


def write_segments_as_geojson(
    segments: Segment3DArray,
    output_path: str,
    crs_name: str = "unknown",
) -> None:
    """
    Exportiert Liniensegmente als GeoJSON-Datei.

    segments : list[ ((x1,y1),(x2,y2)), ... ] oder list[ ((x1,y1,z1),(x2,y2,z2)), ... ]
    output_path : str                    # Ausgabepfad für die GeoJSON-Datei
    crs_name : str                      # Name des Koordinatenreferenzsystems
    """
    segments = np.asarray(segments, dtype=float)

    # Bestimme ob 2D oder 3D Segmente
    has_z = segments.shape[2] == 3 if len(segments) > 0 else False

    # Erstelle GeoJSON-Struktur
    geojson_data = {
        "type": "FeatureCollection",
        "name": "segments",
        "features": [],
        "crs": {"type": "name", "properties": {"name": crs_name}},
    }

    # Füge Liniensegmente als Features hinzu
    for i, segment in enumerate(segments):
        if has_z:  # 3D: ((x1,y1,z1),(x2,y2,z2))
            (x1, y1, z1), (x2, y2, z2) = segment
            coordinates = [
                [float(x1), float(y1), float(z1)],
                [float(x2), float(y2), float(z2)],
            ]
        else:  # 2D: ((x1,y1),(x2,y2))
            (x1, y1), (x2, y2) = segment
            coordinates = [
                [float(x1), float(y1), 0],
                [float(x2), float(y2), 0],
            ]

        feature = {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coordinates},
            "properties": {
                "segment_index": i,
                "has_z": True,
            },
        }

        geojson_data["features"].append(feature)

    # GeoJSON speichern
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2, ensure_ascii=False)


def export_sample_vectors_to_obj(
    sample_data: list,
    tangential_length: float,
    normal_length: float,
):
    """
    Exportiert Sample-Point-Vektoren in eine OBJ-Datei.

    Für jeden Sample-Point werden exportiert:
    - Tangentialvektor (t_hat) skaliert mit tangential_half_width
    - Normalenvektor (n_hat) skaliert mit normal_length
    - Resultierender Punkt (c_xy) als Punkt
    """
    with open("./sample_vectors.obj", "w") as f:
        f.write("# Sample Point Vectors Export\n")
        f.write(f"# Tangential half width: {tangential_length}\n")
        f.write(f"# Normal length: {normal_length}\n")
        f.write(f"# Total sample points: {len(sample_data)}\n\n")

        vertex_count = 0

        for i, data in enumerate(sample_data):
            t_hat = data["t_hat"]
            n_hat = data["n_hat"]
            c_xy = data["c_xy"]
            sample_xy = data["sample_point_xy"]
            z = data["z"]

            f.write(f"# Sample point {i}\n")

            # Startpunkt (Sample-Point-Position)
            start_3d = [sample_xy[0], sample_xy[1], z]
            f.write(f"v {start_3d[0]:.6f} {start_3d[1]:.6f} {start_3d[2]:.6f}\n")

            # Endpunkt des Tangentialvektors (beide Richtungen)
            t_end1_3d = [
                start_3d[0] + t_hat[0] * tangential_length,
                start_3d[1] + t_hat[1] * tangential_length,
                z,
            ]
            t_end2_3d = [
                start_3d[0] - t_hat[0] * tangential_length,
                start_3d[1] - t_hat[1] * tangential_length,
                z,
            ]
            f.write(f"v {t_end1_3d[0]:.6f} {t_end1_3d[1]:.6f} {t_end1_3d[2]:.6f}\n")
            f.write(f"v {t_end2_3d[0]:.6f} {t_end2_3d[1]:.6f} {t_end2_3d[2]:.6f}\n")

            # Endpunkt des Normalenvektors (beide Richtungen)
            n_end1_3d = [
                start_3d[0] + n_hat[0] * normal_length,
                start_3d[1] + n_hat[1] * normal_length,
                z,
            ]
            n_end2_3d = [
                start_3d[0] - n_hat[0] * normal_length,
                start_3d[1] - n_hat[1] * normal_length,
                z,
            ]
            f.write(f"v {n_end1_3d[0]:.6f} {n_end1_3d[1]:.6f} {n_end1_3d[2]:.6f}\n")
            f.write(f"v {n_end2_3d[0]:.6f} {n_end2_3d[1]:.6f} {n_end2_3d[2]:.6f}\n")

            # Resultierender Punkt (c_xy)
            result_3d = [c_xy[0], c_xy[1], z]
            f.write(f"v {result_3d[0]:.6f} {result_3d[1]:.6f} {result_3d[2]:.6f}\n")

            # Linien definieren (OBJ verwendet 1-basierte Indizes)
            base_idx = vertex_count + 1

            # Tangentialvektor-Linien (Kreuz)
            f.write(f"l {base_idx} {base_idx + 1}\n")  # Start -> t_end1
            f.write(f"l {base_idx} {base_idx + 2}\n")  # Start -> t_end2

            # Normalenvektor-Linien (Kreuz)
            f.write(f"l {base_idx} {base_idx + 3}\n")  # Start -> n_end1
            f.write(f"l {base_idx} {base_idx + 4}\n")  # Start -> n_end2

            # Linie zum resultierenden Punkt
            f.write(f"l {base_idx} {base_idx + 5}\n")  # Start -> result

            vertex_count += 6  # 6 Vertices pro Sample-Point
            f.write("\n")
