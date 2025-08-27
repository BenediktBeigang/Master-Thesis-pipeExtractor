import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import hough_line_peaks
import laspy


def save_slice_las(
    sliced_xyz: np.ndarray,
    slice_idx: int,
    z_center: float,
    output_dir: str,
) -> None:
    """
    Speichert die Punkte eines Slices als separate LAS-Datei.
    """
    os.makedirs(os.path.join(output_dir, "las"), exist_ok=True)

    las_path = os.path.join(output_dir, f"las/slice_{slice_idx:03d}_points.las")

    try:
        # Erstelle ein neues LAS-File
        header = laspy.LasHeader(point_format=0, version="1.2")
        header.x_scale = 0.001
        header.y_scale = 0.001
        header.z_scale = 0.001

        # Setze Offset basierend auf den Punktdaten
        header.x_offset = np.floor(sliced_xyz[:, 0].min())
        header.y_offset = np.floor(sliced_xyz[:, 1].min())
        header.z_offset = np.floor(sliced_xyz[:, 2].min())

        # Erstelle LAS-File
        las = laspy.LasData(header)

        # Setze Koordinaten
        las.x = sliced_xyz[:, 0]
        las.y = sliced_xyz[:, 1]
        las.z = sliced_xyz[:, 2]

        # Speichere die Datei
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
    os.makedirs(os.path.join(output_dir, "obj"), exist_ok=True)

    obj_path = os.path.join(output_dir, f"obj/slice_{slice_idx:03d}_lines.obj")

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
