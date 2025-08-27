#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D-Liniendetektion in Z-Slices einer LAS-Punktwolke per Rasterisierung + Hough,
Export aller Linien als OBJ (Polylines) für CloudCompare.

Abhängigkeiten:
    numpy
    laspy        (>=2.0 empfohlen; Fallback für 1.7.0 integriert)
    scikit-image (skimage)
    matplotlib   (für Bildexport)

Beispiel:
    python slice_hough_obj.py \
        --input in.las \
        --thickness 0.10 \
        --cell-size 0.03 \
        --min-count 3 \
        --use-canny \
        --canny-sigma 1.2 \
        --min-line-length-m 0.40 \
        --max-line-gap-m 0.10 \
        --top-k 300 \
        --output lines.obj \
        --save-images \
        --image-output-dir slice_images
"""

import shutil
import argparse
import datetime
import time
import math
import sys
import os
from typing import Tuple, List
from PipeCluster import clean_pipes
from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import probabilistic_hough_line, hough_line, hough_line_peaks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from exportSlices import save_slice_images, save_slice_las, save_slice_obj
from lasUtil import load_las_xyz


def get_z_slices(xyz: np.ndarray, thickness: float) -> List[Tuple[float, float, float]]:
    """
    Berechnet alle Z-Slice Parameter (z_center, zmin, zmax) basierend auf der Bounding Box.

    Returns:
        Liste von (z_center, zmin, zmax) Tupeln für jeden Slice
    """
    z_min = float(xyz[:, 2].min())
    z_max = float(xyz[:, 2].max())

    print(f"Z-Bounds der Punktwolke: {z_min:.3f} .. {z_max:.3f}")

    # Berechne Anzahl der Slices
    z_range = z_max - z_min
    num_slices = max(1, math.ceil(z_range / thickness))

    slices = []
    for i in range(num_slices):
        z_center = z_min + (i + 0.5) * thickness
        zmin = z_center - 0.5 * thickness
        zmax = z_center + 0.5 * thickness

        # Stelle sicher, dass der erste Slice bei z_min beginnt
        if i == 0:
            zmin = z_min
        # Stelle sicher, dass der letzte Slice bei z_max endet
        if i == num_slices - 1:
            zmax = z_max
            z_center = 0.5 * (zmin + zmax)

        slices.append((z_center, zmin, zmax))

    print(f"Erstelle {num_slices} Slices mit Dicke {thickness:.3f}m")
    return slices


def slice_by_z(xyz: np.ndarray, zmin: float, zmax: float) -> np.ndarray:
    """Extrahiert Punkte in einem Z-Bereich"""
    mask = (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    return xyz[mask]


def rasterize_xy(
    xy: np.ndarray, cell_size: float, bounds: Tuple[float, float, float, float] = None
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Rasterisiert XY-Punkte in ein 2D-Count-Grid via histogram2d.
    Rückgabe:
        H: (ny, nx) array mit Zählwerten
        edges: (y_edges, x_edges)
    """
    xs = xy[:, 0]
    ys = xy[:, 1]

    if bounds is None:
        xmin, xmax = xs.min(), xs.max()
        ymin, ymax = ys.min(), ys.max()
    else:
        xmin, xmax, ymin, ymax = bounds

    # Sicherheits-Pad bei degenerierten Fällen
    if xmax <= xmin:
        xmax = xmin + cell_size
    if ymax <= ymin:
        ymax = ymin + cell_size

    nx = max(1, math.ceil((xmax - xmin) / cell_size))
    ny = max(1, math.ceil((ymax - ymin) / cell_size))

    H, y_edges, x_edges = np.histogram2d(
        ys, xs, bins=(ny, nx), range=((ymin, ymax), (xmin, xmax))
    )
    return H.astype(np.float64), (y_edges, x_edges)


def make_binary_from_counts(
    H: np.ndarray, min_count: int = 3, use_canny: bool = True, canny_sigma: float = 1.2
) -> np.ndarray:
    """
    Erzeugt ein Binärbild für Hough.
    - Entweder simple Count-Schwelle (H >= min_count),
    - oder Canny auf normalisierten Counts (robuster bei ungleichmäßiger Dichte).
    """
    if use_canny:
        # leichte Glättung + Normalisierung
        if H.max() > 0:
            Hs = gaussian(H / (H.max() + 1e-9), sigma=1.0, preserve_range=True)
        else:
            Hs = H
        edges = canny(Hs, sigma=canny_sigma)
        return edges.astype(bool)
    else:
        return (H >= max(1, int(min_count))).astype(bool)


def hough_segments(
    binary: np.ndarray,
    cell_size: float,
    min_line_length_m: float = 0.4,
    max_line_gap_m: float = 0.10,
) -> Tuple[
    List[Tuple[Tuple[float, float], Tuple[float, float]]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Probabilistic Hough auf Binärbild.
    Rückgabe:
        - Liste von Segmenten in Pixelkoordinaten [( (x0,y0), (x1,y1) ), ...]
        - Hough-Space (accumulator)
        - angles array
        - distances array
    """
    min_len_px = max(1, int(round(min_line_length_m / cell_size)))
    max_gap_px = max(0, int(round(max_line_gap_m / cell_size)))

    # Standard Hough Transform für Hough Space Visualization
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 180, endpoint=False)
    h, theta, d = hough_line(binary, theta=tested_angles)

    # Probabilistic Hough für Liniensegmente
    lines = probabilistic_hough_line(
        binary,
        threshold=10,  # Abstimmbar; höher = weniger Kandidaten
        line_length=min_len_px,  # Mindestlänge in Pixeln
        line_gap=max_gap_px,  # maximaler Spalt in Pixeln
    )

    return lines, h, theta, d


def pixel_to_world(
    segment_px: Tuple[Tuple[float, float], Tuple[float, float]],
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    z_value: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Wandelt ein Segment in Pixelkoordinaten (x=Spalte, y=Zeile) ins Weltkoordinaten-System um.
    Nutzen den Zellenmittelpunkt: x = x_edges[col].., y = y_edges[row]..
    """
    (x0p, y0p), (x1p, y1p) = segment_px

    # Pixel -> Zellmitte in Weltkoordinaten
    def px_to_world(xp, yp):
        # xp ist Spaltenindex (x), yp ist Zeilenindex (y)
        # x_edges: Länge nx+1, y_edges: Länge ny+1
        # Zellmitte = edges[idx] + 0.5 * cell_size
        # Achtung: xp/yp können float sein; hier runden wir auf int für die Zellenadresse
        xi = int(round(xp))
        yi = int(round(yp))
        xi = np.clip(xi, 0, len(x_edges) - 2)
        yi = np.clip(yi, 0, len(y_edges) - 2)
        x = x_edges[xi] + 0.5 * (x_edges[xi + 1] - x_edges[xi])
        y = y_edges[yi] + 0.5 * (y_edges[yi + 1] - y_edges[yi])
        return (float(x), float(y), float(z_value))

    return px_to_world(x0p, y0p), px_to_world(x1p, y1p)


def process_single_slice(
    xyz: np.ndarray,
    z_center: float,
    zmin: float,
    zmax: float,
    args,
    slice_idx: int,
    slice_in_range: bool,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Verarbeitet einen einzelnen Z-Slice und gibt die gefundenen Liniensegmente zurück.
    """
    sliced = slice_by_z(xyz, zmin, zmax)

    if sliced.shape[0] == 0:
        print(f"Slice {slice_idx}: Keine Punkte im Z-Bereich {zmin:.3f} .. {zmax:.3f}")
        return []

    print(f"Slice {slice_idx}: {sliced.shape[0]} Punkte bei Z={z_center:.3f}")

    xy = sliced[:, :2]
    H, (y_edges, x_edges) = rasterize_xy(xy, cell_size=args.cell_size)

    binary = make_binary_from_counts(
        H,
        min_count=args.min_count,
        use_canny=args.use_canny,
        canny_sigma=args.canny_sigma,
    )

    seg_px, hough_space, theta, d = hough_segments(
        binary,
        cell_size=args.cell_size,
        min_line_length_m=args.min_line_length_m,
        max_line_gap_m=args.max_line_gap_m,
    )

    if len(seg_px) == 0:
        print(f"  → Keine Linien gefunden")
        return []

    # optional: nur Top-K längste Segmente pro Slice
    if args.top_k_per_slice and args.top_k_per_slice > 0:
        # Länge in Pixel, dann sortieren
        lens = [math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]) for s in seg_px]
        order = np.argsort(lens)[::-1][: args.top_k_per_slice]
        seg_px = [seg_px[i] for i in order]

    segments_world = [pixel_to_world(seg, y_edges, x_edges, z_center) for seg in seg_px]
    print(f"  → {len(segments_world)} Linien gefunden")

    # Speichere Bilder und OBJ für diesen Slice wenn gewünscht
    if args.save_images and slice_in_range:
        save_slice_images(
            binary,
            seg_px,
            hough_space,
            theta,
            d,
            y_edges,
            x_edges,
            slice_idx,
            z_center,
            args.image_output_dir,
        )

        # Speichere OBJ für diesen Slice
        if len(segments_world) > 0:
            save_slice_obj(segments_world, slice_idx, z_center, args.image_output_dir)

        # Speichere LAS für diesen Slice
        save_slice_las(sliced, slice_idx, z_center, args.image_output_dir)

    return segments_world


def write_obj_lines(
    segments_world: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    out_path: str,
) -> None:
    """
    Schreibt Polylinien als OBJ:
        v x y z
        v x y z
        l i j
    CloudCompare kann diese Linien laden.
    """
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# OBJ generated by slice_hough_obj.py\n")
        f.write(f"# Total lines: {len(segments_world)}\n")
        vert_idx = 1
        for p0, p1 in segments_world:
            f.write(f"v {p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f}\n")
            f.write(f"v {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
            f.write(f"l {vert_idx} {vert_idx+1}\n")
            vert_idx += 2


def segments_to_pipes_format(
    segments_world: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
) -> List[dict]:
    """
    Konvertiert Hough-Segmente ins found_pipes Format für das Clustering.
    """
    pipes = []
    for i, (p0, p1) in enumerate(segments_world):
        # Berechne eine einfache "Fitness" basierend auf der Segmentlänge
        length = math.sqrt(
            (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
        )

        pipe = {
            "pipe_number": i + 1,
            "individual_0": -1,  # Nicht relevant für Hough-Segmente
            "individual_1": -1,  # Nicht relevant für Hough-Segmente
            "fitness": length,  # Verwende Länge als Fitness-Ersatz
            "p1_x": float(p0[0]),
            "p1_y": float(p0[1]),
            "p1_z": float(p0[2]),
            "p2_x": float(p1[0]),
            "p2_y": float(p1[1]),
            "p2_z": float(p1[2]),
        }
        pipes.append(pipe)

    return pipes


def pipes_format_to_segments(
    pipes: List[dict],
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Konvertiert das found_pipes Format zurück zu Segmenten für OBJ-Export.
    """
    segments = []
    for pipe in pipes:
        p0 = (pipe["p1_x"], pipe["p1_y"], pipe["p1_z"])
        p1 = (pipe["p2_x"], pipe["p2_y"], pipe["p2_z"])
        segments.append((p0, p1))

    return segments


def main():
    startTime = time.time()

    ap = argparse.ArgumentParser(
        description="Z-Slice Hough-Liniendetektion aus LAS → OBJ (CloudCompare)"
    )
    ap.add_argument("--input", required=True, help="Pfad zur LAS/LAZ-Datei")
    ap.add_argument(
        "--thickness",
        type=float,
        default=0.1,
        help="Dicke der Z-Slices (Meter)",
    )
    ap.add_argument(
        "--cell-size",
        type=float,
        default=0.02,
        help="Raster-Zellgröße in m (Default: 0.05)",
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Schwellwert für Counts (ohne Canny)",
    )
    ap.add_argument(
        "--use-canny",
        action="store_true",
        default=True,
        help="Kanten via Canny (empfohlen)",
    )
    ap.add_argument(
        "--canny-sigma",
        type=float,
        default=1.6,
        help="Canny Sigma (bei --use-canny)",
    )
    ap.add_argument(
        "--min-line-length-m",
        type=float,
        default=1.5,
        help="Mindest-Linienlänge (Meter)",
    )
    ap.add_argument(
        "--max-line-gap-m",
        type=float,
        default=0.5,
        help="Max. Lückenspanne zwischen Segmenten (Meter)",
    )
    ap.add_argument(
        "--top-k-per-slice",
        type=int,
        default=0,
        help="Nur die längsten K Segmente pro Slice behalten (0 = alle)",
    )
    ap.add_argument(
        "--top-k-total",
        type=int,
        default=0,
        help="Nur die längsten K Segmente insgesamt behalten (0 = alle)",
    )
    ap.add_argument(
        "--clustering",
        action="store_true",
        default=False,
        help="Clustering via PipeCluster",
    )
    ap.add_argument(
        "--output",
        default="houghOutput.obj",
        help="Ausgabe-OBJ",
    )
    ap.add_argument(
        "--save-images",
        action="store_true",
        default=True,
        help="Speichere Bilder für jeden Slice (Binärbild + Hough-Raum)",
    )
    ap.add_argument(
        "--save-slices-in-range",
        type=str,
        default="17;23",
        help="Nur Slices in diesem Z-Bereich speichern (min;max), z.B. 17;19",
    )
    ap.add_argument(
        "--image-output-dir",
        default="slice_images",
        help="Ausgabeordner für Slice-Bilder",
    )
    args = ap.parse_args()

    print(f"Lade Punktwolke: {args.input}")
    xyz = load_las_xyz(args.input)
    if xyz.size == 0:
        print("Leere Punktwolke.", file=sys.stderr)
        sys.exit(1)

    print(f"Punktwolke geladen: {xyz.shape[0]} Punkte")

    # Berechne alle Z-Slices
    slices = get_z_slices(xyz, args.thickness)

    # Verarbeite alle Slices
    all_segments = []
    total_processed = 0

    minSliceId = int(args.save_slices_in_range.split(";")[0])
    maxSliceId = int(args.save_slices_in_range.split(";")[1])
    print(f"Speichere nur Slices im Bereich {minSliceId} .. {maxSliceId}")

    if os.path.exists(args.image_output_dir):
        # Lösche spezifische Unterordner
        subdirs_to_clean = ["hough", "lines", "obj"]
        for subdir in subdirs_to_clean:
            subdir_path = os.path.join(args.image_output_dir, subdir)
            if os.path.exists(subdir_path):
                shutil.rmtree(subdir_path)
                print(f"Verzeichnis {subdir_path} geleert")

        # Erstelle alle Ordner wieder neu
        for subdir in subdirs_to_clean:
            subdir_path = os.path.join(args.image_output_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Verzeichnis {subdir_path} neu erstellt")

    for i, (z_center, zmin, zmax) in enumerate(slices):
        slice_in_range = i >= minSliceId and i <= maxSliceId
        print(f"\nVerarbeite Slice {i}: {minSliceId} .. {maxSliceId}, Z={z_center:.3f}")
        segments = process_single_slice(
            xyz, z_center, zmin, zmax, args, i, slice_in_range
        )
        all_segments.extend(segments)
        total_processed += 1

        if total_processed % 10 == 0:
            print(f"Verarbeitet: {total_processed}/{len(slices)} Slices")

    if len(all_segments) == 0:
        print("Keine Linien in der gesamten Punktwolke gefunden.", file=sys.stderr)
        # trotzdem eine leere OBJ schreiben, damit der Pipeline-Schritt nicht bricht
        open(args.output, "w").write("# Empty OBJ (no lines detected)\n")
        sys.exit(0)

    if args.clustering:
        print(f"Vor Clustering: {len(all_segments)} Segmente")

        # --- Clustering der Segmente ---
        # Konvertiere Segmente ins Pipes-Format
        pipes_format = segments_to_pipes_format(all_segments)

        # Importiere und verwende das Clustering
        try:
            clustered_pipes = clean_pipes(pipes_format)
            print(f"Nach Clustering: {len(clustered_pipes)} optimierte Segmente")

            # Konvertiere zurück zu Segmenten
            all_segments = pipes_format_to_segments(clustered_pipes)

        except ImportError:
            print("PipeCluster nicht verfügbar, überspringe Clustering")
        except Exception as e:
            print(f"Clustering fehlgeschlagen: {e}, verwende ursprüngliche Segmente")

    # Optional: nur Top-K längste Segmente insgesamt (nach Clustering)
    if args.top_k_total and args.top_k_total > 0:
        print(f"Reduziere auf die {args.top_k_total} längsten Segmente...")
        # Länge in 3D berechnen
        lens = [
            math.sqrt(
                (s[1][0] - s[0][0]) ** 2
                + (s[1][1] - s[0][1]) ** 2
                + (s[1][2] - s[0][2]) ** 2
            )
            for s in all_segments
        ]
        order = np.argsort(lens)[::-1][: args.top_k_total]
        all_segments = [all_segments[i] for i in order]

    if not args.save_images:
        write_obj_lines(
            all_segments,
            f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.output}",
        )

    print(f"\nFertig!")
    print(f"Verarbeitete Slices: {len(slices)}")
    print(f"Gefundene Linien gesamt: {len(all_segments)}")
    print(f"Ausgabedatei: {args.output}")
    if args.save_images:
        print(f"Slice-Bilder gespeichert in: {args.image_output_dir}")
    endTime = time.time()
    print(f"Benötigte Zeit: {endTime - startTime:.2f} Sekunden")


if __name__ == "__main__":
    main()
