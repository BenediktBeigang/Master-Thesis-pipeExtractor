import math
from typing import List, Tuple
import numpy as np

from export import save_slice_las, save_slice_obj
from mapping import pixel_to_world

from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import probabilistic_hough_line, hough_line


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

    segments_world = [pixel_to_world(seg, y_edges, x_edges, z_center) for seg in seg_px]
    print(f"  → {len(segments_world)} Linien gefunden")

    # Speichere Bilder und OBJ für diesen Slice wenn gewünscht
    # if slice_in_range and len(segments_world) > 0:
    #     save_slice_obj(segments_world, slice_idx, z_center, "./output/obj/")
    #     save_slice_las(sliced, slice_idx, "./output/las/")

    return segments_world
