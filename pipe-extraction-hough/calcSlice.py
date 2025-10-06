import math
from typing import List, Tuple
import numpy as np

from mapping import pixel_to_world

from skimage.feature import canny
from skimage.filters import gaussian
from skimage.transform import probabilistic_hough_line
from custom_types import Segment2DArray, Segment3DArray


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
    xy: np.ndarray, cell_size: float
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Rasterisiert XY-Punkte in ein 2D-Count-Grid via histogram2d.
    Rückgabe:
        H: (ny, nx) array mit Zählwerten
        edges: (y_edges, x_edges)
    """
    xs = xy[:, 0]
    ys = xy[:, 1]

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()

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


def make_binary_from_counts(H: np.ndarray, canny_sigma: float = 1.2) -> np.ndarray:
    """
    Erzeugt ein Binärbild für Hough.
    - Entweder simple Count-Schwelle (H >= min_count),
    - oder Canny auf normalisierten Counts (robuster bei ungleichmäßiger Dichte).
    """
    if H.max() > 0:
        Hs = gaussian(H / (H.max() + 1e-9), sigma=1.0, preserve_range=True)
    else:
        Hs = H
    edges = canny(Hs, sigma=canny_sigma)
    return edges.astype(bool)


def hough_segments(
    binary: np.ndarray,
    cell_size: float,
    min_line_length_m: float = 0.4,
    max_line_gap_m: float = 0.10,
) -> Segment2DArray:
    """
    Detects line segments in a binary image using probabilistic Hough transform.

    ## How it works
    This function applies the probabilistic Hough line detection algorithm to find
    straight line segments in a binary image. It converts metric distance parameters
    to pixel coordinates based on the given cell size.

    Args:
        binary (np.ndarray): Binary input image where line detection is performed.
                           Should be a 2D boolean or binary array.
        cell_size (float): Size of each pixel/cell in meters. Used to convert
                         metric parameters to pixel coordinates.
        min_line_length_m (float, optional): Minimum line length in meters that
                                           will be detected. Defaults to 0.4m.
        max_line_gap_m (float, optional): Maximum gap in meters between line
                                        segments that can be linked together.
                                        Defaults to 0.10m.

    Returns:
        Segment2DArray: Array of detected line segments, each defined by start and end points in pixel coordinates.
    """
    min_len_px = max(1, int(round(min_line_length_m / cell_size)))
    max_gap_px = max(0, int(round(max_line_gap_m / cell_size)))

    lines = probabilistic_hough_line(
        binary,
        threshold=10,  # higher = less candidates
        line_length=min_len_px,  # min length of line in pixels
        line_gap=max_gap_px,  # max gap between segments to link them
        rng=42,  # for reproducibility
    )

    return lines


def process_single_slice(
    xyz: np.ndarray,
    z_center: float,
    zmin: float,
    zmax: float,
    args,
    slice_idx: int,
) -> Segment3DArray:
    """
    Processes a single horizontal slice of point cloud data to detect line segments.

    This function extracts points within a specific Z-range, rasterizes them into a 2D grid,
    applies edge detection, and uses probabilistic Hough transform to find line segments
    that potentially represent pipes or other linear structures.

    Args:
        xyz (np.ndarray): Input point cloud data as Nx3 array (x, y, z coordinates)
        z_center (float): Center Z-coordinate of the slice for 3D reconstruction
        zmin (float): Minimum Z-coordinate of the slice boundary
        zmax (float): Maximum Z-coordinate of the slice boundary
        args: Configuration object containing processing parameters:
            - cell_size: Raster cell size in meters
            - canny_sigma: Sigma parameter for Canny edge detection
            - min_line_length_m: Minimum line length in meters for Hough detection
            - max_line_gap_m: Maximum gap in meters between line segments
        slice_idx (int): Index of the current slice (for debugging/logging purposes)

    Returns:
        Segment3DArray: List of 3D line segments found in this slice. Each segment
                       is defined by start and end points in world coordinates.
                       Returns empty list if no segments are detected.
    """
    sliced = slice_by_z(xyz, zmin, zmax)

    if sliced.shape[0] == 0:
        return []

    xy = sliced[:, :2]
    H, (y_edges, x_edges) = rasterize_xy(xy, cell_size=args.cell_size)

    binary = make_binary_from_counts(
        H,
        canny_sigma=args.canny_sigma,
    )

    seg_px = hough_segments(
        binary,
        cell_size=args.cell_size,
        min_line_length_m=args.min_line_length_m,
        max_line_gap_m=args.max_line_gap_m,
    )

    return (
        []
        if len(seg_px) == 0
        else [pixel_to_world(seg, y_edges, x_edges, z_center) for seg in seg_px]
    )
