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
    Compute all Z-slice parameters (z_center, zmin, zmax) based on the point cloud bounding box.

    This function determines the Z extents of the input point cloud and partitions that
    range into horizontal slices of the given thickness. The first slice starts at the
    minimum Z, the last slice ends at the maximum Z. Slice centers are returned along
    with their min/max bounds.

    Parameters
    ----------
    xyz : np.ndarray
        Nx3 array of point coordinates (x, y, z).
    thickness : float
        Desired slice thickness in the same units as the point cloud (e.g. meters).

    Returns
    -------
    List[Tuple[float, float, float]]
        List of tuples (z_center, zmin, zmax) for each slice.
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
    """Extract points within a Z range.

    Parameters
    ----------
    xyz : np.ndarray
        Nx3 array of points (x, y, z).
    zmin : float
        Lower Z bound (inclusive).
    zmax : float
        Upper Z bound (inclusive).

    Returns
    -------
    np.ndarray
        Subset of input points whose z coordinate lies between zmin and zmax.
    """
    mask = (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax)
    return xyz[mask]


def rasterize_xy(
    xy: np.ndarray, cell_size: float
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Rasterize XY points into a 2D count grid using numpy histogram2d.

    The function computes a regular grid aligned with the XY extents of the provided
    points and counts how many points fall into each cell.

    Parameters
    ----------
    xy : np.ndarray
        Nx2 array of XY coordinates.
    cell_size : float
        Edge length of a single grid cell in the same units as xy (e.g. meters).

    Returns
    -------
    Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        H : 2D array of shape (ny, nx) with per-cell counts (float64).
        edges : Tuple of (y_edges, x_edges) returned by numpy.histogram2d.
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


def make_binary_from_counts(H: np.ndarray, canny_sigma: float) -> np.ndarray:
    """
    Smooth count grid and extract edges as a binary image.

    This function normalizes the count grid, applies a Gaussian filter for smoothing,
    and then runs the Canny edge detector to produce a boolean edge image suitable
    for Hough line detection.

    Parameters
    ----------
    H : np.ndarray
        2D array of per-cell point counts.
    canny_sigma : float
        Gaussian sigma parameter passed to skimage.feature.canny.

    Returns
    -------
    np.ndarray
        Boolean 2D array where True indicates an edge.
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
    threshold: int,
    min_line_length_m: float,
    max_line_gap_m: float,
) -> Segment2DArray:
    """
    Detects line segments in a binary image using probabilistic Hough transform.

    ## How it works
    This function applies the probabilistic Hough line detection algorithm to find
    straight line segments in a binary image. It converts metric distance parameters
    to pixel coordinates based on the given cell size.

    Parameters
    ----------
    binary : np.ndarray
        Binary input image where line detection is performed.
        Should be a 2D boolean or binary array.
    cell_size : float
        Size of each pixel/cell in meters. Used to convert
        metric parameters to pixel coordinates.
    min_line_length_m : float
        Minimum line length in meters that will be detected. Defaults to 0.4m.
    max_line_gap_m : float
        Maximum gap in meters between line segments that can be linked together.
        Defaults to 0.10m.

    Returns
    -------
        Segment2DArray: Array of detected line segments, each defined by start and end points in pixel coordinates.
    """
    min_len_px = max(1, int(round(min_line_length_m / cell_size)))
    max_gap_px = max(0, int(round(max_line_gap_m / cell_size)))

    lines = probabilistic_hough_line(
        binary,
        threshold=threshold,  # higher = less candidates
        line_length=min_len_px,  # min length of line in pixels
        line_gap=max_gap_px,  # max gap between segments to link them
        rng=42,  # for reproducibility
    )

    return lines


def find_lines_in_slice(
    xyz: np.ndarray,
    z_center: float,
    zmin: float,
    zmax: float,
    args: dict,
    slice_idx: int,
) -> Segment3DArray:
    """
    Processes a single horizontal slice of point cloud data to detect line segments.

    ## How it works
    This function extracts points within a specific Z-range, rasterizes them into a 2D grid,
    applies edge detection, and uses probabilistic Hough transform to find line segments
    that potentially represent pipes or other linear structures.

    Parameters
    ----------
    xyz : np.ndarray
        Input point cloud data as Nx3 array (x, y, z coordinates)
    z_center : float
        Center Z-coordinate of the slice for 3D reconstruction
    zmin : float
        Minimum Z-coordinate of the slice boundary
    zmax : float
        Maximum Z-coordinate of the slice boundary
    args : dict
        Configuration object containing processing Hough parameters
    slice_idx : int
        Index of the current slice (for debugging/logging purposes)

    Returns
    -------
        Segment3DArray: List of 3D line segments found in this slice. Each segment
                       is defined by start and end points in world coordinates.
                       Returns empty list if no segments are detected.
    """
    sliced = slice_by_z(xyz, zmin, zmax)

    if sliced.shape[0] == 0:
        return []

    xy = sliced[:, :2]
    H, (y_edges, x_edges) = rasterize_xy(xy, cell_size=args["cell_size"])

    binary = make_binary_from_counts(
        H,
        canny_sigma=args["canny_sigma"],
    )

    seg_px = hough_segments(
        binary,
        cell_size=args["cell_size"],
        threshold=args["threshold"],
        min_line_length_m=args["min_line_length"],
        max_line_gap_m=args["max_line_gap"],
    )

    return (
        []
        if len(seg_px) == 0
        else [pixel_to_world(seg, y_edges, x_edges, z_center) for seg in seg_px]
    )
