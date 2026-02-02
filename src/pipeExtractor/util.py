import os
import shutil
import numpy as np
import pdal
import json

PIPE_CLASS = 1
PIPE_COMPONENT_CLASS = 2


def get_classification_array(las):
    cls = getattr(las, "classification", None)
    if cls is not None:
        return cls

    for name in ("classification", "Classification"):
        try:
            return las[name]
        except Exception:
            pass

    dims = getattr(las.point_format, "dimension_names", [])
    for name in ("classification", "Classification"):
        if name in dims:
            return las[name]

    print("LAS does not provide field 'Classification' or 'classification'.")
    return None


def load_las(pc_path: str, ignoreZ: bool = False, sample_radius: float = None):
    """
    Load LAS file.
    Splits into pipes and pipe components.
    Optionally apply PDAL sampling.

    Parameters
    ----------
    pc_path : str
        Path to the input LAS file
    ignoreZ : bool
        Whether to ignore Z coordinates
    sample_radius : float, optional
        Radius for Poisson sampling in meters. If None, no sampling is applied.

    Returns
    -------
    xyz_pipes : np.ndarray
        Points classified as pipes (Class 1)
    xyz_pipeComponents : np.ndarray
        Points classified as pipe components (Class 2)
    """
    # PDAL setup
    pdal_pipeline = {"pipeline": [pc_path]}
    if sample_radius is not None:
        print(f"PDAL will sample with point to point distance: {sample_radius} m")
        pdal_pipeline["pipeline"].append(
            {"type": "filters.sample", "radius": sample_radius}
        )
    pipeline = pdal.Pipeline(json.dumps(pdal_pipeline))

    # PDAL execution
    print(f"Read {pc_path} with PDAL...")
    count = pipeline.execute()
    arrays = pipeline.arrays
    points = arrays[0]

    # Split by classification
    print("Splitting points by classification...")
    xyz = np.column_stack((points["X"], points["Y"], points["Z"]))
    classification = points["Classification"]
    xyz_pipes = xyz[classification == PIPE_CLASS]
    xyz_pipeComponents = xyz[classification == PIPE_COMPONENT_CLASS]

    if ignoreZ:
        xyz_pipes = xyz_pipes[:, :2]
        xyz_pipeComponents = xyz_pipeComponents[:, :2]

    print(f"Loaded {count} points from {pc_path}")
    print(f"  - Pipes (Class 1): {len(xyz_pipes)} points")
    print(f"  - Components (Class 2): {len(xyz_pipeComponents)} points")

    return xyz_pipes, xyz_pipeComponents


def prepare_output_directory(output_dir: str, clean: bool = True):
    subdirs_to_clean = ["obj", "geojson", "metrics", "plots"]
    if os.path.exists(output_dir) and clean:
        # Lösche spezifische Unterordner
        for subdir in subdirs_to_clean:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                shutil.rmtree(subdir_path)
                print(f"Verzeichnis {subdir_path} geleert")

    # Erstelle alle Ordner neu
    for subdir in subdirs_to_clean:
        subdir_path = os.path.join(output_dir, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Verzeichnis {subdir_path} neu erstellt")

    print("---------------------------")
    print()


def load_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Konnte Konfigurationsdatei nicht laden: {e}")


def point_to_line_distance(point, line_start, line_end):
    """
    Berechnet den kürzesten Abstand eines Punktes zu einer Linie (definiert durch zwei Punkte).
    """
    line_vec = line_end - line_start
    line_length_sq = np.dot(line_vec, line_vec)

    if line_length_sq == 0:
        # Linie ist ein Punkt
        return np.linalg.norm(point - line_start)

    # Projektion des Punktes auf die Linie
    t = np.dot(point - line_start, line_vec) / line_length_sq
    t = max(0, min(1, t))  # Begrenze t auf [0,1] für Segment

    projection = line_start + t * line_vec
    return np.linalg.norm(point - projection)


def project_point_to_line(point, line_origin, line_dir):
    """Projiziere `point` auf die parametische Linie (origin + t * dir)."""
    v = line_dir
    ap = point - line_origin
    denom = np.dot(v, v)
    if denom == 0:
        return line_origin.copy()
    t = np.dot(ap, v) / denom
    return line_origin + t * v


def poisson_disk_on_points_xy(xy: np.ndarray, radius: float) -> np.ndarray:
    if xy.size == 0:
        return np.zeros(0, dtype=bool)

    n = xy.shape[0]
    rng = np.random.default_rng(42)
    order = rng.permutation(n)

    # cell size = radius (check 3x3 neighbors)
    cell_size = float(radius)
    mins = xy.min(axis=0)
    cell_coords = np.floor((xy - mins) / cell_size).astype(np.int32)

    grid: dict[tuple[int, int], list[int]] = {}
    kept_mask = np.zeros(n, dtype=bool)
    r2 = radius * radius

    # Localize for speed
    _cell_coords = cell_coords
    _xy = xy
    _grid_get = grid.get
    _grid_setdefault = grid.setdefault

    accepted = 0
    for idx in order:
        cc = (_cell_coords[idx, 0], _cell_coords[idx, 1])
        found = False

        # check neighbor cells (3x3)
        cx, cy = cc
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nb = _grid_get((cx + dx, cy + dy))
                if not nb:
                    continue
                # check distances to already accepted points in that cell
                for a_idx in nb:
                    d2 = (_xy[a_idx, 0] - _xy[idx, 0]) ** 2 + (
                        _xy[a_idx, 1] - _xy[idx, 1]
                    ) ** 2
                    if d2 < r2:
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            kept_mask[idx] = True
            _grid_setdefault(cc, []).append(int(idx))
            accepted += 1

    return kept_mask
