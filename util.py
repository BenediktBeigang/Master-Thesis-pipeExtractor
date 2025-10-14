import os
import shutil
import numpy as np
import laspy
import json
from numpy.typing import NDArray


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
    raise RuntimeError("LAS enthält kein Feld 'Classification' / 'classification'.")


def load_las(
    path: str,
    ignoreZ: bool = False,
    filterClass: int | None = None,
) -> np.ndarray:
    """
    Loads a LAS/LAZ file and returns a Nx3 numpy array of points.

    Parameters
    ----------
    path : str
        Path to the LAS/LAZ file.
    ignoreZ : bool, optional
        If True, only X and Y coordinates are returned (Z is ignored). Default is False.
    filterClass : int, optional
        If provided, only points with the specified classification are returned. Default is None.

    Returns
    -------
    Point3DArray or Point2DArray
        Nx3 array of points (X, Y, Z) or Nx2 if ignoreZ is True.
    """
    try:
        print(f"Load LAS-File: {path}")
        las = laspy.read(path)

        # Bool-Maske für Klassifikation (falls gesetzt)
        mask: NDArray[np.bool_] | None = None
        if filterClass is not None:
            cls = get_classification_array(las)
            mask = cls == filterClass

        # Sicherstellen, dass wir echte np.ndarray[float64] erhalten
        x_all: NDArray[np.float64] = np.asarray(las.x, dtype=np.float64)
        y_all: NDArray[np.float64] = np.asarray(las.y, dtype=np.float64)

        if ignoreZ:
            x = x_all[mask] if mask is not None else x_all
            y = y_all[mask] if mask is not None else y_all
            return np.column_stack((x, y))

        z_all: NDArray[np.float64] = np.asarray(las.z, dtype=np.float64)
        x = x_all[mask] if mask is not None else x_all
        y = y_all[mask] if mask is not None else y_all
        z = z_all[mask] if mask is not None else z_all

        return np.column_stack((x, y, z))

    except Exception as e:
        raise RuntimeError(f"Konnte LAS nicht laden: {e}")


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
