import os
import shutil
import numpy as np
import laspy
import json


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


def load_las(path: str, ignoreZ: bool = False, filterClass: int = None) -> np.ndarray:
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

        mask = (
            None
            if filterClass is None
            else filterClass == get_classification_array(las)
        )

        # Hole Koordinaten ggf. gefiltert
        x = las.x[mask] if mask is not None else las.x
        y = las.y[mask] if mask is not None else las.y

        if ignoreZ:
            return np.vstack((x, y)).T.astype(np.float64)

        z = las.z[mask] if mask is not None else las.z
        return np.vstack((x, y, z)).T.astype(np.float64)

    except Exception as e:
        raise RuntimeError(f"Konnte LAS nicht laden: {e}")


def prepare_output_directory(output_dir: str):
    subdirs_to_clean = ["obj", "geojson"]
    if os.path.exists(output_dir):
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
