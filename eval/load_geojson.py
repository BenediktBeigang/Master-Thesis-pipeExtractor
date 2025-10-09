import json
from typing import Iterable, List

import numpy as np
from custom_types import PipeComponent, Point3D, Segment3D


def _as_point3d(coord: Iterable[float]) -> np.ndarray:
    arr = np.asarray(coord, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Ungültige Koordinate: {coord}")
    if arr.size == 3:
        return arr
    if arr.size == 2:
        return np.array([arr[0], arr[1], np.nan], dtype=float)
    raise ValueError(f"Koordinatenlänge wird nicht unterstützt: {coord}")


def _safe_iterable(value: Iterable) -> list:
    return list(value) if value is not None else []


def load_geojson(file_path: str) -> tuple[List[Segment3D], list[Point3D]]:
    """
    Loads a GeoJSON file and extracts LineString geometries as 3D segments and Point geometries as 3D points.

    Parameters
    ----------
    file_path : str
        Path to the GeoJSON file.

    Returns
    -------
    tuple
        A tuple containing:
        - List[Segment3D]: An array of 3D segments (each segment is a numpy array of shape (2, 3)).
        - A list of 3D points (each point is a numpy array of shape (3,)).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pipes: List[Segment3D] = []
    pipe_components: List[PipeComponent] = []

    for feature in data.get("features", []):
        if feature.get("type") != "Feature":
            continue

        geometry = feature.get("geometry") or {}
        match geometry.get("type"):
            case "Point":
                coord = _safe_iterable(geometry.get("coordinates"))[:3]
                pipe_components.append(coord)
            case "LineString":
                coords = _safe_iterable(geometry.get("coordinates"))[:2]
                if len(coords) == 2:
                    start, end = (_as_point3d(c) for c in coords)
                    pipes.append(np.vstack([start, end]))

    return pipes, pipe_components
