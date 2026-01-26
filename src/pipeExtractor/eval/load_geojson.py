import json
from typing import Iterable, List

import numpy as np
from custom_types import (
    ListOfPoint3DArrays,
    PipeComponent,
    PipeComponentArray,
    Point3DArray,
    Segment3D,
    Segment3DArray,
)


def _as_point3d(coord: Iterable[float]) -> np.ndarray:
    arr = np.asarray(coord, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Ungültige Koordinate: {coord}")
    if arr.size == 3:
        return arr
    if arr.size == 2:
        return np.array([arr[0], arr[1], np.nan], dtype=float)
    raise ValueError(f"Koordinatenlänge wird nicht unterstützt: {coord}")


def _safe_iterable(value: Iterable | None) -> list:
    return list(value) if value is not None else []


def load_geojson(
    file_path: str,
) -> tuple[Segment3DArray, PipeComponentArray | None, ListOfPoint3DArrays]:
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
        - Segment3DArray: An array of 3D segments/pipes (each segment is a numpy array of shape (2, 3)).
        - PipeComponentArray | None: A list of pipe components (each component is a tuple of three 3D points).
        - ListOfPoint3DArrays: A list of arrays of 3D points representing a pipe as a chain of sample points (each array represents a chain of points).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pipes: Segment3DArray = []
    pipes_fromChains: ListOfPoint3DArrays = []
    pipe_components: PipeComponentArray = []

    for feature in data.get("features", []):
        if feature.get("type") != "Feature":
            continue

        geometry = feature.get("geometry") or {}
        match geometry.get("type"):
            case "Point":
                coord = _safe_iterable(geometry.get("coordinates"))[:3]
                point3d = _as_point3d(coord)

                # Extrahiere bbox falls vorhanden, sonst verwende den Punkt selbst
                bbox = feature.get("bbox")
                if bbox and len(bbox) == 6:
                    bbox_min = np.array(bbox[:3], dtype=float)
                    bbox_max = np.array(bbox[3:], dtype=float)
                else:
                    bbox_min = bbox_max = point3d

                pipe_component = (bbox_min, bbox_max, point3d)
                pipe_components.append(pipe_component)
            case "LineString":
                if feature.get("id") and "chain_" in feature.get("id"):
                    continue

                coords = _safe_iterable(geometry.get("coordinates"))[:2]
                if len(coords) == 2:
                    start, end = (_as_point3d(c) for c in coords)
                    pipes.append(np.vstack([start, end]))

                if len(coords) > 2:
                    for i in range(len(coords) - 1):
                        start, end = (_as_point3d(c) for c in coords[i : i + 2])
                        pipes_fromChains.append(np.vstack([start, end]))

    return pipes, pipe_components, pipes_fromChains
