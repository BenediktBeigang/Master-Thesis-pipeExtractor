import json
import numpy as np
from typing import Optional
from custom_types import PipeComponentArray, Point3DArray, Segment3DArray


def _bbox_from_min_max(bmin: np.ndarray, bmax: np.ndarray) -> list[float]:
    if bmin.shape[0] == 3 and bmax.shape[0] == 3:
        return [
            float(bmin[0]),
            float(bmin[1]),
            float(bmin[2]),
            float(bmax[0]),
            float(bmax[1]),
            float(bmax[2]),
        ]
    return [float(bmin[0]), float(bmin[1]), float(bmax[0]), float(bmax[1])]


def export_geojson(
    pipes: Segment3DArray | None,
    snapped_chains: list[Point3DArray] | None,
    pipeComponents: PipeComponentArray | None,
    pointscloudName: str,
    epsg: Optional[int] = None,
) -> None:
    if pipes is None and pipeComponents is None:
        return print("WARNING: No pipes or pipe-components to export in a geojson.")

    features: list[dict] = []

    # --- Pipes ---
    if pipes is not None:
        P = np.asarray(pipes)
        for i, seg in enumerate(P):
            a, b = seg
            coords = [
                [float(a[0]), float(a[1]), float(a[2])],
                [float(b[0]), float(b[1]), float(b[2])],
            ]
            features.append(
                {
                    "type": "Feature",
                    "id": f"pipe_{int(i)}",
                    "geometry": {"type": "LineString", "coordinates": coords},
                    "properties": {
                        "feature_type": "pipe",
                        **({"proj:epsg": int(epsg)} if epsg else {}),
                    },
                }
            )

    # --- Snapped Chains ---
    if snapped_chains is not None:
        for i, chain in enumerate(snapped_chains):
            coords = np.asarray(chain, dtype=float)
            if coords.ndim != 2:
                raise ValueError("Snapped chain muss 2D sein.")
            if coords.shape[1] == 3:
                coords_iter = coords
            elif coords.shape[0] == 3:
                coords_iter = coords.T
            else:
                raise ValueError("Snapped chain muss 3D-Koordinaten haben.")
            features.append(
                {
                    "type": "Feature",
                    "id": f"chain_{i}",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [float(x), float(y), float(z)] for x, y, z in coords_iter
                        ],
                    },
                    "properties": {
                        "feature_type": "snapped_chain",
                        **({"proj:epsg": int(epsg)} if epsg else {}),
                    },
                }
            )

    # --- Components ---
    if pipeComponents is not None:
        for j, (bmin, bmax, mean) in enumerate(pipeComponents):
            bmin = np.asarray(bmin).astype(float).ravel()
            bmax = np.asarray(bmax).astype(float).ravel()
            mean = np.asarray(mean).astype(float).ravel()

            geom_coords = [float(mean[0]), float(mean[1])] + ([float(mean[2])])
            feature = {
                "type": "Feature",
                "id": f"comp_{j}",
                "bbox": _bbox_from_min_max(bmin, bmax),
                "geometry": {"type": "Point", "coordinates": geom_coords},
                "properties": {
                    "feature_type": "component",
                    **({"proj:epsg": int(epsg)} if epsg else {}),
                },
            }
            features.append(feature)

    fc = {"type": "FeatureCollection", "name": pointscloudName, "features": features}

    with open(pointscloudName, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2, ensure_ascii=False)
