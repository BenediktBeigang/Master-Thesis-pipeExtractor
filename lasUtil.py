import numpy as np
import laspy


def load_las_xyz(path: str, ignoreZ: bool = False) -> np.ndarray:
    """
    Lädt die XYZ-Koordinaten aus einer LAS-Datei.
    Wenn ignoreZ gesetzt ist, werden nur X und Y geladen (2D).
    """
    try:
        las = laspy.read(path)
        if ignoreZ:
            return np.vstack((las.x, las.y)).T.astype(np.float64)
        return np.vstack((las.x, las.y, las.z)).T.astype(np.float64)
    except Exception as e:
        raise RuntimeError(f"Konnte LAS nicht laden: {e}")
