import os
import shutil
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


def prepare_output_directory(output_dir: str):
    if os.path.exists(output_dir):
        # Lösche spezifische Unterordner
        subdirs_to_clean = ["hough", "lines", "obj", "las"]
        for subdir in subdirs_to_clean:
            subdir_path = os.path.join(output_dir, subdir)
            if os.path.exists(subdir_path):
                shutil.rmtree(subdir_path)
                print(f"Verzeichnis {subdir_path} geleert")

        # Erstelle alle Ordner wieder neu
        for subdir in subdirs_to_clean:
            subdir_path = os.path.join(output_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            print(f"Verzeichnis {subdir_path} neu erstellt")

    print("---------------------------")
    print()
