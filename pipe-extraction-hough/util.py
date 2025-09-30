import os
import shutil
import numpy as np
import laspy


def load_las(path: str, ignoreZ: bool = False) -> np.ndarray:
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
