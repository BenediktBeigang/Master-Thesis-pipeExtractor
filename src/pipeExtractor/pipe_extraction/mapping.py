import math
from typing import List
import numpy as np


def pixel_to_world(
    segment_px: tuple[tuple[float, float], tuple[float, float]],
    y_edges: np.ndarray,
    x_edges: np.ndarray,
    z_value: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Wandelt ein Segment in Pixelkoordinaten (x=Spalte, y=Zeile) ins Weltkoordinaten-System um.
    Nutzen den Zellenmittelpunkt: x = x_edges[col].., y = y_edges[row]..
    """
    (x0p, y0p), (x1p, y1p) = segment_px

    # Pixel -> Zellmitte in Weltkoordinaten
    def px_to_world(xp, yp):
        # xp ist Spaltenindex (x), yp ist Zeilenindex (y)
        # x_edges: Länge nx+1, y_edges: Länge ny+1
        # Zellmitte = edges[idx] + 0.5 * cell_size
        # Achtung: xp/yp können float sein; hier runden wir auf int für die Zellenadresse
        xi = int(round(xp))
        yi = int(round(yp))
        xi = np.clip(xi, 0, len(x_edges) - 2)
        yi = np.clip(yi, 0, len(y_edges) - 2)
        x = x_edges[xi] + 0.5 * (x_edges[xi + 1] - x_edges[xi])
        y = y_edges[yi] + 0.5 * (y_edges[yi + 1] - y_edges[yi])
        return (float(x), float(y), float(z_value))

    return px_to_world(x0p, y0p), px_to_world(x1p, y1p)


def segments_to_pipes_format(
    segments_world: List[tuple[tuple[float, float, float], tuple[float, float, float]]],
) -> List[dict]:
    """
    Konvertiert Hough-Segmente ins found_pipes Format für das Clustering.
    """
    pipes = []
    for i, (p0, p1) in enumerate(segments_world):
        # Berechne eine einfache "Fitness" basierend auf der Segmentlänge
        length = math.sqrt(
            (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2
        )

        pipe = {
            "pipe_number": i + 1,
            "individual_0": -1,  # Nicht relevant für Hough-Segmente
            "individual_1": -1,  # Nicht relevant für Hough-Segmente
            "fitness": length,  # Verwende Länge als Fitness-Ersatz
            "p1_x": float(p0[0]),
            "p1_y": float(p0[1]),
            "p1_z": float(p0[2]),
            "p2_x": float(p1[0]),
            "p2_y": float(p1[1]),
            "p2_z": float(p1[2]),
        }
        pipes.append(pipe)

    return pipes


def pipes_format_to_segments(
    pipes: List[dict],
) -> List[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """
    Konvertiert das found_pipes Format zurück zu Segmenten für OBJ-Export.
    """
    segments = []
    for pipe in pipes:
        p0 = (pipe["p1_x"], pipe["p1_y"], pipe["p1_z"])
        p1 = (pipe["p2_x"], pipe["p2_y"], pipe["p2_z"])
        segments.append((p0, p1))

    return segments
