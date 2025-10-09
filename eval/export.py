def write_segments_to_obj(
    missed_segments: list,
    false_positive_segments: list,
    pointcloudName: str,
):
    """
    Schreibt missed und false-positive Segmente in eine OBJ-Datei.

    Args:
        missed_segments: Liste von Ground Truth Segmenten die nicht erkannt wurden
        false_positive_segments: Liste von Detection Segmenten die keine Ground Truth entsprechen
        output_path: Pfad zur Ausgabe-OBJ-Datei
    """
    with open(f"./output/obj/{pointcloudName}_missed.obj", "w") as f:
        f.write("# OBJ file with missed and false positive segments\n")
        f.write("# Red lines: Missed segments (Ground Truth not detected)\n")
        f.write(
            "# Blue lines: False positive segments (Detections without Ground Truth)\n\n"
        )

        vertex_index = 1

        # Schreibe missed segments (rot)
        if missed_segments:
            f.write("# Missed segments (red)\n")

            for segment in missed_segments:
                p1, p2 = segment
                # Schreibe Vertices
                f.write(f"v {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
                f.write(f"v {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
                # Schreibe Linie
                f.write(f"l {vertex_index} {vertex_index + 1}\n")
                vertex_index += 2

            f.write("\n")

    with open(f"./output/obj/{pointcloudName}_falsePositive.obj", "w") as f:
        f.write("# OBJ file with missed and false positive segments\n")
        f.write("# Red lines: Missed segments (Ground Truth not detected)\n")
        f.write(
            "# Blue lines: False positive segments (Detections without Ground Truth)\n\n"
        )

        vertex_index = 1

        # Schreibe false positive segments (blau)
        if false_positive_segments:
            f.write("# False positive segments (blue)\n")

            for segment in false_positive_segments:
                p1, p2 = segment
                # Schreibe Vertices
                f.write(f"v {p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}\n")
                f.write(f"v {p2[0]:.6f} {p2[1]:.6f} {p2[2]:.6f}\n")
                # Schreibe Linie
                f.write(f"l {vertex_index} {vertex_index + 1}\n")
                vertex_index += 2


def export_segments_to_obj(segments, output_path):
    """
    Exportiert eine Liste von Segmenten als OBJ-Datei.

    Args:
        segments (list): Liste von Segmenten, wobei jedes Segment ein Tupel aus zwei Punkten ist.
                        Jeder Punkt ist ein Tupel (x, y, z) oder (x, y).
        output_path (str): Pfad zur Ausgabe-OBJ-Datei
    """
    with open(output_path, "w") as f:
        # OBJ Header
        f.write("# OBJ file generated from GeoJSON segments\n")
        f.write(
            "# Each line segment is represented as two vertices connected by a line\n\n"
        )

        vertex_index = 1

        # Schreibe alle Vertices und Lines
        for segment in segments:
            start_point, end_point = segment

            # Stelle sicher, dass Punkte 3D sind (füge z=0 hinzu falls nötig)
            if len(start_point) == 2:
                start_point = (start_point[0], start_point[1], 0.0)
            if len(end_point) == 2:
                end_point = (end_point[0], end_point[1], 0.0)

            # Schreibe Vertices
            f.write(f"v {start_point[0]} {start_point[1]} {start_point[2]}\n")
            f.write(f"v {end_point[0]} {end_point[1]} {end_point[2]}\n")

            # Schreibe Line (verbindet die beiden gerade hinzugefügten Vertices)
            f.write(f"l {vertex_index} {vertex_index + 1}\n")

            vertex_index += 2

        f.write(f"\n# Total segments: {len(segments)}\n")
