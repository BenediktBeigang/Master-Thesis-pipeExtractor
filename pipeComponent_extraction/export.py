def write_obj_boxes(cluster_boxes, output_path):
    try:
        lines = []
        vert_count = 0
        for label, (mins, maxs) in cluster_boxes.items():
            xmin, ymin, zmin = mins
            xmax, ymax, zmax = maxs
            verts = [
                (xmin, ymin, zmin),  # 0
                (xmax, ymin, zmin),  # 1
                (xmax, ymax, zmin),  # 2
                (xmin, ymax, zmin),  # 3
                (xmin, ymin, zmax),  # 4
                (xmax, ymin, zmax),  # 5
                (xmax, ymax, zmax),  # 6
                (xmin, ymax, zmax),  # 7
            ]
            for v in verts:
                lines.append(f"v {v[0]} {v[1]} {v[2]}")

            # Kanten der Bounding Box als Linien
            edges = [
                # Untere Fläche (z = zmin)
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                # Obere Fläche (z = zmax)
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                # Vertikale Kanten
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for a, b in edges:
                lines.append(f"l {vert_count + a + 1} {vert_count + b + 1}")

            vert_count += 8
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"Fehler beim Schreiben der OBJ-Datei: {e}")
