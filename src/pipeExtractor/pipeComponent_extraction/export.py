from custom_types import PipeComponentArray


def write_obj_pipeComponents(components: PipeComponentArray, output_path):
    try:
        lines = []
        vert_count = 0
        for mins, maxs, centroid in components:
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

            centroid_index = vert_count + len(verts) + 1
            lines.append(f"v {centroid[0]} {centroid[1]} {centroid[2]}")
            lines.append(f"p {centroid_index}")

            eps = 0.01
            cross_verts = [
                (centroid[0] - eps, centroid[1], centroid[2]),
                (centroid[0] + eps, centroid[1], centroid[2]),
                (centroid[0], centroid[1] - eps, centroid[2]),
                (centroid[0], centroid[1] + eps, centroid[2]),
                (centroid[0], centroid[1], centroid[2] - eps),
                (centroid[0], centroid[1], centroid[2] + eps),
            ]
            for v in cross_verts:
                lines.append(f"v {v[0]} {v[1]} {v[2]}")
            cross_start = centroid_index + 1
            cross_edges = [(0, 1), (2, 3), (4, 5)]
            for a, b in cross_edges:
                lines.append(f"l {cross_start + a} {cross_start + b}")

            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for a, b in edges:
                lines.append(f"l {vert_count + a + 1} {vert_count + b + 1}")

            vert_count += len(verts) + 1 + len(cross_verts)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
    except Exception as e:
        print(f"Fehler beim Schreiben der OBJ-Datei: {e}")
