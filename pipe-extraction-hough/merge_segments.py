import numpy as np


def mean_z_height(segment):
    # Z-Koordinaten
    z1 = segment[0, 2]
    z2 = segment[1, 2]

    # Mittlere Z-Höhe der beiden Punkte
    return (z1 + z2) / 2.0


def merge_segments_in_clusters(
    segments, clusters, gap_threshold, min_length, z_max=False
):
    if not segments:
        return []

    numpy_segments = np.asarray(segments, dtype=float)
    if (
        numpy_segments.ndim != 3
        or numpy_segments.shape[1] != 2
        or numpy_segments.shape[2] != 3
    ):
        raise ValueError(f"Only 3D segments are supported. {numpy_segments.shape}")

    result_segments = []

    # go through every cluster
    for cid, idx in clusters.items():
        idx = np.asarray(idx, dtype=int)
        if idx.size == 0:
            continue

        segments_of_cluster = numpy_segments[idx]
        points = segments_of_cluster.reshape(-1, 3)  # einfach liste aus Punkten

        # 1) Schwerpunkt (nur XY für Richtung)
        mean_xy = points[:, :2].mean(axis=0)

        # 2) Axiale Mittelrichtung aus XY
        v = segments_of_cluster[:, 1, :2] - segments_of_cluster[:, 0, :2]
        phi = np.mod(np.arctan2(v[:, 1], v[:, 0]), np.pi)
        c, s = np.cos(2 * phi).mean(), np.sin(2 * phi).mean()
        mean_phi = 0.5 * np.arctan2(s, c)
        u = np.array([np.cos(mean_phi), np.sin(mean_phi)])  # Einheitsrichtung in XY
        u_norm = np.linalg.norm(u)
        u = u / u_norm if u_norm > 0 else np.array([1.0, 0.0])

        segments_ready_to_merge = []

        for segment in segments_of_cluster:
            p1_xy = segment[0, :2]
            p2_xy = segment[1, :2]

            # Projektion auf Hauptgerade
            v1, v2 = p1_xy - mean_xy, p2_xy - mean_xy
            pos1, pos2 = np.dot(v1, u), np.dot(v2, u)

            left = min(pos1, pos2)
            right = max(pos1, pos2)

            segments_ready_to_merge.append([left, right, mean_z_height(segment)])

        # sort with min of pos1,pos2
        segments_ready_to_merge.sort(key=lambda x: x[0])

        final_cluster_segments_1d = []
        for left, right, z in segments_ready_to_merge:
            # First segment
            if len(final_cluster_segments_1d) == 0:
                final_cluster_segments_1d.append([left, right, z])
                continue

            # Segment inside current one
            if final_cluster_segments_1d[-1][1] >= max(left, right):
                continue

            # New segment
            if final_cluster_segments_1d[-1][1] + gap_threshold < min(left, right):
                final_cluster_segments_1d.append([left, right, z])
                continue

            # Merge with the last segment
            final_cluster_segments_1d[-1] = [
                final_cluster_segments_1d[-1][0],  # old left
                max(left, right),  # new right
                (
                    max(final_cluster_segments_1d[-1][2], z)
                    if z_max
                    else (final_cluster_segments_1d[-1][2] + z) / 2
                ),  # new z
            ]

        # filter out short segments and convert back to 3D world coordinates
        for seg in final_cluster_segments_1d:
            if seg[1] - seg[0] < min_length:
                continue

            # Convert back to world coordinates
            p1_world = mean_xy + seg[0] * u
            p2_world = mean_xy + seg[1] * u

            # Create 3D points with mean Z height
            p1_3d = np.array([p1_world[0], p1_world[1], seg[2]])
            p2_3d = np.array([p2_world[0], p2_world[1], seg[2]])

            result_segments.append([p1_3d, p2_3d])
            # if z_max:
            #     seg_idx = len(result_segments) - 1
            #     print(
            #         f"[merge_segments] Cluster {cid} -> finales Segment #{seg_idx}: {p1_3d} -> {p2_3d}"
            #     )

    return result_segments
