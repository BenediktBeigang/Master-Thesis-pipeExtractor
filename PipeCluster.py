import numpy as np


# --- Zweistufiges Clustering der gefundenen Rohre ---
def normalize_vector(v):
    """Normalisiert einen Vektor"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def get_pipe_vector(pipe):
    """Berechnet den normalisierten Richtungsvektor eines Rohres"""
    p1 = np.array([pipe["p1_x"], pipe["p1_y"], pipe["p1_z"]])
    p2 = np.array([pipe["p2_x"], pipe["p2_y"], pipe["p2_z"]])
    return normalize_vector(p2 - p1)


def get_pipe_midpoint(pipe):
    """Berechnet den Mittelpunkt eines Rohres"""
    p1 = np.array([pipe["p1_x"], pipe["p1_y"], pipe["p1_z"]])
    p2 = np.array([pipe["p2_x"], pipe["p2_y"], pipe["p2_z"]])
    return (p1 + p2) / 2


def vector_angle_similarity(v1, v2):
    """Berechnet die Ähnlichkeit zweier Vektoren basierend auf dem Winkel"""
    dot_product = np.abs(np.dot(v1, v2))  # Absolut, da Richtung egal ist
    return dot_product  # cos(angle), 1 = parallel, 0 = orthogonal


def orthogonal_distance_to_line(point, line_point, line_vector):
    """Berechnet die orthogonale Distanz eines Punktes zu einer Linie"""
    to_point = point - line_point
    projection_length = np.dot(to_point, line_vector)
    projection = projection_length * line_vector
    orthogonal = to_point - projection
    return np.linalg.norm(orthogonal)


def cluster_pipes_by_vector(pipes, angle_threshold=0.9):
    """Clustert Rohre basierend auf Vektorähnlichkeit"""
    if not pipes:
        return []

    clusters = []
    used = set()

    for i, pipe in enumerate(pipes):
        if i in used:
            continue

        cluster = [pipe]
        used.add(i)
        pipe_vector = get_pipe_vector(pipe)

        for j, other_pipe in enumerate(pipes):
            if j in used:
                continue

            other_vector = get_pipe_vector(other_pipe)
            similarity = vector_angle_similarity(pipe_vector, other_vector)

            if similarity >= angle_threshold:
                cluster.append(other_pipe)
                used.add(j)

        clusters.append(cluster)

    return clusters


def cluster_by_orthogonal_distance(pipes, distance_threshold=2.0):
    """Clustert Rohre basierend auf orthogonaler Distanz zum ersten Rohr"""
    if len(pipes) <= 1:
        return [pipes]

    clusters = []
    used = set()

    for i, reference_pipe in enumerate(pipes):
        if i in used:
            continue

        cluster = [reference_pipe]
        used.add(i)

        ref_midpoint = get_pipe_midpoint(reference_pipe)
        ref_vector = get_pipe_vector(reference_pipe)

        for j, other_pipe in enumerate(pipes):
            if j in used:
                continue

            other_midpoint = get_pipe_midpoint(other_pipe)
            distance = orthogonal_distance_to_line(
                other_midpoint, ref_midpoint, ref_vector
            )

            if distance <= distance_threshold:
                cluster.append(other_pipe)
                used.add(j)

        clusters.append(cluster)

    return clusters


def find_furthest_points_in_cluster(cluster):
    """Findet das Punktpaar mit der größten Distanz in einem Cluster"""
    if len(cluster) <= 1:
        return cluster[0] if cluster else None

    all_points = []
    for pipe in cluster:
        p1 = np.array([pipe["p1_x"], pipe["p1_y"], pipe["p1_z"]])
        p2 = np.array([pipe["p2_x"], pipe["p2_y"], pipe["p2_z"]])
        all_points.extend([p1, p2])

    max_distance = 0
    best_pair = None

    for i, p1 in enumerate(all_points):
        for j, p2 in enumerate(all_points[i + 1 :], i + 1):
            distance = np.linalg.norm(p2 - p1)
            if distance > max_distance:
                max_distance = distance
                best_pair = (p1, p2)

    if best_pair is None:
        return cluster[0]

    # Erstelle ein neues Rohr-Objekt mit den weitesten Punkten
    p1, p2 = best_pair
    # Berechne Fitness basierend auf dem ersten Rohr im Cluster
    avg_fitness = np.mean([pipe["fitness"] for pipe in cluster])

    optimized_pipe = {
        "pipe_number": cluster[0]["pipe_number"],
        "individual_0": -1,  # Nicht mehr gültig nach Optimierung
        "individual_1": -1,  # Nicht mehr gültig nach Optimierung
        "fitness": avg_fitness,
        "p1_x": float(p1[0]),
        "p1_y": float(p1[1]),
        "p1_z": float(p1[2]),
        "p2_x": float(p2[0]),
        "p2_y": float(p2[1]),
        "p2_z": float(p2[2]),
        "cluster_size": len(cluster),
        "original_pipes": len(cluster),
    }

    return optimized_pipe


def clean_pipes(foundPipes):
    # Clustering durchführen
    print(f"\n=== Clustering von {len(foundPipes)} gefundenen Rohren ===")

    # Stufe 1: Clustering nach Vektorähnlichkeit
    vector_clusters = cluster_pipes_by_vector(foundPipes, angle_threshold=0.95)
    print(f"Nach Vektor-Clustering: {len(vector_clusters)} Cluster")

    # Stufe 2: Clustering nach orthogonaler Distanz
    final_clusters = []
    for i, vector_cluster in enumerate(vector_clusters):
        print(f"Cluster {i+1} hat {len(vector_cluster)} Rohre")
        distance_clusters = cluster_by_orthogonal_distance(
            vector_cluster, distance_threshold=1.0
        )
        final_clusters.extend(distance_clusters)

    print(f"Nach Distanz-Clustering: {len(final_clusters)} finale Cluster")

    # Stufe 3: Weiteste Punkte in jedem Cluster finden und zu einem Rohr zusammenfassen
    optimized_pipes = []
    for i, cluster in enumerate(final_clusters):
        print(f"Finale Cluster {i+1}: {len(cluster)} Rohre")
        optimized_pipe = find_furthest_points_in_cluster(cluster)
        if optimized_pipe:
            optimized_pipes.append(optimized_pipe)

    return optimized_pipes
