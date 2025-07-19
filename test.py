import math
import numpy as np
import random
import laspy
import pandas as pd
from pyntcloud import PyntCloud
import datetime

# --- Laden der Punktwolke ---
las = laspy.read("./testPipeCloud.las")  # LAS-Datei laden
points = np.vstack((las.x, las.y, las.z)).T  # Nx3-Array aus den Koordinaten
cloud = PyntCloud(pd.DataFrame(points, columns=["x", "y", "z"]))  # PyntCloud-Instanz

# --- KDTree für Nachbarschaftsanfragen ---
kdtree_id = cloud.add_structure("kdtree")  # KDTree-Struktur erstellen
tree = cloud.structures[kdtree_id]

# Parameter des evolutionären Algorithmus
POP_SIZE = 20
N_GENERATIONS = 100
RADIUS = 0.10  # 10 cm
HEIGHT_W = 1
SAMPLE_DISTANCE = 0.5  # 5m
BOUNDING_BOX_DISTANCE = 1  # 50cm Abstand für Bounding Box

ELITE_RATIO = 0.3  # 30% Elite direkt übernehmen
CROSSOVER_RATIO = 0.5  # 50% durch Crossover/Mutation
RANDOM_RATIO = 0.2  # 20% komplett neue Individuen

# Parameter für Mehrfachsuche
NUM_PIPES = 50  # Anzahl der zu suchenden Rohre

# Hilfsvariablen
pts = cloud.points[["x", "y", "z"]].values
# Für Höhennormierung
z_min = np.min(pts[:, 2])
z_max = np.max(pts[:, 2])
z_range = z_max - z_min

# Globale Datenstruktur für gefundene Rohre
found_pipes = []


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


def is_point_in_pipe_bounding_box(point, pipe_p1, pipe_p2, distance):
    """
    Prüft ob ein Punkt innerhalb der Bounding Box eines Rohres liegt.
    Die Bounding Box ist der Bereich um das Rohrsegment mit dem gegebenen Abstand.
    """
    return point_to_line_distance(point, pipe_p1, pipe_p2) <= distance


def is_individual_valid(ind):
    """
    Prüft ob ein Individuum gültig ist (nicht beide Punkte in derselben Pipe-Bounding-Box).
    """
    if len(found_pipes) == 0:
        return True

    p1, p2 = pts[ind[0]], pts[ind[1]]

    for pipe in found_pipes:
        pipe_p1 = np.array([pipe["p1_x"], pipe["p1_y"], pipe["p1_z"]])
        pipe_p2 = np.array([pipe["p2_x"], pipe["p2_y"], pipe["p2_z"]])

        # Prüfe ob BEIDE Punkte des Individuums in der Bounding Box dieser Pipe sind
        p1_in_box = is_point_in_pipe_bounding_box(
            p1, pipe_p1, pipe_p2, BOUNDING_BOX_DISTANCE
        )
        p2_in_box = is_point_in_pipe_bounding_box(
            p2, pipe_p1, pipe_p2, BOUNDING_BOX_DISTANCE
        )

        if p1_in_box and p2_in_box:
            return False  # Ungültig, da beide Punkte in derselben Pipe-Box

    return True


def count_points_sampled_along_segment_bool(p1, p2, radius, sample_distance):
    """
    Abtastung entlang des Segments alle sample_distance Meter.
    An jedem Abtastpunkt werden Nachbarn im Umkreis von radius gezählt.
    Falls die Anzahl der Nachbarn größer als der halbe Median der Abtastpunkte ist,
    wird der Zähler inkrementiert, sonst dekrementiert.
    Nutzt KDTree für effiziente Nachbarschaftssuche.
    """
    # Richtungsvektor und Länge
    v = p2 - p1
    L = np.linalg.norm(v)
    if L == 0:
        return 0

    v_norm = v / L

    # Abtastpunkte entlang des Segments berechnen
    num_samples = int(L / sample_distance) + 1

    # Für jeden Abtastpunkt Nachbarn zählen
    sample_counts = []
    for i in range(num_samples):
        t = min(i * sample_distance, L)  # Position entlang des Segments
        sample_point = p1 + t * v_norm

        # KDTree verwenden für effiziente Nachbarschaftssuche
        neighbors = tree.query_ball_point(sample_point, r=radius)
        sample_counts.append(len(neighbors))

    # median bestimmen
    sample_counts.sort()
    mid_index = math.floor(len(sample_counts) // 2)
    median = sample_counts[mid_index]

    # gehe jeden sample_count durch und inkrementiere total_count, wenn größer als der halbe median und dekrementiere, wenn kleiner
    total_count = 0
    highest_empty_sequence = 0
    max_highest_empty_sequence_max = 0
    for count in sample_counts:
        if count > median / 2:
            total_count += 1
            highest_empty_sequence = 0
        else:
            total_count -= 1
            highest_empty_sequence += 1
            max_highest_empty_sequence_max = max(
                max_highest_empty_sequence_max, highest_empty_sequence
            )

    # wenn die höchste leere Sequenz größer als 1m ist gebe -1000 zurück
    if max_highest_empty_sequence_max * sample_distance > 2.0:
        return -1000
    return total_count


# --- Individuen-Definition ---
def create_individual():
    max_attempts = 100
    for _ in range(max_attempts):
        ind = (random.randrange(len(pts)), random.randrange(len(pts)))
        if is_individual_valid(ind):
            return ind
    # Falls nach max_attempts kein gültiges Individuum gefunden wurde, nehme das letzte
    return ind


# --- Fitness-Funktion ---
def fitness(ind):
    # Ungültige Individuen erhalten sehr schlechte Fitness
    if not is_individual_valid(ind):
        return -1000

    p1, p2 = pts[ind[0]], pts[ind[1]]

    count_near = count_points_sampled_along_segment_bool(
        p1, p2, RADIUS, SAMPLE_DISTANCE
    )

    avg_height = np.mean([p1[2], p2[2]])
    normalized_height = (
        ((avg_height - z_min) / z_range) * HEIGHT_W if z_range > 0 else 0
    )
    return count_near + normalized_height


# --- Mutation ---
def mutate(ind):
    max_attempts = 10
    for _ in range(max_attempts):
        if random.random() < 0.5:
            # Komplett zufällig
            i = random.choice([0, 1])
            new_idx = random.randrange(len(pts))
            new_ind = list(ind)
            new_ind[i] = new_idx
            new_ind = tuple(new_ind)
        else:
            # Punkt aus Nachbarschaft
            i = random.choice([0, 1])
            idx = ind[i]
            neigh = tree.query_ball_point(pts[idx], r=RADIUS * 2)
            if len(neigh) > 1:
                new_idx = random.choice(neigh)
                new_ind = list(ind)
                new_ind[i] = new_idx
                new_ind = tuple(new_ind)
            else:
                new_ind = ind

        if is_individual_valid(new_ind):
            return new_ind

    return ind  # Falls keine gültige Mutation gefunden wurde


# --- Rekombination (Crossover) ---
def crossover(p1, p2):
    # Tausche zufällig eine Achse
    if random.random() < 0.5:
        child = (p1[0], p2[1])
    else:
        child = (p2[0], p1[1])

    # Prüfe Gültigkeit, falls ungültig, nehme einen Elternteil
    if not is_individual_valid(child):
        return p1 if random.random() < 0.5 else p2

    return child


# --- Hauptschleife für mehrere Rohre ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

for pipe_num in range(NUM_PIPES):
    print(f"=== Suche nach Rohr {pipe_num + 1}/{NUM_PIPES} ===")

    # --- Evolutionäre Hauptschleife ---
    history = []
    population = [create_individual() for _ in range(POP_SIZE)]

    for gen in range(N_GENERATIONS):
        # Fitness berechnen & Selektion
        scored = sorted(
            ((ind, fitness(ind)) for ind in population),
            key=lambda x: x[1],
            reverse=True,
        )

        # Speichere die besten 10 Ergebnisse für die Historie
        top_10 = scored[:10]
        for rank, (ind, fit) in enumerate(top_10):
            if fit <= -1000:  # Überspringe ungültige Individuen
                continue

            p1_gen, p2_gen = pts[ind[0]], pts[ind[1]]

            # Prüfe ob diese Kombination bereits in der History ist
            new_entry = {
                "generation": gen + 1,
                "rank": rank + 1,
                "individual_0": ind[0],
                "individual_1": ind[1],
                "fitness": fit,
                "p1_x": float(p1_gen[0]),
                "p1_y": float(p1_gen[1]),
                "p1_z": float(p1_gen[2]),
                "p2_x": float(p2_gen[0]),
                "p2_y": float(p2_gen[1]),
                "p2_z": float(p2_gen[2]),
            }

            # Prüfe auf Duplikate (beide Richtungen des Segments)
            is_duplicate = False
            for existing in history:
                # Prüfe ob gleiche Punkte (in beide Richtungen)
                if (
                    existing["individual_0"] == ind[0]
                    and existing["individual_1"] == ind[1]
                ) or (
                    existing["individual_0"] == ind[1]
                    and existing["individual_1"] == ind[0]
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                history.append(new_entry)

        # History auf die besten 20 begrenzen
        history = sorted(history, key=lambda x: x["fitness"], reverse=True)[:20]

        next_gen = []

        # 1. Elite direkt übernehmen
        elite_count = int(POP_SIZE * ELITE_RATIO)
        elite = [ind for ind, fit in scored[:elite_count] if fit > -1000]
        next_gen.extend(elite)

        # 2. Crossover/Mutation
        crossover_count = int(POP_SIZE * CROSSOVER_RATIO)
        valid_parents = [ind for ind, fit in scored if fit > -1000]
        parent_pool = valid_parents[: min(len(valid_parents), POP_SIZE // 2)]

        for _ in range(crossover_count):
            if len(parent_pool) >= 2:
                pa, pb = random.sample(parent_pool, 2)
                child = crossover(pa, pb)
                child = mutate(child)
                next_gen.append(child)
            else:
                next_gen.append(create_individual())

        # 3. Komplett neue Individuen
        random_count = POP_SIZE - len(next_gen)
        for _ in range(random_count):
            next_gen.append(create_individual())

        population = next_gen

    # --- Bestes Ergebnis dieser Iteration ---
    valid_scored = [(ind, fit) for ind, fit in scored if fit > -1000]
    if valid_scored:
        best_ind, best_fit = valid_scored[0]
        p1, p2 = pts[best_ind[0]], pts[best_ind[1]]

        # Bestes Ergebnis zu gefundenen Rohren hinzufügen
        best_pipe = {
            "pipe_number": pipe_num + 1,
            "individual_0": best_ind[0],
            "individual_1": best_ind[1],
            "fitness": best_fit,
            "p1_x": float(p1[0]),
            "p1_y": float(p1[1]),
            "p1_z": float(p1[2]),
            "p2_x": float(p2[0]),
            "p2_y": float(p2[1]),
            "p2_z": float(p2[2]),
        }
        found_pipes.append(best_pipe)

        # print(f"Rohr {pipe_num + 1} - Beste Endpunkte:", p1, p2)
        # print(f"Rohr {pipe_num + 1} - Beste Fitness:", best_fit)

        # History für dieses Rohr speichern
        if history:
            history_df = pd.DataFrame(history)
            history_df.to_csv(
                # f"./output/{timestamp}_pipe_{pipe_num + 1}_evolution_history.csv",
                index=False,
            )
    else:
        print(f"Kein gültiges Rohr {pipe_num + 1} gefunden!")


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


# Clustering durchführen
print(f"\n=== Clustering von {len(found_pipes)} gefundenen Rohren ===")

# Stufe 1: Clustering nach Vektorähnlichkeit
vector_clusters = cluster_pipes_by_vector(found_pipes, angle_threshold=0.95)
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

# Optimierung: Weiteste Punkte in jedem Cluster finden
optimized_pipes = []
for i, cluster in enumerate(final_clusters):
    print(f"Finale Cluster {i+1}: {len(cluster)} Rohre")
    optimized_pipe = find_furthest_points_in_cluster(cluster)
    if optimized_pipe:
        optimized_pipes.append(optimized_pipe)

# --- Finale Ausgabe ---
print(
    f"\n=== Zusammenfassung: {len(optimized_pipes)} optimierte Rohre nach Clustering ==="
)
for i, pipe in enumerate(optimized_pipes):
    original_count = pipe.get("original_pipes", 1)
    print(
        f"Rohr {i+1}: Fitness={pipe['fitness']:.2f}, zusammengefasst aus {original_count} ursprünglichen Rohren"
    )

# Alle optimierten Rohre in OBJ-Datei schreiben
with open(f"./output/{timestamp}_all_pipes.obj", "w") as f:
    for pipe in optimized_pipes:
        f.write(f"v {pipe['p1_x']} {pipe['p1_y']} {pipe['p1_z']}\n")
        f.write(f"v {pipe['p2_x']} {pipe['p2_y']} {pipe['p2_z']}\n")
        f.write("l -1 -2\n")

# Optimierte Rohre als CSV speichern
if optimized_pipes:
    pipes_df = pd.DataFrame(optimized_pipes)
    # pipes_df.to_csv(f"./output/{timestamp}_optimized_pipes.csv", index=False)

# Auch die ursprünglichen Rohre speichern für Vergleich
if found_pipes:
    original_pipes_df = pd.DataFrame(found_pipes)
    # original_pipes_df.to_csv(f"./output/{timestamp}_original_pipes.csv", index=False)
