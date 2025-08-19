import math
import numpy as np
import random
from individualValidation import is_individual_valid
import laspy
from pyntcloud import PyntCloud
import pandas as pd


# --- KDTree für Nachbarschaftsanfragen ---
kdtree_id = None
tree = None
pts = None

# Globale Variablen in der Punktwolke
z_min = None
z_max = None
z_range = None

NEIGHBORHOOD_RADIUS = 0.10  # 10 cm
SAMPLE_DISTANCE = 0.5  # 50cm
HEIGHT_WEIGHT = 1  # Gewichtung der Höhe in der Fitness-Funktion


def initialize_globals(pointcloudPath: str):
    global kdtree_id, tree, pts, z_min, z_max, z_range

    # --- Laden der Punktwolke ---
    las = laspy.read(pointcloudPath)  # LAS-Datei laden
    points = np.vstack((las.x, las.y, las.z)).T  # type: ignore
    cloud = PyntCloud(
        pd.DataFrame(points, columns=["x", "y", "z"])
    )  # PyntCloud-Instanz

    # --- KDTree für Nachbarschaftsanfragen ---
    kdtree_id = cloud.add_structure("kdtree")  # KDTree-Struktur erstellen
    tree = cloud.structures[kdtree_id]
    pts = cloud.points[["x", "y", "z"]].values

    # Globale Variablen in der Punktwolke
    z_min = np.min(pts[:, 2])
    z_max = np.max(pts[:, 2])
    z_range = z_max - z_min


def count_points_sampled_along_segment_bool(p1, p2, radius, sample_distance):
    """
    Abtastung entlang des Segments alle sample_distance Meter.
    An jedem Abtastpunkt werden Nachbarn im Umkreis von radius gezählt.
    Falls die Anzahl der Nachbarn größer als der halbe Median der Abtastpunkte ist,
    wird der Zähler inkrementiert, sonst dekrementiert.
    Nutzt KDTree für effiziente Nachbarschaftssuche.
    """
    if tree is None:
        raise ValueError(
            "KDTree ist nicht initialisiert. Bitte initialize_globals aufrufen."
        )

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
        t = min(i * sample_distance, float(L))  # Position entlang des Segments
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
def create_individual(pts, found_pipes):
    max_attempts = 100
    for _ in range(max_attempts):
        ind = (random.randrange(len(pts)), random.randrange(len(pts)))
        if is_individual_valid(ind, found_pipes, pts):
            return ind
    # Falls nach max_attempts kein gültiges Individuum gefunden wurde, nehme das letzte
    return ind


# --- Fitness-Funktion ---
def fitness(ind, found_pipes, pts):
    if z_min is None or z_range is None:
        raise ValueError(
            "KDTree ist nicht initialisiert. Bitte initialize_globals aufrufen."
        )

    # Ungültige Individuen erhalten sehr schlechte Fitness
    if not is_individual_valid(ind, found_pipes, pts):
        return -1000

    p1, p2 = pts[ind[0]], pts[ind[1]]

    count_near = count_points_sampled_along_segment_bool(
        p1, p2, NEIGHBORHOOD_RADIUS, SAMPLE_DISTANCE
    )

    avg_height = np.mean([p1[2], p2[2]])
    normalized_height = (
        ((avg_height - z_min) / z_range) * HEIGHT_WEIGHT if z_range > 0 else 0
    )
    return count_near + normalized_height


# --- Mutation ---
def mutate(ind, foundPipes, pts):
    if tree is None:
        raise ValueError(
            "KDTree ist nicht initialisiert. Bitte initialize_globals aufrufen."
        )

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
            neigh = tree.query_ball_point(pts[idx], r=NEIGHBORHOOD_RADIUS * 2)
            if len(neigh) > 1:
                new_idx = random.choice(neigh)
                new_ind = list(ind)
                new_ind[i] = new_idx
                new_ind = tuple(new_ind)
            else:
                new_ind = ind

        if is_individual_valid(new_ind, foundPipes, pts):
            return new_ind

    return ind  # Falls keine gültige Mutation gefunden wurde


# --- Rekombination (Crossover) ---
def crossover(p1, p2, foundPipes, pts):
    # Tausche zufällig eine Achse
    if random.random() < 0.5:
        child = (p1[0], p2[1])
    else:
        child = (p2[0], p1[1])

    # Prüfe Gültigkeit, falls ungültig, nehme einen Elternteil
    if not is_individual_valid(child, foundPipes, pts):
        return p1 if random.random() < 0.5 else p2

    return child


def find_one_pipe(
    pipeIndex: int,
    generation_count: int,
    populationSize,
    eliteRatio,
    crossoverRatio,
    foundPipes,
):
    if pts is None:
        raise ValueError(
            "KDTree ist nicht initialisiert. Bitte initialize_globals aufrufen."
        )

    history = []
    population = [create_individual(pts, foundPipes) for _ in range(populationSize)]

    for gen in range(generation_count):
        # Fitness berechnen & Selektion
        scored = sorted(
            ((ind, fitness(ind, foundPipes, pts)) for ind in population),
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
        elite_count = int(populationSize * eliteRatio)
        elite = [ind for ind, fit in scored[:elite_count] if fit > -1000]
        next_gen.extend(elite)

        # 2. Crossover/Mutation
        crossover_count = int(populationSize * crossoverRatio)
        valid_parents = [ind for ind, fit in scored if fit > -1000]
        parent_pool = valid_parents[: min(len(valid_parents), populationSize // 2)]

        for _ in range(crossover_count):
            if len(parent_pool) >= 2:
                pa, pb = random.sample(parent_pool, 2)
                child = crossover(pa, pb, foundPipes, pts)
                child = mutate(child, foundPipes, pts)
                next_gen.append(child)
            else:
                next_gen.append(create_individual(pts, foundPipes))

        # 3. Komplett neue Individuen
        random_count = populationSize - len(next_gen)
        for _ in range(random_count):
            next_gen.append(create_individual(pts, foundPipes))

        population = next_gen

    # --- Bestes Ergebnis dieser Iteration ---
    valid_scored = [(ind, fit) for ind, fit in scored if fit > -1000]

    if not valid_scored:
        return None

    best_ind, best_fit = valid_scored[0]
    p1, p2 = pts[best_ind[0]], pts[best_ind[1]]

    # Bestes Ergebnis zu gefundenen Rohren hinzufügen
    best_pipe = {
        "pipe_number": pipeIndex + 1,
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

    return best_pipe
