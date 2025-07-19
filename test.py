import math
import numpy as np
import random
import laspy
import pandas as pd
from pyntcloud import PyntCloud
import datetime

# --- Laden der Punktwolke ---
las = laspy.read("./testPipeCloud.las")  # LAS-Datei laden
points = np.vstack(
    (las.x, las.y, las.z)
).T  # Nx3-Array aus den Koordinaten :contentReference[oaicite:6]{index=6}
cloud = PyntCloud(
    pd.DataFrame(points, columns=["x", "y", "z"])
)  # PyntCloud-Instanz :contentReference[oaicite:7]{index=7}

# --- KDTree für Nachbarschaftsanfragen ---
kdtree_id = cloud.add_structure(
    "kdtree"
)  # KDTree-Struktur erstellen :contentReference[oaicite:8]{index=8}
# Hinweis: PyntCloud bietet aktuell keinen Octree; KDTree ist jedoch sehr performant :contentReference[oaicite:9]{index=9}
tree = cloud.structures[kdtree_id]

# Parameter des evolutionären Algorithmus
POP_SIZE = 100
N_GENERATIONS = 100
RADIUS = 0.10  # 10 cm
HEIGHT_W = 1
SAMPLE_DISTANCE = 0.5  # 5m

ELITE_RATIO = 0.3  # 10% Elite direkt übernehmen
CROSSOVER_RATIO = 0.5  # 50% durch Crossover/Mutation
RANDOM_RATIO = 0.2  # 40% komplett neue Individuen

# Hilfsvariablen
pts = cloud.points[["x", "y", "z"]].values
# Für Höhennormierung
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
    for count in sample_counts:
        if count > median / 2:
            total_count += 1
        else:
            total_count -= 1

    return total_count


# --- Individuen-Definition ---
def create_individual():
    return (random.randrange(len(pts)), random.randrange(len(pts)))


# --- Fitness-Funktion ---
def fitness(ind):
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
    if random.random() < 0.5:
        # Komplett zufällig
        i = random.choice([0, 1])
        new_idx = random.randrange(len(pts))
        ind = list(ind)
        ind[i] = new_idx
        return tuple(ind)
    else:
        # Punkt aus Nachbarschaft
        i = random.choice([0, 1])
        idx = ind[i]
        neigh = tree.query_ball_point(pts[idx], r=RADIUS * 2)
        if len(neigh) > 1:
            new_idx = random.choice(neigh)
            ind = list(ind)
            ind[i] = new_idx
            return tuple(ind)
        return ind


# --- Rekombination (Crossover) ---
def crossover(p1, p2):
    # Tausche zufällig eine Achse
    if random.random() < 0.5:
        return (p1[0], p2[1])
    else:
        return (p2[0], p1[1])


# --- Evolutionäre Hauptschleife ---
history = []
population = [create_individual() for _ in range(POP_SIZE)]
for gen in range(N_GENERATIONS):
    print(f"Generation {gen+1}/{N_GENERATIONS}")
    # Fitness berechnen & Selektion
    scored = sorted(
        ((ind, fitness(ind)) for ind in population), key=lambda x: x[1], reverse=True
    )

    # Speichere die besten 10 Ergebnisse für die Historie
    top_10 = scored[:10]
    for rank, (ind, fit) in enumerate(top_10):
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

    # 1. Elite direkt übernehmen (10%)
    elite_count = int(POP_SIZE * ELITE_RATIO)
    elite = [ind for ind, _ in scored[:elite_count]]
    next_gen.extend(elite)

    # 2. Crossover/Mutation (50%)
    crossover_count = int(POP_SIZE * CROSSOVER_RATIO)
    parent_pool = [ind for ind, _ in scored[: POP_SIZE // 2]]  # Beste 50% als Eltern

    for _ in range(crossover_count):
        pa, pb = random.sample(parent_pool, 2)
        child = crossover(pa, pb)
        child = mutate(child)
        next_gen.append(child)

    # 3. Komplett neue Individuen (40%)
    random_count = POP_SIZE - len(next_gen)  # Rest auffüllen
    for _ in range(random_count):
        next_gen.append(create_individual())

    population = next_gen

# --- Ergebnis ausgeben ---
best_ind, best_fit = max(
    ((ind, fitness(ind)) for ind in population), key=lambda x: x[1]
)
p1, p2 = pts[best_ind[0]], pts[best_ind[1]]
print("Best ray endpoints:", p1, p2)
print("Best fitness:", best_fit)

# write history to CSV
history_df = pd.DataFrame(history)
# history_df.to_csv("evolution_history.csv", index=False)

# Schreibe ALLE Vektoren der History in die OBJ-Datei
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
with open(f"./output/{timestamp}_best_pipe_endpoints.obj", "w") as f:
    top_3_history = history_df.head(3)
    for idx, row in top_3_history.iterrows():
        f.write(f"v {row['p1_x']} {row['p1_y']} {row['p1_z']}\n")
        f.write(f"v {row['p2_x']} {row['p2_y']} {row['p2_z']}\n")
        f.write("l -1 -2\n")
