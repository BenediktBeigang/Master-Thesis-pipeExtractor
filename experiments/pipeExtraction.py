import datetime
from PipeEvoSearch import find_one_pipe, initialize_globals
from experiments.PipeCluster import clean_pipes
import os
import json
import math


NUM_PIPES = 100  # Anzahl der zu suchenden Rohre
found_pipes = []  # Globale Datenstruktur für gefundene Rohre

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

initialize_globals("./test.las")  # Initialisiere globale Variablen

# --- Hauptschleife für mehrere Rohre ---
for pipe_num in range(NUM_PIPES):
    print(f"=== Suche nach Rohr {pipe_num + 1}/{NUM_PIPES} ===")
    best_pipe = find_one_pipe(
        pipeIndex=pipe_num,
        generation_count=100,
        populationSize=20,
        eliteRatio=0.3,
        crossoverRatio=0.5,
        foundPipes=found_pipes,
    )
    if best_pipe is None:
        continue
    found_pipes.append(best_pipe)

# Clustering durchführen
optimized_pipes = clean_pipes(found_pipes)

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
os.makedirs("./output", exist_ok=True)
with open(f"./output/{timestamp}_all_pipes.obj", "w") as f:
    for pipe in optimized_pipes:
        f.write(f"v {pipe['p1_x']} {pipe['p1_y']} {pipe['p1_z']}\n")
        f.write(f"v {pipe['p2_x']} {pipe['p2_y']} {pipe['p2_z']}\n")
        f.write("l -1 -2\n")

# JSON-Export mit Vektoren und Längen
json_data = []
for i, pipe in enumerate(optimized_pipes):
    # Berechne die Länge des Rohrs
    length = math.sqrt(
        (pipe["p2_x"] - pipe["p1_x"]) ** 2
        + (pipe["p2_y"] - pipe["p1_y"]) ** 2
        + (pipe["p2_z"] - pipe["p1_z"]) ** 2
    )

    pipe_data = {
        "id": i + 1,
        "point1": {"x": pipe["p1_x"], "y": pipe["p1_y"], "z": pipe["p1_z"]},
        "point2": {"x": pipe["p2_x"], "y": pipe["p2_y"], "z": pipe["p2_z"]},
        "length": length,
        "fitness": pipe["fitness"],
        "original_pipes": pipe.get("original_pipes", 1),
    }
    json_data.append(pipe_data)

# JSON-Datei speichern
with open(f"./output/{timestamp}_all_pipes.json", "w") as f:
    json.dump(json_data, f, indent=2)

print(f"\nDateien gespeichert:")
print(f"- OBJ: ./output/{timestamp}_all_pipes.obj")
print(f"- JSON: ./output/{timestamp}_all_pipes.json")
