import numpy as np
import laspy
import pandas as pd
from pyntcloud import PyntCloud
import datetime
from PipeEvoSearch import find_one_pipe, initialize_globals
from PipeCluster import clean_pipes
import os


NUM_PIPES = 50  # Anzahl der zu suchenden Rohre
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
        print(f"Rohr {pipe_num + 1} konnte nicht gefunden werden.")
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
